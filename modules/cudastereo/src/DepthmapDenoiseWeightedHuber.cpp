/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

//! OpenDTAM Variant of Chambolle & Pock denoising
//!
//! The complicated half of the DTAM algorithm's mapping core,
//! but can be used independently to refine depthmaps.
//!
//! Written by Paul Foster for GSoC 2014 OpenDTAM project.
//! High level algorithm described by Richard Newcombe, Steven J. Lovegrove, and Andrew J. Davison. 
//! "DTAM: Dense tracking and mapping in real-time."
//! Which was in turn based on Chambolle & Pock's
//! "A first-order primal-dual algorithm for convex problems with applications to imaging."


#include "cuda/DepthmapDenoiseWeightedHuber.cuh"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/core.hpp"


using namespace std;
using namespace cv;
using namespace cv::cuda;
namespace cv{
    namespace cuda{
        class DepthmapDenoiseWeightedHuberImpl : public DepthmapDenoiseWeightedHuber
        {
        public:
            //CostVolume cv;//The cost volume we are attached to
            
            DepthmapDenoiseWeightedHuberImpl(const cv::cuda::GpuMat& visibleLightImage=cv::cuda::GpuMat(),cv::cuda::Stream cvStream=cv::cuda::Stream::Null());
            cv::cuda::GpuMat operator()(InputArray ain,
                                        float epsilon,
                                        float theta);

            cv::cuda::GpuMat visibleLightImage;
            //buffers
            cv::cuda::GpuMat _qx,_qy,_d,_a,_g,_g1,_gx,_gy;
            cv::cuda::GpuMat stableDepth;


            //in case you want to do these explicitly
            void allocate(int rows,int cols, InputArray gxin=cv::cuda::GpuMat(), InputArray gyin=cv::cuda::GpuMat());
            void cacheGValues();

        private:
            int rows;
            int cols;

            void computeSigmas(float epsilon,float theta);

            //internal parameter values
            float sigma_d,sigma_q;

            //flags
            bool cachedG;
            int alloced;
            int dInited;

        public:
            cv::cuda::Stream cvStream;
        }; 

        DepthmapDenoiseWeightedHuberImpl::DepthmapDenoiseWeightedHuberImpl(const cv::cuda::GpuMat& _visibleLightImage,
                                                                Stream _cvStream) : 
                                                                visibleLightImage(_visibleLightImage), 
                                                                rows(_visibleLightImage.rows),
                                                                cols(_visibleLightImage.cols),
                                                                cvStream(_cvStream)
        {
            alloced=0;
            cachedG=0; 
            dInited=0;
        }

        Ptr<DepthmapDenoiseWeightedHuber>
        CV_EXPORTS createDepthmapDenoiseWeightedHuber(InputArray visibleLightImage, Stream cvStream){
            return Ptr<DepthmapDenoiseWeightedHuber>(new DepthmapDenoiseWeightedHuberImpl(visibleLightImage.getGpuMat(),cvStream));
        };
    }
}






using namespace std;
using namespace cv::cuda;

#define FLATALLOC(n,cv) n.create(1,cv.rows*cv.cols, CV_32FC1);n=n.reshape(0,cv.rows)
static void memZero(GpuMat& in,Stream& cvStream){
    cudaSafeCall(cudaMemsetAsync(in.data,0,in.rows*in.cols*sizeof(float),cv::cuda::StreamAccessor::getStream(cvStream)));
}

void DepthmapDenoiseWeightedHuberImpl::allocate(int _rows,int _cols,InputArray _gxin,InputArray _gyin){
    const GpuMat& gxin=_gxin.getGpuMat();
    const GpuMat& gyin=_gyin.getGpuMat();
    
    rows=_rows;
    cols=_cols;
    if(!(rows % 32 == 0 && cols % 32 == 0 && cols >= 64)){
        CV_Assert(!"For performance reasons, DepthmapDenoiseWeightedHuber currenty only supports multiple of 32 image sizes with cols >= 64. Pad the image to achieve this.");
    }
    

    if(!_a.data){
        _a.create(1,rows*cols, CV_32FC1);
        _a=_a.reshape(0,rows);
    }
    FLATALLOC(_d, _a);
    cachedG=1;
    if(gxin.empty()||gyin.empty()){
        if(gxin.empty()){
            FLATALLOC(_gx,_d);
            cachedG=0;
        }else{
            _gx=gxin;
        }
        if(gyin.empty()){
            FLATALLOC(_gy,_d);
            cachedG=0;
        }else{
            _gy=gyin;
        }
    }else{
        
        if(!gxin.isContinuous()){
            FLATALLOC(_gx,_d);
            gxin.copyTo(_gx,cvStream);
        }
        if(!gyin.isContinuous()){
            FLATALLOC(_gy,_d);
            gyin.copyTo(_gy,cvStream);
        }
    }
    FLATALLOC(_qx, _d);
    FLATALLOC(_qy, _d);
    FLATALLOC(_g1, _d);
    FLATALLOC(stableDepth,_d);
    memZero(_qx,cvStream);
    memZero(_qy,cvStream);
    alloced=1;
}


void DepthmapDenoiseWeightedHuberImpl::computeSigmas(float epsilon,float theta){
    /*
    //This function is my best guess of what was meant by the line:
    //"Gradient ascent/descent time-steps sigma_q , sigma_d are set optimally
    //for the update scheme provided as detailed in [3]."
    // Where [3] is :
    //A. Chambolle and T. Pock. A first-order primal-dual 
    //algorithm for convex problems with applications to imaging.
    //Journal of Mathematical Imaging and Vision, 40(1):120–
    //145, 2011. 3, 4, 6
    //
    // I converted these mechanically to the best of my ability, but no 
    // explaination is given in [3] as to how they came up with these, just 
    // some proofs beyond my ability.
    //
    // Explainations below are speculation, but I think good ones:
    //
    // L appears to be a bound on the largest vector length that can be 
    // produced by the linear operator from a unit vector. In this case the 
    // linear operator is the differentiation matrix with G weighting 
    // (written AG in the DTAM paper,(but I use GA because we want to weight 
    // the springs rather than the pixels)). Since G has each row sum < 1 and 
    // A is a forward difference matrix (which has each input affecting at most
    // 2 outputs via pairs of +-1 weights) the value is bounded by 4.0.
    //
    // So in a sense, L is an upper bound on the magnification of our step size.
    // 
    // Lambda and alpha are called convexity parameters. They come from the 
    // Huber norm and the (d-a)^2 terms. The convexity parameter of a function 
    // is defined as follows: 
    //  Choose a point on the function and construct a parabola of convexity 
    //    c tangent at that point. Call the point c-convex if the parabola is 
    //    above the function at all other points. 
    //  The smallest c such that the function is c-convex everywhere is the 
    //      convexity parameter.
    //  We can think of this as a measure of the bluntest tip that can trace the 
    //     entire function.
    // This is important because any gradient descent step that would not 
    // cause divergence on the tangent parabola is guaranteed not to diverge 
    // on the base function (since the parabola is always higher(i.e. worse)).
    */
    
        
    float lambda, alpha,gamma,delta,mu,rho,sigma;
    float L=4;//lower is better(longer steps), but in theory only >=4 is guaranteed to converge. For the adventurous, set to 2 or 1.44
    
    lambda=1.0/theta;
    alpha=epsilon;
    
    gamma=lambda;
    delta=alpha;
    
    mu=2.0*sqrt(gamma*delta)/L;

    rho= mu/(2.0*gamma);
    sigma=mu/(2.0*delta);
    
    sigma_d = rho;
    sigma_q = sigma;
}

void DepthmapDenoiseWeightedHuberImpl::cacheGValues(){
    using namespace cv::cuda::device::dtam_denoise;
    localStream = cv::cuda::StreamAccessor::getStream(cvStream);
    if(cachedG)
        return;//already cached
    if(!alloced)
        allocate(rows,cols);

    // Call the gpu function for caching g's
    
    loadConstants(rows, 0, 0, 0, 0, 0, 0, 0,
            0, 0);
    CV_Assert(_g1.isContinuous());
    float* pp = (float*) visibleLightImage.data;//TODO: write a color version.
    float* g1p = (float*)_g1.data;
    float* gxp = (float*)_gx.data;
    float* gyp = (float*)_gy.data;
    computeGCaller(pp,  g1p,  gxp,  gyp, cols);
    cachedG=1;
}

GpuMat DepthmapDenoiseWeightedHuberImpl::operator()(InputArray _ain, float epsilon,float theta){
    const GpuMat& ain=_ain.getGpuMat();
    
    using namespace cv::cuda::device::dtam_denoise;
    localStream = cv::cuda::StreamAccessor::getStream(cvStream);
    
    rows=ain.rows;
    cols=ain.cols;
    
    CV_Assert(ain.cols>0);
    if(!(ain.rows % 32 == 0 && ain.cols % 32 == 0 && ain.cols >= 64)){
        CV_Assert(!"For performance reasons, DepthmapDenoiseWeightedHuber currenty only supports multiple of 32 image sizes with cols >= 64. Pad the image to achieve this.");
    }
    rows=ain.rows;
    cols=ain.cols;
    if(!ain.isContinuous()){
        _a.create(1,rows*cols, CV_32FC1);
        _a=_a.reshape(0,rows);
        ain.copyTo(_a,cvStream);
    }else{
        _a=ain;
    }
    

    
    if(!alloced){
        allocate(rows,cols);
    } 
    
    if(!visibleLightImage.empty())
        cacheGValues();
    if(!cachedG){
        _gx.setTo(1,cvStream);
        _gy.setTo(1,cvStream);
    }
    if(!dInited){
        _a.copyTo(_d,cvStream);
        dInited=1;
    }
    
    computeSigmas(epsilon,theta);
    
    float* d = (float*) _d.data;
    float* a = (float*) _a.data;
    float* gxpt = (float*)_gx.data;
    float* gypt = (float*)_gy.data;
    float* gqxpt = (float*)_qx.data;
    float* gqypt = (float*)_qy.data;

   loadConstants(rows, cols, 0, 0, 0, 0, 0, 0,
           0, 0);
    updateQDCaller  ( gqxpt, gqypt, d, a,
            gxpt, gypt, cols, sigma_q, sigma_d, epsilon, theta);
    cudaDeviceSynchronize();
    cudaSafeCall(cudaGetLastError());
    return _d;
}



