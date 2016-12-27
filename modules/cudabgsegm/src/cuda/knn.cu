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

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/limits.hpp"

#define CN 3 // Set number of channels as constant for now.

namespace cv {
namespace cuda {
namespace device {
namespace knn {

__constant__ int c_nN;
__constant__ int c_nkNN;
__constant__ float c_Tb;
__constant__ bool c_bShadowDetection;
__constant__ unsigned char c_nShadowDetection;
__constant__ float c_Tau;

void loadConstants(int nN, int nkNN, float Tb,
                   bool bShadowDetection, unsigned char nShadowDetection, float Tau) {
    cudaSafeCall( cudaMemcpyToSymbol(c_nN, &nN, sizeof(int)) );
    cudaSafeCall( cudaMemcpyToSymbol(c_nkNN, &nkNN, sizeof(int)) );
    cudaSafeCall( cudaMemcpyToSymbol(c_Tb, &Tb, sizeof(float)) );
    cudaSafeCall( cudaMemcpyToSymbol(c_bShadowDetection, &bShadowDetection, sizeof(bool)) );
    cudaSafeCall( cudaMemcpyToSymbol(c_nShadowDetection, &nShadowDetection, sizeof(unsigned char)) );
    cudaSafeCall( cudaMemcpyToSymbol(c_Tau, &Tau, sizeof(float)) );
}

__global__ void check_pix_bg(PtrStepSz<uchar3> frame, PtrStepSz<uchar1> fgmask,
                             PtrStepSz<uchar4> bgmodel, PtrStepSz<uchar1> include ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= frame.cols || y >= frame.rows)
        return;

    int Pbf = 0; // the total probability that this pixel is background
    int Pb = 0; //background model probability
    include(y,x).x = 0; //do we include this pixel into background model?
    long posPixel = x + y*frame.cols;
    float dData[CN];

    float dist2;

    for (int n = 0; n < c_nN*3; n++) {
        dist2 = 0.0f;
        dData[0] = (float)bgmodel(n, posPixel).x - frame(y,x).x;
        dData[1] = (float)bgmodel(n, posPixel).y - frame(y,x).y;
        dData[2] = (float)bgmodel(n, posPixel).z - frame(y,x).z;
        dist2 = dData[0]*dData[0] + dData[1]*dData[1] + dData[2]*dData[2];

        if (dist2 < c_Tb) {
            Pbf++;//all
            //background only
            if(bgmodel(n, posPixel).w) {//indicator
                Pb++;
                if (Pb >= c_nkNN) {//Tb
                    // This pixel is background
                    include(y,x).x = 1;//include
                    fgmask(y,x).x = 0;
                    return;
                }
            }
        }
    }

    //include?
    if (Pbf >= c_nkNN) {//m_nTbf)
        include(y,x).x = 1;
    }

    // Shadow detection
    if (c_bShadowDetection) {
        int Ps = 0; // the total probability that this pixel is background shadow
        for (int n = 0; n < c_nN*3; n++) {
            if(bgmodel(n, posPixel).w) {//indicator
                float numerator = 0.0f;
                float denominator = 0.0f;
                numerator += (float)frame(y,x).x*bgmodel(n, posPixel).x;
                numerator += (float)frame(y,x).y*bgmodel(n, posPixel).y;
                numerator += (float)frame(y,x).z*bgmodel(n, posPixel).z;
                denominator += (float)bgmodel(n, posPixel).x*bgmodel(n, posPixel).x;
                denominator += (float)bgmodel(n, posPixel).y*bgmodel(n, posPixel).y;
                denominator += (float)bgmodel(n, posPixel).z*bgmodel(n, posPixel).z;

                // no division by zero allowed
                if( denominator == 0 ) {
                    fgmask(y,x).x = 255;
                    return;
                }

                // if tau < a < 1 then also check the color distortion
                if( numerator <= denominator && numerator >= c_Tau*denominator ) {
                    float a = numerator / denominator;
                    dist2 = 0.0f;

                    dData[0] = a*(float)bgmodel(n, posPixel).x - frame(y,x).x;
                    dData[1] = a*(float)bgmodel(n, posPixel).y - frame(y,x).y;
                    dData[2] = a*(float)bgmodel(n, posPixel).z - frame(y,x).z;
                    dist2 = dData[0]*dData[0] + dData[1]*dData[1] + dData[2]*dData[2];

                    if (dist2 < c_Tb*a*a) {
                        Ps++;
                        if (Ps >= c_nkNN) {//shadow
                            fgmask(y,x).x = c_nShadowDetection;
                            return;
                        }
                    }
                }
            }
        }
    }
    fgmask(y,x).x = 255;
}

__global__ void update_pix_bg(PtrStepSz<uchar3> frame, PtrStepSz<uchar4> bgmodel,
                              PtrStepSz<uchar3> aModelIndex, PtrStepSz<uchar1> include,
                              const uchar3 nCounter, const uchar3 nNextUpdate) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= frame.cols || y >= frame.rows)
        return;

    long posPixel = x + y*frame.cols;
    long offsetShort = aModelIndex(y,x).x;
    long offsetMid  =  aModelIndex(y,x).y + c_nN * 1;
    long offsetLong =  aModelIndex(y,x).z + c_nN * 2;

    // Long update?
    if (nCounter.x == nNextUpdate.x ) {
        // add the oldest pixel from Mid to the list of values (for each color)
        bgmodel(offsetLong, posPixel) = bgmodel(offsetMid, posPixel);
        // increase the index
        if(++aModelIndex(y,x).x >= c_nN) { aModelIndex(y,x).x = 0; };
    }

    // Mid update?
    if (nCounter.y == nNextUpdate.y ) {
        // add this pixel to the list of values (for each color)
        bgmodel(offsetMid, posPixel) = bgmodel(offsetShort, posPixel);
        // increase the index
        if(++aModelIndex(y,x).y >= c_nN) { aModelIndex(y,x).y = 0; };
    }

    // Short update?
    if (nCounter.z == nNextUpdate.z ) {
        // add this pixel to the list of values (for each color)
        bgmodel(offsetShort, posPixel).x = frame(y,x).x;
        bgmodel(offsetShort, posPixel).y = frame(y,x).y;
        bgmodel(offsetShort, posPixel).z = frame(y,x).z;
        //set the include flag
        bgmodel(offsetShort, posPixel).w = include(y,x).x;
        // increase the index
        if(++aModelIndex(y,x).z >= c_nN) { aModelIndex(y,x).z = 0; };
    }
}

__global__ void get_bg_img(PtrStepSz<uchar3> bgImg, PtrStepSz<uchar4> bgmodel) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= bgImg.cols || y >= bgImg.rows)
        return;

    long posPixel = x + y*bgImg.cols;

    for (int n = 0; n < c_nN*3; n++) {
        if ( bgmodel(n, posPixel).w ) {
            bgImg(y,x).x = bgmodel(n, posPixel).x;
            bgImg(y,x).y = bgmodel(n, posPixel).y;
            bgImg(y,x).z = bgmodel(n, posPixel).z;
            break;
        }
    }
}

void cvCheckPixelBackground_gpu(PtrStepSzb frame, PtrStepSzb fgmask, PtrStepSzb bgmodel, PtrStepSzb include, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid(divUp(frame.cols, block.x), divUp(frame.rows, block.y));

    check_pix_bg<<<grid, block, 0, stream>>>((PtrStepSz<uchar3>)frame, (PtrStepSz<uchar1>)fgmask,
            (PtrStepSz<uchar4>)bgmodel, (PtrStepSz<uchar1>)include);
}

void cvUpdatePixelBackground_gpu(PtrStepSzb frame, PtrStepSzb bgmodel,
                                 PtrStepSzb aModelIndex, PtrStepSzb include,
                                 unsigned char *nCounter, unsigned char *nNextUpdate,
                                 cudaStream_t stream) {

    dim3 block(32, 8);
    dim3 grid(divUp(frame.cols, block.x), divUp(frame.rows, block.y));

    uchar3 cntr = make_uchar3(nCounter[0], nCounter[1], nCounter[2]);
    uchar3 nup = make_uchar3(nNextUpdate[0], nNextUpdate[1], nNextUpdate[2]);

    update_pix_bg<<<grid, block, 0, stream>>>((PtrStepSz<uchar3>)frame, (PtrStepSz<uchar4>)bgmodel,
            (PtrStepSz<uchar3>)aModelIndex, (PtrStepSz<uchar1>)include, cntr, nup);
}

void getBackgroundImage_gpu(PtrStepSzb bgImg, PtrStepSzb bgmodel, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid(divUp(bgImg.cols, block.x), divUp(bgImg.rows, block.y));

    get_bg_img<<<grid, block, 0, stream>>>((PtrStepSz<uchar3>)bgImg, (PtrStepSz<uchar4>)bgmodel);
}


}
}
}
}

#endif /* CUDA_DISABLER */
