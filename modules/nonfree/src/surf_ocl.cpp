/*M/////////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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
#include "precomp.hpp"

#ifdef HAVE_OPENCV_OCL
#include <cstdio>
#include <sstream>
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

namespace cv
{
    namespace ocl
    {
        // The number of degrees between orientation samples in calcOrientation
        const static int ORI_SEARCH_INC = 5;
        // The local size of the calcOrientation kernel
        const static int ORI_LOCAL_SIZE = (360 / ORI_SEARCH_INC);

        static void openCLExecuteKernelSURF(Context *clCxt, const cv::ocl::ProgramEntry* source, string kernelName, size_t globalThreads[3],
            size_t localThreads[3],  std::vector< std::pair<size_t, const void *> > &args, int channels, int depth)
        {
            std::stringstream optsStr;
            optsStr << "-D ORI_LOCAL_SIZE=" << ORI_LOCAL_SIZE << " ";
            optsStr << "-D ORI_SEARCH_INC=" << ORI_SEARCH_INC << " ";
            cl_kernel kernel;
            kernel = openCLGetKernelFromSource(clCxt, source, kernelName, optsStr.str().c_str());
            size_t wave_size = queryWaveFrontSize(kernel);
            CV_Assert(clReleaseKernel(kernel) == CL_SUCCESS);
            optsStr << "-D WAVE_SIZE=" << wave_size;
            openCLExecuteKernel(clCxt, source, kernelName, globalThreads, localThreads, args, channels, depth, optsStr.str().c_str());
        }

    }
}

static inline int calcSize(int octave, int layer)
{
    /* Wavelet size at first layer of first octave. */
    const int HAAR_SIZE0 = 9;

    /* Wavelet size increment between layers. This should be an even number,
    such that the wavelet sizes in an octave are either all even or all odd.
    This ensures that when looking for the neighbors of a sample, the layers

    above and below are aligned correctly. */
    const int HAAR_SIZE_INC = 6;

    return (HAAR_SIZE0 + HAAR_SIZE_INC * layer) << octave;
}


class SURF_OCL_Invoker
{
public:
    // facilities
    void bindImgTex(const oclMat &img, cl_mem &texture);

    //void loadGlobalConstants(int maxCandidates, int maxFeatures, int img_rows, int img_cols, int nOctaveLayers, float hessianThreshold);
    //void loadOctaveConstants(int octave, int layer_rows, int layer_cols);

    // kernel callers declarations
    void icvCalcLayerDetAndTrace_gpu(oclMat &det, oclMat &trace, int octave, int nOctaveLayers, int layer_rows);

    void icvFindMaximaInLayer_gpu(const oclMat &det, const oclMat &trace, oclMat &maxPosBuffer, oclMat &maxCounter, int counterOffset,
                                  int octave, bool use_mask, int nLayers, int layer_rows, int layer_cols);

    void icvInterpolateKeypoint_gpu(const oclMat &det, const oclMat &maxPosBuffer, int maxCounter,
                                    oclMat &keypoints, oclMat &counters, int octave, int layer_rows, int maxFeatures);

    void icvCalcOrientation_gpu(const oclMat &keypoints, int nFeatures);

    void icvSetUpright_gpu(const oclMat &keypoints, int nFeatures);

    void compute_descriptors_gpu(const oclMat &descriptors, const oclMat &keypoints, int nFeatures);
    // end of kernel callers declarations

    SURF_OCL_Invoker(SURF_OCL &theSurf, const oclMat &img, const oclMat &mask) :
        surf_(theSurf),
        img_cols(img.cols), img_rows(img.rows),
        use_mask(!mask.empty()), counters(oclMat()),
        imgTex(NULL), sumTex(NULL), maskSumTex(NULL), _img(img)
    {
        CV_Assert(!img.empty() && img.type() == CV_8UC1);
        CV_Assert(mask.empty() || (mask.size() == img.size() && mask.type() == CV_8UC1));
        CV_Assert(surf_.nOctaves > 0 && surf_.nOctaveLayers > 0);

        const int min_size = calcSize(surf_.nOctaves - 1, 0);
        CV_Assert(img_rows - min_size >= 0);
        CV_Assert(img_cols - min_size >= 0);

        const int layer_rows = img_rows >> (surf_.nOctaves - 1);
        const int layer_cols = img_cols >> (surf_.nOctaves - 1);
        const int min_margin = ((calcSize((surf_.nOctaves - 1), 2) >> 1) >> (surf_.nOctaves - 1)) + 1;
        CV_Assert(layer_rows - 2 * min_margin > 0);
        CV_Assert(layer_cols - 2 * min_margin > 0);

        maxFeatures   = std::min(static_cast<int>(img.size().area() * theSurf.keypointsRatio), 65535);
        maxCandidates = std::min(static_cast<int>(1.5 * maxFeatures), 65535);

        CV_Assert(maxFeatures > 0);

        counters.create(1, surf_.nOctaves + 1, CV_32SC1);
        counters.setTo(Scalar::all(0));

        integral(img, surf_.sum);

        bindImgTex(img, imgTex);
        bindImgTex(surf_.sum, sumTex);
        finish();

        maskSumTex = 0;

        if (use_mask)
        {
            CV_Error(CV_StsBadFunc, "Masked SURF detector is not implemented yet");
            //!FIXME
            // temp fix for missing min overload
            //oclMat temp(mask.size(), mask.type());
            //temp.setTo(Scalar::all(1.0));
            ////cv::ocl::min(mask, temp, surf_.mask1);           ///////// disable this
            //integral(surf_.mask1, surf_.maskSum);
            //bindImgTex(surf_.maskSum, maskSumTex);
        }
    }

    void detectKeypoints(oclMat &keypoints)
    {
        // create image pyramid buffers
        // different layers have same sized buffers, but they are sampled from Gaussian kernel.
        ensureSizeIsEnough(img_rows * (surf_.nOctaveLayers + 2), img_cols, CV_32FC1, surf_.det);
        ensureSizeIsEnough(img_rows * (surf_.nOctaveLayers + 2), img_cols, CV_32FC1, surf_.trace);

        ensureSizeIsEnough(1, maxCandidates, CV_32SC4, surf_.maxPosBuffer);
        ensureSizeIsEnough(SURF_OCL::ROWS_COUNT, maxFeatures, CV_32FC1, keypoints);
        keypoints.setTo(Scalar::all(0));

        for (int octave = 0; octave < surf_.nOctaves; ++octave)
        {
            const int layer_rows = img_rows >> octave;
            const int layer_cols = img_cols >> octave;

            //loadOctaveConstants(octave, layer_rows, layer_cols);

            icvCalcLayerDetAndTrace_gpu(surf_.det, surf_.trace, octave, surf_.nOctaveLayers, layer_rows);

            icvFindMaximaInLayer_gpu(surf_.det, surf_.trace, surf_.maxPosBuffer, counters, 1 + octave,
                                     octave, use_mask, surf_.nOctaveLayers, layer_rows, layer_cols);

            int maxCounter = ((Mat)counters).at<int>(1 + octave);
            maxCounter = std::min(maxCounter, static_cast<int>(maxCandidates));

            if (maxCounter > 0)
            {
                icvInterpolateKeypoint_gpu(surf_.det, surf_.maxPosBuffer, maxCounter,
                                           keypoints, counters, octave, layer_rows, maxFeatures);
            }
        }
        int featureCounter = Mat(counters).at<int>(0);
        featureCounter = std::min(featureCounter, static_cast<int>(maxFeatures));

        keypoints.cols = featureCounter;

        if (surf_.upright)
        {
            //keypoints.row(SURF_OCL::ANGLE_ROW).setTo(Scalar::all(90.0));
            setUpright(keypoints);
        }
        else
        {
            findOrientation(keypoints);
        }
    }

    void setUpright(oclMat &keypoints)
    {
        const int nFeatures = keypoints.cols;
        if(nFeatures > 0)
        {
            icvSetUpright_gpu(keypoints, keypoints.cols);
        }
    }

    void findOrientation(oclMat &keypoints)
    {
        const int nFeatures = keypoints.cols;
        if (nFeatures > 0)
        {
            icvCalcOrientation_gpu(keypoints, nFeatures);
        }
    }

    void computeDescriptors(const oclMat &keypoints, oclMat &descriptors, int descriptorSize)
    {
        const int nFeatures = keypoints.cols;
        if (nFeatures > 0)
        {
            ensureSizeIsEnough(nFeatures, descriptorSize, CV_32F, descriptors);
            compute_descriptors_gpu(descriptors, keypoints, nFeatures);
        }
    }

    ~SURF_OCL_Invoker()
    {
        if(imgTex)
            openCLFree(imgTex);
        if(sumTex)
            openCLFree(sumTex);
        if(maskSumTex)
            openCLFree(maskSumTex);
    }

private:
    SURF_OCL &surf_;

    int img_cols, img_rows;

    bool use_mask;

    int maxCandidates;
    int maxFeatures;

    oclMat counters;

    // texture buffers
    cl_mem imgTex;
    cl_mem sumTex;
    cl_mem maskSumTex;

    const oclMat _img; // make a copy for non-image2d_t supported platform

    SURF_OCL_Invoker &operator= (const SURF_OCL_Invoker &right)
    {
        (*this) = right;
        return *this;
    } // remove warning C4512
};

cv::ocl::SURF_OCL::SURF_OCL()
{
    hessianThreshold = 100.0f;
    extended = false;
    nOctaves = 4;
    nOctaveLayers = 3;
    keypointsRatio = 0.01f;
    upright = false;
}

cv::ocl::SURF_OCL::SURF_OCL(double _threshold, int _nOctaves, int _nOctaveLayers, bool _extended, float _keypointsRatio, bool _upright)
{
    hessianThreshold = saturate_cast<float>(_threshold);
    extended = _extended;
    nOctaves = _nOctaves;
    nOctaveLayers = _nOctaveLayers;
    keypointsRatio = _keypointsRatio;
    upright = _upright;
}

int cv::ocl::SURF_OCL::descriptorSize() const
{
    return extended ? 128 : 64;
}

int cv::ocl::SURF_OCL::descriptorType() const
{
    return CV_32F;
}

void cv::ocl::SURF_OCL::uploadKeypoints(const vector<KeyPoint> &keypoints, oclMat &keypointsGPU)
{
    if (keypoints.empty())
        keypointsGPU.release();
    else
    {
        Mat keypointsCPU(SURF_OCL::ROWS_COUNT, static_cast<int>(keypoints.size()), CV_32FC1);

        float *kp_x = keypointsCPU.ptr<float>(SURF_OCL::X_ROW);
        float *kp_y = keypointsCPU.ptr<float>(SURF_OCL::Y_ROW);
        int *kp_laplacian = keypointsCPU.ptr<int>(SURF_OCL::LAPLACIAN_ROW);
        int *kp_octave = keypointsCPU.ptr<int>(SURF_OCL::OCTAVE_ROW);
        float *kp_size = keypointsCPU.ptr<float>(SURF_OCL::SIZE_ROW);
        float *kp_dir = keypointsCPU.ptr<float>(SURF_OCL::ANGLE_ROW);
        float *kp_hessian = keypointsCPU.ptr<float>(SURF_OCL::HESSIAN_ROW);

        for (size_t i = 0, size = keypoints.size(); i < size; ++i)
        {
            const KeyPoint &kp = keypoints[i];
            kp_x[i] = kp.pt.x;
            kp_y[i] = kp.pt.y;
            kp_octave[i] = kp.octave;
            kp_size[i] = kp.size;
            kp_dir[i] = kp.angle;
            kp_hessian[i] = kp.response;
            kp_laplacian[i] = 1;
        }

        keypointsGPU.upload(keypointsCPU);
    }
}

void cv::ocl::SURF_OCL::downloadKeypoints(const oclMat &keypointsGPU, vector<KeyPoint> &keypoints)
{
    const int nFeatures = keypointsGPU.cols;

    if (nFeatures == 0)
        keypoints.clear();
    else
    {
        CV_Assert(keypointsGPU.type() == CV_32FC1 && keypointsGPU.rows == ROWS_COUNT);

        Mat keypointsCPU(keypointsGPU);

        keypoints.resize(nFeatures);

        float *kp_x = keypointsCPU.ptr<float>(SURF_OCL::X_ROW);
        float *kp_y = keypointsCPU.ptr<float>(SURF_OCL::Y_ROW);
        int *kp_laplacian = keypointsCPU.ptr<int>(SURF_OCL::LAPLACIAN_ROW);
        int *kp_octave = keypointsCPU.ptr<int>(SURF_OCL::OCTAVE_ROW);
        float *kp_size = keypointsCPU.ptr<float>(SURF_OCL::SIZE_ROW);
        float *kp_dir = keypointsCPU.ptr<float>(SURF_OCL::ANGLE_ROW);
        float *kp_hessian = keypointsCPU.ptr<float>(SURF_OCL::HESSIAN_ROW);

        for (int i = 0; i < nFeatures; ++i)
        {
            KeyPoint &kp = keypoints[i];
            kp.pt.x = kp_x[i];
            kp.pt.y = kp_y[i];
            kp.class_id = kp_laplacian[i];
            kp.octave = kp_octave[i];
            kp.size = kp_size[i];
            kp.angle = kp_dir[i];
            kp.response = kp_hessian[i];
        }
    }
}

void cv::ocl::SURF_OCL::downloadDescriptors(const oclMat &descriptorsGPU, vector<float> &descriptors)
{
    if (descriptorsGPU.empty())
        descriptors.clear();
    else
    {
        CV_Assert(descriptorsGPU.type() == CV_32F);

        descriptors.resize(descriptorsGPU.rows * descriptorsGPU.cols);
        Mat descriptorsCPU(descriptorsGPU.size(), CV_32F, &descriptors[0]);
        descriptorsGPU.download(descriptorsCPU);
    }
}

void cv::ocl::SURF_OCL::operator()(const oclMat &img, const oclMat &mask, oclMat &keypoints)
{
    if (!img.empty())
    {
        SURF_OCL_Invoker theSurf(*this, img, mask);

        theSurf.detectKeypoints(keypoints);
    }
}

void cv::ocl::SURF_OCL::operator()(const oclMat &img, const oclMat &mask, oclMat &keypoints, oclMat &descriptors,
                                   bool useProvidedKeypoints)
{
    if (!img.empty())
    {
        SURF_OCL_Invoker theSurf(*this, img, mask);

        if (!useProvidedKeypoints)
            theSurf.detectKeypoints(keypoints);
        else if (!upright)
        {
            theSurf.findOrientation(keypoints);
        }

        theSurf.computeDescriptors(keypoints, descriptors, descriptorSize());
    }
}

void cv::ocl::SURF_OCL::operator()(const oclMat &img, const oclMat &mask, vector<KeyPoint> &keypoints)
{
    oclMat keypointsGPU;

    (*this)(img, mask, keypointsGPU);

    downloadKeypoints(keypointsGPU, keypoints);
}

void cv::ocl::SURF_OCL::operator()(const oclMat &img, const oclMat &mask, vector<KeyPoint> &keypoints,
                                   oclMat &descriptors, bool useProvidedKeypoints)
{
    oclMat keypointsGPU;

    if (useProvidedKeypoints)
        uploadKeypoints(keypoints, keypointsGPU);

    (*this)(img, mask, keypointsGPU, descriptors, useProvidedKeypoints);

    downloadKeypoints(keypointsGPU, keypoints);
}

void cv::ocl::SURF_OCL::operator()(const oclMat &img, const oclMat &mask, vector<KeyPoint> &keypoints,
                                   vector<float> &descriptors, bool useProvidedKeypoints)
{
    oclMat descriptorsGPU;

    (*this)(img, mask, keypoints, descriptorsGPU, useProvidedKeypoints);

    downloadDescriptors(descriptorsGPU, descriptors);
}


void cv::ocl::SURF_OCL::operator()(InputArray img, InputArray mask,
                                   CV_OUT vector<KeyPoint>& keypoints) const
{
    this->operator()(img, mask, keypoints, noArray(), false);
}

void cv::ocl::SURF_OCL::operator()(InputArray img, InputArray mask, vector<KeyPoint> &keypoints,
                                   OutputArray descriptors, bool useProvidedKeypoints) const
{
    oclMat _img, _mask;
    if(img.kind() == _InputArray::OCL_MAT)
        _img = *(oclMat*)img.obj;
    else
        _img.upload(img.getMat());
    if(_img.channels() != 1)
    {
        oclMat temp;
        cvtColor(_img, temp, COLOR_BGR2GRAY);
        _img = temp;
    }

    if( !mask.empty() )
    {
        if(mask.kind() == _InputArray::OCL_MAT)
            _mask = *(oclMat*)mask.obj;
        else
            _mask.upload(mask.getMat());
    }

    SURF_OCL_Invoker theSurf((SURF_OCL&)*this, _img, _mask);
    oclMat keypointsGPU;

    if (!useProvidedKeypoints || !upright)
        ((SURF_OCL*)this)->uploadKeypoints(keypoints, keypointsGPU);

    if (!useProvidedKeypoints)
        theSurf.detectKeypoints(keypointsGPU);
    else if (!upright)
        theSurf.findOrientation(keypointsGPU);
    if(keypointsGPU.cols*keypointsGPU.rows != 0)
        ((SURF_OCL*)this)->downloadKeypoints(keypointsGPU, keypoints);

    if( descriptors.needed() )
    {
        oclMat descriptorsGPU;
        theSurf.computeDescriptors(keypointsGPU, descriptorsGPU, descriptorSize());
        Size sz = descriptorsGPU.size();
        if( descriptors.kind() == _InputArray::STD_VECTOR )
        {
            CV_Assert(descriptors.type() == CV_32F);
            std::vector<float>* v = (std::vector<float>*)descriptors.obj;
            v->resize(sz.width*sz.height);
            Mat m(sz, CV_32F, &v->at(0));
            descriptorsGPU.download(m);
        }
        else
        {
            descriptors.create(sz, CV_32F);
            Mat m = descriptors.getMat();
            descriptorsGPU.download(m);
        }
    }
}

void cv::ocl::SURF_OCL::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
{
    (*this)(image, mask, keypoints, noArray(), false);
}

void cv::ocl::SURF_OCL::computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
{
    (*this)(image, Mat(), keypoints, descriptors, true);
}

void cv::ocl::SURF_OCL::releaseMemory()
{
    sum.release();
    mask1.release();
    maskSum.release();
    intBuffer.release();
    det.release();
    trace.release();
    maxPosBuffer.release();
}


// bind source buffer to image oject.
void SURF_OCL_Invoker::bindImgTex(const oclMat &img, cl_mem &texture)
{
    if(texture)
    {
        openCLFree(texture);
    }
    texture = bindTexture(img);
}

////////////////////////////
// kernel caller definitions
void SURF_OCL_Invoker::icvCalcLayerDetAndTrace_gpu(oclMat &det, oclMat &trace, int octave, int nOctaveLayers, int c_layer_rows)
{
    const int min_size = calcSize(octave, 0);
    const int max_samples_i = 1 + ((img_rows - min_size) >> octave);
    const int max_samples_j = 1 + ((img_cols - min_size) >> octave);

    Context *clCxt = det.clCxt;
    string kernelName = "icvCalcLayerDetAndTrace";
    std::vector< std::pair<size_t, const void *> > args;

    if(sumTex)
    {
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&sumTex));
    }
    else
    {
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&surf_.sum.data)); // if image2d is not supported
    }
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&det.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&trace.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&det.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&trace.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&img_rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&img_cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&nOctaveLayers));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&octave));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&c_layer_rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&surf_.sum.step));

    size_t localThreads[3]  = {16, 16, 1};
    size_t globalThreads[3] =
    {
        divUp(max_samples_j, localThreads[0]) * localThreads[0],
        divUp(max_samples_i, localThreads[1]) * localThreads[1] *(nOctaveLayers + 2),
        1
    };
    openCLExecuteKernelSURF(clCxt, &surf, kernelName, globalThreads, localThreads, args, -1, -1);
}

void SURF_OCL_Invoker::icvFindMaximaInLayer_gpu(const oclMat &det, const oclMat &trace, oclMat &maxPosBuffer, oclMat &maxCounter, int counterOffset,
        int octave, bool useMask, int nLayers, int layer_rows, int layer_cols)
{
    const int min_margin = ((calcSize(octave, 2) >> 1) >> octave) + 1;

    Context *clCxt = det.clCxt;
    string kernelName = useMask ? "icvFindMaximaInLayer_withmask" : "icvFindMaximaInLayer";
    std::vector< std::pair<size_t, const void *> > args;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&det.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&trace.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&maxPosBuffer.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&maxCounter.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&counterOffset));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&det.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&trace.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&img_rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&img_cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&nLayers));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&octave));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layer_rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layer_cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&maxCandidates));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&surf_.hessianThreshold));

    if(useMask)
    {
        if(maskSumTex)
        {
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&maskSumTex));
        }
        else
        {
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&surf_.maskSum.data));
        }
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&surf_.maskSum.step));
    }
    size_t localThreads[3]  = {16, 16, 1};
    size_t globalThreads[3] = {divUp(layer_cols - 2 * min_margin, localThreads[0] - 2) *localThreads[0],
                               divUp(layer_rows - 2 * min_margin, localThreads[1] - 2) *nLayers *localThreads[1],
                               1
                              };

    openCLExecuteKernelSURF(clCxt, &surf, kernelName, globalThreads, localThreads, args, -1, -1);
}

void SURF_OCL_Invoker::icvInterpolateKeypoint_gpu(const oclMat &det, const oclMat &maxPosBuffer, int maxCounter,
        oclMat &keypoints, oclMat &counters_, int octave, int layer_rows, int max_features)
{
    Context *clCxt = det.clCxt;
    string kernelName = "icvInterpolateKeypoint";
    std::vector< std::pair<size_t, const void *> > args;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&det.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&maxPosBuffer.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypoints.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&counters_.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&det.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypoints.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&img_rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&img_cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&octave));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layer_rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&max_features));

    size_t localThreads[3]  = {3, 3, 3};
    size_t globalThreads[3] = {maxCounter *localThreads[0], localThreads[1], 1};

    openCLExecuteKernelSURF(clCxt, &surf, kernelName, globalThreads, localThreads, args, -1, -1);
}

void SURF_OCL_Invoker::icvCalcOrientation_gpu(const oclMat &keypoints, int nFeatures)
{
    Context *clCxt = counters.clCxt;
    string kernelName = "icvCalcOrientation";

    std::vector< std::pair<size_t, const void *> > args;

    if(sumTex)
    {
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&sumTex));
    }
    else
    {
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&surf_.sum.data)); // if image2d is not supported
    }
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypoints.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypoints.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&img_rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&img_cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&surf_.sum.step));

    size_t localThreads[3]  = {ORI_LOCAL_SIZE, 1, 1};
    size_t globalThreads[3] = {nFeatures * localThreads[0], 1, 1};

    openCLExecuteKernelSURF(clCxt, &surf, kernelName, globalThreads, localThreads, args, -1, -1);
}

void SURF_OCL_Invoker::icvSetUpright_gpu(const oclMat &keypoints, int nFeatures)
{
    Context *clCxt = counters.clCxt;
    string kernelName = "icvSetUpright";

    std::vector< std::pair<size_t, const void *> > args;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypoints.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypoints.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&nFeatures));

    size_t localThreads[3]  = {256, 1, 1};
    size_t globalThreads[3] = {saturate_cast<size_t>(nFeatures), 1, 1};

    openCLExecuteKernelSURF(clCxt, &surf, kernelName, globalThreads, localThreads, args, -1, -1);
}


void SURF_OCL_Invoker::compute_descriptors_gpu(const oclMat &descriptors, const oclMat &keypoints, int nFeatures)
{
    // compute unnormalized descriptors, then normalize them - odd indexing since grid must be 2D
    Context *clCxt = descriptors.clCxt;
    string kernelName;
    std::vector< std::pair<size_t, const void *> > args;
    size_t localThreads[3]  = {1, 1, 1};
    size_t globalThreads[3] = {1, 1, 1};

    if(descriptors.cols == 64)
    {
        kernelName = "compute_descriptors64";

        localThreads[0] = 6;
        localThreads[1] = 6;

        globalThreads[0] = nFeatures * localThreads[0];
        globalThreads[1] = 16 * localThreads[1];

        args.clear();
        if(imgTex)
        {
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&imgTex));
        }
        else
        {
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&_img.data));
        }
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&descriptors.data));
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypoints.data));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&descriptors.step));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypoints.step));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&_img.rows));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&_img.cols));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&_img.step));

        openCLExecuteKernelSURF(clCxt, &surf, kernelName, globalThreads, localThreads, args, -1, -1);

        kernelName = "normalize_descriptors64";

        localThreads[0] = 64;
        localThreads[1] = 1;

        globalThreads[0] = nFeatures * localThreads[0];
        globalThreads[1] = localThreads[1];

        args.clear();
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&descriptors.data));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&descriptors.step));

        openCLExecuteKernelSURF(clCxt, &surf, kernelName, globalThreads, localThreads, args, -1, -1);
    }
    else
    {
        kernelName = "compute_descriptors128";

        localThreads[0] = 6;
        localThreads[1] = 6;

        globalThreads[0] = nFeatures * localThreads[0];
        globalThreads[1] = 16 * localThreads[1];

        args.clear();
        if(imgTex)
        {
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&imgTex));
        }
        else
        {
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&_img.data));
        }
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&descriptors.data));
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypoints.data));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&descriptors.step));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypoints.step));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&_img.rows));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&_img.cols));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&_img.step));

        openCLExecuteKernelSURF(clCxt, &surf, kernelName, globalThreads, localThreads, args, -1, -1);

        kernelName = "normalize_descriptors128";

        localThreads[0] = 128;
        localThreads[1] = 1;

        globalThreads[0] = nFeatures * localThreads[0];
        globalThreads[1] = localThreads[1];

        args.clear();
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&descriptors.data));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&descriptors.step));

        openCLExecuteKernelSURF(clCxt, &surf, kernelName, globalThreads, localThreads, args, -1, -1);
    }
}

#endif //HAVE_OPENCV_OCL
