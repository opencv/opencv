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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

using namespace cv;
using namespace cv::gpu;
using namespace std;

#if !defined (HAVE_CUDA)

cv::gpu::SURF_GPU::SURF_GPU() { throw_nogpu(); }
cv::gpu::SURF_GPU::SURF_GPU(double, int, int, bool, float, bool) { throw_nogpu(); }
int cv::gpu::SURF_GPU::descriptorSize() const { throw_nogpu(); return 0;}
void cv::gpu::SURF_GPU::uploadKeypoints(const vector<KeyPoint>&, GpuMat&) { throw_nogpu(); }
void cv::gpu::SURF_GPU::downloadKeypoints(const GpuMat&, vector<KeyPoint>&) { throw_nogpu(); }
void cv::gpu::SURF_GPU::downloadDescriptors(const GpuMat&, vector<float>&) { throw_nogpu(); }
void cv::gpu::SURF_GPU::operator()(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::SURF_GPU::operator()(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::SURF_GPU::operator()(const GpuMat&, const GpuMat&, vector<KeyPoint>&) { throw_nogpu(); }
void cv::gpu::SURF_GPU::operator()(const GpuMat&, const GpuMat&, vector<KeyPoint>&, GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::SURF_GPU::operator()(const GpuMat&, const GpuMat&, vector<KeyPoint>&, vector<float>&, bool) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace surf
{
    void icvCalcLayerDetAndTrace_gpu(const PtrStepf& det, const PtrStepf& trace, int img_rows, int img_cols, int octave, int nOctaveLayers);

    void icvFindMaximaInLayer_gpu(const PtrStepf& det, const PtrStepf& trace, int4* maxPosBuffer, unsigned int* maxCounter,
        int img_rows, int img_cols, int octave, bool use_mask, int nLayers);

    void icvInterpolateKeypoint_gpu(const PtrStepf& det, const int4* maxPosBuffer, unsigned int maxCounter, KeyPoint_GPU* featuresBuffer, unsigned int* featureCounter);

    void icvCalcOrientation_gpu(const KeyPoint_GPU* featureBuffer, int nFeatures, KeyPoint_GPU* keypoints, unsigned int* keypointCounter);

    void compute_descriptors_gpu(const DevMem2Df& descriptors, const KeyPoint_GPU* features, int nFeatures);
}}}

using namespace cv::gpu::surf;

namespace
{
    class SURF_GPU_Invoker : private CvSURFParams
    {
    public:
        SURF_GPU_Invoker(SURF_GPU& surf, const GpuMat& img, const GpuMat& mask) :
            CvSURFParams(surf),

            sum(surf.sum), mask1(surf.mask1), maskSum(surf.maskSum), intBuffer(surf.intBuffer), det(surf.det), trace(surf.trace),

            maxPosBuffer(surf.maxPosBuffer), featuresBuffer(surf.featuresBuffer), keypointsBuffer(surf.keypointsBuffer),

            img_cols(img.cols), img_rows(img.rows),

            use_mask(!mask.empty()),

            upright(surf.upright)
        {
            CV_Assert(!img.empty() && img.type() == CV_8UC1);
            CV_Assert(mask.empty() || (mask.size() == img.size() && mask.type() == CV_8UC1));
            CV_Assert(nOctaves > 0 && nOctaveLayers > 0);
            CV_Assert(TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS));

            maxKeypoints = min(static_cast<int>(img.size().area() * surf.keypointsRatio), 65535);
            maxFeatures = min(static_cast<int>(1.5 * maxKeypoints), 65535);
            maxCandidates = min(static_cast<int>(1.5 * maxFeatures), 65535);

            CV_Assert(maxKeypoints > 0);
            
            cudaSafeCall( cudaMalloc((void**)&d_counters, (nOctaves + 2) * sizeof(unsigned int)) );
            cudaSafeCall( cudaMemset(d_counters, 0, (nOctaves + 2) * sizeof(unsigned int)) );

            uploadConstant("cv::gpu::surf::c_max_candidates",    maxCandidates);
            uploadConstant("cv::gpu::surf::c_max_features",      maxFeatures);
            uploadConstant("cv::gpu::surf::c_max_keypoints",     maxKeypoints);
            uploadConstant("cv::gpu::surf::c_img_rows",          img_rows);
            uploadConstant("cv::gpu::surf::c_img_cols",          img_cols);
            uploadConstant("cv::gpu::surf::c_nOctaveLayers",     nOctaveLayers);
            uploadConstant("cv::gpu::surf::c_hessianThreshold",  static_cast<float>(hessianThreshold));

            bindTexture("cv::gpu::surf::imgTex", (DevMem2D)img);

            integralBuffered(img, sum, intBuffer);
            bindTexture("cv::gpu::surf::sumTex", (DevMem2D_<unsigned int>)sum);

            if (use_mask)
            {
                min(mask, 1.0, mask1);
                integralBuffered(mask1, maskSum, intBuffer);

                bindTexture("cv::gpu::surf::maskSumTex", (DevMem2D_<unsigned int>)maskSum);
            }
        }

        ~SURF_GPU_Invoker()
        {
            cudaSafeCall( cudaFree(d_counters) );

            unbindTexture("cv::gpu::surf::imgTex");
            unbindTexture("cv::gpu::surf::sumTex");
            if (use_mask)
                unbindTexture("cv::gpu::surf::maskSumTex");
        }

        void detectKeypoints(GpuMat& keypoints)
        {
            ensureSizeIsEnough(img_rows * (nOctaveLayers + 2), img_cols, CV_32FC1, det);
            ensureSizeIsEnough(img_rows * (nOctaveLayers + 2), img_cols, CV_32FC1, trace);
            
            ensureSizeIsEnough(1, maxCandidates, CV_32SC4, maxPosBuffer);
            ensureSizeIsEnough(1, maxFeatures, CV_32FC(6), featuresBuffer);

            for (int octave = 0; octave < nOctaves; ++octave)
            {
                const int layer_rows = img_rows >> octave;
                const int layer_cols = img_cols >> octave;

                uploadConstant("cv::gpu::surf::c_octave",     octave);
                uploadConstant("cv::gpu::surf::c_layer_rows", layer_rows);
                uploadConstant("cv::gpu::surf::c_layer_cols", layer_cols);

                icvCalcLayerDetAndTrace_gpu(det, trace, img_rows, img_cols, octave, nOctaveLayers);

                icvFindMaximaInLayer_gpu(det, trace, maxPosBuffer.ptr<int4>(), d_counters + 2 + octave,
                    img_rows, img_cols, octave, use_mask, nOctaveLayers);

                unsigned int maxCounter;
                cudaSafeCall( cudaMemcpy(&maxCounter, d_counters + 2 + octave, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
                maxCounter = std::min(maxCounter, static_cast<unsigned int>(maxCandidates));

                if (maxCounter > 0)
                {
                    icvInterpolateKeypoint_gpu(det, maxPosBuffer.ptr<int4>(), maxCounter, 
                        featuresBuffer.ptr<KeyPoint_GPU>(), d_counters);
                }
            }
            unsigned int featureCounter;
            cudaSafeCall( cudaMemcpy(&featureCounter, d_counters, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
            featureCounter = std::min(featureCounter, static_cast<unsigned int>(maxFeatures));

            if (!upright)
                findOrientation(featuresBuffer.colRange(0, featureCounter), keypoints);
            else
            {
                if (featureCounter > 0)
                    featuresBuffer.colRange(0, featureCounter).copyTo(keypoints);
                else
                    keypoints.release();
            }
        }

        void findOrientation(const GpuMat& features, GpuMat& keypoints)
        {
            if (features.cols > 0)
            {
                ensureSizeIsEnough(1, maxKeypoints, CV_32FC(6), keypointsBuffer);

                icvCalcOrientation_gpu(features.ptr<KeyPoint_GPU>(), features.cols, keypointsBuffer.ptr<KeyPoint_GPU>(), 
                    d_counters + 1);

                unsigned int keypointsCounter;
                cudaSafeCall( cudaMemcpy(&keypointsCounter, d_counters + 1, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
                keypointsCounter = std::min(keypointsCounter, static_cast<unsigned int>(maxKeypoints));

                if (keypointsCounter > 0)
                    keypointsBuffer.colRange(0, keypointsCounter).copyTo(keypoints);
                else
                    keypoints.release();
            }
        }

        void computeDescriptors(const GpuMat& keypoints, GpuMat& descriptors, int descriptorSize)
        {
            if (keypoints.cols > 0)
            {
                descriptors.create(keypoints.cols, descriptorSize, CV_32F);
                compute_descriptors_gpu(descriptors, keypoints.ptr<KeyPoint_GPU>(), keypoints.cols);
            }
        }

    private:
        GpuMat& sum;
        GpuMat& mask1;
        GpuMat& maskSum;
        GpuMat& intBuffer;

        GpuMat& det;
        GpuMat& trace;

        GpuMat& maxPosBuffer;
        GpuMat& featuresBuffer;
        GpuMat& keypointsBuffer;

        int img_cols, img_rows;

        bool use_mask;

        bool upright;

        int maxCandidates;
        int maxFeatures;
        int maxKeypoints;

        unsigned int* d_counters;
    };
}

cv::gpu::SURF_GPU::SURF_GPU()
{
    hessianThreshold = 100;
    extended = 1;
    nOctaves = 4;
    nOctaveLayers = 2;
    keypointsRatio = 0.01f;
    upright = false;
}

cv::gpu::SURF_GPU::SURF_GPU(double _threshold, int _nOctaves, int _nOctaveLayers, bool _extended, float _keypointsRatio, bool _upright)
{
    hessianThreshold = _threshold;
    extended = _extended;
    nOctaves = _nOctaves;
    nOctaveLayers = _nOctaveLayers;
    keypointsRatio = _keypointsRatio;
    upright = _upright;
}

int cv::gpu::SURF_GPU::descriptorSize() const
{
    return extended ? 128 : 64;
}

void cv::gpu::SURF_GPU::uploadKeypoints(const vector<KeyPoint>& keypoints, GpuMat& keypointsGPU)
{
    if (keypoints.empty())
        keypointsGPU.release();
    else
    {
        Mat keypointsCPU(1, keypoints.size(), CV_32FC(6));

        for (size_t i = 0; i < keypoints.size(); ++i)
        {
            const KeyPoint& kp = keypoints[i];
            KeyPoint_GPU& gkp = keypointsCPU.ptr<KeyPoint_GPU>()[i];

            gkp.x = kp.pt.x;
            gkp.y = kp.pt.y;

            gkp.laplacian = 1.0f;

            gkp.size = kp.size;

            gkp.dir = kp.angle;
            gkp.hessian = kp.response;
        }

        keypointsGPU.upload(keypointsCPU);
    }
}

namespace
{
    int calcSize(int octave, int layer)
    {
        /* Wavelet size at first layer of first octave. */
        const int HAAR_SIZE0 = 9;

        /* Wavelet size increment between layers. This should be an even number,
         such that the wavelet sizes in an octave are either all even or all odd.
         This ensures that when looking for the neighbours of a sample, the layers
         above and below are aligned correctly. */
        const int HAAR_SIZE_INC = 6;

        return (HAAR_SIZE0 + HAAR_SIZE_INC * layer) << octave;
    }

    int getPointOctave(const KeyPoint_GPU& kpt, const CvSURFParams& params)
    {
        int best_octave = 0;
        float min_diff = numeric_limits<float>::max();
        for (int octave = 1; octave < params.nOctaves; ++octave)
        {
            for (int layer = 0; layer < params.nOctaveLayers; ++layer)
            {
                float diff = std::abs(kpt.size - (float)calcSize(octave, layer));
                if (min_diff > diff)
                {
                    min_diff = diff;
                    best_octave = octave;
                    if (min_diff == 0)
                        return best_octave;
                }
            }
        }
        return best_octave;
    }
}

void cv::gpu::SURF_GPU::downloadKeypoints(const GpuMat& keypointsGPU, vector<KeyPoint>& keypoints)
{
    if (keypointsGPU.empty())
        keypoints.clear();
    else
    {
        CV_Assert(keypointsGPU.type() == CV_32FC(6) && keypointsGPU.isContinuous());

        Mat keypointsCPU = keypointsGPU;
        keypoints.resize(keypointsGPU.cols);

        for (int i = 0; i < keypointsGPU.cols; ++i)
        {
            KeyPoint& kp = keypoints[i];
            const KeyPoint_GPU& gkp = keypointsCPU.ptr<KeyPoint_GPU>()[i];

            kp.pt.x = gkp.x;
            kp.pt.y = gkp.y;

            kp.size = gkp.size;

            kp.angle = gkp.dir;

            kp.response = gkp.hessian;

            kp.octave = getPointOctave(gkp, *this);

            kp.class_id = static_cast<int>(gkp.laplacian);
        }
    }
}

void cv::gpu::SURF_GPU::downloadDescriptors(const GpuMat& descriptorsGPU, vector<float>& descriptors)
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

void cv::gpu::SURF_GPU::operator()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints)
{
    if (!img.empty())
    {
        SURF_GPU_Invoker surf(*this, img, mask);

        surf.detectKeypoints(keypoints);
    }
}

void cv::gpu::SURF_GPU::operator()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints, GpuMat& descriptors, 
                                   bool useProvidedKeypoints)
{
    if (!img.empty())
    {
        SURF_GPU_Invoker surf(*this, img, mask);
    
        if (!useProvidedKeypoints)
            surf.detectKeypoints(keypoints);
        else if (!upright)
        {
            GpuMat keypointsBuf;
            surf.findOrientation(keypoints, keypointsBuf);
            keypointsBuf.copyTo(keypoints);
        }

        surf.computeDescriptors(keypoints, descriptors, descriptorSize());
    }
}

void cv::gpu::SURF_GPU::operator()(const GpuMat& img, const GpuMat& mask, vector<KeyPoint>& keypoints)
{
    GpuMat keypointsGPU;

    (*this)(img, mask, keypointsGPU);

    downloadKeypoints(keypointsGPU, keypoints);
}

void cv::gpu::SURF_GPU::operator()(const GpuMat& img, const GpuMat& mask, vector<KeyPoint>& keypoints, 
    GpuMat& descriptors, bool useProvidedKeypoints)
{
    GpuMat keypointsGPU;

    if (useProvidedKeypoints)
        uploadKeypoints(keypoints, keypointsGPU);    

    (*this)(img, mask, keypointsGPU, descriptors, useProvidedKeypoints);

    downloadKeypoints(keypointsGPU, keypoints);
}

void cv::gpu::SURF_GPU::operator()(const GpuMat& img, const GpuMat& mask, vector<KeyPoint>& keypoints, 
    vector<float>& descriptors, bool useProvidedKeypoints)
{
    GpuMat descriptorsGPU;

    (*this)(img, mask, keypoints, descriptorsGPU, useProvidedKeypoints);

    downloadDescriptors(descriptorsGPU, descriptors);
}

#endif /* !defined (HAVE_CUDA) */
