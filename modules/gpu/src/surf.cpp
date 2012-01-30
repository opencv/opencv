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
void cv::gpu::SURF_GPU::releaseMemory() { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace device 
{
    namespace surf
    {
        void loadGlobalConstants(int maxCandidates, int maxFeatures, int img_rows, int img_cols, int nOctaveLayers, float hessianThreshold);
        void loadOctaveConstants(int octave, int layer_rows, int layer_cols);

        void bindImgTex(DevMem2Db img);
        void bindSumTex(DevMem2D_<unsigned int> sum);
        void bindMaskSumTex(DevMem2D_<unsigned int> maskSum);

        void icvCalcLayerDetAndTrace_gpu(const PtrStepf& det, const PtrStepf& trace, int img_rows, int img_cols, int octave, int nOctaveLayers);

        void icvFindMaximaInLayer_gpu(const PtrStepf& det, const PtrStepf& trace, int4* maxPosBuffer, unsigned int* maxCounter,
            int img_rows, int img_cols, int octave, bool use_mask, int nLayers);

        void icvInterpolateKeypoint_gpu(const PtrStepf& det, const int4* maxPosBuffer, unsigned int maxCounter, 
            float* featureX, float* featureY, int* featureLaplacian, float* featureSize, float* featureHessian, 
            unsigned int* featureCounter);

        void icvCalcOrientation_gpu(const float* featureX, const float* featureY, const float* featureSize, float* featureDir, int nFeatures);

        void compute_descriptors_gpu(const DevMem2Df& descriptors, 
            const float* featureX, const float* featureY, const float* featureSize, const float* featureDir, int nFeatures);
    }
}}}

using namespace ::cv::gpu::device::surf;

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
    
    class SURF_GPU_Invoker : private CvSURFParams
    {
    public:
        SURF_GPU_Invoker(SURF_GPU& surf, const GpuMat& img, const GpuMat& mask) :
            CvSURFParams(surf),

            sum(surf.sum), mask1(surf.mask1), maskSum(surf.maskSum), intBuffer(surf.intBuffer), det(surf.det), trace(surf.trace),

            maxPosBuffer(surf.maxPosBuffer), 

            img_cols(img.cols), img_rows(img.rows),

            use_mask(!mask.empty())
        {
            CV_Assert(!img.empty() && img.type() == CV_8UC1);
            CV_Assert(mask.empty() || (mask.size() == img.size() && mask.type() == CV_8UC1));
            CV_Assert(nOctaves > 0 && nOctaveLayers > 0);
            CV_Assert(TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS));
                
            const int min_size = calcSize(nOctaves - 1, 0);
            CV_Assert(img_rows - min_size >= 0);
            CV_Assert(img_cols - min_size >= 0);
            
            const int layer_rows = img_rows >> (nOctaves - 1);
            const int layer_cols = img_cols >> (nOctaves - 1);
            const int min_margin = ((calcSize((nOctaves - 1), 2) >> 1) >> (nOctaves - 1)) + 1;
            CV_Assert(layer_rows - 2 * min_margin > 0);
            CV_Assert(layer_cols - 2 * min_margin > 0);

            maxFeatures = min(static_cast<int>(img.size().area() * surf.keypointsRatio), 65535);
            maxCandidates = min(static_cast<int>(1.5 * maxFeatures), 65535);

            CV_Assert(maxFeatures > 0);

            counters.create(1, nOctaves + 1, CV_32SC1);
            counters.setTo(Scalar::all(0));

            loadGlobalConstants(maxCandidates, maxFeatures, img_rows, img_cols, nOctaveLayers, static_cast<float>(hessianThreshold));

            bindImgTex(img);

            integralBuffered(img, sum, intBuffer);
            bindSumTex(sum);

            if (use_mask)
            {
                min(mask, 1.0, mask1);
                integralBuffered(mask1, maskSum, intBuffer);
                bindMaskSumTex(maskSum);
            }
        }

        void detectKeypoints(GpuMat& keypoints)
        {
            ensureSizeIsEnough(img_rows * (nOctaveLayers + 2), img_cols, CV_32FC1, det);
            ensureSizeIsEnough(img_rows * (nOctaveLayers + 2), img_cols, CV_32FC1, trace);
            
            ensureSizeIsEnough(1, maxCandidates, CV_32SC4, maxPosBuffer);
            ensureSizeIsEnough(SURF_GPU::SF_FEATURE_STRIDE, maxFeatures, CV_32FC1, keypoints);
            keypoints.setTo(Scalar::all(0));

            for (int octave = 0; octave < nOctaves; ++octave)
            {
                const int layer_rows = img_rows >> octave;
                const int layer_cols = img_cols >> octave;

                loadOctaveConstants(octave, layer_rows, layer_cols);

                icvCalcLayerDetAndTrace_gpu(det, trace, img_rows, img_cols, octave, nOctaveLayers);

                icvFindMaximaInLayer_gpu(det, trace, maxPosBuffer.ptr<int4>(), counters.ptr<unsigned int>() + 1 + octave,
                    img_rows, img_cols, octave, use_mask, nOctaveLayers);

                unsigned int maxCounter;
                cudaSafeCall( cudaMemcpy(&maxCounter, counters.ptr<unsigned int>() + 1 + octave, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
                maxCounter = std::min(maxCounter, static_cast<unsigned int>(maxCandidates));

                if (maxCounter > 0)
                {
                    icvInterpolateKeypoint_gpu(det, maxPosBuffer.ptr<int4>(), maxCounter, 
                        keypoints.ptr<float>(SURF_GPU::SF_X), keypoints.ptr<float>(SURF_GPU::SF_Y),
                        keypoints.ptr<int>(SURF_GPU::SF_LAPLACIAN), keypoints.ptr<float>(SURF_GPU::SF_SIZE),
                        keypoints.ptr<float>(SURF_GPU::SF_HESSIAN), counters.ptr<unsigned int>());
                }
            }
            unsigned int featureCounter;
            cudaSafeCall( cudaMemcpy(&featureCounter, counters.ptr<unsigned int>(), sizeof(unsigned int), cudaMemcpyDeviceToHost) );
            featureCounter = std::min(featureCounter, static_cast<unsigned int>(maxFeatures));

            keypoints.cols = featureCounter;

            if (upright)
                keypoints.row(SURF_GPU::SF_DIR).setTo(Scalar::all(90.0));
            else
                findOrientation(keypoints);
        }

        void findOrientation(GpuMat& keypoints)
        {
            const int nFeatures = keypoints.cols;
            if (nFeatures > 0)
            {
                icvCalcOrientation_gpu(keypoints.ptr<float>(SURF_GPU::SF_X), keypoints.ptr<float>(SURF_GPU::SF_Y),
                    keypoints.ptr<float>(SURF_GPU::SF_SIZE), keypoints.ptr<float>(SURF_GPU::SF_DIR), nFeatures);
            }
        }

        void computeDescriptors(const GpuMat& keypoints, GpuMat& descriptors, int descriptorSize)
        {
            const int nFeatures = keypoints.cols;
            if (nFeatures > 0)
            {
                ensureSizeIsEnough(nFeatures, descriptorSize, CV_32F, descriptors);
                compute_descriptors_gpu(descriptors, keypoints.ptr<float>(SURF_GPU::SF_X), keypoints.ptr<float>(SURF_GPU::SF_Y),
                    keypoints.ptr<float>(SURF_GPU::SF_SIZE), keypoints.ptr<float>(SURF_GPU::SF_DIR), nFeatures);
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

        int img_cols, img_rows;

        bool use_mask;

        int maxCandidates;
        int maxFeatures;

        GpuMat counters;
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
        Mat keypointsCPU(SURF_GPU::SF_FEATURE_STRIDE, static_cast<int>(keypoints.size()), CV_32FC1);

        float* kp_x = keypointsCPU.ptr<float>(SURF_GPU::SF_X);
        float* kp_y = keypointsCPU.ptr<float>(SURF_GPU::SF_Y);
        int* kp_laplacian = keypointsCPU.ptr<int>(SURF_GPU::SF_LAPLACIAN);
        float* kp_size = keypointsCPU.ptr<float>(SURF_GPU::SF_SIZE);
        float* kp_dir = keypointsCPU.ptr<float>(SURF_GPU::SF_DIR);
        float* kp_hessian = keypointsCPU.ptr<float>(SURF_GPU::SF_HESSIAN);

        for (size_t i = 0, size = keypoints.size(); i < size; ++i)
        {
            const KeyPoint& kp = keypoints[i];
            kp_x[i] = kp.pt.x;
            kp_y[i] = kp.pt.y;
            kp_size[i] = kp.size;
            kp_dir[i] = kp.angle;
            kp_hessian[i] = kp.response;
            kp_laplacian[i] = 1;
        }

        keypointsGPU.upload(keypointsCPU);
    }
}

namespace
{
    int getPointOctave(float size, const CvSURFParams& params)
    {
        int best_octave = 0;
        float min_diff = numeric_limits<float>::max();
        for (int octave = 1; octave < params.nOctaves; ++octave)
        {
            for (int layer = 0; layer < params.nOctaveLayers; ++layer)
            {
                float diff = std::abs(size - (float)calcSize(octave, layer));
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
    const int nFeatures = keypointsGPU.cols;

    if (nFeatures == 0)
        keypoints.clear();
    else
    {
        CV_Assert(keypointsGPU.type() == CV_32FC1 && keypointsGPU.rows == SF_FEATURE_STRIDE);
        
        Mat keypointsCPU(keypointsGPU);
        
        keypoints.resize(nFeatures);

        float* kp_x = keypointsCPU.ptr<float>(SF_X);
        float* kp_y = keypointsCPU.ptr<float>(SF_Y);
        int* kp_laplacian = keypointsCPU.ptr<int>(SF_LAPLACIAN);
        float* kp_size = keypointsCPU.ptr<float>(SF_SIZE);
        float* kp_dir = keypointsCPU.ptr<float>(SF_DIR);
        float* kp_hessian = keypointsCPU.ptr<float>(SF_HESSIAN);

        for (int i = 0; i < nFeatures; ++i)
        {
            KeyPoint& kp = keypoints[i];
            kp.pt.x = kp_x[i];
            kp.pt.y = kp_y[i];
            kp.class_id = kp_laplacian[i];
            kp.size = kp_size[i];
            kp.angle = kp_dir[i];
            kp.response = kp_hessian[i];
            kp.octave = getPointOctave(kp.size, *this);
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
            surf.findOrientation(keypoints);
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

void cv::gpu::SURF_GPU::releaseMemory() 
{
    sum.release(); 
    mask1.release();
    maskSum.release();
    intBuffer.release();
    det.release();
    trace.release();
    maxPosBuffer.release();
}

#endif /* !defined (HAVE_CUDA) */
