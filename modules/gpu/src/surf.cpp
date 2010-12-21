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

int cv::gpu::SURF_GPU::descriptorSize() const { throw_nogpu(); return 0;}
void cv::gpu::SURF_GPU::uploadKeypoints(const vector<KeyPoint>&, GpuMat&) { throw_nogpu(); }
void cv::gpu::SURF_GPU::downloadKeypoints(const GpuMat&, vector<KeyPoint>&) { throw_nogpu(); }
void cv::gpu::SURF_GPU::downloadDescriptors(const GpuMat&, vector<float>&) { throw_nogpu(); }
void cv::gpu::SURF_GPU::operator()(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::SURF_GPU::operator()(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool, bool) { throw_nogpu(); }
void cv::gpu::SURF_GPU::operator()(const GpuMat&, const GpuMat&, vector<KeyPoint>&) { throw_nogpu(); }
void cv::gpu::SURF_GPU::operator()(const GpuMat&, const GpuMat&, vector<KeyPoint>&, GpuMat&, bool, bool) { throw_nogpu(); }
void cv::gpu::SURF_GPU::operator()(const GpuMat&, const GpuMat&, vector<KeyPoint>&, vector<float>&, bool, bool) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace surf
{    
    void fasthessian_gpu(PtrStepf hessianBuffer, int nIntervals, int x_size, int y_size);
    
    void nonmaxonly_gpu(PtrStepf hessianBuffer, int4* maxPosBuffer, unsigned int& maxCounter, 
        int nIntervals, int x_size, int y_size, bool use_mask);
    
    void fh_interp_extremum_gpu(PtrStepf hessianBuffer, const int4* maxPosBuffer, unsigned int maxCounter, 
        KeyPoint_GPU* featuresBuffer, unsigned int& featureCounter);
    
    void find_orientation_gpu(KeyPoint_GPU* features, int nFeatures);
    
    void compute_descriptors_gpu(const DevMem2Df& descriptors, const KeyPoint_GPU* features, int nFeatures);
}}}

using namespace cv::gpu::surf;

namespace
{
    class SURF_GPU_Invoker : private SURFParams_GPU
    {
    public:
        SURF_GPU_Invoker(SURF_GPU& surf, const GpuMat& img, const GpuMat& mask) : 
            SURFParams_GPU(surf),

            sum(surf.sum), sumf(surf.sumf),

            mask1(surf.mask1), maskSum(surf.maskSum),

            hessianBuffer(surf.hessianBuffer), 
            maxPosBuffer(surf.maxPosBuffer), 
            featuresBuffer(surf.featuresBuffer),

            img_cols(img.cols), img_rows(img.rows),

            use_mask(!mask.empty()),

            mask_width(0), mask_height(0),

            featureCounter(0), maxCounter(0)
        {
            CV_Assert(img.type() == CV_8UC1);
            CV_Assert(mask.empty() || (mask.size() == img.size() && mask.type() == CV_8UC1));
            CV_Assert(nOctaves > 0 && nIntervals > 2);
            CV_Assert(hasAtomicsSupport(getDevice()));

            max_features = static_cast<int>(img.size().area() * featuresRatio);
            max_candidates = static_cast<int>(1.5 * max_features);

            featuresBuffer.create(1, max_features, CV_32FC(6));
            maxPosBuffer.create(1, max_candidates, CV_32SC4);

            mask_width = l2 * 0.5f;
            mask_height = 1.0f + l1;

            // Dxy gap half-width
            float dxy_center_offset = 0.5f * (l4 + l3);
            // Dxy squares half-width
            float dxy_half_width = 0.5f * l3;

            // rescale edge_scale to fit with the filter dimensions
            float dxy_scale = edgeScale * std::pow((2.f + 2.f * l1) * l2 / (4.f * l3 * l3), 2.f);
            
            // Compute border required such that the filters don't overstep the image boundaries	        
	        float smax0 = 2.0f * initialScale + 0.5f;
            int border0 = static_cast<int>(std::ceil(smax0 * std::max(std::max(mask_width, mask_height), l3 + l4 * 0.5f)));

            int width0 = (img_cols - 2 * border0) / initialStep;
            int height0 = (img_rows - 2 * border0) / initialStep;

            uploadConstant("cv::gpu::surf::c_max_candidates",    max_candidates);
            uploadConstant("cv::gpu::surf::c_max_features",      max_features);
            uploadConstant("cv::gpu::surf::c_nIntervals",        nIntervals);
            uploadConstant("cv::gpu::surf::c_mask_width",        mask_width);
            uploadConstant("cv::gpu::surf::c_mask_height",       mask_height);
            uploadConstant("cv::gpu::surf::c_dxy_center_offset", dxy_center_offset);
            uploadConstant("cv::gpu::surf::c_dxy_half_width",    dxy_half_width);
            uploadConstant("cv::gpu::surf::c_dxy_scale",         dxy_scale);
            uploadConstant("cv::gpu::surf::c_initialScale",      initialScale);
            uploadConstant("cv::gpu::surf::c_threshold",         threshold);
            
            hessianBuffer.create(height0 * nIntervals, width0, CV_32F);

            integral(img, sum);
            sum.convertTo(sumf, CV_32F, 1.0 / 255.0);
            
            bindTexture("cv::gpu::surf::sumTex", (DevMem2Df)sumf);

            if (!mask.empty())
		    {
                min(mask, 1.0, mask1);
                integral(mask1, maskSum);
            
                bindTexture("cv::gpu::surf::maskSumTex", (DevMem2Di)maskSum);
		    }
        }

        ~SURF_GPU_Invoker()
        {
            unbindTexture("cv::gpu::surf::sumTex");
            if (use_mask)
                unbindTexture("cv::gpu::surf::maskSumTex");
        }

        void detectKeypoints(GpuMat& keypoints)
        {
            for(int octave = 0; octave < nOctaves; ++octave)
            {
                int step = initialStep * (1 << octave);

                // Compute border required such that the filters don't overstep the image boundaries
                float d = (initialScale * (1 << octave)) / (nIntervals - 2);
	            float smax = initialScale * (1 << octave) + d * (nIntervals - 2.0f) + 0.5f;
                int border = static_cast<int>(std::ceil(smax * std::max(std::max(mask_width, mask_height), l3 + l4 * 0.5f)));
                
                int x_size = (img_cols - 2 * border) / step;
                int y_size = (img_rows - 2 * border) / step;
                
                if (x_size <= 0 || y_size <= 0)
                    break;

                uploadConstant("cv::gpu::surf::c_octave", octave);
                uploadConstant("cv::gpu::surf::c_x_size", x_size);
                uploadConstant("cv::gpu::surf::c_y_size", y_size);
                uploadConstant("cv::gpu::surf::c_border", border);
                uploadConstant("cv::gpu::surf::c_step",   step);

                fasthessian_gpu(hessianBuffer, nIntervals, x_size, y_size);

                // Reset the candidate count.
                maxCounter = 0;

                nonmaxonly_gpu(hessianBuffer, maxPosBuffer.ptr<int4>(), maxCounter, nIntervals, x_size, y_size, use_mask); 
                
                maxCounter = std::min(maxCounter, static_cast<unsigned int>(max_candidates));

                fh_interp_extremum_gpu(hessianBuffer, maxPosBuffer.ptr<int4>(), maxCounter,
                    featuresBuffer.ptr<KeyPoint_GPU>(), featureCounter);

                featureCounter = std::min(featureCounter, static_cast<unsigned int>(max_features));
            }

            featuresBuffer.colRange(0, featureCounter).copyTo(keypoints);
        }

        void findOrientation(GpuMat& keypoints)
        {
            if (keypoints.cols > 0)
                find_orientation_gpu(keypoints.ptr<KeyPoint_GPU>(), keypoints.cols);
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
        GpuMat& sumf;

        GpuMat& mask1;
        GpuMat& maskSum;

        GpuMat& hessianBuffer;
        GpuMat& maxPosBuffer;
        GpuMat& featuresBuffer;

        int img_cols, img_rows;

        bool use_mask;
        
        float mask_width, mask_height;

        unsigned int featureCounter;
        unsigned int maxCounter;

        int max_candidates;
        int max_features;
    };
}

int cv::gpu::SURF_GPU::descriptorSize() const
{
    return extended ? 128 : 64;
}

void cv::gpu::SURF_GPU::uploadKeypoints(const vector<KeyPoint>& keypoints, GpuMat& keypointsGPU)
{
    Mat keypointsCPU(1, keypoints.size(), CV_32FC(6));

    const KeyPoint* keypoints_ptr = &keypoints[0];
    KeyPoint_GPU* keypointsCPU_ptr = keypointsCPU.ptr<KeyPoint_GPU>();
    for (size_t i = 0; i < keypoints.size(); ++i, ++keypoints_ptr, ++keypointsCPU_ptr)
    {
        const KeyPoint& kp = *keypoints_ptr;
        KeyPoint_GPU& gkp = *keypointsCPU_ptr;

        gkp.x = kp.pt.x;
        gkp.y = kp.pt.y;

        gkp.size = kp.size;

        gkp.octave = static_cast<float>(kp.octave);
        gkp.angle = kp.angle;
        gkp.response = kp.response;
    }

    keypointsGPU.upload(keypointsCPU);
}

void cv::gpu::SURF_GPU::downloadKeypoints(const GpuMat& keypointsGPU, vector<KeyPoint>& keypoints)
{
    CV_Assert(keypointsGPU.type() == CV_32FC(6) && keypointsGPU.rows == 1);

    Mat keypointsCPU = keypointsGPU;
    keypoints.resize(keypointsGPU.cols);

    KeyPoint* keypoints_ptr = &keypoints[0];
    const KeyPoint_GPU* keypointsCPU_ptr = keypointsCPU.ptr<KeyPoint_GPU>();
    for (int i = 0; i < keypointsGPU.cols; ++i, ++keypoints_ptr, ++keypointsCPU_ptr)
    {
        KeyPoint& kp = *keypoints_ptr;
        const KeyPoint_GPU& gkp = *keypointsCPU_ptr;

        kp.pt.x = gkp.x;
        kp.pt.y = gkp.y;

        kp.size = gkp.size;

        kp.octave = static_cast<int>(gkp.octave);
        kp.angle = gkp.angle;
        kp.response = gkp.response;
    }
}

void cv::gpu::SURF_GPU::downloadDescriptors(const GpuMat& descriptorsGPU, vector<float>& descriptors)
{
    CV_Assert(descriptorsGPU.type() == CV_32F);

    descriptors.resize(descriptorsGPU.rows * descriptorsGPU.cols);
    Mat descriptorsCPU(descriptorsGPU.size(), CV_32F, &descriptors[0]);
    descriptorsGPU.download(descriptorsCPU);
}

void cv::gpu::SURF_GPU::operator()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints)
{
    SURF_GPU_Invoker surf(*this, img, mask);

    surf.detectKeypoints(keypoints);

    surf.findOrientation(keypoints);
}

void cv::gpu::SURF_GPU::operator()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints, GpuMat& descriptors, 
                                   bool useProvidedKeypoints, bool calcOrientation)
{
    SURF_GPU_Invoker surf(*this, img, mask);
    
    if (!useProvidedKeypoints)
        surf.detectKeypoints(keypoints);
    
    if (calcOrientation)
        surf.findOrientation(keypoints);

    surf.computeDescriptors(keypoints, descriptors, descriptorSize());
}

void cv::gpu::SURF_GPU::operator()(const GpuMat& img, const GpuMat& mask, vector<KeyPoint>& keypoints)
{
    GpuMat keypointsGPU;

    (*this)(img, mask, keypointsGPU);

    downloadKeypoints(keypointsGPU, keypoints);
}

void cv::gpu::SURF_GPU::operator()(const GpuMat& img, const GpuMat& mask, vector<KeyPoint>& keypoints, 
    GpuMat& descriptors, bool useProvidedKeypoints, bool calcOrientation)
{
    GpuMat keypointsGPU;

    if (useProvidedKeypoints)
        uploadKeypoints(keypoints, keypointsGPU);    

    (*this)(img, mask, keypointsGPU, descriptors, useProvidedKeypoints, calcOrientation);

    downloadKeypoints(keypointsGPU, keypoints);
}

void cv::gpu::SURF_GPU::operator()(const GpuMat& img, const GpuMat& mask, vector<KeyPoint>& keypoints, 
    vector<float>& descriptors, bool useProvidedKeypoints, bool calcOrientation)
{
    GpuMat descriptorsGPU;

    (*this)(img, mask, keypoints, descriptorsGPU, useProvidedKeypoints, calcOrientation);

    downloadDescriptors(descriptorsGPU, descriptors);
}

#endif /* !defined (HAVE_CUDA) */
