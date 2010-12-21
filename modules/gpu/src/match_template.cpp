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

#include "precomp.hpp"
#include <utility>

using namespace cv;
using namespace cv::gpu;

#if !defined (HAVE_CUDA)

void cv::gpu::matchTemplate(const GpuMat&, const GpuMat&, GpuMat&, int) { throw_nogpu(); }

#else

#include <cufft.h>

namespace cv { namespace gpu { namespace imgproc 
{  
    void multiplyAndNormalizeSpects(int n, float scale, const cufftComplex* a,
                                    const cufftComplex* b, cufftComplex* c);

    void matchTemplateNaive_CCORR_8U(
            const DevMem2D image, const DevMem2D templ, DevMem2Df result, int cn);

    void matchTemplateNaive_CCORR_32F(
            const DevMem2D image, const DevMem2D templ, DevMem2Df result, int cn);

    void matchTemplateNaive_SQDIFF_8U(
            const DevMem2D image, const DevMem2D templ, DevMem2Df result, int cn);

    void matchTemplateNaive_SQDIFF_32F(
            const DevMem2D image, const DevMem2D templ, DevMem2Df result, int cn);

    void matchTemplatePrepared_SQDIFF_8U(
            int w, int h, const DevMem2D_<unsigned long long> image_sqsum, 
            unsigned int templ_sqsum, DevMem2Df result, int cn);

    void matchTemplatePrepared_SQDIFF_NORMED_8U(
            int w, int h, const DevMem2D_<unsigned long long> image_sqsum, 
            unsigned int templ_sqsum, DevMem2Df result, int cn);

    void matchTemplatePrepared_CCOFF_8U(
            int w, int h, const DevMem2D_<unsigned int> image_sum,
            unsigned int templ_sum, DevMem2Df result);

    void matchTemplatePrepared_CCOFF_8UC2(
            int w, int h, 
            const DevMem2D_<unsigned int> image_sum_r, 
            const DevMem2D_<unsigned int> image_sum_g,
            unsigned int templ_sum_r, unsigned int templ_sum_g, 
            DevMem2Df result);

    void matchTemplatePrepared_CCOFF_8UC3(
            int w, int h, 
            const DevMem2D_<unsigned int> image_sum_r, 
            const DevMem2D_<unsigned int> image_sum_g,
            const DevMem2D_<unsigned int> image_sum_b,
            unsigned int templ_sum_r, 
            unsigned int templ_sum_g, 
            unsigned int templ_sum_b, 
            DevMem2Df result);

    void matchTemplatePrepared_CCOFF_8UC4(
            int w, int h, 
            const DevMem2D_<unsigned int> image_sum_r, 
            const DevMem2D_<unsigned int> image_sum_g,
            const DevMem2D_<unsigned int> image_sum_b,
            const DevMem2D_<unsigned int> image_sum_a,
            unsigned int templ_sum_r, 
            unsigned int templ_sum_g, 
            unsigned int templ_sum_b, 
            unsigned int templ_sum_a, 
            DevMem2Df result);

    void matchTemplatePrepared_CCOFF_NORMED_8U(
            int w, int h, const DevMem2D_<unsigned int> image_sum, 
            const DevMem2D_<unsigned long long> image_sqsum,
            unsigned int templ_sum, unsigned int templ_sqsum,
            DevMem2Df result);

    void matchTemplatePrepared_CCOFF_NORMED_8UC2(
            int w, int h, 
            const DevMem2D_<unsigned int> image_sum_r, const DevMem2D_<unsigned long long> image_sqsum_r,
            const DevMem2D_<unsigned int> image_sum_g, const DevMem2D_<unsigned long long> image_sqsum_g,
            unsigned int templ_sum_r, unsigned int templ_sqsum_r,
            unsigned int templ_sum_g, unsigned int templ_sqsum_g,
            DevMem2Df result);

    void matchTemplatePrepared_CCOFF_NORMED_8UC3(
            int w, int h, 
            const DevMem2D_<unsigned int> image_sum_r, const DevMem2D_<unsigned long long> image_sqsum_r,
            const DevMem2D_<unsigned int> image_sum_g, const DevMem2D_<unsigned long long> image_sqsum_g,
            const DevMem2D_<unsigned int> image_sum_b, const DevMem2D_<unsigned long long> image_sqsum_b,
            unsigned int templ_sum_r, unsigned int templ_sqsum_r,
            unsigned int templ_sum_g, unsigned int templ_sqsum_g,
            unsigned int templ_sum_b, unsigned int templ_sqsum_b,
            DevMem2Df result);

    void matchTemplatePrepared_CCOFF_NORMED_8UC4(
            int w, int h, 
            const DevMem2D_<unsigned int> image_sum_r, const DevMem2D_<unsigned long long> image_sqsum_r,
            const DevMem2D_<unsigned int> image_sum_g, const DevMem2D_<unsigned long long> image_sqsum_g,
            const DevMem2D_<unsigned int> image_sum_b, const DevMem2D_<unsigned long long> image_sqsum_b,
            const DevMem2D_<unsigned int> image_sum_a, const DevMem2D_<unsigned long long> image_sqsum_a,
            unsigned int templ_sum_r, unsigned int templ_sqsum_r,
            unsigned int templ_sum_g, unsigned int templ_sqsum_g,
            unsigned int templ_sum_b, unsigned int templ_sqsum_b,
            unsigned int templ_sum_a, unsigned int templ_sqsum_a,
            DevMem2Df result);

    void normalize_8U(int w, int h, const DevMem2D_<unsigned long long> image_sqsum, 
                  unsigned int templ_sqsum, DevMem2Df result, int cn);

    void extractFirstChannel_32F(const DevMem2D image, DevMem2Df result, int cn);
}}}


namespace 
{
    // Computes integral image. Result matrix will have data type 32S,
    // while actuall data type is 32U
    void integral_8U_32U(const GpuMat& src, GpuMat& sum);

    // Computes squared integral image. Result matrix will have data type 64F,
    // while actual data type is 64U
    void sqrIntegral_8U_64U(const GpuMat& src, GpuMat& sqsum);

    // Estimates optimal blocks size for FFT method
    void estimateBlockSize(int w, int h, int tw, int th, int& bw, int& bh);

    // Performs FFT-based cross-correlation
    void crossCorr_32F(const GpuMat& image, const GpuMat& templ, GpuMat& result);

    // Evaluates optimal template's area threshold. If 
    // template's area is less  than the threshold, we use naive match 
    // template version, otherwise FFT-based (if available)
    int getTemplateThreshold(int method, int depth);

    void matchTemplate_CCORR_32F(const GpuMat& image, const GpuMat& templ, GpuMat& result);
    void matchTemplate_CCORR_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result);
    void matchTemplate_CCORR_NORMED_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result);

    void matchTemplate_SQDIFF_32F(const GpuMat& image, const GpuMat& templ, GpuMat& result);
    void matchTemplate_SQDIFF_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result);
    void matchTemplate_SQDIFF_NORMED_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result);

    void matchTemplate_CCOFF_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result);
    void matchTemplate_CCOFF_NORMED_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result);


    void integral_8U_32U(const GpuMat& src, GpuMat& sum)
    {
        CV_Assert(src.type() == CV_8U);

        NppStSize32u roiSize;
        roiSize.width = src.cols;
        roiSize.height = src.rows;

        NppSt32u bufSize;
        nppSafeCall(nppiStIntegralGetSize_8u32u(roiSize, &bufSize));
        GpuMat buf(1, bufSize, CV_8U);

        sum.create(src.rows + 1, src.cols + 1, CV_32S);
        nppSafeCall(nppiStIntegral_8u32u_C1R(
                const_cast<NppSt8u*>(src.ptr<NppSt8u>(0)), src.step, 
                sum.ptr<NppSt32u>(0), sum.step, roiSize, 
                buf.ptr<NppSt8u>(0), bufSize));
    }


    void sqrIntegral_8U_64U(const GpuMat& src, GpuMat& sqsum)
    {
        CV_Assert(src.type() == CV_8U);

        NppStSize32u roiSize;
        roiSize.width = src.cols;
        roiSize.height = src.rows;

        NppSt32u bufSize;
        nppSafeCall(nppiStSqrIntegralGetSize_8u64u(roiSize, &bufSize));
        GpuMat buf(1, bufSize, CV_8U);

        sqsum.create(src.rows + 1, src.cols + 1, CV_64F);
        nppSafeCall(nppiStSqrIntegral_8u64u_C1R(
                const_cast<NppSt8u*>(src.ptr<NppSt8u>(0)), src.step, 
                sqsum.ptr<NppSt64u>(0), sqsum.step, roiSize, 
                buf.ptr<NppSt8u>(0), bufSize));
    }


    void estimateBlockSize(int w, int h, int tw, int th, int& bw, int& bh)
    {
        int major, minor;
        getComputeCapability(getDevice(), major, minor);

        int scale = 40;
        int bh_min = 1024;
        int bw_min = 1024;

        if (major >= 2) // Fermi generation or newer
        {
            bh_min = 2048;
            bw_min = 2048;
        }

        bw = std::max(tw * scale, bw_min);
        bh = std::max(th * scale, bh_min);
        bw = std::min(bw, w);
        bh = std::min(bh, h);
    }


    void crossCorr_32F(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
        CV_Assert(image.type() == CV_32F);
        CV_Assert(templ.type() == CV_32F);

        result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);

        Size block_size;
        estimateBlockSize(result.cols, result.rows, templ.cols, templ.rows, 
                          block_size.width, block_size.height);

        Size dft_size;
        dft_size.width = getOptimalDFTSize(block_size.width + templ.cols - 1);
        dft_size.height = getOptimalDFTSize(block_size.width + templ.rows - 1);

        block_size.width = std::min(dft_size.width - templ.cols + 1, result.cols);
        block_size.height = std::min(dft_size.height - templ.rows + 1, result.rows);

        cufftReal* image_data;
        cufftReal* templ_data;
        cufftReal* result_data;
        cudaMalloc((void**)&image_data, sizeof(cufftReal) * dft_size.area());
        cudaMalloc((void**)&templ_data, sizeof(cufftReal) * dft_size.area());
        cudaMalloc((void**)&result_data, sizeof(cufftReal) * dft_size.area());

        int spect_len = dft_size.height * (dft_size.width / 2 + 1);
        cufftComplex* image_spect;
        cufftComplex* templ_spect;
        cufftComplex* result_spect;
        cudaMalloc((void**)&image_spect, sizeof(cufftComplex) * spect_len);
        cudaMalloc((void**)&templ_spect, sizeof(cufftComplex) * spect_len);
        cudaMalloc((void**)&result_spect, sizeof(cufftComplex) * spect_len);

        cufftHandle planR2C, planC2R;
        CV_Assert(cufftPlan2d(&planC2R, dft_size.height, dft_size.width, CUFFT_C2R) == CUFFT_SUCCESS);
        CV_Assert(cufftPlan2d(&planR2C, dft_size.height, dft_size.width, CUFFT_R2C) == CUFFT_SUCCESS);

        GpuMat templ_roi(templ.size(), CV_32S, templ.data, templ.step);
        GpuMat templ_block(dft_size, CV_32S, templ_data, dft_size.width * sizeof(cufftReal));
        copyMakeBorder(templ_roi, templ_block, 0, templ_block.rows - templ_roi.rows, 0, 
                       templ_block.cols - templ_roi.cols, 0);
        CV_Assert(cufftExecR2C(planR2C, templ_data, templ_spect) == CUFFT_SUCCESS);

        GpuMat image_block(dft_size, CV_32S, image_data, dft_size.width * sizeof(cufftReal));

        for (int y = 0; y < result.rows; y += block_size.height)
        {
            for (int x = 0; x < result.cols; x += block_size.width)
            {                
                Size image_roi_size;
                image_roi_size.width = min(x + dft_size.width, image.cols) - x;
                image_roi_size.height = min(y + dft_size.height, image.rows) - y;
                GpuMat image_roi(image_roi_size, CV_32S, (void*)(image.ptr<float>(y) + x), image.step);
                copyMakeBorder(image_roi, image_block, 0, image_block.rows - image_roi.rows, 0, 
                               image_block.cols - image_roi.cols, 0);

                CV_Assert(cufftExecR2C(planR2C, image_data, image_spect) == CUFFT_SUCCESS);
                imgproc::multiplyAndNormalizeSpects(spect_len, 1.f / dft_size.area(), 
                                                    image_spect, templ_spect, result_spect);
                CV_Assert(cufftExecC2R(planC2R, result_spect, result_data) == CUFFT_SUCCESS);

                Size result_roi_size;
                result_roi_size.width = min(x + block_size.width, result.cols) - x;
                result_roi_size.height = min(y + block_size.height, result.rows) - y;
                GpuMat result_roi(result_roi_size, CV_32F, (void*)(result.ptr<float>(y) + x), result.step);
                GpuMat result_block(result_roi_size, CV_32F, result_data, dft_size.width * sizeof(cufftReal));
                result_block.copyTo(result_roi);
            }
        }

        cufftDestroy(planR2C);
        cufftDestroy(planC2R);

        cudaFree(image_spect);
        cudaFree(templ_spect);
        cudaFree(result_spect);
        cudaFree(image_data);
        cudaFree(templ_data);
        cudaFree(result_data);
    }


    int getTemplateThreshold(int method, int depth)
    {
        switch (method)
        {
        case CV_TM_CCORR: 
            if (depth == CV_32F) return 250;
            if (depth == CV_8U) return 300;
            break;
        case CV_TM_SQDIFF:
            if (depth == CV_8U) return 500;
            break;
        }
        CV_Error(CV_StsBadArg, "getTemplateThreshold: unsupported match template mode");
        return 0;
    }

    
    void matchTemplate_CCORR_32F(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
        result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
        if (templ.size().area() < getTemplateThreshold(CV_TM_CCORR, CV_32F))
        {
            imgproc::matchTemplateNaive_CCORR_32F(image, templ, result, image.channels());
            return;
        }

        GpuMat result_;
        crossCorr_32F(image.reshape(1), templ.reshape(1), result_);
        imgproc::extractFirstChannel_32F(result_, result, image.channels());
    }


    void matchTemplate_CCORR_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
        if (templ.size().area() < getTemplateThreshold(CV_TM_CCORR, CV_8U))
        {
            result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
            imgproc::matchTemplateNaive_CCORR_8U(image, templ, result, image.channels());
            return;
        }

        GpuMat imagef, templf;
        image.convertTo(imagef, CV_32F);
        templ.convertTo(templf, CV_32F);
        matchTemplate_CCORR_32F(imagef, templf, result);
    }


    void matchTemplate_CCORR_NORMED_8U(const GpuMat& image, const GpuMat& templ, 
                                       GpuMat& result)
    {
        matchTemplate_CCORR_8U(image, templ, result);

        GpuMat img_sqsum;
        sqrIntegral_8U_64U(image.reshape(1), img_sqsum);

        unsigned int templ_sqsum = (unsigned int)sqrSum(templ.reshape(1))[0];
        imgproc::normalize_8U(templ.cols, templ.rows, img_sqsum, templ_sqsum, 
                              result, image.channels());
    }

    
    void matchTemplate_SQDIFF_32F(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
        result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
        imgproc::matchTemplateNaive_SQDIFF_32F(image, templ, result, image.channels());
    }


    void matchTemplate_SQDIFF_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
        if (templ.size().area() < getTemplateThreshold(CV_TM_SQDIFF, CV_8U))
        {
            result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
            imgproc::matchTemplateNaive_SQDIFF_8U(image, templ, result, image.channels());
            return;
        }

        GpuMat img_sqsum;
        sqrIntegral_8U_64U(image.reshape(1), img_sqsum);

        unsigned int templ_sqsum = (unsigned int)sqrSum(templ.reshape(1))[0];

        matchTemplate_CCORR_8U(image, templ, result);
        imgproc::matchTemplatePrepared_SQDIFF_8U(
                templ.cols, templ.rows, img_sqsum, templ_sqsum, result, image.channels());
    }


    void matchTemplate_SQDIFF_NORMED_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
        GpuMat img_sqsum;
        sqrIntegral_8U_64U(image.reshape(1), img_sqsum);

        unsigned int templ_sqsum = (unsigned int)sqrSum(templ.reshape(1))[0];

        matchTemplate_CCORR_8U(image, templ, result);
        imgproc::matchTemplatePrepared_SQDIFF_NORMED_8U(
                templ.cols, templ.rows, img_sqsum, templ_sqsum, result, image.channels());
    }


    void matchTemplate_CCOFF_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
        matchTemplate_CCORR_8U(image, templ, result);

        if (image.channels() == 1)
        {
            GpuMat image_sum;
            integral_8U_32U(image, image_sum);

            unsigned int templ_sum = (unsigned int)sum(templ)[0];
            imgproc::matchTemplatePrepared_CCOFF_8U(templ.cols, templ.rows, 
                                                    image_sum, templ_sum, result);
        }
        else
        {
            std::vector<GpuMat> images;
            std::vector<GpuMat> image_sums(image.channels());

            split(image, images);
            for (int i = 0; i < image.channels(); ++i)
                integral_8U_32U(images[i], image_sums[i]);

            Scalar templ_sum = sum(templ);

            switch (image.channels())
            {
            case 2:
                imgproc::matchTemplatePrepared_CCOFF_8UC2(
                        templ.cols, templ.rows, image_sums[0], image_sums[1],
                        (unsigned int)templ_sum[0], (unsigned int)templ_sum[1],
                        result);
                break;
            case 3:
                imgproc::matchTemplatePrepared_CCOFF_8UC3(
                        templ.cols, templ.rows, image_sums[0], image_sums[1], image_sums[2],
                        (unsigned int)templ_sum[0], (unsigned int)templ_sum[1], (unsigned int)templ_sum[2],
                        result);
                break;
            case 4:
                imgproc::matchTemplatePrepared_CCOFF_8UC4(
                        templ.cols, templ.rows, image_sums[0], image_sums[1], image_sums[2], image_sums[3],
                        (unsigned int)templ_sum[0], (unsigned int)templ_sum[1], (unsigned int)templ_sum[2],
                        (unsigned int)templ_sum[3], result);
                break;
            default:
                CV_Error(CV_StsBadArg, "matchTemplate: unsupported number of channels");
            }
        }
    }


    void matchTemplate_CCOFF_NORMED_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
        GpuMat imagef, templf;
        image.convertTo(imagef, CV_32F);
        templ.convertTo(templf, CV_32F);
        matchTemplate_CCORR_32F(imagef, templf, result);

        if (image.channels() == 1)
        {
            GpuMat image_sum, image_sqsum;
            integral_8U_32U(image, image_sum);
            sqrIntegral_8U_64U(image, image_sqsum);

            unsigned int templ_sum = (unsigned int)sum(templ)[0];
            unsigned int templ_sqsum = (unsigned int)sqrSum(templ)[0];

            imgproc::matchTemplatePrepared_CCOFF_NORMED_8U(
                    templ.cols, templ.rows, image_sum, image_sqsum, 
                    templ_sum, templ_sqsum, result);
        }
        else
        {
            std::vector<GpuMat> images;
            std::vector<GpuMat> image_sums(image.channels());
            std::vector<GpuMat> image_sqsums(image.channels());

            split(image, images);
            for (int i = 0; i < image.channels(); ++i)
            {
                integral_8U_32U(images[i], image_sums[i]);
                sqrIntegral_8U_64U(images[i], image_sqsums[i]);
            }

            Scalar templ_sum = sum(templ);
            Scalar templ_sqsum = sqrSum(templ);

            switch (image.channels())
            {
            case 2:
                imgproc::matchTemplatePrepared_CCOFF_NORMED_8UC2(
                        templ.cols, templ.rows, 
                        image_sums[0], image_sqsums[0],
                        image_sums[1], image_sqsums[1],
                        (unsigned int)templ_sum[0], (unsigned int)templ_sqsum[0],
                        (unsigned int)templ_sum[1], (unsigned int)templ_sqsum[1],
                        result);
                break;
            case 3:
                imgproc::matchTemplatePrepared_CCOFF_NORMED_8UC3(
                        templ.cols, templ.rows, 
                        image_sums[0], image_sqsums[0],
                        image_sums[1], image_sqsums[1],
                        image_sums[2], image_sqsums[2],
                        (unsigned int)templ_sum[0], (unsigned int)templ_sqsum[0],
                        (unsigned int)templ_sum[1], (unsigned int)templ_sqsum[1],
                        (unsigned int)templ_sum[2], (unsigned int)templ_sqsum[2],
                        result);
                break;
            case 4:
                imgproc::matchTemplatePrepared_CCOFF_NORMED_8UC4(
                        templ.cols, templ.rows, 
                        image_sums[0], image_sqsums[0],
                        image_sums[1], image_sqsums[1],
                        image_sums[2], image_sqsums[2],
                        image_sums[3], image_sqsums[3],
                        (unsigned int)templ_sum[0], (unsigned int)templ_sqsum[0],
                        (unsigned int)templ_sum[1], (unsigned int)templ_sqsum[1],
                        (unsigned int)templ_sum[2], (unsigned int)templ_sqsum[2],
                        (unsigned int)templ_sum[3], (unsigned int)templ_sqsum[3],
                        result);                
                break;
            default:
                CV_Error(CV_StsBadArg, "matchTemplate: unsupported number of channels");
            }
        }
    }
}


void cv::gpu::matchTemplate(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method)
{
    CV_Assert(image.type() == templ.type());
    CV_Assert(image.cols >= templ.cols && image.rows >= templ.rows);

    typedef void (*Caller)(const GpuMat&, const GpuMat&, GpuMat&);

    static const Caller callers8U[] = { ::matchTemplate_SQDIFF_8U, ::matchTemplate_SQDIFF_NORMED_8U, 
                                        ::matchTemplate_CCORR_8U, ::matchTemplate_CCORR_NORMED_8U, 
                                        ::matchTemplate_CCOFF_8U, ::matchTemplate_CCOFF_NORMED_8U };
    static const Caller callers32F[] = { ::matchTemplate_SQDIFF_32F, 0, 
                                         ::matchTemplate_CCORR_32F, 0, 0, 0 };

    const Caller* callers;
    switch (image.depth())
    {
        case CV_8U: callers = callers8U; break;
        case CV_32F: callers = callers32F; break;
        default: CV_Error(CV_StsBadArg, "matchTemplate: unsupported data type");
    }

    Caller caller = callers[method];
    CV_Assert(caller);
    caller(image, templ, result);
}

#endif


