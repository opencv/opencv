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
#include <iostream>
#include <utility>

using namespace cv;
using namespace cv::gpu;

#define BLOCK_VERSION

#if !defined (HAVE_CUDA)

void cv::gpu::matchTemplate(const GpuMat&, const GpuMat&, GpuMat&, int) { throw_nogpu(); }

#else

#include <cufft.h>

namespace cv { namespace gpu { namespace imgproc 
{  
    void multiplyAndNormalizeSpects(int n, float scale, const cufftComplex* a,
                                    const cufftComplex* b, cufftComplex* c);

    void matchTemplateNaive_8U_SQDIFF(
            const DevMem2D image, const DevMem2D templ, DevMem2Df result);

    void matchTemplateNaive_32F_SQDIFF(
            const DevMem2D image, const DevMem2D templ, DevMem2Df result);

    void matchTemplatePrepared_8U_SQDIFF(
            int w, int h, const DevMem2Df image_sumsq, float templ_sumsq,
            DevMem2Df result);
}}}


namespace 
{
    void matchTemplate_32F_SQDIFF(const GpuMat&, const GpuMat&, GpuMat&);
    void matchTemplate_32F_CCORR(const GpuMat&, const GpuMat&, GpuMat&);
    void matchTemplate_8U_SQDIFF(const GpuMat&, const GpuMat&, GpuMat&);
    void matchTemplate_8U_CCORR(const GpuMat&, const GpuMat&, GpuMat&);


#ifdef BLOCK_VERSION
    void estimateBlockSize(int w, int h, int tw, int th, int& bw, int& bh)
    {
        const int scale = 40;
        const int bh_min = 1024;
        const int bw_min = 1024;
        bw = std::max(tw * scale, bw_min);
        bh = std::max(th * scale, bh_min);
        bw = std::min(bw, w);
        bh = std::min(bh, h);
    }
#endif
    
    void matchTemplate_32F_SQDIFF(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
        result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
        imgproc::matchTemplateNaive_32F_SQDIFF(image, templ, result);
    }


#ifdef BLOCK_VERSION
    void matchTemplate_32F_CCORR(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
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
#else
    void matchTemplate_32F_CCORR(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
        Size opt_size;
        opt_size.width = getOptimalDFTSize(image.cols);
        opt_size.height = getOptimalDFTSize(image.rows);

        cufftReal* image_data;
        cufftReal* templ_data;
        cufftReal* result_data;
        cudaMalloc((void**)&image_data, sizeof(cufftReal) * opt_size.area());
        cudaMalloc((void**)&templ_data, sizeof(cufftReal) * opt_size.area());
        cudaMalloc((void**)&result_data, sizeof(cufftReal) * opt_size.area());

        int spect_len = opt_size.height * (opt_size.width / 2 + 1);
        cufftComplex* image_spect;
        cufftComplex* templ_spect;
        cufftComplex* result_spect;
        cudaMalloc((void**)&image_spect, sizeof(cufftComplex) * spect_len);
        cudaMalloc((void**)&templ_spect, sizeof(cufftComplex) * spect_len);
        cudaMalloc((void**)&result_spect, sizeof(cufftComplex) * spect_len);

        GpuMat image_(image.size(), CV_32S, image.data, image.step);
        GpuMat image_cont(opt_size, CV_32S, image_data, opt_size.width * sizeof(cufftReal));
        copyMakeBorder(image_, image_cont, 0, image_cont.rows - image.rows, 0, 
                       image_cont.cols - image.cols, 0);

        GpuMat templ_(templ.size(), CV_32S, templ.data, templ.step);
        GpuMat templ_cont(opt_size, CV_32S, templ_data, opt_size.width * sizeof(cufftReal));
        copyMakeBorder(templ_, templ_cont, 0, templ_cont.rows - templ.rows, 0, 
                       templ_cont.cols - templ.cols, 0);

        cufftHandle planR2C, planC2R;
        CV_Assert(cufftPlan2d(&planC2R, opt_size.height, opt_size.width, CUFFT_C2R) == CUFFT_SUCCESS);
        CV_Assert(cufftPlan2d(&planR2C, opt_size.height, opt_size.width, CUFFT_R2C) == CUFFT_SUCCESS);

        CV_Assert(cufftExecR2C(planR2C, image_data, image_spect) == CUFFT_SUCCESS);
        CV_Assert(cufftExecR2C(planR2C, templ_data, templ_spect) == CUFFT_SUCCESS);
        imgproc::multiplyAndNormalizeSpects(spect_len, 1.f / opt_size.area(), 
                                            image_spect, templ_spect, result_spect);

        CV_Assert(cufftExecC2R(planC2R, result_spect, result_data) == CUFFT_SUCCESS);

        cufftDestroy(planR2C);
        cufftDestroy(planC2R);

        GpuMat result_cont(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F, 
                           result_data, opt_size.width * sizeof(cufftReal));
        result_cont.copyTo(result);

        cudaFree(image_spect);
        cudaFree(templ_spect);
        cudaFree(result_spect);
        cudaFree(image_data);
        cudaFree(templ_data);
        cudaFree(result_data);
    }
#endif


    void matchTemplate_8U_SQDIFF(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
        result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
        imgproc::matchTemplateNaive_8U_SQDIFF(image, templ, result);
    }

    
    void matchTemplate_8U_CCORR(const GpuMat& image, const GpuMat& templ, GpuMat& result)
    {
        GpuMat imagef, templf;
        image.convertTo(imagef, CV_32F);
        templ.convertTo(templf, CV_32F);
        matchTemplate_32F_CCORR(imagef, templf, result);
    }
}


void cv::gpu::matchTemplate(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method)
{
    CV_Assert(image.type() == templ.type());
    CV_Assert(image.cols >= templ.cols && image.rows >= templ.rows);

    typedef void (*Caller)(const GpuMat&, const GpuMat&, GpuMat&);

    static const Caller callers8U[] = { ::matchTemplate_8U_SQDIFF, 0, 
                                        ::matchTemplate_8U_CCORR, 0, 0, 0 };
    static const Caller callers32F[] = { ::matchTemplate_32F_SQDIFF, 0, 
                                         ::matchTemplate_32F_CCORR, 0, 0, 0 };

    const Caller* callers;
    switch (image.type())
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

