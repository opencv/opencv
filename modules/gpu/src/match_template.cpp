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

using namespace cv;
using namespace cv::gpu;

#if !defined (HAVE_CUDA)

void cv::gpu::matchTemplate(const GpuMat&, const GpuMat&, GpuMat&, int, Stream&) { throw_nogpu(); }

#else

namespace cv { namespace gpu { namespace imgproc 
{  
    void matchTemplateNaive_CCORR_8U(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, int cn, cudaStream_t stream);
    void matchTemplateNaive_CCORR_32F(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, int cn, cudaStream_t stream);

    void matchTemplateNaive_SQDIFF_8U(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, int cn, cudaStream_t stream);
    void matchTemplateNaive_SQDIFF_32F(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, int cn, cudaStream_t stream);

    void matchTemplatePrepared_SQDIFF_8U(int w, int h, const DevMem2D_<unsigned long long> image_sqsum, unsigned int templ_sqsum, DevMem2Df result, 
        int cn, cudaStream_t stream);

    void matchTemplatePrepared_SQDIFF_NORMED_8U(int w, int h, const DevMem2D_<unsigned long long> image_sqsum, unsigned int templ_sqsum, DevMem2Df result, 
        int cn, cudaStream_t stream);

    void matchTemplatePrepared_CCOFF_8U(int w, int h, const DevMem2D_<unsigned int> image_sum, unsigned int templ_sum, DevMem2Df result, cudaStream_t stream);
    void matchTemplatePrepared_CCOFF_8UC2(
        int w, int h,
        const DevMem2D_<unsigned int> image_sum_r, 
        const DevMem2D_<unsigned int> image_sum_g, 
        unsigned int templ_sum_r,
        unsigned int templ_sum_g, 
        DevMem2Df result, cudaStream_t stream);
    void matchTemplatePrepared_CCOFF_8UC3(
            int w, int h, 
            const DevMem2D_<unsigned int> image_sum_r, 
            const DevMem2D_<unsigned int> image_sum_g,
            const DevMem2D_<unsigned int> image_sum_b,
            unsigned int templ_sum_r, 
            unsigned int templ_sum_g, 
            unsigned int templ_sum_b, 
            DevMem2Df result, cudaStream_t stream);
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
            DevMem2Df result, cudaStream_t stream);


    void matchTemplatePrepared_CCOFF_NORMED_8U(
            int w, int h, const DevMem2D_<unsigned int> image_sum, 
            const DevMem2D_<unsigned long long> image_sqsum,
            unsigned int templ_sum, unsigned int templ_sqsum,
            DevMem2Df result, cudaStream_t stream);
    void matchTemplatePrepared_CCOFF_NORMED_8UC2(
            int w, int h, 
            const DevMem2D_<unsigned int> image_sum_r, const DevMem2D_<unsigned long long> image_sqsum_r,
            const DevMem2D_<unsigned int> image_sum_g, const DevMem2D_<unsigned long long> image_sqsum_g,
            unsigned int templ_sum_r, unsigned int templ_sqsum_r,
            unsigned int templ_sum_g, unsigned int templ_sqsum_g,
            DevMem2Df result, cudaStream_t stream);
    void matchTemplatePrepared_CCOFF_NORMED_8UC3(
            int w, int h, 
            const DevMem2D_<unsigned int> image_sum_r, const DevMem2D_<unsigned long long> image_sqsum_r,
            const DevMem2D_<unsigned int> image_sum_g, const DevMem2D_<unsigned long long> image_sqsum_g,
            const DevMem2D_<unsigned int> image_sum_b, const DevMem2D_<unsigned long long> image_sqsum_b,
            unsigned int templ_sum_r, unsigned int templ_sqsum_r,
            unsigned int templ_sum_g, unsigned int templ_sqsum_g,
            unsigned int templ_sum_b, unsigned int templ_sqsum_b,
            DevMem2Df result, cudaStream_t stream);
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
            DevMem2Df result, cudaStream_t stream);

    void normalize_8U(int w, int h, const DevMem2D_<unsigned long long> image_sqsum, 
                      unsigned int templ_sqsum, DevMem2Df result, int cn, cudaStream_t stream);

    void extractFirstChannel_32F(const DevMem2Db image, DevMem2Df result, int cn, cudaStream_t stream);
}}}


namespace 
{

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


    int getTemplateThreshold(int method, int depth)
    {
        switch (method)
        {
        case CV_TM_CCORR: 
            if (depth == CV_32F) return 250;
            if (depth == CV_8U) return 300;
            break;
        case CV_TM_SQDIFF:
            if (depth == CV_8U) return 300;
            break;
        }
        CV_Error(CV_StsBadArg, "getTemplateThreshold: unsupported match template mode");
        return 0;
    }

    
    void matchTemplate_CCORR_32F(const GpuMat& image, const GpuMat& templ, GpuMat& result, Stream& stream)
    {
        result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
        if (templ.size().area() < getTemplateThreshold(CV_TM_CCORR, CV_32F))
        {
            imgproc::matchTemplateNaive_CCORR_32F(image, templ, result, image.channels(), StreamAccessor::getStream(stream));
            return;
        }

        GpuMat result_;
        ConvolveBuf buf;
        convolve(image.reshape(1), templ.reshape(1), result_, true, buf, stream);
        imgproc::extractFirstChannel_32F(result_, result, image.channels(), StreamAccessor::getStream(stream));
    }


    void matchTemplate_CCORR_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result, Stream& stream)
    {
        if (templ.size().area() < getTemplateThreshold(CV_TM_CCORR, CV_8U))
        {
            result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
            imgproc::matchTemplateNaive_CCORR_8U(image, templ, result, image.channels(), StreamAccessor::getStream(stream));
            return;
        }

        GpuMat imagef, templf;
        if (stream)
        {
            stream.enqueueConvert(image, imagef, CV_32F);
            stream.enqueueConvert(templ, templf, CV_32F);
        }
        else
        {
            image.convertTo(imagef, CV_32F);
            templ.convertTo(templf, CV_32F);
        }
        matchTemplate_CCORR_32F(imagef, templf, result, stream);
    }


    void matchTemplate_CCORR_NORMED_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result, Stream& stream)
    {
        matchTemplate_CCORR_8U(image, templ, result, stream);

        GpuMat img_sqsum;
        sqrIntegral(image.reshape(1), img_sqsum, stream);

        unsigned int templ_sqsum = (unsigned int)sqrSum(templ.reshape(1))[0];
        imgproc::normalize_8U(templ.cols, templ.rows, img_sqsum, templ_sqsum, 
                              result, image.channels(), StreamAccessor::getStream(stream));
    }

    
    void matchTemplate_SQDIFF_32F(const GpuMat& image, const GpuMat& templ, GpuMat& result, Stream& stream)
    {
        result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
        imgproc::matchTemplateNaive_SQDIFF_32F(image, templ, result, image.channels(), StreamAccessor::getStream(stream));
    }


    void matchTemplate_SQDIFF_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result, Stream& stream)
    {
        if (templ.size().area() < getTemplateThreshold(CV_TM_SQDIFF, CV_8U))
        {
            result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
            imgproc::matchTemplateNaive_SQDIFF_8U(image, templ, result, image.channels(), StreamAccessor::getStream(stream));
            return;
        }

        GpuMat img_sqsum;
        sqrIntegral(image.reshape(1), img_sqsum, stream);

        unsigned int templ_sqsum = (unsigned int)sqrSum(templ.reshape(1))[0];

        matchTemplate_CCORR_8U(image, templ, result, stream);
        imgproc::matchTemplatePrepared_SQDIFF_8U(
                templ.cols, templ.rows, img_sqsum, templ_sqsum, result, image.channels(), StreamAccessor::getStream(stream));
    }


    void matchTemplate_SQDIFF_NORMED_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result, Stream& stream)
    {
        GpuMat img_sqsum;
        sqrIntegral(image.reshape(1), img_sqsum, stream);

        unsigned int templ_sqsum = (unsigned int)sqrSum(templ.reshape(1))[0];

        matchTemplate_CCORR_8U(image, templ, result, stream);
        imgproc::matchTemplatePrepared_SQDIFF_NORMED_8U(
                templ.cols, templ.rows, img_sqsum, templ_sqsum, result, image.channels(), StreamAccessor::getStream(stream));
    }


    void matchTemplate_CCOFF_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result, Stream& stream)
    {
        matchTemplate_CCORR_8U(image, templ, result, stream);

        if (image.channels() == 1)
        {
            GpuMat image_sum;
            integral(image, image_sum, stream);

            unsigned int templ_sum = (unsigned int)sum(templ)[0];
            imgproc::matchTemplatePrepared_CCOFF_8U(templ.cols, templ.rows, 
                                                    image_sum, templ_sum, result, StreamAccessor::getStream(stream));
        }
        else
        {
            std::vector<GpuMat> images;
            std::vector<GpuMat> image_sums(image.channels());

            split(image, images);
            for (int i = 0; i < image.channels(); ++i)
                integral(images[i], image_sums[i], stream);

            Scalar templ_sum = sum(templ);

            switch (image.channels())
            {
            case 2:
                imgproc::matchTemplatePrepared_CCOFF_8UC2(
                        templ.cols, templ.rows, image_sums[0], image_sums[1],
                        (unsigned int)templ_sum[0], (unsigned int)templ_sum[1],
                        result, StreamAccessor::getStream(stream));
                break;
            case 3:
                imgproc::matchTemplatePrepared_CCOFF_8UC3(
                        templ.cols, templ.rows, image_sums[0], image_sums[1], image_sums[2],
                        (unsigned int)templ_sum[0], (unsigned int)templ_sum[1], (unsigned int)templ_sum[2],
                        result, StreamAccessor::getStream(stream));
                break;
            case 4:
                imgproc::matchTemplatePrepared_CCOFF_8UC4(
                        templ.cols, templ.rows, image_sums[0], image_sums[1], image_sums[2], image_sums[3],
                        (unsigned int)templ_sum[0], (unsigned int)templ_sum[1], (unsigned int)templ_sum[2],
                        (unsigned int)templ_sum[3], result, StreamAccessor::getStream(stream));
                break;
            default:
                CV_Error(CV_StsBadArg, "matchTemplate: unsupported number of channels");
            }
        }
    }


    void matchTemplate_CCOFF_NORMED_8U(const GpuMat& image, const GpuMat& templ, GpuMat& result, Stream& stream)
    {
        GpuMat imagef, templf;
        if (stream)
        {
            stream.enqueueConvert(image, imagef, CV_32F);
            stream.enqueueConvert(templ, templf, CV_32F);
        }
        else
        {
            image.convertTo(imagef, CV_32F);
            templ.convertTo(templf, CV_32F);
        }

        matchTemplate_CCORR_32F(imagef, templf, result, stream);

        if (image.channels() == 1)
        {
            GpuMat image_sum, image_sqsum;
            integral(image, image_sum, stream);
            sqrIntegral(image, image_sqsum, stream);

            unsigned int templ_sum = (unsigned int)sum(templ)[0];
            unsigned int templ_sqsum = (unsigned int)sqrSum(templ)[0];

            imgproc::matchTemplatePrepared_CCOFF_NORMED_8U(
                    templ.cols, templ.rows, image_sum, image_sqsum, 
                    templ_sum, templ_sqsum, result, StreamAccessor::getStream(stream));
        }
        else
        {
            std::vector<GpuMat> images;
            std::vector<GpuMat> image_sums(image.channels());
            std::vector<GpuMat> image_sqsums(image.channels());

            split(image, images);
            for (int i = 0; i < image.channels(); ++i)
            {
                integral(images[i], image_sums[i], stream);
                sqrIntegral(images[i], image_sqsums[i], stream);
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
                        result, StreamAccessor::getStream(stream));
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
                        result, StreamAccessor::getStream(stream));
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
                        result, StreamAccessor::getStream(stream));                
                break;
            default:
                CV_Error(CV_StsBadArg, "matchTemplate: unsupported number of channels");
            }
        }
    }
}


void cv::gpu::matchTemplate(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method, Stream& stream)
{
    CV_Assert(image.type() == templ.type());
    CV_Assert(image.cols >= templ.cols && image.rows >= templ.rows);

    typedef void (*Caller)(const GpuMat&, const GpuMat&, GpuMat&, Stream& stream);

    static const Caller callers8U[] = { ::matchTemplate_SQDIFF_8U, ::matchTemplate_SQDIFF_NORMED_8U, 
                                        ::matchTemplate_CCORR_8U, ::matchTemplate_CCORR_NORMED_8U, 
                                        ::matchTemplate_CCOFF_8U, ::matchTemplate_CCOFF_NORMED_8U };
    static const Caller callers32F[] = { ::matchTemplate_SQDIFF_32F, 0, 
                                         ::matchTemplate_CCORR_32F, 0, 0, 0 };

    const Caller* callers = 0;
    switch (image.depth())
    {
        case CV_8U: callers = callers8U; break;
        case CV_32F: callers = callers32F; break;
        default: CV_Error(CV_StsBadArg, "matchTemplate: unsupported data type");
    }

    Caller caller = callers[method];
    CV_Assert(caller);
    caller(image, templ, result, stream);
}

#endif
