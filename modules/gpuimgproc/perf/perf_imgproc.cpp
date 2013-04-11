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

#include "perf_precomp.hpp"

using namespace std;
using namespace testing;
using namespace perf;

//////////////////////////////////////////////////////////////////////
// HistEvenC1

PERF_TEST_P(Sz_Depth, HistEvenC1,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_16S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::histEven(d_src, dst, d_buf, 30, 0, 180);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        const int hbins = 30;
        const float hranges[] = {0.0f, 180.0f};
        const int histSize[] = {hbins};
        const float* ranges[] = {hranges};
        const int channels[] = {0};

        cv::Mat dst;

        TEST_CYCLE() cv::calcHist(&src, 1, channels, cv::Mat(), dst, 1, histSize, ranges);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// HistEvenC4

PERF_TEST_P(Sz_Depth, HistEvenC4,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_16S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, CV_MAKE_TYPE(depth, 4));
    declare.in(src, WARMUP_RNG);

    int histSize[] = {30, 30, 30, 30};
    int lowerLevel[] = {0, 0, 0, 0};
    int upperLevel[] = {180, 180, 180, 180};

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_hist[4];
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::histEven(d_src, d_hist, d_buf, histSize, lowerLevel, upperLevel);

        cv::Mat cpu_hist0, cpu_hist1, cpu_hist2, cpu_hist3;
        d_hist[0].download(cpu_hist0);
        d_hist[1].download(cpu_hist1);
        d_hist[2].download(cpu_hist2);
        d_hist[3].download(cpu_hist3);
        SANITY_CHECK(cpu_hist0);
        SANITY_CHECK(cpu_hist1);
        SANITY_CHECK(cpu_hist2);
        SANITY_CHECK(cpu_hist3);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// CalcHist

PERF_TEST_P(Sz, CalcHist,
            GPU_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src(size, CV_8UC1);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::calcHist(d_src, dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// EqualizeHist

PERF_TEST_P(Sz, EqualizeHist,
            GPU_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src(size, CV_8UC1);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;
        cv::gpu::GpuMat d_hist;
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::equalizeHist(d_src, dst, d_hist, d_buf);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::equalizeHist(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

DEF_PARAM_TEST(Sz_ClipLimit, cv::Size, double);

PERF_TEST_P(Sz_ClipLimit, CLAHE,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(0.0, 40.0)))
{
    const cv::Size size = GET_PARAM(0);
    const double clipLimit = GET_PARAM(1);

    cv::Mat src(size, CV_8UC1);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        cv::Ptr<cv::gpu::CLAHE> clahe = cv::gpu::createCLAHE(clipLimit);
        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() clahe->apply(d_src, dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit);
        cv::Mat dst;

        TEST_CYCLE() clahe->apply(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Canny

DEF_PARAM_TEST(Image_AppertureSz_L2gradient, string, int, bool);

PERF_TEST_P(Image_AppertureSz_L2gradient, Canny,
            Combine(Values("perf/800x600.png", "perf/1280x1024.png", "perf/1680x1050.png"),
                    Values(3, 5),
                    Bool()))
{
    const string fileName = GET_PARAM(0);
    const int apperture_size = GET_PARAM(1);
    const bool useL2gradient = GET_PARAM(2);

    const cv::Mat image = readImage(fileName, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    const double low_thresh = 50.0;
    const double high_thresh = 100.0;

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_image(image);
        cv::gpu::GpuMat dst;
        cv::gpu::CannyBuf d_buf;

        TEST_CYCLE() cv::gpu::Canny(d_image, d_buf, dst, low_thresh, high_thresh, apperture_size, useL2gradient);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::Canny(image, dst, low_thresh, high_thresh, apperture_size, useL2gradient);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MeanShiftFiltering

DEF_PARAM_TEST_1(Image, string);

PERF_TEST_P(Image, MeanShiftFiltering,
            Values<string>("gpu/meanshift/cones.png"))
{
    declare.time(300.0);

    const cv::Mat img = readImage(GetParam());
    ASSERT_FALSE(img.empty());

    cv::Mat rgba;
    cv::cvtColor(img, rgba, cv::COLOR_BGR2BGRA);

    const int sp = 50;
    const int sr = 50;

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(rgba);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::meanShiftFiltering(d_src, dst, sp, sr);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::pyrMeanShiftFiltering(img, dst, sp, sr);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MeanShiftProc

PERF_TEST_P(Image, MeanShiftProc,
            Values<string>("gpu/meanshift/cones.png"))
{
    declare.time(300.0);

    const cv::Mat img = readImage(GetParam());
    ASSERT_FALSE(img.empty());

    cv::Mat rgba;
    cv::cvtColor(img, rgba, cv::COLOR_BGR2BGRA);

    const int sp = 50;
    const int sr = 50;

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(rgba);
        cv::gpu::GpuMat dstr;
        cv::gpu::GpuMat dstsp;

        TEST_CYCLE() cv::gpu::meanShiftProc(d_src, dstr, dstsp, sp, sr);

        GPU_SANITY_CHECK(dstr);
        GPU_SANITY_CHECK(dstsp);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// MeanShiftSegmentation

PERF_TEST_P(Image, MeanShiftSegmentation,
            Values<string>("gpu/meanshift/cones.png"))
{
    declare.time(300.0);

    const cv::Mat img = readImage(GetParam());
    ASSERT_FALSE(img.empty());

    cv::Mat rgba;
    cv::cvtColor(img, rgba, cv::COLOR_BGR2BGRA);

    const int sp = 10;
    const int sr = 10;
    const int minsize = 20;

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(rgba);
        cv::Mat dst;

        TEST_CYCLE() cv::gpu::meanShiftSegmentation(d_src, dst, sp, sr, minsize);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// BlendLinear

PERF_TEST_P(Sz_Depth_Cn, BlendLinear,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_32F),
                    GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat img1(size, type);
    cv::Mat img2(size, type);
    declare.in(img1, img2, WARMUP_RNG);

    const cv::Mat weights1(size, CV_32FC1, cv::Scalar::all(0.5));
    const cv::Mat weights2(size, CV_32FC1, cv::Scalar::all(0.5));

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_img1(img1);
        const cv::gpu::GpuMat d_img2(img2);
        const cv::gpu::GpuMat d_weights1(weights1);
        const cv::gpu::GpuMat d_weights2(weights2);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::blendLinear(d_img1, d_img2, d_weights1, d_weights2, dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate8U

CV_ENUM(TemplateMethod, cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED)
#define ALL_TEMPLATE_METHODS ValuesIn(TemplateMethod::all())

DEF_PARAM_TEST(Sz_TemplateSz_Cn_Method, cv::Size, cv::Size, MatCn, TemplateMethod);

PERF_TEST_P(Sz_TemplateSz_Cn_Method, MatchTemplate8U,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(cv::Size(5, 5), cv::Size(16, 16), cv::Size(30, 30)),
                    GPU_CHANNELS_1_3_4,
                    ALL_TEMPLATE_METHODS))
{
    declare.time(300.0);

    const cv::Size size = GET_PARAM(0);
    const cv::Size templ_size = GET_PARAM(1);
    const int cn = GET_PARAM(2);
    const int method = GET_PARAM(3);

    cv::Mat image(size, CV_MAKE_TYPE(CV_8U, cn));
    cv::Mat templ(templ_size, CV_MAKE_TYPE(CV_8U, cn));
    declare.in(image, templ, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_image(image);
        const cv::gpu::GpuMat d_templ(templ);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::matchTemplate(d_image, d_templ, dst, method);

        GPU_SANITY_CHECK(dst, 1e-5, ERROR_RELATIVE);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::matchTemplate(image, templ, dst, method);

        CPU_SANITY_CHECK(dst);
    }
};

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate32F

PERF_TEST_P(Sz_TemplateSz_Cn_Method, MatchTemplate32F,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(cv::Size(5, 5), cv::Size(16, 16), cv::Size(30, 30)),
                    GPU_CHANNELS_1_3_4,
                    Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_CCORR))))
{
    declare.time(300.0);

    const cv::Size size = GET_PARAM(0);
    const cv::Size templ_size = GET_PARAM(1);
    const int cn = GET_PARAM(2);
    int method = GET_PARAM(3);

    cv::Mat image(size, CV_MAKE_TYPE(CV_32F, cn));
    cv::Mat templ(templ_size, CV_MAKE_TYPE(CV_32F, cn));
    declare.in(image, templ, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_image(image);
        const cv::gpu::GpuMat d_templ(templ);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::matchTemplate(d_image, d_templ, dst, method);

        GPU_SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::matchTemplate(image, templ, dst, method);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// CornerHarris

DEF_PARAM_TEST(Image_Type_Border_BlockSz_ApertureSz, string, MatType, BorderMode, int, int);

PERF_TEST_P(Image_Type_Border_BlockSz_ApertureSz, CornerHarris,
            Combine(Values<string>("gpu/stereobm/aloe-L.png"),
                    Values(CV_8UC1, CV_32FC1),
                    Values(BorderMode(cv::BORDER_REFLECT101), BorderMode(cv::BORDER_REPLICATE), BorderMode(cv::BORDER_REFLECT)),
                    Values(3, 5, 7),
                    Values(0, 3, 5, 7)))
{
    const string fileName = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int borderMode = GET_PARAM(2);
    const int blockSize = GET_PARAM(3);
    const int apertureSize = GET_PARAM(4);

    cv::Mat img = readImage(fileName, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    img.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

    const double k = 0.5;

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat dst;
        cv::gpu::GpuMat d_Dx;
        cv::gpu::GpuMat d_Dy;
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::cornerHarris(d_img, dst, d_Dx, d_Dy, d_buf, blockSize, apertureSize, k, borderMode);

        GPU_SANITY_CHECK(dst, 1e-4);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::cornerHarris(img, dst, blockSize, apertureSize, k, borderMode);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// CornerMinEigenVal

PERF_TEST_P(Image_Type_Border_BlockSz_ApertureSz, CornerMinEigenVal,
            Combine(Values<string>("gpu/stereobm/aloe-L.png"),
                    Values(CV_8UC1, CV_32FC1),
                    Values(BorderMode(cv::BORDER_REFLECT101), BorderMode(cv::BORDER_REPLICATE), BorderMode(cv::BORDER_REFLECT)),
                    Values(3, 5, 7),
                    Values(0, 3, 5, 7)))
{
    const string fileName = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int borderMode = GET_PARAM(2);
    const int blockSize = GET_PARAM(3);
    const int apertureSize = GET_PARAM(4);

    cv::Mat img = readImage(fileName, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    img.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat dst;
        cv::gpu::GpuMat d_Dx;
        cv::gpu::GpuMat d_Dy;
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::cornerMinEigenVal(d_img, dst, d_Dx, d_Dy, d_buf, blockSize, apertureSize, borderMode);

        GPU_SANITY_CHECK(dst, 1e-4);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::cornerMinEigenVal(img, dst, blockSize, apertureSize, borderMode);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// CvtColor

DEF_PARAM_TEST(Sz_Depth_Code, cv::Size, MatDepth, CvtColorInfo);

PERF_TEST_P(Sz_Depth_Code, CvtColor,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_32F),
                    Values(CvtColorInfo(4, 4, cv::COLOR_RGBA2BGRA),
                           CvtColorInfo(4, 1, cv::COLOR_BGRA2GRAY),
                           CvtColorInfo(1, 4, cv::COLOR_GRAY2BGRA),
                           CvtColorInfo(3, 3, cv::COLOR_BGR2XYZ),
                           CvtColorInfo(3, 3, cv::COLOR_XYZ2BGR),
                           CvtColorInfo(3, 3, cv::COLOR_BGR2YCrCb),
                           CvtColorInfo(3, 3, cv::COLOR_YCrCb2BGR),
                           CvtColorInfo(3, 3, cv::COLOR_BGR2YUV),
                           CvtColorInfo(3, 3, cv::COLOR_YUV2BGR),
                           CvtColorInfo(3, 3, cv::COLOR_BGR2HSV),
                           CvtColorInfo(3, 3, cv::COLOR_HSV2BGR),
                           CvtColorInfo(3, 3, cv::COLOR_BGR2HLS),
                           CvtColorInfo(3, 3, cv::COLOR_HLS2BGR),
                           CvtColorInfo(3, 3, cv::COLOR_BGR2Lab),
                           CvtColorInfo(3, 3, cv::COLOR_LBGR2Lab),
                           CvtColorInfo(3, 3, cv::COLOR_BGR2Luv),
                           CvtColorInfo(3, 3, cv::COLOR_LBGR2Luv),
                           CvtColorInfo(3, 3, cv::COLOR_Lab2BGR),
                           CvtColorInfo(3, 3, cv::COLOR_Lab2LBGR),
                           CvtColorInfo(3, 3, cv::COLOR_Luv2RGB),
                           CvtColorInfo(3, 3, cv::COLOR_Luv2LRGB))))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const CvtColorInfo info = GET_PARAM(2);

    cv::Mat src(size, CV_MAKETYPE(depth, info.scn));
    cv::randu(src, 0, depth == CV_8U ? 255.0 : 1.0);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::cvtColor(d_src, dst, info.code, info.dcn);

        GPU_SANITY_CHECK(dst, 1e-4);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::cvtColor(src, dst, info.code, info.dcn);

        CPU_SANITY_CHECK(dst);
    }
}

PERF_TEST_P(Sz_Depth_Code, CvtColorBayer,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U),
                    Values(CvtColorInfo(1, 3, cv::COLOR_BayerBG2BGR),
                           CvtColorInfo(1, 3, cv::COLOR_BayerGB2BGR),
                           CvtColorInfo(1, 3, cv::COLOR_BayerRG2BGR),
                           CvtColorInfo(1, 3, cv::COLOR_BayerGR2BGR),

                           CvtColorInfo(1, 1, cv::COLOR_BayerBG2GRAY),
                           CvtColorInfo(1, 1, cv::COLOR_BayerGB2GRAY),
                           CvtColorInfo(1, 1, cv::COLOR_BayerRG2GRAY),
                           CvtColorInfo(1, 1, cv::COLOR_BayerGR2GRAY))))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const CvtColorInfo info = GET_PARAM(2);

    cv::Mat src(size, CV_MAKETYPE(depth, info.scn));
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::cvtColor(d_src, dst, info.code, info.dcn);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::cvtColor(src, dst, info.code, info.dcn);

        CPU_SANITY_CHECK(dst);
    }
}

CV_ENUM(DemosaicingCode,
        cv::COLOR_BayerBG2BGR, cv::COLOR_BayerGB2BGR, cv::COLOR_BayerRG2BGR, cv::COLOR_BayerGR2BGR,
        cv::COLOR_BayerBG2GRAY, cv::COLOR_BayerGB2GRAY, cv::COLOR_BayerRG2GRAY, cv::COLOR_BayerGR2GRAY,
        cv::gpu::COLOR_BayerBG2BGR_MHT, cv::gpu::COLOR_BayerGB2BGR_MHT, cv::gpu::COLOR_BayerRG2BGR_MHT, cv::gpu::COLOR_BayerGR2BGR_MHT,
        cv::gpu::COLOR_BayerBG2GRAY_MHT, cv::gpu::COLOR_BayerGB2GRAY_MHT, cv::gpu::COLOR_BayerRG2GRAY_MHT, cv::gpu::COLOR_BayerGR2GRAY_MHT)

DEF_PARAM_TEST(Sz_Code, cv::Size, DemosaicingCode);

PERF_TEST_P(Sz_Code, Demosaicing,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    ValuesIn(DemosaicingCode::all())))
{
    const cv::Size size = GET_PARAM(0);
    const int code = GET_PARAM(1);

    cv::Mat src(size, CV_8UC1);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::demosaicing(d_src, dst, code);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        if (code >= cv::COLOR_COLORCVT_MAX)
        {
            FAIL_NO_CPU();
        }
        else
        {
            cv::Mat dst;

            TEST_CYCLE() cv::cvtColor(src, dst, code);

            CPU_SANITY_CHECK(dst);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// SwapChannels

PERF_TEST_P(Sz, SwapChannels,
            GPU_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src(size, CV_8UC4);
    declare.in(src, WARMUP_RNG);

    const int dstOrder[] = {2, 1, 0, 3};

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat dst(src);

        TEST_CYCLE() cv::gpu::swapChannels(dst, dstOrder);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// AlphaComp

CV_ENUM(AlphaOp, cv::gpu::ALPHA_OVER, cv::gpu::ALPHA_IN, cv::gpu::ALPHA_OUT, cv::gpu::ALPHA_ATOP, cv::gpu::ALPHA_XOR, cv::gpu::ALPHA_PLUS, cv::gpu::ALPHA_OVER_PREMUL, cv::gpu::ALPHA_IN_PREMUL, cv::gpu::ALPHA_OUT_PREMUL, cv::gpu::ALPHA_ATOP_PREMUL, cv::gpu::ALPHA_XOR_PREMUL, cv::gpu::ALPHA_PLUS_PREMUL, cv::gpu::ALPHA_PREMUL)
#define ALL_ALPHA_OPS ValuesIn(AlphaOp::all())

DEF_PARAM_TEST(Sz_Type_Op, cv::Size, MatType, AlphaOp);

PERF_TEST_P(Sz_Type_Op, AlphaComp,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8UC4, CV_16UC4, CV_32SC4, CV_32FC4),
                    ALL_ALPHA_OPS))
{
    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int alpha_op = GET_PARAM(2);

    cv::Mat img1(size, type);
    cv::Mat img2(size, type);
    declare.in(img1, img2, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_img1(img1);
        const cv::gpu::GpuMat d_img2(img2);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::alphaComp(d_img1, d_img2, dst, alpha_op);

        GPU_SANITY_CHECK(dst, 1e-3, ERROR_RELATIVE);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// HoughLines

namespace
{
    struct Vec4iComparator
    {
        bool operator()(const cv::Vec4i& a, const cv::Vec4i b) const
        {
            if (a[0] != b[0]) return a[0] < b[0];
            else if(a[1] != b[1]) return a[1] < b[1];
            else if(a[2] != b[2]) return a[2] < b[2];
            else return a[3] < b[3];
        }
    };
    struct Vec3fComparator
    {
        bool operator()(const cv::Vec3f& a, const cv::Vec3f b) const
        {
            if(a[0] != b[0]) return a[0] < b[0];
            else if(a[1] != b[1]) return a[1] < b[1];
            else return a[2] < b[2];
        }
    };
    struct Vec2fComparator
    {
        bool operator()(const cv::Vec2f& a, const cv::Vec2f b) const
        {
            if(a[0] != b[0]) return a[0] < b[0];
            else return a[1] < b[1];
        }
    };
}

PERF_TEST_P(Sz, HoughLines,
            GPU_TYPICAL_MAT_SIZES)
{
    declare.time(30.0);

    const cv::Size size = GetParam();

    const float rho = 1.0f;
    const float theta = static_cast<float>(CV_PI / 180.0);
    const int threshold = 300;

    cv::Mat src(size, CV_8UC1, cv::Scalar::all(0));
    cv::line(src, cv::Point(0, 100), cv::Point(src.cols, 100), cv::Scalar::all(255), 1);
    cv::line(src, cv::Point(0, 200), cv::Point(src.cols, 200), cv::Scalar::all(255), 1);
    cv::line(src, cv::Point(0, 400), cv::Point(src.cols, 400), cv::Scalar::all(255), 1);
    cv::line(src, cv::Point(100, 0), cv::Point(100, src.rows), cv::Scalar::all(255), 1);
    cv::line(src, cv::Point(200, 0), cv::Point(200, src.rows), cv::Scalar::all(255), 1);
    cv::line(src, cv::Point(400, 0), cv::Point(400, src.rows), cv::Scalar::all(255), 1);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_lines;
        cv::gpu::HoughLinesBuf d_buf;

        TEST_CYCLE() cv::gpu::HoughLines(d_src, d_lines, d_buf, rho, theta, threshold);

        cv::Mat gpu_lines(d_lines.row(0));
        cv::Vec2f* begin = gpu_lines.ptr<cv::Vec2f>(0);
        cv::Vec2f* end = begin + gpu_lines.cols;
        std::sort(begin, end, Vec2fComparator());
        SANITY_CHECK(gpu_lines);
    }
    else
    {
        std::vector<cv::Vec2f> cpu_lines;

        TEST_CYCLE() cv::HoughLines(src, cpu_lines, rho, theta, threshold);

        SANITY_CHECK(cpu_lines);
    }
}

//////////////////////////////////////////////////////////////////////
// HoughLinesP

DEF_PARAM_TEST_1(Image, std::string);

PERF_TEST_P(Image, HoughLinesP,
            testing::Values("cv/shared/pic5.png", "stitching/a1.png"))
{
    declare.time(30.0);

    const std::string fileName = getDataPath(GetParam());

    const float rho = 1.0f;
    const float theta = static_cast<float>(CV_PI / 180.0);
    const int threshold = 100;
    const int minLineLenght = 50;
    const int maxLineGap = 5;

    const cv::Mat image = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::Mat mask;
    cv::Canny(image, mask, 50, 100);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_mask(mask);
        cv::gpu::GpuMat d_lines;
        cv::gpu::HoughLinesBuf d_buf;

        TEST_CYCLE() cv::gpu::HoughLinesP(d_mask, d_lines, d_buf, rho, theta, minLineLenght, maxLineGap);

        cv::Mat gpu_lines(d_lines);
        cv::Vec4i* begin = gpu_lines.ptr<cv::Vec4i>();
        cv::Vec4i* end = begin + gpu_lines.cols;
        std::sort(begin, end, Vec4iComparator());
        SANITY_CHECK(gpu_lines);
    }
    else
    {
        std::vector<cv::Vec4i> cpu_lines;

        TEST_CYCLE() cv::HoughLinesP(mask, cpu_lines, rho, theta, threshold, minLineLenght, maxLineGap);

        SANITY_CHECK(cpu_lines);
    }
}

//////////////////////////////////////////////////////////////////////
// HoughCircles

DEF_PARAM_TEST(Sz_Dp_MinDist, cv::Size, float, float);

PERF_TEST_P(Sz_Dp_MinDist, HoughCircles,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(1.0f, 2.0f, 4.0f),
                    Values(1.0f)))
{
    declare.time(30.0);

    const cv::Size size = GET_PARAM(0);
    const float dp = GET_PARAM(1);
    const float minDist = GET_PARAM(2);

    const int minRadius = 10;
    const int maxRadius = 30;
    const int cannyThreshold = 100;
    const int votesThreshold = 15;

    cv::Mat src(size, CV_8UC1, cv::Scalar::all(0));
    cv::circle(src, cv::Point(100, 100), 20, cv::Scalar::all(255), -1);
    cv::circle(src, cv::Point(200, 200), 25, cv::Scalar::all(255), -1);
    cv::circle(src, cv::Point(200, 100), 25, cv::Scalar::all(255), -1);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_circles;
        cv::gpu::HoughCirclesBuf d_buf;

        TEST_CYCLE() cv::gpu::HoughCircles(d_src, d_circles, d_buf, cv::HOUGH_GRADIENT, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);

        cv::Mat gpu_circles(d_circles);
        cv::Vec3f* begin = gpu_circles.ptr<cv::Vec3f>(0);
        cv::Vec3f* end = begin + gpu_circles.cols;
        std::sort(begin, end, Vec3fComparator());
        SANITY_CHECK(gpu_circles);
    }
    else
    {
        std::vector<cv::Vec3f> cpu_circles;

        TEST_CYCLE() cv::HoughCircles(src, cpu_circles, cv::HOUGH_GRADIENT, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);

        SANITY_CHECK(cpu_circles);
    }
}

//////////////////////////////////////////////////////////////////////
// GeneralizedHough

enum { GHT_POSITION = cv::GeneralizedHough::GHT_POSITION,
       GHT_SCALE    = cv::GeneralizedHough::GHT_SCALE,
       GHT_ROTATION = cv::GeneralizedHough::GHT_ROTATION
     };

CV_FLAGS(GHMethod, GHT_POSITION, GHT_SCALE, GHT_ROTATION);

DEF_PARAM_TEST(Method_Sz, GHMethod, cv::Size);

PERF_TEST_P(Method_Sz, GeneralizedHough,
            Combine(Values(GHMethod(GHT_POSITION), GHMethod(GHT_POSITION | GHT_SCALE), GHMethod(GHT_POSITION | GHT_ROTATION), GHMethod(GHT_POSITION | GHT_SCALE | GHT_ROTATION)),
                    GPU_TYPICAL_MAT_SIZES))
{
    declare.time(10);

    const int method = GET_PARAM(0);
    const cv::Size imageSize = GET_PARAM(1);

    const cv::Mat templ = readImage("cv/shared/templ.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(templ.empty());

    cv::Mat image(imageSize, CV_8UC1, cv::Scalar::all(0));
    templ.copyTo(image(cv::Rect(50, 50, templ.cols, templ.rows)));

    cv::RNG rng(123456789);
    const int objCount = rng.uniform(5, 15);
    for (int i = 0; i < objCount; ++i)
    {
        double scale = rng.uniform(0.7, 1.3);
        bool rotate = 1 == rng.uniform(0, 2);

        cv::Mat obj;
        cv::resize(templ, obj, cv::Size(), scale, scale);
        if (rotate)
            obj = obj.t();

        cv::Point pos;

        pos.x = rng.uniform(0, image.cols - obj.cols);
        pos.y = rng.uniform(0, image.rows - obj.rows);

        cv::Mat roi = image(cv::Rect(pos, obj.size()));
        cv::add(roi, obj, roi);
    }

    cv::Mat edges;
    cv::Canny(image, edges, 50, 100);

    cv::Mat dx, dy;
    cv::Sobel(image, dx, CV_32F, 1, 0);
    cv::Sobel(image, dy, CV_32F, 0, 1);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_edges(edges);
        const cv::gpu::GpuMat d_dx(dx);
        const cv::gpu::GpuMat d_dy(dy);
        cv::gpu::GpuMat posAndVotes;

        cv::Ptr<cv::gpu::GeneralizedHough_GPU> d_hough = cv::gpu::GeneralizedHough_GPU::create(method);
        if (method & GHT_ROTATION)
        {
            d_hough->set("maxAngle", 90.0);
            d_hough->set("angleStep", 2.0);
        }

        d_hough->setTemplate(cv::gpu::GpuMat(templ));

        TEST_CYCLE() d_hough->detect(d_edges, d_dx, d_dy, posAndVotes);

        const cv::gpu::GpuMat positions(1, posAndVotes.cols, CV_32FC4, posAndVotes.data);
        GPU_SANITY_CHECK(positions);
    }
    else
    {
        cv::Mat positions;

        cv::Ptr<cv::GeneralizedHough> hough = cv::GeneralizedHough::create(method);
        if (method & GHT_ROTATION)
        {
            hough->set("maxAngle", 90.0);
            hough->set("angleStep", 2.0);
        }

        hough->setTemplate(templ);

        TEST_CYCLE() hough->detect(edges, dx, dy, positions);

        CPU_SANITY_CHECK(positions);
    }
}

//////////////////////////////////////////////////////////////////////
// BilateralFilter

DEF_PARAM_TEST(Sz_Depth_Cn_KernelSz, cv::Size, MatDepth, MatCn, int);

PERF_TEST_P(Sz_Depth_Cn_KernelSz, BilateralFilter,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_32F),
                    GPU_CHANNELS_1_3,
                    Values(3, 5, 9)))
{
    declare.time(60.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int kernel_size = GET_PARAM(3);

    const float sigma_color = 7;
    const float sigma_spatial = 5;
    const int borderMode = cv::BORDER_REFLECT101;

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::bilateralFilter(d_src, dst, kernel_size, sigma_color, sigma_spatial, borderMode);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bilateralFilter(src, dst, kernel_size, sigma_color, sigma_spatial, borderMode);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////
// GoodFeaturesToTrack

DEF_PARAM_TEST(Image_MinDistance, string, double);

PERF_TEST_P(Image_MinDistance, GoodFeaturesToTrack,
            Combine(Values<string>("gpu/perf/aloe.png"),
                    Values(0.0, 3.0)))
{
    const string fileName = GET_PARAM(0);
    const double minDistance = GET_PARAM(1);

    const cv::Mat image = readImage(fileName, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    const int maxCorners = 8000;
    const double qualityLevel = 0.01;

    if (PERF_RUN_GPU())
    {
        cv::gpu::GoodFeaturesToTrackDetector_GPU d_detector(maxCorners, qualityLevel, minDistance);

        const cv::gpu::GpuMat d_image(image);
        cv::gpu::GpuMat pts;

        TEST_CYCLE() d_detector(d_image, pts);

        GPU_SANITY_CHECK(pts);
    }
    else
    {
        cv::Mat pts;

        TEST_CYCLE() cv::goodFeaturesToTrack(image, pts, maxCorners, qualityLevel, minDistance);

        CPU_SANITY_CHECK(pts);
    }
}
