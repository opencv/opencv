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
// Remap

enum { HALF_SIZE=0, UPSIDE_DOWN, REFLECTION_X, REFLECTION_BOTH };
CV_ENUM(RemapMode, HALF_SIZE, UPSIDE_DOWN, REFLECTION_X, REFLECTION_BOTH);

void generateMap(cv::Mat& map_x, cv::Mat& map_y, int remapMode)
{
    for (int j = 0; j < map_x.rows; ++j)
    {
        for (int i = 0; i < map_x.cols; ++i)
        {
            switch (remapMode)
            {
            case HALF_SIZE:
                if (i > map_x.cols*0.25 && i < map_x.cols*0.75 && j > map_x.rows*0.25 && j < map_x.rows*0.75)
                {
                    map_x.at<float>(j,i) = 2.f * (i - map_x.cols * 0.25f) + 0.5f;
                    map_y.at<float>(j,i) = 2.f * (j - map_x.rows * 0.25f) + 0.5f;
                }
                else
                {
                    map_x.at<float>(j,i) = 0.f;
                    map_y.at<float>(j,i) = 0.f;
                }
                break;
            case UPSIDE_DOWN:
                map_x.at<float>(j,i) = static_cast<float>(i);
                map_y.at<float>(j,i) = static_cast<float>(map_x.rows - j);
                break;
            case REFLECTION_X:
                map_x.at<float>(j,i) = static_cast<float>(map_x.cols - i);
                map_y.at<float>(j,i) = static_cast<float>(j);
                break;
            case REFLECTION_BOTH:
                map_x.at<float>(j,i) = static_cast<float>(map_x.cols - i);
                map_y.at<float>(j,i) = static_cast<float>(map_x.rows - j);
                break;
            } // end of switch
        }
    }
}

DEF_PARAM_TEST(Sz_Depth_Cn_Inter_Border_Mode, cv::Size, MatDepth, MatCn, Interpolation, BorderMode, RemapMode);

PERF_TEST_P(Sz_Depth_Cn_Inter_Border_Mode, ImgProc_Remap,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    GPU_CHANNELS_1_3_4,
                    Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC)),
                    ALL_BORDER_MODES,
                    RemapMode::all()))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int interpolation = GET_PARAM(3);
    const int borderMode = GET_PARAM(4);
    const int remapMode = GET_PARAM(5);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    cv::Mat xmap(size, CV_32FC1);
    cv::Mat ymap(size, CV_32FC1);
    generateMap(xmap, ymap, remapMode);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        const cv::gpu::GpuMat d_xmap(xmap);
        const cv::gpu::GpuMat d_ymap(ymap);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::remap(d_src, dst, d_xmap, d_ymap, interpolation, borderMode);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::remap(src, dst, xmap, ymap, interpolation, borderMode);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Resize

DEF_PARAM_TEST(Sz_Depth_Cn_Inter_Scale, cv::Size, MatDepth, MatCn, Interpolation, double);

PERF_TEST_P(Sz_Depth_Cn_Inter_Scale, ImgProc_Resize,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    GPU_CHANNELS_1_3_4,
                    Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC)),
                    Values(0.5, 0.3, 2.0)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int interpolation = GET_PARAM(3);
    const double f = GET_PARAM(4);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::resize(d_src, dst, cv::Size(), f, f, interpolation);

        GPU_SANITY_CHECK(dst, 1e-3, ERROR_RELATIVE);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::resize(src, dst, cv::Size(), f, f, interpolation);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// ResizeArea

DEF_PARAM_TEST(Sz_Depth_Cn_Scale, cv::Size, MatDepth, MatCn, double);

PERF_TEST_P(Sz_Depth_Cn_Scale, ImgProc_ResizeArea,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    GPU_CHANNELS_1_3_4,
                    Values(0.2, 0.1, 0.05)))
{
    declare.time(1.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int interpolation = cv::INTER_AREA;
    const double f = GET_PARAM(3);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::resize(d_src, dst, cv::Size(), f, f, interpolation);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::resize(src, dst, cv::Size(), f, f, interpolation);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// WarpAffine

DEF_PARAM_TEST(Sz_Depth_Cn_Inter_Border, cv::Size, MatDepth, MatCn, Interpolation, BorderMode);

PERF_TEST_P(Sz_Depth_Cn_Inter_Border, ImgProc_WarpAffine,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    GPU_CHANNELS_1_3_4,
                    Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC)),
                    ALL_BORDER_MODES))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int interpolation = GET_PARAM(3);
    const int borderMode = GET_PARAM(4);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    const double aplha = CV_PI / 4;
    const double mat[2 * 3] =
    {
        std::cos(aplha), -std::sin(aplha), src.cols / 2,
        std::sin(aplha),  std::cos(aplha), 0
    };
    const cv::Mat M(2, 3, CV_64F, (void*) mat);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::warpAffine(d_src, dst, M, size, interpolation, borderMode);

        GPU_SANITY_CHECK(dst, 1);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::warpAffine(src, dst, M, size, interpolation, borderMode);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// WarpPerspective

PERF_TEST_P(Sz_Depth_Cn_Inter_Border, ImgProc_WarpPerspective,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    GPU_CHANNELS_1_3_4,
                    Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC)),
                    ALL_BORDER_MODES))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int interpolation = GET_PARAM(3);
    const int borderMode = GET_PARAM(4);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    const double aplha = CV_PI / 4;
    double mat[3][3] = { {std::cos(aplha), -std::sin(aplha), src.cols / 2},
                         {std::sin(aplha),  std::cos(aplha), 0},
                         {0.0,              0.0,             1.0}};
    const cv::Mat M(3, 3, CV_64F, (void*) mat);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::warpPerspective(d_src, dst, M, size, interpolation, borderMode);

        GPU_SANITY_CHECK(dst, 1);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::warpPerspective(src, dst, M, size, interpolation, borderMode);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// CopyMakeBorder

DEF_PARAM_TEST(Sz_Depth_Cn_Border, cv::Size, MatDepth, MatCn, BorderMode);

PERF_TEST_P(Sz_Depth_Cn_Border, ImgProc_CopyMakeBorder,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    GPU_CHANNELS_1_3_4,
                    ALL_BORDER_MODES))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int borderMode = GET_PARAM(3);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::copyMakeBorder(d_src, dst, 5, 5, 5, 5, borderMode);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::copyMakeBorder(src, dst, 5, 5, 5, 5, borderMode);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Threshold

CV_ENUM(ThreshOp, THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV)

DEF_PARAM_TEST(Sz_Depth_Op, cv::Size, MatDepth, ThreshOp);

PERF_TEST_P(Sz_Depth_Op, ImgProc_Threshold,
            Combine(GPU_TYPICAL_MAT_SIZES,
            Values(CV_8U, CV_16U, CV_32F, CV_64F),
            ThreshOp::all()))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int threshOp = GET_PARAM(2);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::threshold(d_src, dst, 100.0, 255.0, threshOp);

        GPU_SANITY_CHECK(dst, 1e-10);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::threshold(src, dst, 100.0, 255.0, threshOp);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Integral

PERF_TEST_P(Sz, ImgProc_Integral,
            GPU_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src(size, CV_8UC1);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::integralBuffered(d_src, dst, d_buf);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::integral(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// IntegralSqr

PERF_TEST_P(Sz, ImgProc_IntegralSqr,
            GPU_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src(size, CV_8UC1);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::sqrIntegral(d_src, dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// HistEvenC1

PERF_TEST_P(Sz_Depth, ImgProc_HistEvenC1,
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

PERF_TEST_P(Sz_Depth, ImgProc_HistEvenC4,
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

PERF_TEST_P(Sz, ImgProc_CalcHist,
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
        cv::Mat dst;

        const int hbins = 256;
        const float hranges[] = {0.0f, 256.0f};
        const int histSize[] = {hbins};
        const float* ranges[] = {hranges};
        const int channels[] = {0};

        TEST_CYCLE() cv::calcHist(&src, 1, channels, cv::Mat(), dst, 1, histSize, ranges);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// EqualizeHist

PERF_TEST_P(Sz, ImgProc_EqualizeHist,
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

PERF_TEST_P(Sz_ClipLimit, ImgProc_CLAHE,
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
// ColumnSum

PERF_TEST_P(Sz, ImgProc_ColumnSum,
            GPU_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src(size, CV_32FC1);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::columnSum(d_src, dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// Canny

DEF_PARAM_TEST(Image_AppertureSz_L2gradient, string, int, bool);

PERF_TEST_P(Image_AppertureSz_L2gradient, ImgProc_Canny,
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

PERF_TEST_P(Image, ImgProc_MeanShiftFiltering,
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

PERF_TEST_P(Image, ImgProc_MeanShiftProc,
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

PERF_TEST_P(Image, ImgProc_MeanShiftSegmentation,
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

PERF_TEST_P(Sz_Depth_Cn, ImgProc_BlendLinear,
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

//////////////////////////////////////////////////////////////////////
// Convolve

DEF_PARAM_TEST(Sz_KernelSz_Ccorr, cv::Size, int, bool);

PERF_TEST_P(Sz_KernelSz_Ccorr, ImgProc_Convolve,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(17, 27, 32, 64),
                    Bool()))
{
    declare.time(10.0);

    const cv::Size size = GET_PARAM(0);
    const int templ_size = GET_PARAM(1);
    const bool ccorr = GET_PARAM(2);

    const cv::Mat image(size, CV_32FC1);
    const cv::Mat templ(templ_size, templ_size, CV_32FC1);
    declare.in(image, templ, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_image = cv::gpu::createContinuous(size, CV_32FC1);
        d_image.upload(image);

        cv::gpu::GpuMat d_templ = cv::gpu::createContinuous(templ_size, templ_size, CV_32FC1);
        d_templ.upload(templ);

        cv::gpu::GpuMat dst;
        cv::gpu::ConvolveBuf d_buf;

        TEST_CYCLE() cv::gpu::convolve(d_image, d_templ, dst, ccorr, d_buf);

        GPU_SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else
    {
        if (ccorr)
            FAIL_NO_CPU();

        cv::Mat dst;

        TEST_CYCLE() cv::filter2D(image, dst, image.depth(), templ);

        CPU_SANITY_CHECK(dst);
    }
}

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate8U

CV_ENUM(TemplateMethod, TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED)

DEF_PARAM_TEST(Sz_TemplateSz_Cn_Method, cv::Size, cv::Size, MatCn, TemplateMethod);

PERF_TEST_P(Sz_TemplateSz_Cn_Method, ImgProc_MatchTemplate8U,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(cv::Size(5, 5), cv::Size(16, 16), cv::Size(30, 30)),
                    GPU_CHANNELS_1_3_4,
                    TemplateMethod::all()))
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

PERF_TEST_P(Sz_TemplateSz_Cn_Method, ImgProc_MatchTemplate32F,
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
};

//////////////////////////////////////////////////////////////////////
// MulSpectrums

CV_FLAGS(DftFlags, 0, DFT_INVERSE, DFT_SCALE, DFT_ROWS, DFT_COMPLEX_OUTPUT, DFT_REAL_OUTPUT)

DEF_PARAM_TEST(Sz_Flags, cv::Size, DftFlags);

PERF_TEST_P(Sz_Flags, ImgProc_MulSpectrums,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(0, DftFlags(cv::DFT_ROWS))))
{
    const cv::Size size = GET_PARAM(0);
    const int flag = GET_PARAM(1);

    cv::Mat a(size, CV_32FC2);
    cv::Mat b(size, CV_32FC2);
    declare.in(a, b, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_a(a);
        const cv::gpu::GpuMat d_b(b);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::mulSpectrums(d_a, d_b, dst, flag);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::mulSpectrums(a, b, dst, flag);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MulAndScaleSpectrums

PERF_TEST_P(Sz, ImgProc_MulAndScaleSpectrums,
            GPU_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    const float scale = 1.f / size.area();

    cv::Mat src1(size, CV_32FC2);
    cv::Mat src2(size, CV_32FC2);
    declare.in(src1,src2, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src1(src1);
        const cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::mulAndScaleSpectrums(d_src1, d_src2, dst, cv::DFT_ROWS, scale, false);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// Dft

PERF_TEST_P(Sz_Flags, ImgProc_Dft,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(0, DftFlags(cv::DFT_ROWS), DftFlags(cv::DFT_INVERSE))))
{
    declare.time(10.0);

    const cv::Size size = GET_PARAM(0);
    const int flag = GET_PARAM(1);

    cv::Mat src(size, CV_32FC2);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::dft(d_src, dst, size, flag);

        GPU_SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::dft(src, dst, flag);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// CornerHarris

DEF_PARAM_TEST(Image_Type_Border_BlockSz_ApertureSz, string, MatType, BorderMode, int, int);

PERF_TEST_P(Image_Type_Border_BlockSz_ApertureSz, ImgProc_CornerHarris,
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

PERF_TEST_P(Image_Type_Border_BlockSz_ApertureSz, ImgProc_CornerMinEigenVal,
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
// BuildWarpPlaneMaps

PERF_TEST_P(Sz, ImgProc_BuildWarpPlaneMaps,
            GPU_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    const cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
    const cv::Mat R = cv::Mat::ones(3, 3, CV_32FC1);
    const cv::Mat T = cv::Mat::zeros(1, 3, CV_32F);

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat map_x;
        cv::gpu::GpuMat map_y;

        TEST_CYCLE() cv::gpu::buildWarpPlaneMaps(size, cv::Rect(0, 0, size.width, size.height), K, R, T, 1.0, map_x, map_y);

        GPU_SANITY_CHECK(map_x);
        GPU_SANITY_CHECK(map_y);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// BuildWarpCylindricalMaps

PERF_TEST_P(Sz, ImgProc_BuildWarpCylindricalMaps,
            GPU_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    const cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
    const cv::Mat R = cv::Mat::ones(3, 3, CV_32FC1);

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat map_x;
        cv::gpu::GpuMat map_y;

        TEST_CYCLE() cv::gpu::buildWarpCylindricalMaps(size, cv::Rect(0, 0, size.width, size.height), K, R, 1.0, map_x, map_y);

        GPU_SANITY_CHECK(map_x);
        GPU_SANITY_CHECK(map_y);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// BuildWarpSphericalMaps

PERF_TEST_P(Sz, ImgProc_BuildWarpSphericalMaps,
            GPU_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    const cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
    const cv::Mat R = cv::Mat::ones(3, 3, CV_32FC1);

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat map_x;
        cv::gpu::GpuMat map_y;

        TEST_CYCLE() cv::gpu::buildWarpSphericalMaps(size, cv::Rect(0, 0, size.width, size.height), K, R, 1.0, map_x, map_y);

        GPU_SANITY_CHECK(map_x);
        GPU_SANITY_CHECK(map_y);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// Rotate

DEF_PARAM_TEST(Sz_Depth_Cn_Inter, cv::Size, MatDepth, MatCn, Interpolation);

PERF_TEST_P(Sz_Depth_Cn_Inter, ImgProc_Rotate,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    GPU_CHANNELS_1_3_4,
                    Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC))))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int interpolation = GET_PARAM(3);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::rotate(d_src, dst, size, 30.0, 0, 0, interpolation);

        GPU_SANITY_CHECK(dst, 1e-3, ERROR_RELATIVE);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// PyrDown

PERF_TEST_P(Sz_Depth_Cn, ImgProc_PyrDown,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::pyrDown(d_src, dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::pyrDown(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// PyrUp

PERF_TEST_P(Sz_Depth_Cn, ImgProc_PyrUp,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::pyrUp(d_src, dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::pyrUp(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// CvtColor

DEF_PARAM_TEST(Sz_Depth_Code, cv::Size, MatDepth, CvtColorInfo);

PERF_TEST_P(Sz_Depth_Code, ImgProc_CvtColor,
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

PERF_TEST_P(Sz_Depth_Code, ImgProc_CvtColorBayer,
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
        COLOR_BayerBG2BGR, COLOR_BayerGB2BGR, COLOR_BayerRG2BGR, COLOR_BayerGR2BGR,
        COLOR_BayerBG2GRAY, COLOR_BayerGB2GRAY, COLOR_BayerRG2GRAY, COLOR_BayerGR2GRAY,
        COLOR_BayerBG2BGR_MHT, COLOR_BayerGB2BGR_MHT, COLOR_BayerRG2BGR_MHT, COLOR_BayerGR2BGR_MHT,
        COLOR_BayerBG2GRAY_MHT, COLOR_BayerGB2GRAY_MHT, COLOR_BayerRG2GRAY_MHT, COLOR_BayerGR2GRAY_MHT)

DEF_PARAM_TEST(Sz_Code, cv::Size, DemosaicingCode);

PERF_TEST_P(Sz_Code, ImgProc_Demosaicing,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    DemosaicingCode::all()))
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

PERF_TEST_P(Sz, ImgProc_SwapChannels,
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

CV_ENUM(AlphaOp, ALPHA_OVER, ALPHA_IN, ALPHA_OUT, ALPHA_ATOP, ALPHA_XOR, ALPHA_PLUS, ALPHA_OVER_PREMUL, ALPHA_IN_PREMUL, ALPHA_OUT_PREMUL, ALPHA_ATOP_PREMUL, ALPHA_XOR_PREMUL, ALPHA_PLUS_PREMUL, ALPHA_PREMUL)

DEF_PARAM_TEST(Sz_Type_Op, cv::Size, MatType, AlphaOp);

PERF_TEST_P(Sz_Type_Op, ImgProc_AlphaComp,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8UC4, CV_16UC4, CV_32SC4, CV_32FC4),
                    AlphaOp::all()))
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
// ImagePyramidBuild

PERF_TEST_P(Sz_Depth_Cn, ImgProc_ImagePyramidBuild,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    const int nLayers = 5;
    const cv::Size dstSize(size.width / 2 + 10, size.height / 2 + 10);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);

        cv::gpu::ImagePyramid d_pyr;

        TEST_CYCLE() d_pyr.build(d_src, nLayers);

        cv::gpu::GpuMat dst;
        d_pyr.getLayer(dst, dstSize);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// ImagePyramidGetLayer

PERF_TEST_P(Sz_Depth_Cn, ImgProc_ImagePyramidGetLayer,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    GPU_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    const int nLayers = 3;
    const cv::Size dstSize(size.width / 2 + 10, size.height / 2 + 10);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        cv::gpu::ImagePyramid d_pyr(d_src, nLayers);

        TEST_CYCLE() d_pyr.getLayer(dst, dstSize);

        GPU_SANITY_CHECK(dst);
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

PERF_TEST_P(Sz, ImgProc_HoughLines,
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

PERF_TEST_P(Image, ImgProc_HoughLinesP,
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

PERF_TEST_P(Sz_Dp_MinDist, ImgProc_HoughCircles,
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

        TEST_CYCLE() cv::gpu::HoughCircles(d_src, d_circles, d_buf, CV_HOUGH_GRADIENT, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);

        cv::Mat gpu_circles(d_circles);
        cv::Vec3f* begin = gpu_circles.ptr<cv::Vec3f>(0);
        cv::Vec3f* end = begin + gpu_circles.cols;
        std::sort(begin, end, Vec3fComparator());
        SANITY_CHECK(gpu_circles);
    }
    else
    {
        std::vector<cv::Vec3f> cpu_circles;

        TEST_CYCLE() cv::HoughCircles(src, cpu_circles, CV_HOUGH_GRADIENT, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);

        SANITY_CHECK(cpu_circles);
    }
}

//////////////////////////////////////////////////////////////////////
// GeneralizedHough

CV_FLAGS(GHMethod, GHT_POSITION, GHT_SCALE, GHT_ROTATION);

DEF_PARAM_TEST(Method_Sz, GHMethod, cv::Size);

PERF_TEST_P(Method_Sz, ImgProc_GeneralizedHough,
            Combine(Values(GHMethod(cv::GHT_POSITION), GHMethod(cv::GHT_POSITION | cv::GHT_SCALE), GHMethod(cv::GHT_POSITION | cv::GHT_ROTATION), GHMethod(cv::GHT_POSITION | cv::GHT_SCALE | cv::GHT_ROTATION)),
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
        if (method & cv::GHT_ROTATION)
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
        if (method & cv::GHT_ROTATION)
        {
            hough->set("maxAngle", 90.0);
            hough->set("angleStep", 2.0);
        }

        hough->setTemplate(templ);

        TEST_CYCLE() hough->detect(edges, dx, dy, positions);

        CPU_SANITY_CHECK(positions);
    }
}
