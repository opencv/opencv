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
CV_ENUM(RemapMode, HALF_SIZE, UPSIDE_DOWN, REFLECTION_X, REFLECTION_BOTH)
#define ALL_REMAP_MODES ValuesIn(RemapMode::all())

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

PERF_TEST_P(Sz_Depth_Cn_Inter_Border_Mode, Remap,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    GPU_CHANNELS_1_3_4,
                    Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC)),
                    ALL_BORDER_MODES,
                    ALL_REMAP_MODES))
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

PERF_TEST_P(Sz_Depth_Cn_Inter_Scale, Resize,
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

PERF_TEST_P(Sz_Depth_Cn_Scale, ResizeArea,
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

PERF_TEST_P(Sz_Depth_Cn_Inter_Border, WarpAffine,
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

PERF_TEST_P(Sz_Depth_Cn_Inter_Border, WarpPerspective,
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
// BuildWarpPlaneMaps

PERF_TEST_P(Sz, BuildWarpPlaneMaps,
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

PERF_TEST_P(Sz, BuildWarpCylindricalMaps,
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

PERF_TEST_P(Sz, BuildWarpSphericalMaps,
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

PERF_TEST_P(Sz_Depth_Cn_Inter, Rotate,
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

PERF_TEST_P(Sz_Depth_Cn, PyrDown,
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

PERF_TEST_P(Sz_Depth_Cn, PyrUp,
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
// ImagePyramidBuild

PERF_TEST_P(Sz_Depth_Cn, ImagePyramidBuild,
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

PERF_TEST_P(Sz_Depth_Cn, ImagePyramidGetLayer,
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
