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

#define ARITHM_MAT_DEPTH Values(CV_8U, CV_16U, CV_32F, CV_64F)

//////////////////////////////////////////////////////////////////////
// Merge

PERF_TEST_P(Sz_Depth_Cn, Merge,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH,
                    Values(2, 3, 4)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    std::vector<cv::Mat> src(channels);
    for (int i = 0; i < channels; ++i)
    {
        src[i].create(size, depth);
        declare.in(src[i], WARMUP_RNG);
    }

    if (PERF_RUN_CUDA())
    {
        std::vector<cv::cuda::GpuMat> d_src(channels);
        for (int i = 0; i < channels; ++i)
            d_src[i].upload(src[i]);

        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::merge(d_src, dst);

        CUDA_SANITY_CHECK(dst, 1e-10);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::merge(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Split

PERF_TEST_P(Sz_Depth_Cn, Split,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH,
                    Values(2, 3, 4)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    cv::Mat src(size, CV_MAKE_TYPE(depth, channels));
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        std::vector<cv::cuda::GpuMat> dst;

        TEST_CYCLE() cv::cuda::split(d_src, dst);

        const cv::cuda::GpuMat& dst0 = dst[0];
        const cv::cuda::GpuMat& dst1 = dst[1];

        CUDA_SANITY_CHECK(dst0, 1e-10);
        CUDA_SANITY_CHECK(dst1, 1e-10);
    }
    else
    {
        std::vector<cv::Mat> dst;

        TEST_CYCLE() cv::split(src, dst);

        const cv::Mat& dst0 = dst[0];
        const cv::Mat& dst1 = dst[1];

        CPU_SANITY_CHECK(dst0);
        CPU_SANITY_CHECK(dst1);
    }
}

//////////////////////////////////////////////////////////////////////
// Transpose

PERF_TEST_P(Sz_Type, Transpose,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8UC1, CV_8UC4, CV_16UC2, CV_16SC2, CV_32SC1, CV_32SC2, CV_64FC1)))
{
    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::transpose(d_src, dst);

        CUDA_SANITY_CHECK(dst, 1e-10);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::transpose(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Flip

enum {FLIP_BOTH = 0, FLIP_X = 1, FLIP_Y = -1};
CV_ENUM(FlipCode, FLIP_BOTH, FLIP_X, FLIP_Y)

DEF_PARAM_TEST(Sz_Depth_Cn_Code, cv::Size, MatDepth, MatCn, FlipCode);

PERF_TEST_P(Sz_Depth_Cn_Code, Flip,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    CUDA_CHANNELS_1_3_4,
                    FlipCode::all()))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int flipCode = GET_PARAM(3);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::flip(d_src, dst, flipCode);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::flip(src, dst, flipCode);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// LutOneChannel

PERF_TEST_P(Sz_Type, LutOneChannel,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8UC1, CV_8UC3)))
{
    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    cv::Mat lut(1, 256, CV_8UC1);
    declare.in(lut, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        cv::Ptr<cv::cuda::LookUpTable> lutAlg = cv::cuda::createLookUpTable(lut);

        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() lutAlg->transform(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::LUT(src, lut, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// LutMultiChannel

PERF_TEST_P(Sz_Type, LutMultiChannel,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values<MatType>(CV_8UC3)))
{
    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    cv::Mat lut(1, 256, CV_MAKE_TYPE(CV_8U, src.channels()));
    declare.in(lut, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        cv::Ptr<cv::cuda::LookUpTable> lutAlg = cv::cuda::createLookUpTable(lut);

        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() lutAlg->transform(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::LUT(src, lut, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// CopyMakeBorder

DEF_PARAM_TEST(Sz_Depth_Cn_Border, cv::Size, MatDepth, MatCn, BorderMode);

PERF_TEST_P(Sz_Depth_Cn_Border, CopyMakeBorder,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F),
                    CUDA_CHANNELS_1_3_4,
                    ALL_BORDER_MODES))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);
    const int borderMode = GET_PARAM(3);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::copyMakeBorder(d_src, dst, 5, 5, 5, 5, borderMode);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::copyMakeBorder(src, dst, 5, 5, 5, 5, borderMode);

        CPU_SANITY_CHECK(dst);
    }
}
