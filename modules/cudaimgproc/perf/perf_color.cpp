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
// CvtColor

DEF_PARAM_TEST(Sz_Depth_Code, cv::Size, MatDepth, CvtColorInfo);

PERF_TEST_P(Sz_Depth_Code, CvtColor,
            Combine(CUDA_TYPICAL_MAT_SIZES,
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

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::cvtColor(d_src, dst, info.code, info.dcn);

        CUDA_SANITY_CHECK(dst, 1e-4);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::cvtColor(src, dst, info.code, info.dcn);

        CPU_SANITY_CHECK(dst);
    }
}

PERF_TEST_P(Sz_Depth_Code, CvtColorBayer,
            Combine(CUDA_TYPICAL_MAT_SIZES,
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

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::cvtColor(d_src, dst, info.code, info.dcn);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::cvtColor(src, dst, info.code, info.dcn);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Demosaicing

CV_ENUM(DemosaicingCode,
        cv::COLOR_BayerBG2BGR, cv::COLOR_BayerGB2BGR, cv::COLOR_BayerRG2BGR, cv::COLOR_BayerGR2BGR,
        cv::COLOR_BayerBG2GRAY, cv::COLOR_BayerGB2GRAY, cv::COLOR_BayerRG2GRAY, cv::COLOR_BayerGR2GRAY,
        cv::cuda::COLOR_BayerBG2BGR_MHT, cv::cuda::COLOR_BayerGB2BGR_MHT, cv::cuda::COLOR_BayerRG2BGR_MHT, cv::cuda::COLOR_BayerGR2BGR_MHT,
        cv::cuda::COLOR_BayerBG2GRAY_MHT, cv::cuda::COLOR_BayerGB2GRAY_MHT, cv::cuda::COLOR_BayerRG2GRAY_MHT, cv::cuda::COLOR_BayerGR2GRAY_MHT)

DEF_PARAM_TEST(Sz_Code, cv::Size, DemosaicingCode);

PERF_TEST_P(Sz_Code, Demosaicing,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    DemosaicingCode::all()))
{
    const cv::Size size = GET_PARAM(0);
    const int code = GET_PARAM(1);

    cv::Mat src(size, CV_8UC1);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::demosaicing(d_src, dst, code);

        CUDA_SANITY_CHECK(dst);
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
            CUDA_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src(size, CV_8UC4);
    declare.in(src, WARMUP_RNG);

    const int dstOrder[] = {2, 1, 0, 3};

    if (PERF_RUN_CUDA())
    {
        cv::cuda::GpuMat dst(src);

        TEST_CYCLE() cv::cuda::swapChannels(dst, dstOrder);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// AlphaComp

CV_ENUM(AlphaOp, cv::cuda::ALPHA_OVER, cv::cuda::ALPHA_IN, cv::cuda::ALPHA_OUT, cv::cuda::ALPHA_ATOP, cv::cuda::ALPHA_XOR, cv::cuda::ALPHA_PLUS, cv::cuda::ALPHA_OVER_PREMUL, cv::cuda::ALPHA_IN_PREMUL, cv::cuda::ALPHA_OUT_PREMUL, cv::cuda::ALPHA_ATOP_PREMUL, cv::cuda::ALPHA_XOR_PREMUL, cv::cuda::ALPHA_PLUS_PREMUL, cv::cuda::ALPHA_PREMUL)

DEF_PARAM_TEST(Sz_Type_Op, cv::Size, MatType, AlphaOp);

PERF_TEST_P(Sz_Type_Op, AlphaComp,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8UC4, CV_16UC4, CV_32SC4, CV_32FC4),
                    AlphaOp::all()))
{
    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int alpha_op = GET_PARAM(2);

    cv::Mat img1(size, type);
    cv::Mat img2(size, type);
    declare.in(img1, img2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_img1(img1);
        const cv::cuda::GpuMat d_img2(img2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::alphaComp(d_img1, d_img2, dst, alpha_op);

        if (CV_MAT_DEPTH(type) < CV_32F)
        {
            CUDA_SANITY_CHECK(dst, 1);
        }
        else
        {
            CUDA_SANITY_CHECK(dst, 1e-3, ERROR_RELATIVE);
        }
    }
    else
    {
        FAIL_NO_CPU();
    }
}
