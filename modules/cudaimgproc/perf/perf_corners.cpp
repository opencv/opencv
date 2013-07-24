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

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_img(img);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::CornernessCriteria> harris = cv::cuda::createHarrisCorner(img.type(), blockSize, apertureSize, k, borderMode);

        TEST_CYCLE() harris->compute(d_img, dst);

        CUDA_SANITY_CHECK(dst, 1e-4);
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

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_img(img);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::CornernessCriteria> minEigenVal = cv::cuda::createMinEigenValCorner(img.type(), blockSize, apertureSize, borderMode);

        TEST_CYCLE() minEigenVal->compute(d_img, dst);

        CUDA_SANITY_CHECK(dst, 1e-4);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::cornerMinEigenVal(img, dst, blockSize, apertureSize, borderMode);

        CPU_SANITY_CHECK(dst);
    }
}
