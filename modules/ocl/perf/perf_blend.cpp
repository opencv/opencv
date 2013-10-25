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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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

using namespace perf;

///////////// blend ////////////////////////

template <typename T>
static void blendLinearGold(const cv::Mat &img1, const cv::Mat &img2,
                            const cv::Mat &weights1, const cv::Mat &weights2,
                            cv::Mat &result_gold)
{
    result_gold.create(img1.size(), img1.type());

    int cn = img1.channels();

    for (int y = 0; y < img1.rows; ++y)
    {
        const float *weights1_row = weights1.ptr<float>(y);
        const float *weights2_row = weights2.ptr<float>(y);
        const T *img1_row = img1.ptr<T>(y);
        const T *img2_row = img2.ptr<T>(y);
        T *result_gold_row = result_gold.ptr<T>(y);

        for (int x = 0; x < img1.cols * cn; ++x)
        {
            int x1 = x * cn;
            float w1 = weights1_row[x];
            float w2 = weights2_row[x];
            result_gold_row[x] = static_cast<T>((img1_row[x1] * w1
                                                 + img2_row[x1] * w2) / (w1 + w2 + 1e-5f));
        }
    }
}

typedef TestBaseWithParam<Size> blendLinearFixture;

PERF_TEST_P(blendLinearFixture, blendLinear, OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();
    const int type = CV_8UC1;

    Mat src1(srcSize, type), src2(srcSize, CV_8UC1), dst;
    Mat weights1(srcSize, CV_32FC1), weights2(srcSize, CV_32FC1);

    declare.in(src1, src2, WARMUP_RNG);
    randu(weights1, 0.0f, 1.0f);
    randu(weights2, 0.0f, 1.0f);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst;
        ocl::oclMat oclWeights1(weights1), oclWeights2(weights2);

        OCL_TEST_CYCLE() cv::ocl::blendLinear(oclSrc1, oclSrc2, oclWeights1, oclWeights2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() blendLinearGold<uchar>(src1, src2, weights1, weights2, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}
