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
using namespace cv;
using std::tr1::get;

///////////// blend ////////////////////////

template <typename T>
static void blendLinearGold(const Mat &img1, const Mat &img2,
                            const Mat &weights1, const Mat &weights2,
                            Mat &result_gold)
{
    CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
    CV_Assert(weights1.size() == weights2.size() && weights1.size() == img1.size() &&
              weights1.type() == CV_32FC1 && weights2.type() == CV_32FC1);

    result_gold.create(img1.size(), img1.type());

    int cn = img1.channels();
    int step1 = img1.cols * img1.channels();

    for (int y = 0; y < img1.rows; ++y)
    {
        const float * const weights1_row = weights1.ptr<float>(y);
        const float * const weights2_row = weights2.ptr<float>(y);
        const T * const img1_row = img1.ptr<T>(y);
        const T * const img2_row = img2.ptr<T>(y);
        T * const result_gold_row = result_gold.ptr<T>(y);

        for (int x = 0; x < step1; ++x)
        {
            int x1 = x / cn;
            float w1 = weights1_row[x1], w2 = weights2_row[x1];
            result_gold_row[x] = saturate_cast<T>(((float)img1_row[x] * w1
                                                 + (float)img2_row[x] * w2) / (w1 + w2 + 1e-5f));
        }
    }
}

typedef void (*blendFunction)(const Mat &img1, const Mat &img2,
                              const Mat &weights1, const Mat &weights2,
                              Mat &result_gold);

typedef Size_MatType BlendLinearFixture;

OCL_PERF_TEST_P(BlendLinearFixture, BlendLinear,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int srcType = get<1>(params);
    const double eps = CV_MAT_DEPTH(srcType) <= CV_32S ? 1.0 : 0.2;

    Mat src1(srcSize, srcType), src2(srcSize, srcType), dst(srcSize, srcType);
    Mat weights1(srcSize, CV_32FC1), weights2(srcSize, CV_32FC1);

    declare.in(src1, src2, WARMUP_RNG).out(dst);
    randu(weights1, 0.0f, 1.0f);
    randu(weights2, 0.0f, 1.0f);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst;
        ocl::oclMat oclWeights1(weights1), oclWeights2(weights2);

        OCL_TEST_CYCLE() ocl::blendLinear(oclSrc1, oclSrc2, oclWeights1, oclWeights2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst, eps);
    }
    else if (RUN_PLAIN_IMPL)
    {
        blendFunction funcs[] = { (blendFunction)blendLinearGold<uchar>, (blendFunction)blendLinearGold<float> };
        int funcIdx = CV_MAT_DEPTH(srcType) == CV_8UC1 ? 0 : 1;

        TEST_CYCLE() (funcs[funcIdx])(src1, src2, weights1, weights2, dst);

        SANITY_CHECK(dst, eps);
    }
    else
        OCL_PERF_ELSE
}
