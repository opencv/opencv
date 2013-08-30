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
//     and/or other oclMaterials provided with the distribution.
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
using std::tr1::tuple;
using std::tr1::get;

///////////// Merge////////////////////////

typedef Size_MatType MergeFixture;

PERF_TEST_P(MergeFixture, Merge,
            ::testing::Combine(::testing::Values(OCL_SIZE_1000, OCL_SIZE_2000),
                               OCL_PERF_ENUM(CV_8U, CV_32F)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int depth = get<1>(params), channels = 3;

    const int dstType = CV_MAKE_TYPE(depth, channels);
    Mat dst(srcSize, dstType);
    vector<Mat> src(channels);
    for (vector<Mat>::iterator i = src.begin(), end = src.end(); i != end; ++i)
    {
        i->create(srcSize, CV_MAKE_TYPE(depth, 1));
        declare.in(*i, WARMUP_RNG);
    }
    declare.out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclDst(srcSize, dstType);
        vector<ocl::oclMat> oclSrc(src.size());
        for (vector<ocl::oclMat>::size_type i = 0, end = src.size(); i < end; ++i)
            oclSrc[i] = src[i];

        OCL_TEST_CYCLE() cv::ocl::merge(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::merge(src, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// Split////////////////////////

typedef Size_MatType SplitFixture;

PERF_TEST_P(SplitFixture, Split,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8U, CV_32F)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int depth = get<1>(params), channels = 3;

    Mat src(srcSize, CV_MAKE_TYPE(depth, channels));
    declare.in(src, WARMUP_RNG);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src);
        vector<ocl::oclMat> oclDst(channels, ocl::oclMat(srcSize, CV_MAKE_TYPE(depth, 1)));

        OCL_TEST_CYCLE() cv::ocl::split(oclSrc, oclDst);

        ASSERT_EQ(3, channels);
        Mat dst0, dst1, dst2;
        oclDst[0].download(dst0);
        oclDst[1].download(dst1);
        oclDst[2].download(dst2);
        SANITY_CHECK(dst0);
        SANITY_CHECK(dst1);
        SANITY_CHECK(dst2);
    }
    else if (RUN_PLAIN_IMPL)
    {
        vector<Mat> dst(channels, Mat(srcSize, CV_MAKE_TYPE(depth, 1)));
        TEST_CYCLE() cv::split(src, dst);

        ASSERT_EQ(3, channels);
        Mat & dst0 = dst[0], & dst1 = dst[1], & dst2 = dst[2];
        SANITY_CHECK(dst0);
        SANITY_CHECK(dst1);
        SANITY_CHECK(dst2);
    }
    else
        OCL_PERF_ELSE
}
