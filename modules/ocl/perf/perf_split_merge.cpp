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
using std::tr1::tuple;
using std::tr1::get;

///////////// Merge////////////////////////

typedef tuple<Size, MatDepth, int> MergeParams;
typedef TestBaseWithParam<MergeParams> MergeFixture;

OCL_PERF_TEST_P(MergeFixture, Merge,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8U, CV_32F),
                                   OCL_PERF_ENUM(2, 3)))
{
    const MergeParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int depth = get<1>(params), cn = get<2>(params),
            dtype = CV_MAKE_TYPE(depth, cn);

    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    Mat dst(srcSize, dtype);
    vector<Mat> src(cn);
    for (vector<Mat>::iterator i = src.begin(), end = src.end(); i != end; ++i)
    {
        i->create(srcSize, CV_MAKE_TYPE(depth, 1));
        declare.in(*i, WARMUP_RNG);
    }
    declare.out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclDst(srcSize, dtype);
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

typedef MergeParams SplitParams;
typedef TestBaseWithParam<SplitParams> SplitFixture;

OCL_PERF_TEST_P(SplitFixture, Split,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8U, CV_32F),
                                   OCL_PERF_ENUM(2, 3)))
{
    const SplitParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int depth = get<1>(params), cn = get<2>(params);
    const int type = CV_MAKE_TYPE(depth, cn);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    Mat src(srcSize, type);
    Mat dst0, dst1, dst2;
    declare.in(src, WARMUP_RNG);

    ASSERT_TRUE(cn == 3 || cn == 2);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src);
        vector<ocl::oclMat> oclDst(cn);
        oclDst[0] = ocl::oclMat(srcSize, depth);
        oclDst[1] = ocl::oclMat(srcSize, depth);
        if (cn == 3)
            oclDst[2] = ocl::oclMat(srcSize, depth);

        OCL_TEST_CYCLE() cv::ocl::split(oclSrc, oclDst);

        oclDst[0].download(dst0);
        oclDst[1].download(dst1);
        if (cn == 3)
            oclDst[2].download(dst2);
    }
    else if (RUN_PLAIN_IMPL)
    {
        vector<Mat> dst(cn);
        dst[0] = Mat(srcSize, depth);
        dst[1] = Mat(srcSize, depth);
        if (cn == 3)
            dst[2] = Mat(srcSize, depth);

        TEST_CYCLE() cv::split(src, dst);

        dst0 = dst[0];
        dst1 = dst[1];
        if (cn == 3)
            dst2 = dst[2];
    }
    else
        OCL_PERF_ELSE

    if (cn == 2)
    {
        SANITY_CHECK(dst0);
        SANITY_CHECK(dst1);
    }
    else if (cn == 3)
    {
        SANITY_CHECK(dst0);
        SANITY_CHECK(dst1);
        SANITY_CHECK(dst2);
    }
}
