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

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

///////////// Merge////////////////////////

typedef tuple<Size, MatDepth, int> MergeParams;
typedef TestBaseWithParam<MergeParams> MergeFixture;

OCL_PERF_TEST_P(MergeFixture, Merge,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8U, CV_32F), Values(2, 3)))
{
    const MergeParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int depth = get<1>(params), cn = get<2>(params), dtype = CV_MAKE_TYPE(depth, cn);

    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    UMat dst(srcSize, dtype);
    vector<UMat> src(cn);
    for (vector<UMat>::iterator i = src.begin(), end = src.end(); i != end; ++i)
    {
        i->create(srcSize, CV_MAKE_TYPE(depth, 1));
        declare.in(*i, WARMUP_RNG);
    }
    declare.out(dst);

    OCL_TEST_CYCLE() cv::merge(src, dst);

    SANITY_CHECK(dst);
}

///////////// Split ////////////////////////

typedef MergeParams SplitParams;
typedef TestBaseWithParam<SplitParams> SplitFixture;

OCL_PERF_TEST_P(SplitFixture, Split,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8U, CV_32F), Values(2, 3)))
{
    const SplitParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int depth = get<1>(params), cn = get<2>(params), type = CV_MAKE_TYPE(depth, cn);

    ASSERT_TRUE(cn == 3 || cn == 2);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type);
    std::vector<UMat> dst(cn, UMat(srcSize, CV_MAKE_TYPE(depth, 1)));

    declare.in(src, WARMUP_RNG);
    for (int i = 0; i < cn; ++i)
        declare.in(dst[i]);

    OCL_TEST_CYCLE() cv::split(src, dst);

    ASSERT_EQ(cn, (int)dst.size());

    if (cn == 2)
    {
        UMat & dst0 = dst[0], & dst1 = dst[1];
        SANITY_CHECK(dst0);
        SANITY_CHECK(dst1);
    }
    else
    {
        UMat & dst0 = dst[0], & dst1 = dst[1], & dst2 = dst[2];
        SANITY_CHECK(dst0);
        SANITY_CHECK(dst1);
        SANITY_CHECK(dst2);
    }
}

///////////// MixChannels ////////////////////////

typedef tuple<Size, MatDepth> MixChannelsParams;
typedef TestBaseWithParam<MixChannelsParams> MixChannelsFixture;

OCL_PERF_TEST_P(MixChannelsFixture, MixChannels,
                ::testing::Combine(Values(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                                   OCL_PERF_ENUM(CV_8U, CV_32F)))
{
    const MixChannelsParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int depth = get<1>(params), type = CV_MAKE_TYPE(depth, 2), n = 2;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    std::vector<UMat> src(n), dst(n);
    for (int i = 0; i < n; ++i)
    {
        src[i] = UMat(srcSize, type);
        dst[i] = UMat(srcSize, type);
        declare.in(src[i], WARMUP_RNG).out(dst[i]);
    }

    int fromTo[] = { 1,2, 2,0, 0,3, 3,1 };

    OCL_TEST_CYCLE() cv::mixChannels(src, dst, fromTo, 4);

    UMat & dst0 = dst[0], & dst1 = dst[1];
    SANITY_CHECK(dst0);
    SANITY_CHECK(dst1);
}

///////////// InsertChannel ////////////////////////

typedef Size_MatDepth InsertChannelFixture;

OCL_PERF_TEST_P(InsertChannelFixture, InsertChannel,
                ::testing::Combine(Values(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                                   OCL_PERF_ENUM(CV_8U, CV_32F)))
{
    const Size_MatDepth_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int depth = get<1>(params), type = CV_MAKE_TYPE(depth, 3);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, depth), dst(srcSize, type, Scalar::all(17));
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::insertChannel(src, dst, 1);

    SANITY_CHECK(dst);
}

///////////// ExtractChannel ////////////////////////

typedef Size_MatDepth ExtractChannelFixture;

OCL_PERF_TEST_P(ExtractChannelFixture, ExtractChannel,
                ::testing::Combine(Values(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                                   OCL_PERF_ENUM(CV_8U, CV_32F)))
{
    const Size_MatDepth_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int depth = get<1>(params), type = CV_MAKE_TYPE(depth, 3);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, depth);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::extractChannel(src, dst, 1);

    SANITY_CHECK(dst);
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
