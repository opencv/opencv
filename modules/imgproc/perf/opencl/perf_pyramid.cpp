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
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

///////////// PyrDown //////////////////////

typedef Size_MatType PyrDownFixture;

OCL_PERF_TEST_P(PyrDownFixture, PyrDown,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const Size dstSize((srcSize.height + 1) >> 1, (srcSize.width + 1) >> 1);
    const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : 1e-5;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(dstSize, type);

    UMat src(srcSize, type), dst(dstSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::pyrDown(src, dst);

    SANITY_CHECK(dst, eps);
}

///////////// PyrUp ////////////////////////

typedef Size_MatType PyrUpFixture;

OCL_PERF_TEST_P(PyrUpFixture, PyrUp,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const Size dstSize(srcSize.height << 1, srcSize.width << 1);
    const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : 1e-5;

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(dstSize, type);

    UMat src(srcSize, type), dst(dstSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::pyrDown(src, dst);

    SANITY_CHECK(dst, eps);
}

///////////// buildPyramid ////////////////////////

typedef Size_MatType BuildPyramidFixture;

OCL_PERF_TEST_P(BuildPyramidFixture, BuildPyramid,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), maxLevel = 5;
    const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : 1e-5;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    std::vector<UMat> dst(maxLevel);
    UMat src(srcSize, type);
    declare.in(src, WARMUP_RNG);

    OCL_TEST_CYCLE() cv::buildPyramid(src, dst, maxLevel);

    UMat dst0 = dst[0], dst1 = dst[1], dst2 = dst[2], dst3 = dst[3], dst4 = dst[4];

    SANITY_CHECK(dst0, eps);
    SANITY_CHECK(dst1, eps);
    SANITY_CHECK(dst2, eps);
    SANITY_CHECK(dst3, eps);
    SANITY_CHECK(dst4, eps);
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
