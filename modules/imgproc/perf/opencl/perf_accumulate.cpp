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
//    Nathan, liujun@multicorewareinc.com
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

/////////////////////////////////// Accumulate ///////////////////////////////////

typedef Size_MatType AccumulateFixture;

OCL_PERF_TEST_P(AccumulateFixture, Accumulate,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int srcType = get<1>(params), cn = CV_MAT_CN(srcType), dstType = CV_32FC(cn);

    checkDeviceMaxMemoryAllocSize(srcSize, dstType);

    UMat src(srcSize, srcType), dst(srcSize, dstType);
    declare.in(src, dst, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::accumulate(src, dst);

    SANITY_CHECK_NOTHING();
}

/////////////////////////////////// AccumulateSquare ///////////////////////////////////

typedef Size_MatType AccumulateSquareFixture;

OCL_PERF_TEST_P(AccumulateSquareFixture, AccumulateSquare,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int srcType = get<1>(params), cn = CV_MAT_CN(srcType), dstType = CV_32FC(cn);

    checkDeviceMaxMemoryAllocSize(srcSize, dstType);

    UMat src(srcSize, srcType), dst(srcSize, dstType);
    declare.in(src, dst, WARMUP_RNG);

    OCL_TEST_CYCLE() cv::accumulateSquare(src, dst);

    SANITY_CHECK_NOTHING();
}

/////////////////////////////////// AccumulateProduct ///////////////////////////////////

typedef Size_MatType AccumulateProductFixture;

OCL_PERF_TEST_P(AccumulateProductFixture, AccumulateProduct,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int srcType = get<1>(params), cn = CV_MAT_CN(srcType), dstType = CV_32FC(cn);

    checkDeviceMaxMemoryAllocSize(srcSize, dstType);

    UMat src1(srcSize, srcType), src2(srcSize, srcType), dst(srcSize, dstType);
    declare.in(src1, src2, dst, WARMUP_RNG);

    OCL_TEST_CYCLE() cv::accumulateProduct(src1, src2, dst);

    SANITY_CHECK_NOTHING();
}

/////////////////////////////////// AccumulateWeighted ///////////////////////////////////

typedef Size_MatType AccumulateWeightedFixture;

OCL_PERF_TEST_P(AccumulateWeightedFixture, AccumulateWeighted,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int srcType = get<1>(params), cn = CV_MAT_CN(srcType), dstType = CV_32FC(cn);

    checkDeviceMaxMemoryAllocSize(srcSize, dstType);

    UMat src(srcSize, srcType), dst(srcSize, dstType);
    declare.in(src, dst, WARMUP_RNG);

    OCL_TEST_CYCLE() cv::accumulateWeighted(src, dst, 2.0);

    SANITY_CHECK_NOTHING();
}

} } // namespace cvtest::ocl

#endif
