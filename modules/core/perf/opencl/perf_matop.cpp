// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

///////////// SetTo ////////////////////////

typedef Size_MatType SetToFixture;

OCL_PERF_TEST_P(SetToFixture, SetTo,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const Scalar s = Scalar::all(17);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type);
    declare.in(src, WARMUP_RNG).out(src);

    OCL_TEST_CYCLE() src.setTo(s);

    SANITY_CHECK(src);
}

///////////// SetTo with mask ////////////////////////

typedef Size_MatType SetToFixture;

OCL_PERF_TEST_P(SetToFixture, SetToWithMask,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const Scalar s = Scalar::all(17);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), mask(srcSize, CV_8UC1);
    declare.in(src, mask, WARMUP_RNG).out(src);

    OCL_TEST_CYCLE() src.setTo(s, mask);

    SANITY_CHECK(src);
}

///////////// ConvertTo ////////////////////////

typedef Size_MatType ConvertToFixture;

OCL_PERF_TEST_P(ConvertToFixture, ConvertTo,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ddepth = CV_MAT_DEPTH(type) == CV_8U ? CV_32F : CV_8U,
        cn = CV_MAT_CN(type), dtype = CV_MAKE_TYPE(ddepth, cn);

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    UMat src(srcSize, type), dst(srcSize, dtype);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.convertTo(dst, dtype);

    SANITY_CHECK(dst);
}

///////////// CopyTo ////////////////////////

typedef Size_MatType CopyToFixture;

OCL_PERF_TEST_P(CopyToFixture, CopyTo,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.copyTo(dst);

    SANITY_CHECK(dst);
}

///////////// CopyTo with mask ////////////////////////

typedef Size_MatType CopyToFixture;

OCL_PERF_TEST_P(CopyToFixture, CopyToWithMask,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type), mask(srcSize, CV_8UC1);
    declare.in(src, mask, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() src.copyTo(dst, mask);

    SANITY_CHECK(dst);
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
