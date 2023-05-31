// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

typedef TestBaseWithParam<tuple<cv::Size, UMatUsageFlags, UMatUsageFlags, UMatUsageFlags>> SizeUsageFlagsFixture;

OCL_PERF_TEST_P(SizeUsageFlagsFixture, UsageFlags_AllocMem,
    ::testing::Combine(
        OCL_TEST_SIZES,
        testing::Values(USAGE_DEFAULT, USAGE_ALLOCATE_HOST_MEMORY, USAGE_ALLOCATE_DEVICE_MEMORY), // USAGE_ALLOCATE_SHARED_MEMORY
        testing::Values(USAGE_DEFAULT, USAGE_ALLOCATE_HOST_MEMORY, USAGE_ALLOCATE_DEVICE_MEMORY), // USAGE_ALLOCATE_SHARED_MEMORY
        testing::Values(USAGE_DEFAULT, USAGE_ALLOCATE_HOST_MEMORY, USAGE_ALLOCATE_DEVICE_MEMORY) // USAGE_ALLOCATE_SHARED_MEMORY
    ))
{
    Size sz = get<0>(GetParam());
    UMatUsageFlags srcAllocMem = get<1>(GetParam());
    UMatUsageFlags dstAllocMem = get<2>(GetParam());
    UMatUsageFlags finalAllocMem = get<3>(GetParam());

    UMat src(sz, CV_8UC1, Scalar::all(128), srcAllocMem);

    OCL_TEST_CYCLE()
    {
        UMat dst(dstAllocMem);

        cv::add(src, Scalar::all(1), dst);
        {
            Mat canvas = dst.getMat(ACCESS_RW);
            cv::putText(canvas, "Test", Point(20, 20), FONT_HERSHEY_PLAIN, 1, Scalar::all(255));
        }
        UMat final(finalAllocMem);
        cv::subtract(dst, Scalar::all(1), final);
    }

    SANITY_CHECK_NOTHING();
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
