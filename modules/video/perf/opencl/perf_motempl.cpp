// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

///////////// UpdateMotionHistory ////////////////////////

typedef TestBaseWithParam<Size> UpdateMotionHistoryFixture;

OCL_PERF_TEST_P(UpdateMotionHistoryFixture, UpdateMotionHistory, OCL_TEST_SIZES)
{
    const Size size = GetParam();
    checkDeviceMaxMemoryAllocSize(size, CV_32FC1);

    UMat silhouette(size, CV_8UC1), mhi(size, CV_32FC1);
    randu(silhouette, -5, 5);
    declare.in(mhi, WARMUP_RNG);

    OCL_TEST_CYCLE() cv::updateMotionHistory(silhouette, mhi, 1, 0.5);

    SANITY_CHECK(mhi);
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
