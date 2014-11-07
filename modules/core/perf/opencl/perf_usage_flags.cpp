// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

typedef TestBaseWithParam<std::tr1::tuple<cv::Size, bool> > UsageFlagsBoolFixture;

OCL_PERF_TEST_P(UsageFlagsBoolFixture, UsageFlags_AllocHostMem, ::testing::Combine(OCL_TEST_SIZES, Bool()))
{
    Size sz = get<0>(GetParam());
    bool allocHostMem = get<1>(GetParam());

    UMat src(sz, CV_8UC1, Scalar::all(128));

    OCL_TEST_CYCLE()
    {
        UMat dst(allocHostMem ? USAGE_ALLOCATE_HOST_MEMORY : USAGE_DEFAULT);

        cv::add(src, Scalar::all(1), dst);
        {
            Mat canvas = dst.getMat(ACCESS_RW);
            cv::putText(canvas, "Test", Point(20, 20), FONT_HERSHEY_PLAIN, 1, Scalar::all(255));
        }
        UMat final;
        cv::subtract(dst, Scalar::all(1), final);
    }

    SANITY_CHECK_NOTHING();
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
