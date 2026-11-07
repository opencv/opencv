// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include "perf_precomp.hpp"
#include "opencv2/core/accel.hpp"

namespace opencv_test { namespace {
using namespace perf;

#define ACCEL_SIZES sz1080p, sz2160p
#define ACCEL_TYPES CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC4

typedef perf::TestBaseWithParam<tuple<Size, MatType> > Accel_Size_MatType;

// the first device of the built-in CUDA backend; the tests are skipped without one
static accel::Device gpu()
{
    accel::Device d = accel::Device::find("cuda");
    if (d.empty())
        throw SkipTestException("no CUDA device for the non-CPU HAL");
    return d;
}

static UMat onDevice(const accel::Device& d, Size sz, int type)
{
    Mat h(sz, type);
    randu(h, 0, 255);
    UMat u;
    u.allocator = d.allocator();
    h.copyTo(u);
    u.allocator = 0;
    return u;
}

PERF_TEST_P(Accel_Size_MatType, flip, testing::Combine(testing::Values(ACCEL_SIZES), testing::Values(ACCEL_TYPES)))
{
    const accel::Device d = gpu();
    UMat src = onDevice(d, get<0>(GetParam()), get<1>(GetParam())), dst;
    cv::flip(src, dst, 1);

    TEST_CYCLE()
    {
        cv::flip(src, dst, 1);
        accel::synchronize(d);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Accel_Size_MatType, transpose, testing::Combine(testing::Values(ACCEL_SIZES), testing::Values(ACCEL_TYPES)))
{
    const accel::Device d = gpu();
    UMat src = onDevice(d, get<0>(GetParam()), get<1>(GetParam())), dst;
    cv::transpose(src, dst);

    TEST_CYCLE()
    {
        cv::transpose(src, dst);
        accel::synchronize(d);
    }

    SANITY_CHECK_NOTHING();
}

}}  // namespace opencv_test
