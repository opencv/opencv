// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include "perf_precomp.hpp"
#include "opencv2/core/accel.hpp"

namespace opencv_test { namespace {
using namespace perf;

CV_ENUM(AccelInter, INTER_NEAREST, INTER_LINEAR)

typedef perf::TestBaseWithParam<tuple<Size, MatType, int> > Accel_GaussianBlur;
typedef perf::TestBaseWithParam<tuple<Size, MatType, AccelInter> > Accel_WarpAffine;

// the first accelerator device of the build (cuda:0, hip:0); the tests are skipped without one
static accel::Device firstDevice()
{
    std::vector<accel::Device> d = accel::Device::enumerate();
    if (d.empty())
        throw SkipTestException("no accelerator device");
    return d[0];
}

static UMat onDevice(const accel::Device& d, Size sz, int type)
{
    Mat h(sz, type);
    randu(h, 0, 255);
    return h.upload(d);
}

PERF_TEST_P(Accel_GaussianBlur, GaussianBlur,
            testing::Combine(testing::Values(sz1080p, sz2160p), testing::Values(CV_8UC1, CV_8UC4, CV_32FC1),
                             testing::Values(3, 5, 11)))
{
    const accel::Device d = firstDevice();
    const Size ksize(get<2>(GetParam()), get<2>(GetParam()));
    UMat src = onDevice(d, get<0>(GetParam()), get<1>(GetParam())), dst;
    cv::GaussianBlur(src, dst, ksize, 0);
    accel::synchronize(d);

    TEST_CYCLE()
    {
        cv::GaussianBlur(src, dst, ksize, 0);
        accel::synchronize(d);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Accel_WarpAffine, warpAffine,
            testing::Combine(testing::Values(sz1080p, sz2160p), testing::Values(CV_8UC1, CV_8UC4, CV_32FC1),
                             AccelInter::all()))
{
    const accel::Device d = firstDevice();
    const Size sz = get<0>(GetParam());
    const int inter = get<2>(GetParam());
    const Mat M = getRotationMatrix2D(Point2f(sz.width / 2.f, sz.height / 2.f), 30., 0.9);
    UMat src = onDevice(d, sz, get<1>(GetParam())), dst;
    cv::warpAffine(src, dst, M, sz, inter);
    accel::synchronize(d);

    TEST_CYCLE()
    {
        cv::warpAffine(src, dst, M, sz, inter);
        accel::synchronize(d);
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
