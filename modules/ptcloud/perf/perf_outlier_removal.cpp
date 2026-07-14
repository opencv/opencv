// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

using namespace perf;

// Mid-size synthetic cloud: a noisy unit sphere plus scattered outliers.
static Mat makePerfCloud(int nInliers, int nOutliers)
{
    std::vector<Point3f> pts;
    pts.reserve(nInliers + nOutliers);
    RNG rng(0x9e3779b9);
    for (int i = 0; i < nInliers; i++)
    {
        float z = rng.uniform(-1.f, 1.f);
        float t = rng.uniform(0.f, 2.f * (float)CV_PI);
        float r = std::sqrt(std::max(0.f, 1.f - z * z));
        pts.emplace_back(r * std::cos(t) + (float)rng.gaussian(0.01),
                         r * std::sin(t) + (float)rng.gaussian(0.01),
                         z + (float)rng.gaussian(0.01));
    }
    for (int i = 0; i < nOutliers; i++)
        pts.emplace_back(rng.uniform(-3.f, 3.f), rng.uniform(-3.f, 3.f), rng.uniform(-3.f, 3.f));
    return Mat(pts).clone();   // N x 1, CV_32FC3
}

typedef TestBaseWithParam<int> Ptcloud_OutlierPerf;

PERF_TEST_P(Ptcloud_OutlierPerf, statistical, testing::Values(100000, 300000))
{
    const int N = GetParam();
    Mat cloud = makePerfCloud(N, N / 50), out;
    TEST_CYCLE() removeStatisticalOutliers(cloud, out, 20, 2.0);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Ptcloud_OutlierPerf, radius, testing::Values(100000, 300000))
{
    const int N = GetParam();
    Mat cloud = makePerfCloud(N, N / 50), out;
    TEST_CYCLE() removeRadiusOutliers(cloud, out, 0.05, 5);
    SANITY_CHECK_NOTHING();
}

}} // namespace
