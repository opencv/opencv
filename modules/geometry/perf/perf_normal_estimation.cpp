// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

using namespace perf;

// Evenly sampled unit sphere (Fibonacci), as an Nx3 CV_32F cloud.
static Mat makeSphere(int n)
{
    Mat pts(n, 3, CV_32F);
    const float ga = (float)(CV_PI * (3.0 - std::sqrt(5.0)));
    for (int i = 0; i < n; i++)
    {
        float z = 1.f - 2.f * (i + 0.5f) / n;
        float r = std::sqrt(std::max(0.f, 1.f - z * z));
        float t = ga * i;
        float* p = pts.ptr<float>(i);
        p[0] = r * std::cos(t); p[1] = r * std::sin(t); p[2] = z;
    }
    return pts;
}

typedef TestBaseWithParam<int> NormalEstimatePerf;

// normalEstimate with an empty nn_idx builds the kd-tree internally (the option-3 path).
PERF_TEST_P(NormalEstimatePerf, internal_knn, testing::Values(50000, 150000))
{
    Mat cloud = makeSphere(GetParam());
    Mat normals, curv;
    TEST_CYCLE() normalEstimate(normals, curv, cloud, noArray(), 30);
    SANITY_CHECK_NOTHING();
}

}} // namespace
