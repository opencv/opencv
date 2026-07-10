// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

using namespace perf;

// Evenly sampled unit sphere (Fibonacci) of the requested size.
static Mat makeSphere(int n)
{
    std::vector<Point3f> pts;
    pts.reserve(n);
    const float ga = (float)(CV_PI * (3.0 - std::sqrt(5.0)));
    for (int i = 0; i < n; i++)
    {
        float z = 1.f - 2.f * (i + 0.5f) / n;
        float r = std::sqrt(std::max(0.f, 1.f - z * z));
        float t = ga * i;
        pts.emplace_back(r * std::cos(t), r * std::sin(t), z);
    }
    return Mat(pts).clone();
}

typedef TestBaseWithParam<int> Ptcloud_NormalPerf;

PERF_TEST_P(Ptcloud_NormalPerf, estimate, testing::Values(50000, 150000))
{
    Mat cloud = makeSphere(GetParam()), normals;
    TEST_CYCLE() estimateNormals(cloud, normals, 30);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Ptcloud_NormalPerf, orient_consistent, testing::Values(50000, 150000))
{
    Mat cloud = makeSphere(GetParam()), normals;
    estimateNormals(cloud, normals, 30);
    TEST_CYCLE() orientNormalsConsistent(cloud, normals, 30);
    SANITY_CHECK_NOTHING();
}

}} // namespace
