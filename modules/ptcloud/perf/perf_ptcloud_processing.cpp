// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"
#include "opencv2/geometry/segment.hpp"   // cv::normalEstimate

namespace opencv_test { namespace {

using namespace perf;

// ---------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------

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

// Uniform random cloud in the unit cube (Nx3 CV_32F).
static Mat makeUniformCloud(int n)
{
    Mat pts(n, 3, CV_32F);
    RNG rng(0x51ba7e);
    rng.fill(pts, RNG::UNIFORM, -1.f, 1.f);
    return pts;
}

// Noisy unit sphere plus scattered outliers (Nx1 CV_32FC3).
static Mat makeNoisyCloud(int nInliers, int nOutliers)
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

// ---------------------------------------------------------------------------------------------
// Ball-pivoting surface reconstruction
// ---------------------------------------------------------------------------------------------

typedef TestBaseWithParam<int> Ptcloud_BpaPerf;

PERF_TEST_P(Ptcloud_BpaPerf, createMeshBPA, testing::Values(20000, 60000))
{
    Mat cloud = makeSphere(GetParam());
    Mat normals, curv;
    normalEstimate(normals, curv, cloud, noArray(), 15);
    normals = normals.reshape(3, (int)cloud.total());
    orientNormalsConsistent(cloud, normals, 15);

    Mat vertices, triangles;
    TEST_CYCLE() createMeshBPA(cloud, normals, vertices, triangles);
    SANITY_CHECK_NOTHING();
}

// ---------------------------------------------------------------------------------------------
// Bounding volumes
// ---------------------------------------------------------------------------------------------

typedef TestBaseWithParam<int> Ptcloud_BoundsPerf;

PERF_TEST_P(Ptcloud_BoundsPerf, aabb, testing::Values(100000, 300000))
{
    Mat cloud = makeUniformCloud(GetParam()), lo, hi;
    TEST_CYCLE() boundingBox3D(cloud, lo, hi);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Ptcloud_BoundsPerf, obb, testing::Values(100000, 300000))
{
    Mat cloud = makeUniformCloud(GetParam()), c, a, h;
    TEST_CYCLE() orientedBoundingBox3D(cloud, c, a, h);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Ptcloud_BoundsPerf, sphere, testing::Values(100000, 300000))
{
    Mat cloud = makeUniformCloud(GetParam()), c;
    TEST_CYCLE() approxEnclosingSphere3D(cloud, c);
    SANITY_CHECK_NOTHING();
}

// ---------------------------------------------------------------------------------------------
// Normal orientation
// ---------------------------------------------------------------------------------------------

typedef TestBaseWithParam<int> Ptcloud_NormalPerf;

PERF_TEST_P(Ptcloud_NormalPerf, orient_consistent, testing::Values(50000, 150000))
{
    Mat cloud = makeSphere(GetParam());
    Mat normals, curv;
    normalEstimate(normals, curv, cloud, noArray(), 30);   // normals to orient (geometry)
    TEST_CYCLE() orientNormalsConsistent(cloud, normals, 30);
    SANITY_CHECK_NOTHING();
}

// ---------------------------------------------------------------------------------------------
// Outlier removal
// ---------------------------------------------------------------------------------------------

typedef TestBaseWithParam<int> Ptcloud_OutlierPerf;

PERF_TEST_P(Ptcloud_OutlierPerf, statistical, testing::Values(100000, 300000))
{
    const int N = GetParam();
    Mat cloud = makeNoisyCloud(N, N / 50), out;
    TEST_CYCLE() removeStatisticalOutliers(cloud, out, 20, 2.0);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Ptcloud_OutlierPerf, radius, testing::Values(100000, 300000))
{
    const int N = GetParam();
    Mat cloud = makeNoisyCloud(N, N / 50), out;
    TEST_CYCLE() removeRadiusOutliers(cloud, out, 0.05, 5);
    SANITY_CHECK_NOTHING();
}

}} // namespace
