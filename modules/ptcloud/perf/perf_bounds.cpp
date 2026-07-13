// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

using namespace perf;

static Mat makeCloud(int n)
{
    Mat pts(n, 3, CV_32F);
    RNG rng(0x51ba7e);
    rng.fill(pts, RNG::UNIFORM, -1.f, 1.f);
    return pts;
}

typedef TestBaseWithParam<int> Ptcloud_BoundsPerf;

PERF_TEST_P(Ptcloud_BoundsPerf, aabb, testing::Values(100000, 300000))
{
    Mat cloud = makeCloud(GetParam()), lo, hi;
    TEST_CYCLE() getPointCloudBounds(cloud, lo, hi);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Ptcloud_BoundsPerf, obb, testing::Values(100000, 300000))
{
    Mat cloud = makeCloud(GetParam()), c, a, h;
    TEST_CYCLE() getOrientedBoundingBox(cloud, c, a, h);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Ptcloud_BoundsPerf, sphere, testing::Values(100000, 300000))
{
    Mat cloud = makeCloud(GetParam()), c;
    TEST_CYCLE() getBoundingSphere(cloud, c);
    SANITY_CHECK_NOTHING();
}

}} // namespace
