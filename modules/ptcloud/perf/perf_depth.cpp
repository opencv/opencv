// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;
using namespace perf;

typedef std::tuple<Size, MatType> DepthTo3dParams;
typedef TestBaseWithParam<DepthTo3dParams> DepthTo3dTest;

static Matx33d makeK()
{
    return Matx33d(525., 0., 319.5, 0., 525., 239.5, 0., 0., 1.);
}

PERF_TEST_P_(DepthTo3dTest, noMask)
{
    Size sz = get<0>(GetParam());
    int dtype = get<1>(GetParam());

    Mat depth(sz, dtype);
    randu(depth, 0.1, 10.0);
    Mat K(3, 3, CV_MAKETYPE(CV_MAT_DEPTH(dtype), 1));
    Mat(makeK()).convertTo(K, CV_MAT_DEPTH(dtype));
    Mat points3d(sz, CV_MAKETYPE(CV_MAT_DEPTH(dtype), 4));

    declare.in(depth).in(K).out(points3d);  // keep the 0.1-10.0 range (no WARMUP_RNG)
    declare.time(50);

    TEST_CYCLE() cv::depthTo3d(depth, K, points3d);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, DepthTo3dTest,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p, Size(127, 61)),  // 127x61 guards the scalar tail
        testing::Values(CV_32FC1, CV_64FC1)
    )
);

// depthTo3dMask accepts CV_16U/CV_16S/CV_32F only; withMask covers CV_32FC1 here.
typedef TestBaseWithParam<DepthTo3dParams> DepthTo3dMaskTest;

PERF_TEST_P_(DepthTo3dMaskTest, withMask)
{
    Size sz = get<0>(GetParam());
    int dtype = get<1>(GetParam());

    Mat depth(sz, dtype);
    randu(depth, 0.1, 10.0);
    Mat mask(sz, CV_8U);
    randu(mask, 0, 2);
    Mat K(3, 3, CV_MAKETYPE(CV_MAT_DEPTH(dtype), 1));
    Mat(makeK()).convertTo(K, CV_MAT_DEPTH(dtype));
    Mat points3d(1, sz.area(), CV_MAKETYPE(CV_MAT_DEPTH(dtype), 4));

    declare.in(depth).in(K).in(mask).out(points3d);
    declare.time(50);

    TEST_CYCLE() cv::depthTo3d(depth, K, points3d, mask);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, DepthTo3dMaskTest,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p, Size(127, 61)),
        testing::Values(CV_32FC1)
    )
);

}} // namespace
