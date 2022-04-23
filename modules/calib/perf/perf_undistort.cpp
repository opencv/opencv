// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

using PerfIntType = perf::TestBaseWithParam<std::tuple<int>>;
PERF_TEST_P(PerfIntType, fisheye_undistortPoints,
                                            (testing::Values(1e2, 1e3, 1e4)))
{
    const cv::Size imageSize(1280, 800);

    /* Set camera matrix */
    const cv::Matx33d K(558.478087865323,  0, 620.458515360843,
                         0, 560.506767351568, 381.939424848348,
                         0,               0,                1);

    /* Set distortion coefficients */
    Mat D(1, 4, CV_64F);
    theRNG().fill(D, RNG::UNIFORM, -1.e-5, 1.e-5);

    int pointsNumber = std::get<0>(GetParam());

    /* Create two-channel points matrix */
    cv::Mat xy[2] = {};
    xy[0].create(pointsNumber, 1, CV_64F);
    theRNG().fill(xy[0], cv::RNG::UNIFORM, 0, imageSize.width); // x
    xy[1].create(pointsNumber, 1, CV_64F);
    theRNG().fill(xy[1], cv::RNG::UNIFORM, 0, imageSize.height); // y

    cv::Mat points;
    merge(xy, 2, points);

    /* Set fixed iteration number to check only c++ code, not algo convergence */
    TermCriteria termCriteria(TermCriteria::MAX_ITER, 10, 0);

    Mat undistortedPoints;
    TEST_CYCLE() fisheye::undistortPoints(points, undistortedPoints, K, D, noArray(), noArray(), termCriteria);

    SANITY_CHECK_NOTHING();
}

}} // namespace
