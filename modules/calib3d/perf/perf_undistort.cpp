// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

namespace opencv_test {

PERF_TEST(Undistort, InitUndistortMap)
{
    Size size_w_h(512 + 3, 512);
    Mat k(3, 3, CV_32FC1);
    Mat d(1, 14, CV_64FC1);
    Mat dst(size_w_h, CV_32FC2);
    declare.in(k, d, WARMUP_RNG).out(dst);
    TEST_CYCLE() initUndistortRectifyMap(k, d, noArray(), k, size_w_h, CV_32FC2, dst, noArray());
    SANITY_CHECK_NOTHING();
}

PERF_TEST(Undistort, DISABLED_InitInverseRectificationMap)
{
    Size size_w_h(512 + 3, 512);
    Mat k(3, 3, CV_32FC1);
    Mat d(1, 14, CV_64FC1);
    Mat dst(size_w_h, CV_32FC2);
    declare.in(k, d, WARMUP_RNG).out(dst);
    TEST_CYCLE() initInverseRectificationMap(k, d, noArray(), k, size_w_h, CV_32FC2, dst, noArray());
    SANITY_CHECK_NOTHING();
}

PERF_TEST(Undistort, fisheye_undistortPoints)
{

    const cv::Size imageSize(1280, 800);

    const cv::Matx33d K(558.478087865323,  0, 620.458515360843,
                         0, 560.506767351568, 381.939424848348,
                         0,               0,                1);

    Mat D(1, 4, CV_64F);
    theRNG().fill(D, RNG::UNIFORM, -1.e-5, 1.e-5);

    cv::Mat xy[2] = {};
    xy[0].create(1000, 1, CV_64F);
    theRNG().fill(xy[0], cv::RNG::UNIFORM, 0, imageSize.width); // x
    xy[1].create(1000, 1, CV_64F);
    theRNG().fill(xy[1], cv::RNG::UNIFORM, 0, imageSize.height); // y

    cv::Mat points;
    merge(xy, 2, points);

    Mat undistortedPoints;
    TEST_CYCLE() fisheye::undistortPoints(points, undistortedPoints, K, D);
    SANITY_CHECK_NOTHING();
}

} // namespace
