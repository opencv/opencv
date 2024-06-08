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

PERF_TEST(Undistort, fisheye_undistortPoints_100k_10iter)
{
    const int pointsNumber = 100000;
    const Size imageSize(1280, 800);

    /* Set camera matrix */
    const Matx33d K(558.478087865323,  0, 620.458515360843,
                         0, 560.506767351568, 381.939424848348,
                         0,               0,                1);

    /* Set distortion coefficients */
    const Matx14d D(2.81e-06, 1.31e-06, -4.42e-06, -1.25e-06);

    /* Create two-channel points matrix */
    Mat xy[2] = {};
    xy[0].create(pointsNumber, 1, CV_64F);
    theRNG().fill(xy[0], RNG::UNIFORM, 0, imageSize.width); // x
    xy[1].create(pointsNumber, 1, CV_64F);
    theRNG().fill(xy[1], RNG::UNIFORM, 0, imageSize.height); // y

    Mat points;
    merge(xy, 2, points);

    /* Set fixed iteration number to check only c++ code, not algo convergence */
    TermCriteria termCriteria(TermCriteria::MAX_ITER, 10, 0);

    Mat undistortedPoints;
    TEST_CYCLE() fisheye::undistortPoints(points, undistortedPoints, K, D, noArray(), noArray(), termCriteria);

    SANITY_CHECK_NOTHING();
}

} // namespace
