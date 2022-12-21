// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(CV_ArucoDrawMarker, regression_1226)
{
    int squares_x = 7;
    int squares_y = 5;
    int bwidth = 1600;
    int bheight = 1200;

    cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(squares_x, squares_y, 1.0, 0.75, dict);
    cv::Size sz(bwidth, bheight);
    cv::Mat mat;

    ASSERT_NO_THROW(
    {
        board->generateImage(sz, mat, 0, 1);
    });
}

}} // namespace
