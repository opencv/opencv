// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#include "../src/epnp.h"

namespace cv {
struct EPnPTestAccess
{
    static void solve(epnp& solver, Mat& matrix, Mat& rhs, Mat& solution)
    {
        solver.qr_solve(matrix, rhs, solution);
    }

};
}

namespace opencv_test {
namespace {

TEST(Calib3d_EPnP, qrScaleIncludesLastRow)
{
    constexpr int rowCount = 6;
    constexpr int columnCount = 4;
    const Mat camera = Mat::eye(3, 3, CV_64F);
    const Mat objectPoints = Mat::zeros(columnCount, 1, CV_64FC3);
    const Mat imagePoints = Mat::zeros(columnCount, 1, CV_64FC2);
    cv::epnp solver(camera, objectPoints, imagePoints);
    Mat matrix = Mat::zeros(rowCount, columnCount, CV_64F);
    matrix.at<double>(rowCount - 1, 0) = 1.0;
    for (int column = 1; column < columnCount; ++column)
    {
        matrix.at<double>(column - 1, column) = 1.0;
    }
    const Mat expected = (Mat_<double>(columnCount, 1) << 1.0, -2.0, 3.0, -4.0);
    Mat rhs = matrix * expected;
    Mat solution = Mat::zeros(columnCount, 1, CV_64F);

    cv::EPnPTestAccess::solve(solver, matrix, rhs, solution);

    for (int column = 0; column < columnCount; ++column)
    {
        EXPECT_DOUBLE_EQ(expected.at<double>(column), solution.at<double>(column));
    }
}

}
}
