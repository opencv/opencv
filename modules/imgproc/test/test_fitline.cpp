// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Imgproc_fitLine_vector_3d, regression)
{
    std::vector<Point3f> points_vector;

    Point3f p21(4,4,4);
    Point3f p22(8,8,8);

    points_vector.push_back(p21);
    points_vector.push_back(p22);

    std::vector<float> line;

    cv::fitLine(points_vector, line, DIST_L2, 0 ,0 ,0);

    ASSERT_EQ(line.size(), (size_t)6);

}

TEST(Imgproc_fitLine_vector_2d, regression)
{
    std::vector<Point2f> points_vector;

    Point2f p21(4,4);
    Point2f p22(8,8);
    Point2f p23(16,16);

    points_vector.push_back(p21);
    points_vector.push_back(p22);
    points_vector.push_back(p23);

    std::vector<float> line;

    cv::fitLine(points_vector, line, DIST_L2, 0 ,0 ,0);

    ASSERT_EQ(line.size(), (size_t)4);
}

TEST(Imgproc_fitLine_Mat_2dC2, regression)
{
    cv::Mat mat1 = Mat::zeros(3, 1, CV_32SC2);
    std::vector<float> line1;

    cv::fitLine(mat1, line1, DIST_L2, 0 ,0 ,0);

    ASSERT_EQ(line1.size(), (size_t)4);
}

TEST(Imgproc_fitLine_Mat_2dC1, regression)
{
    cv::Matx<int, 3, 2> mat2;
    std::vector<float> line2;

    cv::fitLine(mat2, line2, DIST_L2, 0 ,0 ,0);

    ASSERT_EQ(line2.size(), (size_t)4);
}

TEST(Imgproc_fitLine_Mat_3dC3, regression)
{
    cv::Mat mat1 = Mat::zeros(2, 1, CV_32SC3);
    std::vector<float> line1;

    cv::fitLine(mat1, line1, DIST_L2, 0 ,0 ,0);

    ASSERT_EQ(line1.size(), (size_t)6);
}

TEST(Imgproc_fitLine_Mat_3dC1, regression)
{
    cv::Mat mat2 = Mat::zeros(2, 3, CV_32SC1);
    std::vector<float> line2;

    cv::fitLine(mat2, line2, DIST_L2, 0 ,0 ,0);

    ASSERT_EQ(line2.size(), (size_t)6);
}

}} // namespace
