// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {


TEST(Imgproc_IntersectConvexConvex, no_intersection)
{
    std::vector<cv::Point> convex1;
    convex1.push_back(cv::Point(290, 126));
    convex1.push_back(cv::Point(284, 132));
    convex1.push_back(cv::Point(281, 133));
    convex1.push_back(cv::Point(256, 124));
    convex1.push_back(cv::Point(249, 116));
    convex1.push_back(cv::Point(234, 91));
    convex1.push_back(cv::Point(232, 86));
    convex1.push_back(cv::Point(232, 79));
    convex1.push_back(cv::Point(251, 69));
    convex1.push_back(cv::Point(257, 68));
    convex1.push_back(cv::Point(297, 85));
    convex1.push_back(cv::Point(299, 87));

    std::vector<cv::Point> convex2;
    convex2.push_back(cv::Point(192, 236));
    convex2.push_back(cv::Point(190, 245));
    convex2.push_back(cv::Point(177, 260));
    convex2.push_back(cv::Point(154, 271));
    convex2.push_back(cv::Point(142, 270));
    convex2.push_back(cv::Point(135, 263));
    convex2.push_back(cv::Point(131, 254));
    convex2.push_back(cv::Point(132, 240));
    convex2.push_back(cv::Point(172, 213));
    convex2.push_back(cv::Point(176, 216));

    std::vector<cv::Point> intersection;
    float area = cv::intersectConvexConvex(convex1, convex2, intersection);

    EXPECT_TRUE(intersection.empty());
    EXPECT_NEAR(area, 0, std::numeric_limits<float>::epsilon());
}


TEST(Imgproc_IntersectConvexConvex, no_intersection_with_1_vertex_on_edge_1)
{
    std::vector<cv::Point> convex1;
    convex1.push_back(cv::Point(0,0));
    convex1.push_back(cv::Point(740, 0));
    convex1.push_back(cv::Point(740, 540));
    convex1.push_back(cv::Point(0, 540));

    std::vector<cv::Point> convex2;
    convex2.push_back(cv::Point(0, 210));
    convex2.push_back(cv::Point(-30, 210));
    convex2.push_back(cv::Point(-37, 170));
    convex2.push_back(cv::Point(-7, 172));

    std::vector<cv::Point> intersection;
    float area = cv::intersectConvexConvex(convex1, convex2, intersection);

    EXPECT_TRUE(intersection.empty());
    EXPECT_NEAR(area, 0, std::numeric_limits<float>::epsilon());
}


TEST(Imgproc_IntersectConvexConvex, no_intersection_with_1_vertex_on_edge_2)
{
    std::vector<cv::Point> convex1;
    convex1.push_back(cv::Point(0,0));
    convex1.push_back(cv::Point(740, 0));
    convex1.push_back(cv::Point(740, 540));
    convex1.push_back(cv::Point(0, 540));

    std::vector<cv::Point> convex2;
    convex2.push_back(cv::Point(740, 210));
    convex2.push_back(cv::Point(750, 100));
    convex2.push_back(cv::Point(790, 250));
    convex2.push_back(cv::Point(800, 500));

    std::vector<cv::Point> intersection;
    float area = cv::intersectConvexConvex(convex1, convex2, intersection);

    EXPECT_TRUE(intersection.empty());
    EXPECT_NEAR(area, 0, std::numeric_limits<float>::epsilon());
}


TEST(Imgproc_IntersectConvexConvex, intersection_with_1_vertex_on_edge)
{
    std::vector<cv::Point> convex1;
    convex1.push_back(cv::Point(0,0));
    convex1.push_back(cv::Point(740, 0));
    convex1.push_back(cv::Point(740, 540));
    convex1.push_back(cv::Point(0, 540));

    std::vector<cv::Point> convex2;
    convex2.push_back(cv::Point(30, 210));
    convex2.push_back(cv::Point(0,210));
    convex2.push_back(cv::Point(7, 172));
    convex2.push_back(cv::Point(37, 170));

    std::vector<cv::Point> intersection;
    float area = cv::intersectConvexConvex(convex1, convex2, intersection);

    std::vector<cv::Point> expected_intersection;
    expected_intersection.push_back(cv::Point(0, 210));
    expected_intersection.push_back(cv::Point(7, 172));
    expected_intersection.push_back(cv::Point(37, 170));
    expected_intersection.push_back(cv::Point(30, 210));

    EXPECT_EQ(intersection, expected_intersection);
    EXPECT_NEAR(area, 1163, std::numeric_limits<float>::epsilon());
}


TEST(Imgproc_IntersectConvexConvex, intersection_with_2_vertices_on_edge)
{
    std::vector<cv::Point> convex1;
    convex1.push_back(cv::Point(0,0));
    convex1.push_back(cv::Point(740, 0));
    convex1.push_back(cv::Point(740, 540));
    convex1.push_back(cv::Point(0, 540));

    std::vector<cv::Point> convex2;
    convex2.push_back(cv::Point(30, 210));
    convex2.push_back(cv::Point(37, 170));
    convex2.push_back(cv::Point(0,210));
    convex2.push_back(cv::Point(0, 300));

    std::vector<cv::Point> intersection;
    float area = cv::intersectConvexConvex(convex1, convex2, intersection);

    std::vector<cv::Point> expected_intersection;
    expected_intersection.push_back(cv::Point(0, 300));
    expected_intersection.push_back(cv::Point(0, 210));
    expected_intersection.push_back(cv::Point(37, 170));
    expected_intersection.push_back(cv::Point(30, 210));

    EXPECT_EQ(intersection, expected_intersection);
    EXPECT_NEAR(area, 1950, std::numeric_limits<float>::epsilon());
}


TEST(Imgproc_IntersectConvexConvex, intersection_1)
{
    std::vector<cv::Point> convex1;
    convex1.push_back(cv::Point(0,0));
    convex1.push_back(cv::Point(740, 0));
    convex1.push_back(cv::Point(740, 540));
    convex1.push_back(cv::Point(0, 540));

    std::vector<cv::Point> convex2;
    convex2.push_back(cv::Point(20,210));
    convex2.push_back(cv::Point(30, 210));
    convex2.push_back(cv::Point(37, 170));
    convex2.push_back(cv::Point(7, 172));

    std::vector<cv::Point> intersection;
    float area = cv::intersectConvexConvex(convex1, convex2, intersection);

    std::vector<cv::Point> expected_intersection;
    expected_intersection.push_back(cv::Point(7, 172));
    expected_intersection.push_back(cv::Point(37, 170));
    expected_intersection.push_back(cv::Point(30, 210));
    expected_intersection.push_back(cv::Point(20, 210));

    EXPECT_EQ(intersection, expected_intersection);
    EXPECT_NEAR(area, 783, std::numeric_limits<float>::epsilon());
}


TEST(Imgproc_IntersectConvexConvex, intersection_2)
{
    std::vector<cv::Point> convex1;
    convex1.push_back(cv::Point(0,0));
    convex1.push_back(cv::Point(740, 0));
    convex1.push_back(cv::Point(740, 540));
    convex1.push_back(cv::Point(0, 540));

    std::vector<cv::Point> convex2;
    convex2.push_back(cv::Point(-2,210));
    convex2.push_back(cv::Point(-5, 300));
    convex2.push_back(cv::Point(37, 150));
    convex2.push_back(cv::Point(7, 172));

    std::vector<cv::Point> intersection;
    float area = cv::intersectConvexConvex(convex1, convex2, intersection);

    std::vector<cv::Point> expected_intersection;
    expected_intersection.push_back(cv::Point(0, 202));
    expected_intersection.push_back(cv::Point(7, 172));
    expected_intersection.push_back(cv::Point(37, 150));
    expected_intersection.push_back(cv::Point(0, 282));

    EXPECT_EQ(intersection, expected_intersection);
    EXPECT_NEAR(area, 1857.19836425781, std::numeric_limits<float>::epsilon());
}


TEST(Imgproc_IntersectConvexConvex, intersection_3)
{
    std::vector<cv::Point> convex1;
    convex1.push_back(cv::Point(15, 0));
    convex1.push_back(cv::Point(740, 0));
    convex1.push_back(cv::Point(740, 540));
    convex1.push_back(cv::Point(15, 540));

    std::vector<cv::Point> convex2;
    convex2.push_back(cv::Point(0,210));
    convex2.push_back(cv::Point(30, 210));
    convex2.push_back(cv::Point(37, 170));
    convex2.push_back(cv::Point(7, 172));

    std::vector<cv::Point> intersection;
    float area = cv::intersectConvexConvex(convex1, convex2, intersection);

    std::vector<cv::Point> expected_intersection;
    expected_intersection.push_back(cv::Point(15, 171));
    expected_intersection.push_back(cv::Point(37, 170));
    expected_intersection.push_back(cv::Point(30, 210));
    expected_intersection.push_back(cv::Point(15, 210));

    EXPECT_EQ(intersection, expected_intersection);

    EXPECT_NEAR(area, 723.866760253906, std::numeric_limits<float>::epsilon());
}


TEST(Imgproc_IntersectConvexConvex, intersection_4)
{
    std::vector<cv::Point> convex1;
    convex1.push_back(cv::Point(15, 0));
    convex1.push_back(cv::Point(740, 0));
    convex1.push_back(cv::Point(740, 540));
    convex1.push_back(cv::Point(15, 540));

    std::vector<cv::Point> convex2;
    convex2.push_back(cv::Point(15, 0));
    convex2.push_back(cv::Point(740, 0));
    convex2.push_back(cv::Point(740, 540));
    convex2.push_back(cv::Point(15, 540));

    std::vector<cv::Point> intersection;
    float area = cv::intersectConvexConvex(convex1, convex2, intersection);

    std::vector<cv::Point> expected_intersection;
    expected_intersection.push_back(cv::Point(15, 0));
    expected_intersection.push_back(cv::Point(740, 0));
    expected_intersection.push_back(cv::Point(740, 540));
    expected_intersection.push_back(cv::Point(15, 540));

    EXPECT_EQ(intersection, expected_intersection);
    EXPECT_NEAR(area, 391500, std::numeric_limits<float>::epsilon());
}


} // namespace
} // opencv_test
