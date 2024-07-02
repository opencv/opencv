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

// The inputs are not convex and cuased buffer overflow
// See https://github.com/opencv/opencv/issues/25259
TEST(Imgproc_IntersectConvexConvex, not_convex)
{
    std::vector<cv::Point2f> convex1 = {
        { 46.077175f , 228.66121f  }, {  5.428622f , 250.05899f  }, {207.51741f  , 109.645676f },
        {175.94789f  ,  32.6566f   }, {217.4915f   , 252.66176f  }, {187.09386f  ,   6.3988557f},
        { 52.20488f  ,  69.266205f }, { 38.188286f , 134.48068f  }, {246.4742f   ,  31.41043f  },
        {178.97946f  , 169.52287f  }, {103.40764f  , 153.30397f  }, {160.67746f  ,  17.166115f },
        {152.44255f  , 135.35f     }, {197.03804f  , 193.04782f  }, {248.28397f  ,  56.821487f },
        { 10.907227f ,  82.55291f  }, {109.67949f  ,  70.7405f   }, { 58.96842f  , 150.132f    },
        {150.7613f   , 129.54753f  }, {254.98463f  , 228.21748f  }, {139.02563f  , 193.89336f  },
        { 84.79946f  , 162.25363f  }, { 39.83567f  ,  44.626484f }, {107.034996f , 209.38887f  },
        { 67.61073f  ,  17.119232f }, {208.8617f   ,  33.67367f  }, {182.65207f  ,   8.291072f },
        { 72.89319f  ,  42.51845f  }, {202.4902f   , 123.97209f  }, { 79.945076f , 140.99268f  },
        {225.8952f   ,  66.226326f }, { 34.08404f  , 219.2208f   }, {243.1221f   ,  60.95162f  }
    };
    std::vector<cv::Point2f> convex2 = {
        {144.33624f  , 247.15732f  }, {  5.656847f ,  17.461054f }, {230.54338f  ,   2.0446582f},
        {143.0578f   , 215.27856f  }, {250.44626f  ,  82.54287f  }, {  0.3846766f,  11.101262f },
        { 70.81022f  ,  17.243904f }, { 77.18812f  ,  75.760666f }, {190.34933f  , 234.30962f  },
        {230.10204f  , 133.67998f  }, { 58.903755f , 252.96451f  }, {213.57228f  , 155.7058f   },
        {190.80992f  , 212.90802f  }, {203.4356f   ,  36.55016f  }, { 32.276424f ,   2.5646307f},
        { 39.73823f  ,  87.23782f  }, {112.46902f  , 101.81753f  }, { 58.154305f , 238.40395f  },
        {187.01064f  ,  96.24343f  }, { 44.42692f  ,  10.573529f }, {118.76949f  , 233.35114f  },
        { 86.26109f  , 120.93148f  }, {217.94751f  , 130.5933f   }, {148.2687f   ,  68.56015f  },
        {187.44174f  , 214.32857f  }, {247.19875f  , 180.8494f   }, { 17.986013f ,  61.451443f },
        {254.74344f  , 204.71747f  }, {211.92726f  , 132.0139f   }, { 51.36624f  , 116.63085f  },
        { 83.80044f  , 124.20074f  }, {122.125854f ,  25.182402f }, { 39.08164f  , 180.08517f  }
    };
    std::vector<cv::Point> intersection;

    float area = cv::intersectConvexConvex(convex1, convex2, intersection, false);
    EXPECT_TRUE(intersection.empty());
    EXPECT_LE(area, 0.f);
}

} // namespace
} // opencv_test
