/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  OpenCV: minEnclosingCircles exhaustive tests
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

/****************************************************************************************\
*                        minEnclosingCircles Full Test Suite                              *
\****************************************************************************************/

static vector<Point> makeCircle(int cx, int cy, int r, int pts)
{
    vector<Point> c;
    for (int i = 0; i < pts; i++)
    {
        double a = (double)i * 2.0 * CV_PI / pts;
        c.push_back(Point(cx + (int)(r * cos(a)), cy + (int)(r * sin(a))));
    }
    return c;
}

TEST(minEnclosingCircles, basic_batch)
{
    vector<vector<Point>> contours;
    contours.push_back(makeCircle(100, 100, 20, 20));
    contours.push_back(makeCircle(300, 200, 40, 30));
    contours.push_back(makeCircle(200, 300, 60, 40));

    Mat centers, radii;
    minEnclosingCircles(contours, centers, radii);

    ASSERT_EQ(centers.rows, 3);
    ASSERT_EQ(radii.rows, 3);
    ASSERT_EQ(centers.cols, 2);
    ASSERT_EQ(radii.cols, 1);
    ASSERT_EQ(centers.type(), CV_32F);
    ASSERT_EQ(radii.type(), CV_32F);
}

TEST(minEnclosingCircles, filter_by_radius)
{
    vector<vector<Point>> contours;
    contours.push_back(makeCircle(100, 100, 20, 20));
    contours.push_back(makeCircle(300, 200, 40, 20));
    contours.push_back(makeCircle(500, 300, 60, 20));

    Mat centers, radii;
    minEnclosingCircles(contours, centers, radii, 35.0);

    ASSERT_EQ(centers.rows, 2);
    for (int i = 0; i < radii.rows; i++)
        EXPECT_GE(radii.at<float>(i, 0), 35.0f);
}

TEST(minEnclosingCircles, sort_by_x)
{
    vector<vector<Point>> contours;
    contours.push_back(makeCircle(300, 100, 20, 20));
    contours.push_back(makeCircle(100, 200, 20, 20));
    contours.push_back(makeCircle(500, 300, 20, 20));

    Mat centers, radii;
    minEnclosingCircles(contours, centers, radii, 0, MEC_SORT_BY_X);

    EXPECT_LE(centers.at<float>(0, 0), centers.at<float>(1, 0));
    EXPECT_LE(centers.at<float>(1, 0), centers.at<float>(2, 0));
}

TEST(minEnclosingCircles, sort_by_y)
{
    vector<vector<Point>> contours;
    contours.push_back(makeCircle(100, 300, 20, 20));
    contours.push_back(makeCircle(100, 100, 20, 20));
    contours.push_back(makeCircle(100, 500, 20, 20));

    Mat centers, radii;
    minEnclosingCircles(contours, centers, radii, 0, MEC_SORT_BY_Y);

    EXPECT_LE(centers.at<float>(0, 1), centers.at<float>(1, 1));
    EXPECT_LE(centers.at<float>(1, 1), centers.at<float>(2, 1));
}

TEST(minEnclosingCircles, sort_by_radius)
{
    vector<vector<Point>> contours;
    contours.push_back(makeCircle(100, 100, 60, 30));
    contours.push_back(makeCircle(300, 200, 20, 20));
    contours.push_back(makeCircle(500, 300, 40, 25));

    Mat centers, radii;
    minEnclosingCircles(contours, centers, radii, 0, MEC_SORT_BY_RADIUS);

    EXPECT_LE(radii.at<float>(0, 0), radii.at<float>(1, 0));
    EXPECT_LE(radii.at<float>(1, 0), radii.at<float>(2, 0));
}

TEST(minEnclosingCircles, empty_input)
{
    vector<vector<Point>> contours;
    Mat centers, radii;

    minEnclosingCircles(contours, centers, radii);

    EXPECT_EQ(centers.rows, 0);
    EXPECT_EQ(radii.rows, 0);
}

TEST(minEnclosingCircles, empty_contour_inside)
{
    vector<vector<Point>> contours(3);
    contours[0] = makeCircle(100, 100, 20, 20);
    contours[1].clear();
    contours[2] = makeCircle(300, 200, 30, 20);

    Mat centers, radii;
    minEnclosingCircles(contours, centers, radii);

    EXPECT_EQ(centers.rows, 2);
}

TEST(minEnclosingCircles, consistency_with_single)
{
    vector<Point> contour = makeCircle(200, 200, 35, 30);

    Point2f c1;
    float r1;
    minEnclosingCircle(contour, c1, r1);

    vector<vector<Point>> contours(1);
    contours[0] = contour;

    Mat centers, radii;
    minEnclosingCircles(contours, centers, radii);

    EXPECT_NEAR(centers.at<float>(0, 0), c1.x, 1e-3);
    EXPECT_NEAR(centers.at<float>(0, 1), c1.y, 1e-3);
    EXPECT_NEAR(radii.at<float>(0, 0), r1, 1e-3);
}

TEST(minEnclosingCircles, degenerate_contours)
{
    vector<vector<Point>> contours(5);

    contours[0].push_back(Point(100, 100));
    contours[1].push_back(Point(200, 200));
    contours[1].push_back(Point(240, 200));
    contours[2] = makeCircle(300, 300, 30, 20);
    contours[3].push_back(Point(400, 400));
    contours[3].push_back(Point(400, 400));
    contours[4].push_back(Point(500, 100));
    contours[4].push_back(Point(520, 100));
    contours[4].push_back(Point(540, 100));

    Mat centers, radii;
    minEnclosingCircles(contours, centers, radii);

    ASSERT_EQ(centers.rows, 5);

    EXPECT_LT(radii.at<float>(0, 0), 1.0f);
    EXPECT_NEAR(centers.at<float>(1, 0), 220.0f, 5.0f);
    EXPECT_NEAR(radii.at<float>(1, 0), 20.0f, 5.0f);
    EXPECT_LT(radii.at<float>(3, 0), 1.0f);
    EXPECT_NEAR(centers.at<float>(4, 0), 520.0f, 5.0f);
}

TEST(minEnclosingCircles, stress_test)
{
    const int N = 200;
    RNG rng(12345);

    vector<vector<Point>> contours(N);
    for (int i = 0; i < N; i++)
    {
        int cx = rng.uniform(0, 5000);
        int cy = rng.uniform(0, 5000);
        int r  = rng.uniform(10, 100);
        int pts = rng.uniform(5, 30);
        contours[i] = makeCircle(cx, cy, r, pts);
    }

    Mat centers, radii;
    minEnclosingCircles(contours, centers, radii);

    ASSERT_EQ(centers.rows, N);
    ASSERT_EQ(radii.rows, N);

    Mat centers2, radii2;
    minEnclosingCircles(contours, centers2, radii2, 30.0, MEC_SORT_BY_X);

    for (int i = 0; i < radii2.rows; i++)
        EXPECT_GE(radii2.at<float>(i, 0), 30.0f);

    for (int i = 1; i < centers2.rows; i++)
        EXPECT_LE(centers2.at<float>(i-1, 0), centers2.at<float>(i, 0));
}

}} // namespace opencv_test
