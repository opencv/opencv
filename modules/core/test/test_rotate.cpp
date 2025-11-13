// This file adds missing test cases for cv::rotate()
// Covers small square and rectangular matrices.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;
using cv::norm;

TEST(Rotate, SmallSquareAndRectangularMatrices)
{
    Mat r;

    // ----------------------
    // 2x2 matrix
    {
        Mat m2x2 = (Mat_<int>(2,2) << 1,2,3,4);
        Mat e;

        e = (Mat_<int>(2,2) << 3,1,4,2);
        rotate(m2x2, r, ROTATE_90_CLOCKWISE);
        EXPECT_EQ(0, norm(r, e, NORM_INF));

        e = (Mat_<int>(2,2) << 4,3,2,1);
        rotate(m2x2, r, ROTATE_180);
        EXPECT_EQ(0, norm(r, e, NORM_INF));

        e = (Mat_<int>(2,2) << 2,4,1,3);
        rotate(m2x2, r, ROTATE_90_COUNTERCLOCKWISE);
        EXPECT_EQ(0, norm(r, e, NORM_INF));
    }

    // ----------------------
    // 3x3 matrix
    {
        Mat m3x3 = (Mat_<int>(3,3) << 1,2,3,4,5,6,7,8,9);
        Mat e;

        e = (Mat_<int>(3,3) << 7,4,1,8,5,2,9,6,3);
        rotate(m3x3, r, ROTATE_90_CLOCKWISE);
        EXPECT_EQ(0, norm(r, e, NORM_INF));

        e = (Mat_<int>(3,3) << 9,8,7,6,5,4,3,2,1);
        rotate(m3x3, r, ROTATE_180);
        EXPECT_EQ(0, norm(r, e, NORM_INF));

        e = (Mat_<int>(3,3) << 3,6,9,2,5,8,1,4,7);
        rotate(m3x3, r, ROTATE_90_COUNTERCLOCKWISE);
        EXPECT_EQ(0, norm(r, e, NORM_INF));
    }

    // ----------------------
    // 4x4 matrix
    {
        Mat m4x4 = (Mat_<int>(4,4) << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
        Mat e;

        e = (Mat_<int>(4,4) << 13,9,5,1,14,10,6,2,15,11,7,3,16,12,8,4);
        rotate(m4x4, r, ROTATE_90_CLOCKWISE);
        EXPECT_EQ(0, norm(r, e, NORM_INF));

        e = (Mat_<int>(4,4) << 16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);
        rotate(m4x4, r, ROTATE_180);
        EXPECT_EQ(0, norm(r, e, NORM_INF));

        e = (Mat_<int>(4,4) << 4,8,12,16,3,7,11,15,2,6,10,14,1,5,9,13);
        rotate(m4x4, r, ROTATE_90_COUNTERCLOCKWISE);
        EXPECT_EQ(0, norm(r, e, NORM_INF));
    }

    // ----------------------
    // 2x3 rectangular matrix
    {
        Mat m2x3 = (Mat_<int>(2,3) << 1,2,3,4,5,6);
        Mat e;

        e = (Mat_<int>(3,2) << 4,1,5,2,6,3);
        rotate(m2x3, r, ROTATE_90_CLOCKWISE);
        EXPECT_EQ(0, norm(r, e, NORM_INF));

        e = (Mat_<int>(2,3) << 6,5,4,3,2,1);
        rotate(m2x3, r, ROTATE_180);
        EXPECT_EQ(0, norm(r, e, NORM_INF));

        e = (Mat_<int>(3,2) << 3,6,2,5,1,4);
        rotate(m2x3, r, ROTATE_90_COUNTERCLOCKWISE);
        EXPECT_EQ(0, norm(r, e, NORM_INF));
    }

    // ----------------------
    // 3x2 rectangular matrix
    {
        Mat m3x2 = (Mat_<int>(3,2) << 1,2,3,4,5,6);
        Mat e;

        e = (Mat_<int>(2,3) << 5,3,1,6,4,2);
        rotate(m3x2, r, ROTATE_90_CLOCKWISE);
        EXPECT_EQ(0, norm(r, e, NORM_INF));

        e = (Mat_<int>(3,2) << 6,5,4,3,2,1);
        rotate(m3x2, r, ROTATE_180);
        EXPECT_EQ(0, norm(r, e, NORM_INF));

        e = (Mat_<int>(2,3) << 2,4,6,1,3,5);
        rotate(m3x2, r, ROTATE_90_COUNTERCLOCKWISE);
        EXPECT_EQ(0, norm(r, e, NORM_INF));
    }

    // ----------------------
    // Empty matrix
    {
        Mat empty;
        rotate(empty, r, ROTATE_90_CLOCKWISE);
        EXPECT_TRUE(r.empty());
        rotate(empty, r, ROTATE_180);
        EXPECT_TRUE(r.empty());
        rotate(empty, r, ROTATE_90_COUNTERCLOCKWISE);
        EXPECT_TRUE(r.empty());
    }
}

}} // namespace


