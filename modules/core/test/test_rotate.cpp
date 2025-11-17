// This file adds missing test cases for cv::rotate()
// Covers small square, rectangular, 1xN, Nx1, and float matrices.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Rotate, SmallSquareAndRectangularMatrices)
{
    cv::Mat r;

    // ----------------------
    // 2x2 matrix
    {
        cv::Mat m2x2 = (cv::Mat_<int>(2,2) << 1,2,3,4);
        cv::Mat e;

        e = (cv::Mat_<int>(2,2) << 3,1,4,2);
        cv::rotate(m2x2, r, cv::ROTATE_90_CLOCKWISE);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);

        e = (cv::Mat_<int>(2,2) << 4,3,2,1);
        cv::rotate(m2x2, r, cv::ROTATE_180);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);

        e = (cv::Mat_<int>(2,2) << 2,4,1,3);
        cv::rotate(m2x2, r, cv::ROTATE_90_COUNTERCLOCKWISE);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);
    }

    // ----------------------
    // 3x3 matrix
    {
        cv::Mat m3x3 = (cv::Mat_<int>(3,3) << 1,2,3,4,5,6,7,8,9);
        cv::Mat e;

        e = (cv::Mat_<int>(3,3) << 7,4,1,8,5,2,9,6,3);
        cv::rotate(m3x3, r, cv::ROTATE_90_CLOCKWISE);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);

        e = (cv::Mat_<int>(3,3) << 9,8,7,6,5,4,3,2,1);
        cv::rotate(m3x3, r, cv::ROTATE_180);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);

        e = (cv::Mat_<int>(3,3) << 3,6,9,2,5,8,1,4,7);
        cv::rotate(m3x3, r, cv::ROTATE_90_COUNTERCLOCKWISE);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);
    }

    // ----------------------
    // 2x3 rectangular matrix
    {
        cv::Mat m2x3 = (cv::Mat_<int>(2,3) << 1,2,3,4,5,6);
        cv::Mat e;

        e = (cv::Mat_<int>(3,2) << 4,1,5,2,6,3);
        cv::rotate(m2x3, r, cv::ROTATE_90_CLOCKWISE);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);

        e = (cv::Mat_<int>(2,3) << 6,5,4,3,2,1);
        cv::rotate(m2x3, r, cv::ROTATE_180);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);

        e = (cv::Mat_<int>(3,2) << 3,6,2,5,1,4);
        cv::rotate(m2x3, r, cv::ROTATE_90_COUNTERCLOCKWISE);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);
    }

    // ----------------------
    // 3x2 rectangular matrix
    {
        cv::Mat m3x2 = (cv::Mat_<int>(3,2) << 1,2,3,4,5,6);
        cv::Mat e;

        e = (cv::Mat_<int>(2,3) << 5,3,1,6,4,2);
        cv::rotate(m3x2, r, cv::ROTATE_90_CLOCKWISE);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);

        e = (cv::Mat_<int>(3,2) << 6,5,4,3,2,1);
        cv::rotate(m3x2, r, cv::ROTATE_180);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);

        e = (cv::Mat_<int>(2,3) << 2,4,6,1,3,5);
        cv::rotate(m3x2, r, cv::ROTATE_90_COUNTERCLOCKWISE);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);
    }

    // ----------------------
    // 1xN matrix
    {
        cv::Mat m1x4 = (cv::Mat_<int>(1,4) << 1,2,3,4);
        cv::Mat e;

        e = (cv::Mat_<int>(4,1) << 1,2,3,4);
        cv::rotate(m1x4, r, cv::ROTATE_90_CLOCKWISE);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);
    }

    // ----------------------
    // Nx1 matrix
    {
        cv::Mat m4x1 = (cv::Mat_<int>(4,1) << 1,2,3,4);
        cv::Mat e;

        e = (cv::Mat_<int>(1,4) << 4,3,2,1);
        cv::rotate(m4x1, r, cv::ROTATE_180);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);
    }

    // ----------------------
    // Float matrix
    {
        cv::Mat mf = (cv::Mat_<float>(2,2) << 1.1f, 2.2f, 3.3f, 4.4f);
        cv::Mat e = (cv::Mat_<float>(2,2) << 3.3f,1.1f,4.4f,2.2f);
        cv::rotate(mf, r, cv::ROTATE_90_CLOCKWISE);
        EXPECT_TRUE(cv::countNonZero(r != e) == 0);
    }

    // ----------------------
    // Empty matrix
    {
        cv::Mat empty;
        cv::rotate(empty, r, cv::ROTATE_90_CLOCKWISE);
        EXPECT_TRUE(r.empty());
        cv::rotate(empty, r, cv::ROTATE_180);
        EXPECT_TRUE(r.empty());
        cv::rotate(empty, r, cv::ROTATE_90_COUNTERCLOCKWISE);
        EXPECT_TRUE(r.empty());
    }
}

}} // namespace



