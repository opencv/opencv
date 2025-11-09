#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/mat.hpp>
#include <gtest/gtest.h>

namespace opencv_test { namespace {

TEST(Rotate, SmallSquareAndRectangularMatrices)
{
    using namespace cv;

    Mat r, e;

    // 2x2
    Mat m2x2 = (Mat_<int>(2,2) << 1,2,3,4);
    rotate(m2x2, r, ROTATE_90_CLOCKWISE);
    e = (Mat_<int>(2,2) << 3,1,4,2);
    EXPECT_EQ(0, norm(r, e, NORM_INF));

    rotate(m2x2, r, ROTATE_180);
    e = (Mat_<int>(2,2) << 4,3,2,1);
    EXPECT_EQ(0, norm(r, e, NORM_INF));

    rotate(m2x2, r, ROTATE_90_COUNTERCLOCKWISE);
    e = (Mat_<int>(2,2) << 2,4,1,3);
    EXPECT_EQ(0, norm(r, e, NORM_INF));

    // 3x3
    Mat m3x3 = (Mat_<int>(3,3) << 1,2,3,4,5,6,7,8,9);
    rotate(m3x3, r, ROTATE_90_CLOCKWISE);
    e = (Mat_<int>(3,3) << 7,4,1,8,5,2,9,6,3);
    EXPECT_EQ(0, norm(r, e, NORM_INF));

    rotate(m3x3, r, ROTATE_180);
    e = (Mat_<int>(3,3) << 9,8,7,6,5,4,3,2,1);
    EXPECT_EQ(0, norm(r, e, NORM_INF));

    rotate(m3x3, r, ROTATE_90_COUNTERCLOCKWISE);
    e = (Mat_<int>(3,3) << 3,6,9,2,5,8,1,4,7);
    EXPECT_EQ(0, norm(r, e, NORM_INF));

    // 4x4
    Mat m4x4 = (Mat_<int>(4,4) << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
    rotate(m4x4, r, ROTATE_90_CLOCKWISE);
    e = (Mat_<int>(4,4) << 13,9,5,1,14,10,6,2,15,11,7,3,16,12,8,4);
    EXPECT_EQ(0, norm(r, e, NORM_INF));

    rotate(m4x4, r, ROTATE_180);
    e = (Mat_<int>(4,4) << 16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);
    EXPECT_EQ(0, norm(r, e, NORM_INF));

    rotate(m4x4, r, ROTATE_90_COUNTERCLOCKWISE);
    e = (Mat_<int>(4,4) << 4,8,12,16,3,7,11,15,2,6,10,14,1,5,9,13);
    EXPECT_EQ(0, norm(r, e, NORM_INF));

    // Non-square 2x3
    Mat m2x3 = (Mat_<int>(2,3) << 1,2,3,4,5,6);
    rotate(m2x3, r, ROTATE_90_CLOCKWISE);
    e = (Mat_<int>(3,2) << 4,1,5,2,6,3);
    EXPECT_EQ(0, norm(r, e, NORM_INF));

    // Empty matrix
    Mat empty;
    rotate(empty, r, ROTATE_90_CLOCKWISE);
    EXPECT_TRUE(r.empty());
}

} } // namespace opencv_test
