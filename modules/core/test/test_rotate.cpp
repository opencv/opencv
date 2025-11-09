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
