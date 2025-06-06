#include "test_precomp.hpp"

namespace opencv_test {

TEST(Core_UMat, construct_from_vector)
{
    std::vector<int> src = {1, 2, 3, 4};
    UMat um(src); // copyData parameter is deprecated and ignored

    src[0] = 100; // modify source to ensure data was copied

    Mat result;
    um.copyTo(result);

    ASSERT_EQ(4, result.rows);
    ASSERT_EQ(1, result.cols);
    ASSERT_EQ(CV_32S, result.type());
    EXPECT_EQ(1, result.at<int>(0));
    EXPECT_EQ(2, result.at<int>(1));
    EXPECT_EQ(3, result.at<int>(2));
    EXPECT_EQ(4, result.at<int>(3));
}

} // namespace opencv_test
