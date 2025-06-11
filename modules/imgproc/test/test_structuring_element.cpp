#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(MorphShapes, getStructuringElementDiamond)
{
    cv::Mat element = cv::getStructuringElement(cv::MORPH_DIAMOND, cv::Size(5,5));
    cv::Mat expected = (cv::Mat_<uchar>(5,5) <<
        0,0,1,0,0,
        0,1,1,1,0,
        1,1,1,1,1,
        0,1,1,1,0,
        0,0,1,0,0);
    EXPECT_EQ(0, cvtest::norm(element, expected, cv::NORM_INF));
}

}} // namespace
