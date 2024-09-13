#include "test_precomp.hpp"

#include "opencv2/video.hpp"
#include <iostream>

#define TEST_AVAILABLE 1
#ifdef TEST_AVAILABLE

namespace opencv_test { namespace {

cv::Mat getTestCostMatrix() {
    return (cv::Mat_<float> (3, 3) << 1000, 2, 11,
                                      6, 1000, 1,
                                      5, 12, 1000);
}

std::map<int, int> getExpectedAssignment() {
    return std::map<int, int> {
        {0, 2},
        {1, 0},
        {2, 1}
    };
}

TEST(Video_Lapjv, testSquare) {
    auto cost = getTestCostMatrix();
    auto expectedAssignment = getExpectedAssignment();

    std::map<int, int> assignments = cv::lapjv(cost);

    for (const auto& assignment : assignments) {
        int i = assignment.first;
        int j = assignment.second;
        int expected_j = expectedAssignment.at(i);
        EXPECT_EQ(j, expected_j);
    }
}

TEST(Video_Lapjv, testEmpty) {
    cv::Mat emptyMat;
    auto ret = cv::lapjv(emptyMat);
    EXPECT_TRUE(ret.empty());
}


}} // namespace
#endif
/* End of file. */
