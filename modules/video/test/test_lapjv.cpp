#include "test_precomp.hpp"

#include "opencv2/video/lapjv.hpp"
#include <iostream>


using namespace cv;

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

    std::map<int, int> assignments = lapjv(cost);

    for (const auto& assignment : assignments) {
        int i = assignment.first;
        int j = assignment.second;
        int expected_j = expectedAssignment.at(i);
        EXPECT_EQ(j, expected_j);
    }
}

TEST(Video_Lapjv, testEmpty) {
    cv::Mat emptyMat;
    auto ret = lapjv(emptyMat);
    EXPECT_TRUE(ret.empty());
}

// TEST(Video_Lapjv, testInf) {

//     cv::Mat cost(10, 10, CV_32F);
//     cost.setTo(cv::Scalar::all(std::numeric_limits<double>::infinity()));

//     map<int,int> assignments = lapjv(cost);

//     for (const auto& assignment : assignments) {
//         EXPECT_EQ(assignment.second, -1);  // No assignment (unassigned)
//     }
// }

// TEST(Video_Lapjv, testNonSquare) {

//     cv::Mat nonSquareMat(3, 2, CV_64F, cv::Scalar(0.)); // Create a non-square matrix
//     /*
//     cost_t* cost_ptr[nonSquareMat.rows];
//     for (int i = 0; i < nonSquareMat.rows; ++i) {
//         cost_ptr[i] = nonSquareMat.ptr<double>(i);
//     }
//     */

//     cv::Mat ind0, ind1;
//     // Wrap the code that raises an exception in a try-catch block
//     try {
//         //lapjv_internal(nonSquareMat.rows, cost_ptr, ind0.ptr<int_t>(), ind1.ptr<int_t>());
//         //lapjv(cost_ptr);
//         //lapjv(nonSquareMat);
//         // If the above line doesn't throw an exception, the test fails
//         ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
//     } catch (const cv::Exception& e) {
//         // Check if the exception message contains "non-square"
//         if (e.what() && strstr(e.what(), "non-square")) {
//             // The expected exception was thrown, the test passes
//             ts->set_failed_test_info(cvtest::TS::OK);
//         } else {
//             // A different exception occurred, the test fails
//             ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
//         }
//     }
// }

// TEST(Video_Lapjv, testCostLimit) {
// {
//     cv::Mat cost = get_dense_8x8_int();
//     cv::Mat subCost = cost.rowRange(0, 3).colRange(0, 3);
//     std::map<int, int> expectedAssignments =
//     {
//         {0, 1}, // Row 0 is assigned to column 2
//         {1, 2},  // Row 1 is assigned to column 1
//         {2, 0}, //...
//     };
//     /*
//     cost_t* cost_ptr[subCost.rows];
//     for (int i = 0; i < subCost.rows; ++i) {
//         cost_ptr[i] = subCost.ptr<double>(i);
//     }
//     */

//     cv::Mat ind0, ind1;
//     //double assignments = lapjv_internal(subCost.rows, cost_ptr, ind0.ptr<int_t>(), ind1.ptr<int_t>());
//     //map<int,int> assignments = lapjv(cost_ptr);
//     //map<int,int> assignments = lapjv(subCost);
//     /*
//     for (const auto& assignment : assignments) {
//         int i = assignment.first;
//         int j = assignment.second;
//         int expected_j = expectedAssignments[i];
//         EXPECT_EQ(j, expected_j);  // Compare the assignments
//     }
//     */

//     // Assertions using EXPECT_NEAR and EXPECT_TRUE
//     /*
//     EXPECT_NEAR(assignments, 3.0, 1e-10);
//     cv::Mat expectedInd0 = (cv::Mat_<int>(1, 3) << 1, 2, -1);
//     cv::Mat expectedInd1 = (cv::Mat_<int>(1, 3) << -1, 0, 1);
//     EXPECT_TRUE(cv::countNonZero(ind0 != expectedInd0) == 0);
//     EXPECT_TRUE(cv::countNonZero(ind1 != expectedInd1) == 0);
//     */
// }
}} // namespace
#endif
/* End of file. */
