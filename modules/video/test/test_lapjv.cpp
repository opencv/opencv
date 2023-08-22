#include "test_precomp.hpp"
//#include "opencv2/video/lapjv"
#include "opencv2/video/tracking.hpp"
//#include "../src/lapjv/lapjv.hpp"
#include "opencv2/video/lapjv.hpp"

using namespace cv;

namespace opencv_test { namespace {

class CV_LapjvTest : public cvtest::BaseTest
{
public:
    CV_LapjvTest();
protected:
    void run(int);
    void testArrLoop();
    void testLapjvEmpty();
    void testLapjvNonSquareFail();
    void testLapjvNonContiguous();
    void testLapjvExtension();
    void testLapjvNoExtension();
    void testLapjvCostLimit(); // Declare the testLapjvCostLimit function
    void testSquare(const cv::Mat& cost, cv::Mat expectedassignment);
    void testAllInf();

    cv::Mat get_dense_8x8_int();
};


CV_LapjvTest::CV_LapjvTest()
{
}

cv::Mat CV_LapjvTest::get_dense_8x8_int()
{
    double cost_data[] = {
        1000, 2, 11, 10, 8, 7, 6, 5,
        6, 1000, 1, 8, 8, 4, 6, 7,
        5, 12, 1000, 11, 8, 12, 3, 11,
        11, 9, 10, 1000, 1, 9, 8, 10,
        11, 11, 9, 4, 1000, 2, 10, 9,
        12, 8, 5, 2, 11, 1000, 11, 9,
        10, 11, 12, 10, 9, 12, 1000, 3,
        10, 10, 10, 10, 6, 3, 1, 1000
    };

    cv::Mat cost(8, 8, CV_64F, cost_data);
    return cost.clone();
}

void CV_LapjvTest::testArrLoop()
{
    int shape[2] = { 7, 3 };
    double cc[] = {
        2.593883482138951146e-01, 3.080381437461217620e-01,
        1.976243020727339317e-01, 2.462740976049606068e-01,
        4.203993396282833528e-01, 4.286184525458427985e-01,
        1.706431415909629434e-01, 2.192929371231896185e-01,
        2.117769622802734286e-01, 2.604267578125001315e-01 };
    int ii[] = { 0, 0, 1, 1, 2, 2, 5, 5, 6, 6 };
    int jj[] = { 0, 1, 0, 1, 1, 2, 0, 1, 0, 1 };

    cv::Mat cost(2, shape, CV_64F, cv::Scalar(1000.));
    for (int i = 0; i < 10; ++i) {
        cost.at<double>(ii[i], jj[i]) = cc[i];
    }

    double* cost_data[shape[0]];
    for (int i = 0; i < shape[0]; ++i) {
        cost_data[i] = &cost.at<double>(i, 0);
    }

    std::map<int, int> expectedAssignments = { {0, 0}, {1, 1}, {2, 2}, {3, -1}, {4, -1}, {5, 0}, {6, 1} };

    cv::Mat ind0, ind1;
    //std::map<int,int> assignments = lapjv(cost_data);
    std::map<int,int> assignments = lapjv(cost);
    //double assignments = lapjv_internal(shape[0], cost_data, ind0.ptr<int_t>(), ind1.ptr<int_t>());

    for (const auto& entry : expectedAssignments) {
        int row = entry.first;
        int expectedCol = entry.second;

        // Expect that the assigned column for the row matches the expected column
        EXPECT_EQ(assignments[row], expectedCol);
    }
    // Assertions using EXPECT_NEAR and EXPECT_TRUE
    //EXPECT_NEAR(assignments, 0.8455356917416, 1e-10);
    //EXPECT_TRUE(cv::countNonZero(ind0 != (cv::Mat_<int>(1, 3) << 5, 1, 2)) == 0 ||
    //        cv::countNonZero(ind0 != (cv::Mat_<int>(1, 3) << 1, 5, 2)) == 0);

}

void CV_LapjvTest::testLapjvEmpty()
{
    cv::Mat emptyMat; // Create an empty matrix
    cost_t* cost_ptr[emptyMat.rows];
    int_t* ind0 = new int_t[emptyMat.rows];
    int_t* ind1 = new int_t[emptyMat.cols];
    // Wrap the code that raises an exception in a try-catch block
    try {
        //lapjv_internal(0, cost_ptr, ind0, ind1);
        //lapjv(cost_ptr);
        lapjv(emptyMat);

        // If the above line doesn't throw an exception, the test fails
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
    } catch (const cv::Exception& e) {
        // Check if the exception message contains "empty matrix"
        if (e.what() && strstr(e.what(), "empty matrix")) {
            // The expected exception was thrown, the test passes
            ts->set_failed_test_info(cvtest::TS::OK);
        } else {
            // A different exception occurred, the test fails
            ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
        }
    }
}

void CV_LapjvTest::testLapjvNonContiguous()
{
    cv::Mat cost = get_dense_8x8_int();

    cv::Mat subCost = cost.rowRange(0, 3).colRange(0, 3); // Create a non-contiguous submatrix

    cost_t* cost_ptr[subCost.rows];
    for (int i = 0; i < subCost.rows; ++i) {
        cost_ptr[i] = subCost.ptr<double>(i);
    }

    double expectedSubCostData[] = {
        1000, 2, 11,
        6, 1000, 1,
        5, 12, 1000
    };

    cv::Mat ind0, ind1;
    //double assignments = lapjv_internal(subCost.rows, cost_ptr, ind0.ptr<int_t>(), ind1.ptr<int_t>());
    //std::map<int,int> assignments = lapjv(cost_ptr);
    std::map<int,int> assignments = lapjv(subCost);

    EXPECT_EQ(assignments[0], 1);  // Row 0 assigned to Column 1
    EXPECT_EQ(assignments[1], 2);  // Row 1 assigned to Column 2
    EXPECT_EQ(assignments[2], 0);  // Row 2 assigned to Column 0



    // Assertions using EXPECT_NEAR and EXPECT_TRUE
    /*
    EXPECT_NEAR(assignments, 19.0, 1e-10);
    cv::Mat expectedInd0 = (cv::Mat_<int>(1, 3) << 1, 0, 2);
    cv::Mat expectedInd1 = (cv::Mat_<int>(1, 3) << 0, 1, 2);
    EXPECT_TRUE(cv::countNonZero(ind0 != expectedInd0) == 0);
    EXPECT_TRUE(cv::countNonZero(ind1 != expectedInd1) == 0);
    */
}


void CV_LapjvTest::testLapjvNonSquareFail()
{
    cv::Mat nonSquareMat(3, 2, CV_64F, cv::Scalar(0.)); // Create a non-square matrix

    cost_t* cost_ptr[nonSquareMat.rows];
    for (int i = 0; i < nonSquareMat.rows; ++i) {
        cost_ptr[i] = nonSquareMat.ptr<double>(i);
    }

    cv::Mat ind0, ind1;
    // Wrap the code that raises an exception in a try-catch block
    try {
        //lapjv_internal(nonSquareMat.rows, cost_ptr, ind0.ptr<int_t>(), ind1.ptr<int_t>());
        //lapjv(cost_ptr);
        lapjv(nonSquareMat);
        // If the above line doesn't throw an exception, the test fails
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
    } catch (const cv::Exception& e) {
        // Check if the exception message contains "non-square"
        if (e.what() && strstr(e.what(), "non-square")) {
            // The expected exception was thrown, the test passes
            ts->set_failed_test_info(cvtest::TS::OK);
        } else {
            // A different exception occurred, the test fails
            ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
        }
    }
}

void CV_LapjvTest::testLapjvExtension()
{
    cv::Mat cost = get_dense_8x8_int();
    cv::Mat subCost = cost.rowRange(0, 2).colRange(0, 4);

    std::map<int, int> expectedAssignments =
    {
        {0, 1}, // Row 0 is assigned to column 2
        {1, 2}  // Row 1 is assigned to column 1
    };

    cost_t* cost_ptr[subCost.rows];
    for (int i = 0; i < subCost.rows; ++i) {
    cost_ptr[i] = subCost.ptr<double>(i);
    }

    cv::Mat ind0, ind1;
    //double assignments = lapjv_internal(subCost.rows, cost_ptr,ind0.ptr<int_t>(), ind1.ptr<int_t>());
    //map<int,int> assignments = lapjv(cost_ptr);
    map<int,int> assignments = lapjv(subCost);

    // Assertions using EXPECT_NEAR and EXPECT_TRUE
    for (const auto& assignment : assignments) {
        int i = assignment.first;
        int j = assignment.second;
        int expected_j = expectedAssignments[i];
        EXPECT_EQ(j, expected_j);  // Compare the assignments
    }
    /*
    EXPECT_TRUE(cv::countNonZero(ind0 != expectedInd0) == 0);
    EXPECT_TRUE(cv::countNonZero(ind1 != expectedInd1) == 0);
    EXPECT_EQ(ind0.rows, subCost.rows);
    EXPECT_EQ(ind1.rows, subCost.cols);
    */
}

void CV_LapjvTest::testLapjvNoExtension()
{
    cv::Mat cost = get_dense_8x8_int();

    std::map<int, int> expectedAssignments =
    {
        {0, 1}, // Row 0 is assigned to column 2
        {1, 2},  // Row 1 is assigned to column 1
        {2, 0}, //...
        {3, 3}
    };

    cv::Mat c = (cv::Mat_<double>(6, 4) <<
                 cost.at<double>(0, 0), cost.at<double>(0, 1), cost.at<double>(0, 2), cost.at<double>(0, 3),
                 cost.at<double>(1, 0), cost.at<double>(1, 1), cost.at<double>(1, 2), cost.at<double>(1, 3),
                 1001, 1001, 1001, 2001,
                 2001, 1001, 1001, 1001);

    cost_t* cost_ptr[c.rows];
    for (int i = 0; i < c.rows; ++i) {
        cost_ptr[i] = c.ptr<double>(i);
    }

    cv::Mat ind0, ind1;
    //double assignments = lapjv_internal(c.rows, cost_ptr, ind0.ptr<int_t>(), ind1.ptr<int_t>());
    //map<int,int> assignments = lapjv(cost_ptr);
    map<int,int> assignments = lapjv(c);

    for (const auto& assignment : assignments) {
        int i = assignment.first;
        int j = assignment.second;
        int expected_j = expectedAssignments[i];
        EXPECT_EQ(j, expected_j);  // Compare the assignments
    }

    // Assertions using EXPECT_NEAR and EXPECT_TRUE
    /*
    EXPECT_NEAR(assignments - 2002, 3.0, 1e-10);
    cv::Mat expectedInd0 = (cv::Mat_<int>(1, 4) << 1, 2, 0, 3);
    cv::Mat expectedInd1 = (cv::Mat_<int>(1, 4) << 2, 0, 1, 3);
    EXPECT_TRUE(cv::countNonZero(ind0 != expectedInd0) == 0);
    EXPECT_TRUE(cv::countNonZero(ind1 != expectedInd1) == 0);
    */
}

void CV_LapjvTest::testLapjvCostLimit()
{
    cv::Mat cost = get_dense_8x8_int();
    cv::Mat subCost = cost.rowRange(0, 3).colRange(0, 3);
    std::map<int, int> expectedAssignments =
    {
        {0, 1}, // Row 0 is assigned to column 2
        {1, 2},  // Row 1 is assigned to column 1
        {2, 0}, //...
    };

    cost_t* cost_ptr[subCost.rows];
    for (int i = 0; i < subCost.rows; ++i) {
        cost_ptr[i] = subCost.ptr<double>(i);
    }

    cv::Mat ind0, ind1;
    //double assignments = lapjv_internal(subCost.rows, cost_ptr, ind0.ptr<int_t>(), ind1.ptr<int_t>());
    //map<int,int> assignments = lapjv(cost_ptr);
    map<int,int> assignments = lapjv(subCost);

    for (const auto& assignment : assignments) {
        int i = assignment.first;
        int j = assignment.second;
        int expected_j = expectedAssignments[i];
        EXPECT_EQ(j, expected_j);  // Compare the assignments
    }

    // Assertions using EXPECT_NEAR and EXPECT_TRUE
    /*
    EXPECT_NEAR(assignments, 3.0, 1e-10);
    cv::Mat expectedInd0 = (cv::Mat_<int>(1, 3) << 1, 2, -1);
    cv::Mat expectedInd1 = (cv::Mat_<int>(1, 3) << -1, 0, 1);
    EXPECT_TRUE(cv::countNonZero(ind0 != expectedInd0) == 0);
    EXPECT_TRUE(cv::countNonZero(ind1 != expectedInd1) == 0);
    */
}

void CV_LapjvTest::testSquare(const cv::Mat& cost, cv::Mat expectedAssignment)
{
    cost_t* cost_ptr[cost.rows];
    for (int i = 0; i < cost.rows; ++i) {
        cost_ptr[i] = new cost_t[cost.cols]; // Allocate memory for each row
        const double* src_row = cost.ptr<double>(i); // Get the source row
        for (int j = 0; j < cost.cols; ++j) {
            cost_ptr[i][j] = static_cast<cost_t>(src_row[j]); // Convert and copy element
        }
    }

    cv::Mat ind0, ind1;
    //double assignments = lapjv_internal(cost.rows, cost_ptr, ind0.ptr<int_t>(), ind1.ptr<int_t>());
    //map<int,int> assignments = lapjv(cost_ptr);
    map<int,int> assignments = lapjv(cost);

    for (const auto& assignment : assignments) {
        int i = assignment.first;
        int j = assignment.second;
        int expected_j = expectedAssignment.at<float>(i);
        EXPECT_EQ(j, expected_j);  // Compare the assignments
    }

    //EXPECT_NEAR(assignments, expectedAssignment, 1e-10);
}

void CV_LapjvTest::testAllInf()
{
    cv::Mat cost(5, 5, CV_64F);
    cost.setTo(cv::Scalar::all(std::numeric_limits<double>::infinity()));

    cost_t* cost_ptr[cost.rows];
    for (int i = 0; i < cost.rows; ++i) {
        cost_ptr[i] = cost.ptr<double>(i);
    }

    cv::Mat ind0, ind1;
    //double assignments = lapjv_internal(cost.rows, cost_ptr, ind0.ptr<int_t>(), ind1.ptr<int_t>());
    //map<int,int> assignments = lapjv(cost_ptr);
    map<int,int> assignments = lapjv(cost);

    // Assertions using EXPECT_NEAR and EXPECT_TRUE
    EXPECT_EQ(ind0.rows, cost.rows);
    EXPECT_EQ(ind1.rows, cost.cols);
    //EXPECT_NEAR(assignments, std::numeric_limits<double>::infinity(), 1e-10);
    for (const auto& assignment : assignments) {
        EXPECT_EQ(assignment.second, -1);  // No assignment (unassigned)
    }
}





void CV_LapjvTest::run(int)
{
    // Call your testArrLoop and testLapjvEmpty functions
    testArrLoop();
    testLapjvEmpty();
    testLapjvNonSquareFail();
    testLapjvNonContiguous();
    testLapjvExtension();
    testLapjvNoExtension();
    testLapjvCostLimit(); // Declare the testLapjvCostLimit function

    cv::Mat cost1 = (cv::Mat_<double>(5, 5) <<
                11.0, 20.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                12.0, std::numeric_limits<double>::infinity(), 12.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(), 11.0, 10.0, 15.0, 9.0,
                15.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), 22.0, std::numeric_limits<double>::infinity(),
                13.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), 15.0);

    testSquare(cost1, cv::Mat {0, 2, 1, 3, 4});
    testAllInf();

    // Indicate test success
    ts->set_failed_test_info(cvtest::TS::OK);
}


TEST(Video_Lapjv, accuracy) { CV_LapjvTest test; test.safe_run(); }

}} // namespace
/* End of file. */
