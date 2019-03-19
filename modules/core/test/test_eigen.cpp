/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "test_precomp.hpp"

namespace opencv_test { namespace {

#define sign(a) a > 0 ? 1 : a == 0 ? 0 : -1

#define CORE_EIGEN_ERROR_COUNT 1
#define CORE_EIGEN_ERROR_SIZE  2
#define CORE_EIGEN_ERROR_DIFF  3
#define CORE_EIGEN_ERROR_ORTHO 4
#define CORE_EIGEN_ERROR_ORDER 5

#define MESSAGE_ERROR_COUNT "Matrix of eigen values must have the same rows as source matrix and 1 column."
#define MESSAGE_ERROR_SIZE "Source matrix and matrix of eigen vectors must have the same sizes."
#define MESSAGE_ERROR_DIFF_1 "Accuracy of eigen values computing less than required."
#define MESSAGE_ERROR_DIFF_2 "Accuracy of eigen vectors computing less than required."
#define MESSAGE_ERROR_ORTHO "Matrix of eigen vectors is not orthogonal."
#define MESSAGE_ERROR_ORDER "Eigen values are not sorted in descending order."

const int COUNT_NORM_TYPES = 3;
const int NORM_TYPE[COUNT_NORM_TYPES] = {cv::NORM_L1, cv::NORM_L2, cv::NORM_INF};

enum TASK_TYPE_EIGEN {VALUES, VECTORS};

class Core_EigenTest: public cvtest::BaseTest
{
public:

    Core_EigenTest();
    ~Core_EigenTest();

protected:

    bool test_values(const cv::Mat& src);												// complex test for eigen without vectors
    bool check_full(int type);													// compex test for symmetric matrix
    virtual void run (int) = 0;													// main testing method

protected:

    float eps_val_32, eps_vec_32;
    float eps_val_64, eps_vec_64;
    int ntests;

    bool check_pair_count(const cv::Mat& src, const cv::Mat& evalues, int low_index = -1, int high_index = -1);
    bool check_pair_count(const cv::Mat& src, const cv::Mat& evalues, const cv::Mat& evectors, int low_index = -1, int high_index = -1);
    bool check_pairs_order(const cv::Mat& eigen_values);											// checking order of eigen values & vectors (it should be none up)
    bool check_orthogonality(const cv::Mat& U);												// checking is matrix of eigen vectors orthogonal
    bool test_pairs(const cv::Mat& src);													// complex test for eigen with vectors

    void print_information(const size_t norm_idx, const cv::Mat& src, double diff, double max_diff);
};

class Core_EigenTest_Scalar : public Core_EigenTest
{
public:
    Core_EigenTest_Scalar() : Core_EigenTest() {}
    ~Core_EigenTest_Scalar();

    virtual void run(int) = 0;
};

class Core_EigenTest_Scalar_32 : public Core_EigenTest_Scalar
{
public:
    Core_EigenTest_Scalar_32() : Core_EigenTest_Scalar() {}
    ~Core_EigenTest_Scalar_32();

    void run(int);
};

class Core_EigenTest_Scalar_64 : public Core_EigenTest_Scalar
{
public:
    Core_EigenTest_Scalar_64() : Core_EigenTest_Scalar() {}
    ~Core_EigenTest_Scalar_64();
    void run(int);
};

class Core_EigenTest_32 : public Core_EigenTest
{
public:
    Core_EigenTest_32(): Core_EigenTest() {}
    ~Core_EigenTest_32() {}
    void run(int);
};

class Core_EigenTest_64 : public Core_EigenTest
{
public:
    Core_EigenTest_64(): Core_EigenTest() {}
    ~Core_EigenTest_64() {}
    void run(int);
};

Core_EigenTest_Scalar::~Core_EigenTest_Scalar() {}
Core_EigenTest_Scalar_32::~Core_EigenTest_Scalar_32() {}
Core_EigenTest_Scalar_64::~Core_EigenTest_Scalar_64() {}

void Core_EigenTest_Scalar_32::run(int)
{
    for (int i = 0; i < ntests; ++i)
    {
        float value = cv::randu<float>();
        cv::Mat src(1, 1, CV_32FC1, Scalar::all((float)value));
        test_values(src);
    }
}

void Core_EigenTest_Scalar_64::run(int)
{
    for (int i = 0; i < ntests; ++i)
    {
        float value = cv::randu<float>();
        cv::Mat src(1, 1, CV_64FC1, Scalar::all((double)value));
        test_values(src);
    }
}

void Core_EigenTest_32::run(int) { check_full(CV_32FC1); }
void Core_EigenTest_64::run(int) { check_full(CV_64FC1); }

Core_EigenTest::Core_EigenTest()
: eps_val_32(1e-3f), eps_vec_32(1e-3f),
  eps_val_64(1e-4f), eps_vec_64(1e-4f), ntests(100) {}
Core_EigenTest::~Core_EigenTest() {}

bool Core_EigenTest::check_pair_count(const cv::Mat& src, const cv::Mat& evalues, int low_index, int high_index)
{
    int n = src.rows, s = sign(high_index);
    if (!( (evalues.rows == n - max<int>(0, low_index) - ((int)((n/2.0)*(s*s-s)) + (1+s-s*s)*(n - (high_index+1)))) && (evalues.cols == 1)))
    {
        std::cout << endl; std::cout << "Checking sizes of eigen values matrix " << evalues << "..." << endl;
        std::cout << "Number of rows: " << evalues.rows << "   Number of cols: " << evalues.cols << endl;
        std::cout << "Size of src symmetric matrix: " << src.rows << " * " << src.cols << endl; std::cout << endl;
        CV_Error(CORE_EIGEN_ERROR_COUNT, MESSAGE_ERROR_COUNT);
    }
    return true;
}

bool Core_EigenTest::check_pair_count(const cv::Mat& src, const cv::Mat& evalues, const cv::Mat& evectors, int low_index, int high_index)
{
    int n = src.rows, s = sign(high_index);
    int right_eigen_pair_count = n - max<int>(0, low_index) - ((int)((n/2.0)*(s*s-s)) + (1+s-s*s)*(n - (high_index+1)));

    if (!(evectors.rows == right_eigen_pair_count && evectors.cols == right_eigen_pair_count))
    {
        std::cout << endl; std::cout << "Checking sizes of eigen vectors matrix " << evectors << "..." << endl;
        std::cout << "Number of rows: " << evectors.rows << "   Number of cols: " << evectors.cols << endl;
        std:: cout << "Size of src symmetric matrix: " << src.rows << " * " << src.cols << endl; std::cout << endl;
        CV_Error (CORE_EIGEN_ERROR_SIZE, MESSAGE_ERROR_SIZE);
    }

    if (!(evalues.rows == right_eigen_pair_count && evalues.cols == 1))
    {
        std::cout << endl; std::cout << "Checking sizes of eigen values matrix " << evalues << "..." << endl;
        std::cout << "Number of rows: " << evalues.rows << "   Number of cols: " << evalues.cols << endl;
        std:: cout << "Size of src symmetric matrix: " << src.rows << " * " << src.cols << endl; std::cout << endl;
        CV_Error (CORE_EIGEN_ERROR_COUNT, MESSAGE_ERROR_COUNT);
    }

    return true;
}

void Core_EigenTest::print_information(const size_t norm_idx, const cv::Mat& src, double diff, double max_diff)
{
    switch (NORM_TYPE[norm_idx])
    {
    case cv::NORM_L1: std::cout << "L1"; break;
    case cv::NORM_L2: std::cout << "L2"; break;
    case cv::NORM_INF: std::cout << "INF"; break;
    default: break;
    }

    cout << "-criteria... " << endl;
    cout << "Source size: " << src.rows << " * " << src.cols << endl;
    cout << "Difference between original eigen vectors matrix and result: " << diff << endl;
    cout << "Maximum allowed difference: " << max_diff << endl; cout << endl;
}

bool Core_EigenTest::check_orthogonality(const cv::Mat& U)
{
    int type = U.type();
    double eps_vec = type == CV_32FC1 ? eps_vec_32 : eps_vec_64;
    cv::Mat UUt; cv::mulTransposed(U, UUt, false);

    cv::Mat E = Mat::eye(U.rows, U.cols, type);

    for (int i = 0; i < COUNT_NORM_TYPES; ++i)
    {
        double diff = cvtest::norm(UUt, E, NORM_TYPE[i] | cv::NORM_RELATIVE);
        if (diff > eps_vec)
        {
            std::cout << endl; std::cout << "Checking orthogonality of matrix " << U << ": ";
            print_information(i, U, diff, eps_vec);
            CV_Error(CORE_EIGEN_ERROR_ORTHO, MESSAGE_ERROR_ORTHO);
        }
    }

    return true;
}

bool Core_EigenTest::check_pairs_order(const cv::Mat& eigen_values)
{
    switch (eigen_values.type())
    {
    case CV_32FC1:
        {
            for (int i = 0; i < (int)(eigen_values.total() - 1); ++i)
                if (!(eigen_values.at<float>(i, 0) > eigen_values.at<float>(i+1, 0)))
                {
                std::cout << endl; std::cout << "Checking order of eigen values vector " << eigen_values << "..." << endl;
                std::cout << "Pair of indexes with non descending of eigen values: (" << i << ", " << i+1 << ")." << endl;
                std::cout << endl;
                CV_Error(CORE_EIGEN_ERROR_ORDER, MESSAGE_ERROR_ORDER);
            }

            break;
        }

    case CV_64FC1:
        {
            for (int i = 0; i < (int)(eigen_values.total() - 1); ++i)
                if (!(eigen_values.at<double>(i, 0) > eigen_values.at<double>(i+1, 0)))
                {
                    std::cout << endl; std::cout << "Checking order of eigen values vector " << eigen_values << "..." << endl;
                    std::cout << "Pair of indexes with non descending of eigen values: (" << i << ", " << i+1 << ")." << endl;
                    std::cout << endl;
                    CV_Error(CORE_EIGEN_ERROR_ORDER, "Eigen values are not sorted in descending order.");
                }

            break;
        }

    default:;
    }

    return true;
}

bool Core_EigenTest::test_pairs(const cv::Mat& src)
{
    int type = src.type();
    double eps_vec = type == CV_32FC1 ? eps_vec_32 : eps_vec_64;

    cv::Mat eigen_values, eigen_vectors;

    cv::eigen(src, eigen_values, eigen_vectors);

    if (!check_pair_count(src, eigen_values, eigen_vectors))
        return false;

    if (!check_orthogonality (eigen_vectors))
        return false;

    if (!check_pairs_order(eigen_values))
        return false;

    cv::Mat eigen_vectors_t; cv::transpose(eigen_vectors, eigen_vectors_t);

    // Check:
    // src * eigenvector = eigenval * eigenvector
    cv::Mat lhs(src.rows, src.cols, type);
    cv::Mat rhs(src.rows, src.cols, type);

    lhs = src*eigen_vectors_t;

    for (int i = 0; i < src.cols; ++i)
    {
        double eigenval = 0;
        switch (type)
        {
        case CV_32FC1: eigenval = eigen_values.at<float>(i, 0); break;
        case CV_64FC1: eigenval = eigen_values.at<double>(i, 0); break;
        }
        cv::Mat rhs_v = eigenval * eigen_vectors_t.col(i);
        rhs_v.copyTo(rhs.col(i));
    }

    for (int i = 0; i < COUNT_NORM_TYPES; ++i)
    {
        double diff = cvtest::norm(lhs, rhs, NORM_TYPE[i] | cv::NORM_RELATIVE);
        if (diff > eps_vec)
        {
            std::cout << endl; std::cout << "Checking accuracy of eigen vectors computing for matrix " << src << ": ";
            print_information(i, src, diff, eps_vec);
            CV_Error(CORE_EIGEN_ERROR_DIFF, MESSAGE_ERROR_DIFF_2);
        }
    }

    return true;
}

bool Core_EigenTest::test_values(const cv::Mat& src)
{
    int type = src.type();
    double eps_val = type == CV_32FC1 ? eps_val_32 : eps_val_64;

    cv::Mat eigen_values_1, eigen_values_2, eigen_vectors;

    if (!test_pairs(src)) return false;

    cv::eigen(src, eigen_values_1, eigen_vectors);
    cv::eigen(src, eigen_values_2);

    if (!check_pair_count(src, eigen_values_2)) return false;

    for (int i = 0; i < COUNT_NORM_TYPES; ++i)
    {
        double diff = cvtest::norm(eigen_values_1, eigen_values_2, NORM_TYPE[i] | cv::NORM_RELATIVE);
        if (diff > eps_val)
        {
            std::cout << endl; std::cout << "Checking accuracy of eigen values computing for matrix " << src << ": ";
            print_information(i, src, diff, eps_val);
            CV_Error(CORE_EIGEN_ERROR_DIFF, MESSAGE_ERROR_DIFF_1);
        }
    }

    return true;
}

bool Core_EigenTest::check_full(int type)
{
    const int MAX_DEGREE = 7;

    RNG rng = cv::theRNG(); // fix the seed

    for (int i = 0; i < ntests; ++i)
    {
        int src_size = (int)(std::pow(2.0, (rng.uniform(0, MAX_DEGREE) + 1.)));

        cv::Mat src(src_size, src_size, type);

        for (int j = 0; j < src.rows; ++j)
            for (int k = j; k < src.cols; ++k)
                if (type == CV_32FC1)  src.at<float>(k, j) = src.at<float>(j, k) = cv::randu<float>();
        else	src.at<double>(k, j) = src.at<double>(j, k) = cv::randu<double>();

        if (!test_values(src)) return false;
    }

    return true;
}

TEST(Core_Eigen, scalar_32) {Core_EigenTest_Scalar_32 test; test.safe_run(); }
TEST(Core_Eigen, scalar_64) {Core_EigenTest_Scalar_64 test; test.safe_run(); }
TEST(Core_Eigen, vector_32) { Core_EigenTest_32 test; test.safe_run(); }
TEST(Core_Eigen, vector_64) { Core_EigenTest_64 test; test.safe_run(); }

template<typename T>
static void testEigen(const Mat_<T>& src, const Mat_<T>& expected_eigenvalues, bool runSymmetric = false)
{
    SCOPED_TRACE(runSymmetric ? "cv::eigen" : "cv::eigenNonSymmetric");

    int type = traits::Type<T>::value;
    const T eps = src.type() == CV_32F ? 1e-4f : 1e-6f;

    Mat eigenvalues, eigenvectors, eigenvalues0;

    if (runSymmetric)
    {
        cv::eigen(src, eigenvalues0, noArray());
        cv::eigen(src, eigenvalues, eigenvectors);
    }
    else
    {
        cv::eigenNonSymmetric(src, eigenvalues0, noArray());
        cv::eigenNonSymmetric(src, eigenvalues, eigenvectors);
    }
#if 0
    std::cout << "src = " << src << std::endl;
    std::cout << "eigenvalues.t() = " << eigenvalues.t() << std::endl;
    std::cout << "eigenvectors = " << eigenvectors << std::endl;
#endif
    ASSERT_EQ(type, eigenvalues0.type());
    ASSERT_EQ(type, eigenvalues.type());
    ASSERT_EQ(type, eigenvectors.type());

    ASSERT_EQ(src.rows, eigenvalues.rows);
    ASSERT_EQ(eigenvalues.rows, eigenvectors.rows);
    ASSERT_EQ(src.rows, eigenvectors.cols);

    EXPECT_LT(cvtest::norm(eigenvalues, eigenvalues0, NORM_INF), eps);

    // check definition: src*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()
    for (int i = 0; i < src.rows; i++)
    {
        EXPECT_NEAR(eigenvalues.at<T>(i), expected_eigenvalues(i), eps) << "i=" << i;
        Mat lhs = src*eigenvectors.row(i).t();
        Mat rhs = eigenvalues.at<T>(i)*eigenvectors.row(i).t();
        EXPECT_LT(cvtest::norm(lhs, rhs, NORM_INF), eps)
                << "i=" << i << " eigenvalue=" << eigenvalues.at<T>(i) << std::endl
                << "lhs=" << lhs.t() << std::endl
                << "rhs=" << rhs.t();
    }
}

template<typename T>
static void testEigenSymmetric3x3()
{
    /*const*/ T values_[] = {
            2, -1, 0,
            -1, 2, -1,
            0, -1, 2
    };
    Mat_<T> src(3, 3, values_);

    /*const*/ T expected_eigenvalues_[] = { 3.414213562373095f, 2, 0.585786437626905f };
    Mat_<T> expected_eigenvalues(3, 1, expected_eigenvalues_);

    testEigen(src, expected_eigenvalues);
    testEigen(src, expected_eigenvalues, true);
}
TEST(Core_EigenSymmetric, float3x3) { testEigenSymmetric3x3<float>(); }
TEST(Core_EigenSymmetric, double3x3) { testEigenSymmetric3x3<double>(); }

template<typename T>
static void testEigenSymmetric5x5()
{
    /*const*/ T values_[5*5] = {
            5, -1, 0, 2, 1,
            -1, 4, -1, 0, 0,
            0, -1, 3, 1, -1,
            2, 0, 1, 4, 0,
            1, 0, -1, 0, 1
    };
    Mat_<T> src(5, 5, values_);

    /*const*/ T expected_eigenvalues_[] = { 7.028919644935684f, 4.406130784616501f, 3.73626552682258f, 1.438067799899037f, 0.390616243726198f };
    Mat_<T> expected_eigenvalues(5, 1, expected_eigenvalues_);

    testEigen(src, expected_eigenvalues);
    testEigen(src, expected_eigenvalues, true);
}
TEST(Core_EigenSymmetric, float5x5) { testEigenSymmetric5x5<float>(); }
TEST(Core_EigenSymmetric, double5x5) { testEigenSymmetric5x5<double>(); }


template<typename T>
static void testEigen2x2()
{
    /*const*/ T values_[] = { 4, 1, 6, 3 };
    Mat_<T> src(2, 2, values_);

    /*const*/ T expected_eigenvalues_[] = { 6, 1 };
    Mat_<T> expected_eigenvalues(2, 1, expected_eigenvalues_);

    testEigen(src, expected_eigenvalues);
}
TEST(Core_EigenNonSymmetric, float2x2) { testEigen2x2<float>(); }
TEST(Core_EigenNonSymmetric, double2x2) { testEigen2x2<double>(); }

template<typename T>
static void testEigen3x3()
{
    /*const*/ T values_[3*3] = {
            3,1,0,
            0,3,1,
            0,0,3
    };
    Mat_<T> src(3, 3, values_);

    /*const*/ T expected_eigenvalues_[] = { 3, 3, 3 };
    Mat_<T> expected_eigenvalues(3, 1, expected_eigenvalues_);

    testEigen(src, expected_eigenvalues);
}
TEST(Core_EigenNonSymmetric, float3x3) { testEigen3x3<float>(); }
TEST(Core_EigenNonSymmetric, double3x3) { testEigen3x3<double>(); }

typedef testing::TestWithParam<int> Core_EigenZero;
TEST_P(Core_EigenZero, double)
{
    int N = GetParam();
    Mat_<double> srcZero = Mat_<double>::zeros(N, N);
    Mat_<double> expected_eigenvalueZero = Mat_<double>::zeros(N, 1);  // 1D Mat
    testEigen(srcZero, expected_eigenvalueZero);
    testEigen(srcZero, expected_eigenvalueZero, true);
}
INSTANTIATE_TEST_CASE_P(/**/, Core_EigenZero, testing::Values(2, 3, 5));

TEST(Core_EigenNonSymmetric, convergence)
{
    Matx33d m(
        0, -1, 0,
        1, 0, 1,
        0, -1, 0);
    Mat eigenvalues, eigenvectors;
    // eigen values are complex, algorithm doesn't converge
    try
    {
        cv::eigenNonSymmetric(m, eigenvalues, eigenvectors);
        std::cout << Mat(eigenvalues.t()) << std::endl;
    }
    catch (const cv::Exception& e)
    {
        EXPECT_EQ(Error::StsNoConv, e.code) << e.what();
    }
    catch (...)
    {
        FAIL() << "Unknown exception has been raised";
    }
}

}} // namespace
