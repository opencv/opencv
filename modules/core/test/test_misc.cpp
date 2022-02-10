// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include <cmath>

namespace opencv_test { namespace {

TEST(Core_OutputArrayCreate, _1997)
{
    struct local {
        static void create(OutputArray arr, Size submatSize, int type)
        {
            int sizes[] = {submatSize.width, submatSize.height};
            arr.create(sizeof(sizes)/sizeof(sizes[0]), sizes, type);
        }
    };

    Mat mat(Size(512, 512), CV_8U);
    Size submatSize = Size(256, 256);

    ASSERT_NO_THROW(local::create( mat(Rect(Point(), submatSize)), submatSize, mat.type() ));
}

TEST(Core_SaturateCast, NegativeNotClipped)
{
    double d = -1.0;
    unsigned int val = cv::saturate_cast<unsigned int>(d);

    ASSERT_EQ(0xffffffff, val);
}

template<typename T, typename U>
static double maxAbsDiff(const T &t, const U &u)
{
  Mat_<double> d;
  absdiff(t, u, d);
  double ret;
  minMaxLoc(d, NULL, &ret);
  return ret;
}

TEST(Core_OutputArrayAssign, _Matxd_Matd)
{
    Mat expected = (Mat_<double>(2,3) << 1, 2, 3, .1, .2, .3);
    Matx23d actualx;

    {
        OutputArray oa(actualx);
        oa.assign(expected);
    }

    Mat actual = (Mat) actualx;

    EXPECT_LE(maxAbsDiff(expected, actual), 0.0);
}

TEST(Core_OutputArrayAssign, _Matxd_Matf)
{
    Mat expected = (Mat_<float>(2,3) << 1, 2, 3, .1, .2, .3);
    Matx23d actualx;

    {
        OutputArray oa(actualx);
        oa.assign(expected);
    }

    Mat actual = (Mat) actualx;

    EXPECT_LE(maxAbsDiff(expected, actual), FLT_EPSILON);
}

TEST(Core_OutputArrayAssign, _Matxf_Matd)
{
    Mat expected = (Mat_<double>(2,3) << 1, 2, 3, .1, .2, .3);
    Matx23f actualx;

    {
        OutputArray oa(actualx);
        oa.assign(expected);
    }

    Mat actual = (Mat) actualx;

    EXPECT_LE(maxAbsDiff(expected, actual), FLT_EPSILON);
}

TEST(Core_OutputArrayAssign, _Matxd_UMatd)
{
    Mat expected = (Mat_<double>(2,3) << 1, 2, 3, .1, .2, .3);
    UMat uexpected = expected.getUMat(ACCESS_READ);
    Matx23d actualx;

    {
        OutputArray oa(actualx);
        oa.assign(uexpected);
    }

    Mat actual = (Mat) actualx;

    EXPECT_LE(maxAbsDiff(expected, actual), 0.0);
}

TEST(Core_OutputArrayAssign, _Matxd_UMatf)
{
    Mat expected = (Mat_<float>(2,3) << 1, 2, 3, .1, .2, .3);
    UMat uexpected = expected.getUMat(ACCESS_READ);
    Matx23d actualx;

    {
        OutputArray oa(actualx);
        oa.assign(uexpected);
    }

    Mat actual = (Mat) actualx;

    EXPECT_LE(maxAbsDiff(expected, actual), FLT_EPSILON);
}

TEST(Core_OutputArrayAssign, _Matxf_UMatd)
{
    Mat expected = (Mat_<double>(2,3) << 1, 2, 3, .1, .2, .3);
    UMat uexpected = expected.getUMat(ACCESS_READ);
    Matx23f actualx;

    {
        OutputArray oa(actualx);
        oa.assign(uexpected);
    }

    Mat actual = (Mat) actualx;

    EXPECT_LE(maxAbsDiff(expected, actual), FLT_EPSILON);
}



int fixedType_handler(OutputArray dst)
{
    int type = CV_32FC2; // return points only {x, y}
    if (dst.fixedType())
    {
        type = dst.type();
        CV_Assert(type == CV_32FC2 || type == CV_32FC3); // allow points + confidence level: {x, y, confidence}
    }
    const int N = 100;
    dst.create(Size(1, N), type);
    Mat m = dst.getMat();
    if (m.type() == CV_32FC2)
    {
        for (int i = 0; i < N; i++)
            m.at<Vec2f>(i) = Vec2f((float)i, (float)(i*2));
    }
    else if (m.type() == CV_32FC3)
    {
        for (int i = 0; i < N; i++)
            m.at<Vec3f>(i) = Vec3f((float)i, (float)(i*2), 1.0f / (i + 1));
    }
    else
    {
        CV_Assert(0 && "Internal error");
    }
    return CV_MAT_CN(type);
}

TEST(Core_OutputArray, FixedType)
{
    Mat_<Vec2f> pointsOnly;
    int num_pointsOnly = fixedType_handler(pointsOnly);
    EXPECT_EQ(2, num_pointsOnly);

    Mat_<Vec3f> pointsWithConfidence;
    int num_pointsWithConfidence = fixedType_handler(pointsWithConfidence);
    EXPECT_EQ(3, num_pointsWithConfidence);

    Mat defaultResult;
    int num_defaultResult = fixedType_handler(defaultResult);
    EXPECT_EQ(2, num_defaultResult);
}

TEST(Core_OutputArrayCreate, _13772)
{
    cv::Mat1d mat;
    cv::OutputArray o(mat);
    ASSERT_NO_THROW(o.create(3, 5, CV_64F, -1, true));
}



TEST(Core_String, find_last_of__with__empty_string)
{
    cv::String s;
    size_t p = s.find_last_of('q', 0);
    // npos is not exported: EXPECT_EQ(cv::String::npos, p);
    EXPECT_EQ(std::string::npos, p);
}

TEST(Core_String, end_method_regression)
{
    cv::String old_string = "012345";
    cv::String new_string(old_string.begin(), old_string.end());
    EXPECT_EQ(6u, new_string.size());
}

TEST(Core_Copy, repeat_regression_8972)
{
    Mat src = (Mat_<int>(1, 4) << 1, 2, 3, 4);

    ASSERT_ANY_THROW({
                         repeat(src, 5, 1, src);
                     });
}


class ThrowErrorParallelLoopBody : public cv::ParallelLoopBody
{
public:
    ThrowErrorParallelLoopBody(cv::Mat& dst, int i) : dst_(dst), i_(i) {}
    ~ThrowErrorParallelLoopBody() {}
    void operator()(const cv::Range& r) const
    {
        for (int i = r.start; i < r.end; i++)
        {
            CV_Assert(i != i_);
            dst_.row(i).setTo(1);
        }
    }
protected:
    Mat dst_;
    int i_;
};

TEST(Core_Parallel, propagate_exceptions)
{
    Mat dst1(1000, 100, CV_8SC1, Scalar::all(0));
    ASSERT_NO_THROW({
        parallel_for_(cv::Range(0, dst1.rows), ThrowErrorParallelLoopBody(dst1, -1));
    });

    Mat dst2(1000, 100, CV_8SC1, Scalar::all(0));
    ASSERT_THROW({
        parallel_for_(cv::Range(0, dst2.rows), ThrowErrorParallelLoopBody(dst2, dst2.rows / 2));
    }, cv::Exception);
}

TEST(Core_Version, consistency)
{
    // this test verifies that OpenCV version loaded in runtime
    //   is the same this test has been built with
    EXPECT_EQ(CV_VERSION_MAJOR, cv::getVersionMajor());
    EXPECT_EQ(CV_VERSION_MINOR, cv::getVersionMinor());
    EXPECT_EQ(CV_VERSION_REVISION, cv::getVersionRevision());
    EXPECT_EQ(String(CV_VERSION), cv::getVersionString());
}



//
// Test core/check.hpp macros
//

void test_check_eq_1(int value_1, int value_2)
{
    CV_CheckEQ(value_1, value_2, "Validation check failed");
}
TEST(Core_Check, testEQ_int_fail)
{
    try
    {
        test_check_eq_1(123, 5678);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Validation check failed (expected: 'value_1 == value_2'), where\n"
">     'value_1' is 123\n"
"> must be equal to\n"
">     'value_2' is 5678\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}
TEST(Core_Check, testEQ_int_pass)
{
    EXPECT_NO_THROW(
    {
        test_check_eq_1(1234, 1234);
    });
}


void test_check_eq_2(float value_1, float value_2)
{
    CV_CheckEQ(value_1, value_2, "Validation check failed (float)");
}
TEST(Core_Check, testEQ_float_fail)
{
    try
    {
        test_check_eq_2(1234.5f, 1234.55f);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Validation check failed (float) (expected: 'value_1 == value_2'), where\n"
">     'value_1' is 1234.5\n"  // TODO Locale handling (use LC_ALL=C on Linux)
"> must be equal to\n"
">     'value_2' is 1234.55\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}
TEST(Core_Check, testEQ_float_pass)
{
    EXPECT_NO_THROW(
    {
        test_check_eq_2(1234.6f, 1234.6f);
    });
}


void test_check_eq_3(double value_1, double value_2)
{
    CV_CheckEQ(value_1, value_2, "Validation check failed (double)");
}
TEST(Core_Check, testEQ_double_fail)
{
    try
    {
        test_check_eq_3(1234.5, 1234.56);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Validation check failed (double) (expected: 'value_1 == value_2'), where\n"
">     'value_1' is 1234.5\n"  // TODO Locale handling (use LC_ALL=C on Linux)
"> must be equal to\n"
">     'value_2' is 1234.56\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}
TEST(Core_Check, testEQ_double_pass)
{
    EXPECT_NO_THROW(
    {
        test_check_eq_3(1234.0f, 1234.0f);
    });
}


void test_check_ne_1(int value_1, int value_2)
{
    CV_CheckNE(value_1, value_2, "Validation NE check failed");
}
TEST(Core_Check, testNE_int_fail)
{
    try
    {
        test_check_ne_1(123, 123);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Validation NE check failed (expected: 'value_1 != value_2'), where\n"
">     'value_1' is 123\n"
"> must be not equal to\n"
">     'value_2' is 123\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}
TEST(Core_Check, testNE_int_pass)
{
    EXPECT_NO_THROW(
    {
        test_check_ne_1(123, 1234);
    });
}


void test_check_le_1(int value_1, int value_2)
{
    CV_CheckLE(value_1, value_2, "Validation LE check failed");
}
TEST(Core_Check, testLE_int_fail)
{
    try
    {
        test_check_le_1(1234, 123);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Validation LE check failed (expected: 'value_1 <= value_2'), where\n"
">     'value_1' is 1234\n"
"> must be less than or equal to\n"
">     'value_2' is 123\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}
TEST(Core_Check, testLE_int_pass)
{
    EXPECT_NO_THROW(
    {
        test_check_le_1(1234, 1234);
    });
    EXPECT_NO_THROW(
    {
        test_check_le_1(123, 1234);
    });
}

void test_check_lt_1(int value_1, int value_2)
{
    CV_CheckLT(value_1, value_2, "Validation LT check failed");
}
TEST(Core_Check, testLT_int_fail)
{
    try
    {
        test_check_lt_1(1234, 123);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Validation LT check failed (expected: 'value_1 < value_2'), where\n"
">     'value_1' is 1234\n"
"> must be less than\n"
">     'value_2' is 123\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}
TEST(Core_Check, testLT_int_fail_eq)
{
    try
    {
        test_check_lt_1(123, 123);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Validation LT check failed (expected: 'value_1 < value_2'), where\n"
">     'value_1' is 123\n"
"> must be less than\n"
">     'value_2' is 123\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}
TEST(Core_Check, testLT_int_pass)
{
    EXPECT_NO_THROW(
    {
        test_check_lt_1(123, 1234);
    });
}


void test_check_ge_1(int value_1, int value_2)
{
    CV_CheckGE(value_1, value_2, "Validation GE check failed");
}
TEST(Core_Check, testGE_int_fail)
{
    try
    {
        test_check_ge_1(123, 1234);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Validation GE check failed (expected: 'value_1 >= value_2'), where\n"
">     'value_1' is 123\n"
"> must be greater than or equal to\n"
">     'value_2' is 1234\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}
TEST(Core_Check, testGE_int_pass)
{
    EXPECT_NO_THROW(
    {
        test_check_ge_1(1234, 1234);
    });
    EXPECT_NO_THROW(
    {
        test_check_ge_1(1234, 123);
    });
}

void test_check_gt_1(int value_1, int value_2)
{
    CV_CheckGT(value_1, value_2, "Validation GT check failed");
}
TEST(Core_Check, testGT_int_fail)
{
    try
    {
        test_check_gt_1(123, 1234);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Validation GT check failed (expected: 'value_1 > value_2'), where\n"
">     'value_1' is 123\n"
"> must be greater than\n"
">     'value_2' is 1234\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}
TEST(Core_Check, testGT_int_fail_eq)
{
    try
    {
        test_check_gt_1(123, 123);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Validation GT check failed (expected: 'value_1 > value_2'), where\n"
">     'value_1' is 123\n"
"> must be greater than\n"
">     'value_2' is 123\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}
TEST(Core_Check, testGT_int_pass)
{
    EXPECT_NO_THROW(
    {
        test_check_gt_1(1234, 123);
    });
}


void test_check_MatType_1(int src_type)
{
    CV_CheckTypeEQ(src_type, CV_32FC1, "Unsupported source type");
}
TEST(Core_Check, testMatType_pass)
{
    EXPECT_NO_THROW(
    {
        test_check_MatType_1(CV_MAKE_TYPE(CV_32F, 1));
    });
}
TEST(Core_Check, testMatType_fail_1)
{
    try
    {
        test_check_MatType_1(CV_8UC1);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Unsupported source type (expected: 'src_type == CV_32FC1'), where\n"
">     'src_type' is 0 (CV_8UC1)\n"
"> must be equal to\n"
">     'CV_32FC1' is 5 (CV_32FC1)\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}

void test_check_MatType_2(int src_type)
{
    CV_CheckType(src_type, src_type == CV_32FC1 || src_type == CV_32FC3, "Unsupported src");
}
TEST(Core_Check, testMatType_fail_2)
{
    try
    {
        test_check_MatType_2(CV_8UC1);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Unsupported src:\n"
">     'src_type == CV_32FC1 || src_type == CV_32FC3'\n"
"> where\n"
">     'src_type' is 0 (CV_8UC1)\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}

void test_check_MatDepth_1(int src_depth)
{
    CV_CheckDepthEQ(src_depth, CV_32F, "Unsupported source depth");
}
TEST(Core_Check, testMatDepth_pass)
{
    EXPECT_NO_THROW(
    {
        test_check_MatDepth_1(CV_MAKE_TYPE(CV_32F, 1));
    });
}
TEST(Core_Check, testMatDepth_fail_1)
{
    try
    {
        test_check_MatDepth_1(CV_8U);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Unsupported source depth (expected: 'src_depth == CV_32F'), where\n"
">     'src_depth' is 0 (CV_8U)\n"
"> must be equal to\n"
">     'CV_32F' is 5 (CV_32F)\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}

void test_check_MatDepth_2(int src_depth)
{
    CV_CheckDepth(src_depth, src_depth == CV_32F || src_depth == CV_64F, "Unsupported src");
}
TEST(Core_Check, testMatDepth_fail_2)
{
    try
    {
        test_check_MatDepth_2(CV_8U);
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Unsupported src:\n"
">     'src_depth == CV_32F || src_depth == CV_64F'\n"
"> where\n"
">     'src_depth' is 0 (CV_8U)\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}


void test_check_Size_1(const Size& srcSz)
{
    CV_Check(srcSz, srcSz == Size(4, 3), "Unsupported src size");
}
TEST(Core_Check, testSize_1)
{
    try
    {
        test_check_Size_1(Size(2, 1));
        FAIL() << "Unreachable code called";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_STREQ(e.err.c_str(),
"> Unsupported src size:\n"
">     'srcSz == Size(4, 3)'\n"
"> where\n"
">     'srcSz' is [2 x 1]\n"
);
    }
    catch (const std::exception& e)
    {
        FAIL() << "Unexpected C++ exception: " << e.what();
    }
    catch (...)
    {
        FAIL() << "Unexpected unknown exception";
    }
}

TEST(Core_Allocation, alignedAllocation)
{
    // iterate from size=1 to approximate byte size of 8K 32bpp image buffer
    for (int i = 0; i < 200; i++) {
        const size_t size = static_cast<size_t>(std::pow(1.091, (double)i));
        void * const buf = cv::fastMalloc(size);
        ASSERT_NE((uintptr_t)0, (uintptr_t)buf)
            << "failed to allocate memory";
        ASSERT_EQ((uintptr_t)0, (uintptr_t)buf % CV_MALLOC_ALIGN)
            << "memory not aligned to " << CV_MALLOC_ALIGN;
        cv::fastFree(buf);
    }
}


#if !(defined(__GNUC__) && __GNUC__ < 5)  // GCC 4.8 emits: 'is_trivially_copyable' is not a member of 'std'
TEST(Core_Types, trivially_copyable)
{
    EXPECT_TRUE(std::is_trivially_copyable<cv::Complexd>::value);
    EXPECT_TRUE(std::is_trivially_copyable<cv::Point>::value);
    EXPECT_TRUE(std::is_trivially_copyable<cv::Point3f>::value);
    EXPECT_TRUE(std::is_trivially_copyable<cv::Size>::value);
    EXPECT_TRUE(std::is_trivially_copyable<cv::Range>::value);
    EXPECT_TRUE(std::is_trivially_copyable<cv::Rect>::value);
    EXPECT_TRUE(std::is_trivially_copyable<cv::RotatedRect>::value);
    //EXPECT_TRUE(std::is_trivially_copyable<cv::Scalar>::value);  // derived from Vec (Matx)
}

TEST(Core_Types, trivially_copyable_extra)
{
    EXPECT_TRUE(std::is_trivially_copyable<cv::KeyPoint>::value);
    EXPECT_TRUE(std::is_trivially_copyable<cv::DMatch>::value);
    EXPECT_TRUE(std::is_trivially_copyable<cv::TermCriteria>::value);
    EXPECT_TRUE(std::is_trivially_copyable<cv::Moments>::value);
}
#endif

template <typename T> class Rect_Test : public testing::Test {};

TYPED_TEST_CASE_P(Rect_Test);

// Reimplement C++11 std::numeric_limits<>::lowest.
template<typename T> T cv_numeric_limits_lowest();
template<> int cv_numeric_limits_lowest<int>() { return INT_MIN; }
template<> float cv_numeric_limits_lowest<float>() { return -FLT_MAX; }
template<> double cv_numeric_limits_lowest<double>() { return -DBL_MAX; }

TYPED_TEST_P(Rect_Test, Overflows) {
  typedef Rect_<TypeParam> R;
  TypeParam num_max = std::numeric_limits<TypeParam>::max();
  TypeParam num_lowest = cv_numeric_limits_lowest<TypeParam>();
  EXPECT_EQ(R(0, 0, 10, 10), R(0, 0, 10, 10) & R(0, 0, 10, 10));
  EXPECT_EQ(R(5, 6, 4, 3), R(0, 0, 10, 10) & R(5, 6, 4, 3));
  EXPECT_EQ(R(5, 6, 3, 2), R(0, 0, 8, 8) & R(5, 6, 4, 3));
  // Test with overflowing dimenions.
  EXPECT_EQ(R(5, 0, 5, 10), R(0, 0, 10, 10) & R(5, 0, num_max, num_max));
  // Test with overflowing dimensions for floats/doubles.
  EXPECT_EQ(R(num_max, 0, num_max / 4, 10), R(num_max, 0, num_max / 2, 10) & R(num_max, 0, num_max / 4, 10));
  // Test with overflowing coordinates.
  EXPECT_EQ(R(), R(20, 0, 10, 10) & R(num_lowest, 0, 10, 10));
  EXPECT_EQ(R(), R(20, 0, 10, 10) & R(0, num_lowest, 10, 10));
  EXPECT_EQ(R(), R(num_lowest, 0, 10, 10) & R(0, num_lowest, 10, 10));
}
REGISTER_TYPED_TEST_CASE_P(Rect_Test, Overflows);

typedef ::testing::Types<int, float, double> RectTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Negative_Test, Rect_Test, RectTypes);


}} // namespace
