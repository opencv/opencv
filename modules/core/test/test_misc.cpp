#include "test_precomp.hpp"

using namespace cv;
using namespace std;

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


TEST(Core_String, find_last_of__with__empty_string)
{
    cv::String s;
    size_t p = s.find_last_of("q", 0);
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
