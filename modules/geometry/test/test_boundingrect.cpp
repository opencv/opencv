// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/core/types.hpp"
#include "test_precomp.hpp"

using namespace cv;
using namespace std;

namespace opencv_test { namespace {


template <typename T>
cv::Rect calcBoundingRect(Mat pts)
{
    CV_Assert(pts.type() == CV_32FC2 || pts.type() == CV_32SC2);
    CV_Assert(pts.size().width == 1 && pts.size().height > 0);
    const int N = pts.size().height;
    // NOTE: using ::lowest(), not ::min()
    T min_w = std::numeric_limits<T>::max(), max_w = std::numeric_limits<T>::lowest();
    T min_h = min_w, max_h = max_w;
    for (int i = 0; i < N; ++i)
    {
        const Point_<T> & pt = pts.at<Point_<T>>(i, 0);
        min_w = std::min<T>(pt.x, min_w);
        max_w = std::max<T>(pt.x, max_w);
        min_h = std::min<T>(pt.y, min_h);
        max_h = std::max<T>(pt.y, max_h);
    }
    return Rect(cvFloor(min_w), cvFloor(min_h), cvFloor(max_w) - cvFloor(min_w) + 1, cvFloor(max_h) - cvFloor(min_h) + 1);
}

typedef ::testing::TestWithParam<int> Imgproc_BoundingRect_Types;

TEST_P(Imgproc_BoundingRect_Types, accuracy)
{
    const int depth = GetParam();
    RNG& rng = ::cvtest::TS::ptr()->get_rng();
    for (int k = 0; k < 1000; ++k)
    {
        SCOPED_TRACE(cv::format("k=%d", k));
        const int sz = rng.uniform(1, 10000);
        Mat src(sz, 1, CV_MAKETYPE(depth, 2));
        rng.fill(src, RNG::UNIFORM, Scalar(-100000, -100000), Scalar(100000, 100000));
        Rect reference;
        if (depth == CV_32F)
            reference = calcBoundingRect<float>(src);
        else if (depth == CV_32S)
            reference = calcBoundingRect<int>(src);
        else
            CV_Error(Error::StsError, "Test error");
        Rect result = cv::boundingRect(src);
        EXPECT_EQ(reference, result);
    }
}

TEST_P(Imgproc_BoundingRect_Types, alignment)
{
    const int depth = GetParam();
    const int SZ = 100;
    int idata[SZ];
    float fdata[SZ];
    for (int i = 0; i < SZ; ++i)
    {
        idata[i] = i;
        fdata[i] = (float)i;
    }
    for (int i = 0; i < 10; ++i)
    {
        for (int len = 1; len < 40; ++len)
        {
            SCOPED_TRACE(cv::format("i=%d, len=%d", i, len));
            Mat sub(len, 1, CV_MAKETYPE(depth, 2), (depth == CV_32S) ? (void*)(idata + i) : (void*)(fdata + i));
            EXPECT_NO_THROW(boundingRect(sub));
        }
    }
}

INSTANTIATE_TEST_CASE_P(, Imgproc_BoundingRect_Types, ::testing::Values(CV_32S, CV_32F));


TEST(Imgproc_BoundingRect, bug_24217)
{
    for (int image_width = 3; image_width < 20; image_width++)
    {
        for (int image_height = 1; image_height < 15; image_height++)
        {
            cv::Rect rect(0, image_height - 1, 3, 1);

            cv::Mat image(cv::Size(image_width, image_height), CV_8UC1, cv::Scalar(0));
            image(rect) = 255;

            ASSERT_EQ(boundingRect(image), rect);
        }
    }
}

// See https://github.com/opencv/opencv/issues/29578
// cv::Mat_<bool>::depth() returns CV_Bool in 5.0, which regressed boundingRect
// (it used to be treated as CV_8U in 4.x). A boolean mask must be handled as a
// byte-wise mask, matching the CV_8UC1 result.
TEST(Imgproc_BoundingRect, bool_mask_29578)
{
    for (int image_width = 3; image_width < 20; image_width++)
    {
        for (int image_height = 1; image_height < 15; image_height++)
        {
            cv::Rect rect(0, image_height - 1, 3, 1);

            cv::Mat_<bool> mask(cv::Size(image_width, image_height), false);
            mask(rect).setTo(true);

            cv::Mat_<uchar> ref(cv::Size(image_width, image_height), (uchar)0);
            ref(rect).setTo(255);

            EXPECT_EQ(mask.depth(), CV_Bool);
            cv::Rect result;
            ASSERT_NO_THROW(result = boundingRect(mask));
            EXPECT_EQ(result, rect);
            EXPECT_EQ(result, boundingRect(ref));
        }
    }
}

// See https://github.com/opencv/opencv/issues/29837
// A CV_32F coordinate outside of the int range used to reach cvFloor(), which is undefined
// there: points spanning 1e10 collapsed to a 1x1 rect at INT_MIN, and +inf/-inf gave
// INT_MIN/INT_MAX, an inverted rectangle. They are saturated to the representable bounds now.
TEST(Imgproc_BoundingRect, out_of_int_range_29837)
{
    const int imin = std::numeric_limits<int>::min();
    const int imax = std::numeric_limits<int>::max();
    const float inf = std::numeric_limits<float>::infinity();

    // in-range point sets are unaffected
    std::vector<Point2f> in_range { Point2f(1e4f, 1e4f), Point2f(2e4f, 2e4f) };
    EXPECT_EQ(boundingRect(in_range), Rect(10000, 10000, 10001, 10001));

    // out of range: saturated, orientation preserved
    std::vector<Point2f> above { Point2f(1e10f, 1e10f), Point2f(2e10f, 2e10f) };
    EXPECT_EQ(boundingRect(above), Rect(imax, imax, 1, 1));

    std::vector<Point2f> below { Point2f(-1e10f, -1e10f), Point2f(-2e10f, -2e10f) };
    EXPECT_EQ(boundingRect(below), Rect(imin, imin, 1, 1));

    // infinities saturate the same way, and the sides are clamped instead of overflowing
    std::vector<Point2f> plus_inf { Point2f(inf, 0.f), Point2f(1.f, 1.f), Point2f(2.f, 0.f) };
    EXPECT_EQ(boundingRect(plus_inf), Rect(1, 0, imax, 2));

    std::vector<Point2f> minus_inf { Point2f(-inf, 0.f), Point2f(1.f, 1.f), Point2f(2.f, 0.f) };
    EXPECT_EQ(boundingRect(minus_inf), Rect(imin, 0, imax, 2));

    // long enough to go through the SIMD path as well
    std::vector<Point2f> simd(64, Point2f(5.f, 5.f));
    simd[37] = Point2f(1e10f, -1e10f);
    EXPECT_EQ(boundingRect(simd), Rect(5, imin, imax - 4, imax));

    // CV_32S needs no conversion, but its sides can overflow int just the same
    std::vector<Point> ints { Point(imin, imin), Point(imax, imax) };
    EXPECT_EQ(boundingRect(ints), Rect(imin, imin, imax, imax));
}

}} // namespace
