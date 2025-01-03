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

}} // namespace
