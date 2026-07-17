// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#ifdef HAVE_METAL
#include "opencv2/core/metal.hpp"
#endif

namespace opencv_test { namespace {

#ifdef HAVE_METAL

TEST(Imgproc_Metal_Threshold, Threshold8UAllTypes)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(37, 41, CV_8UC3);
    randu(src, 0, 255);

    const int types[] = { THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV };
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); ++i)
    {
        Mat expected;
        double expectedThresh = cv::threshold(src, expected, 127, 203, types[i]);

        UMat usrc, udst;
        src.copyTo(usrc);
        double actualThresh = cv::threshold(usrc, udst, 127, 203, types[i]);

        Mat dst;
        udst.copyTo(dst);

        EXPECT_EQ(actualThresh, expectedThresh);
        EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
    }
}

TEST(Imgproc_Metal_Threshold, Threshold32FChannels)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(31, 37, CV_32FC4);
    randu(src, -10.0f, 10.0f);

    Mat expected;
    cv::threshold(src, expected, 1.25, 5.5, THRESH_BINARY);

    UMat usrc, udst;
    src.copyTo(usrc);
    cv::threshold(usrc, udst, 1.25, 5.5, THRESH_BINARY);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Imgproc_Metal_Threshold, DeviceMemoryUsageThreshold)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(31, 37, CV_8UC4);
    randu(src, 0, 255);

    Mat expected;
    cv::threshold(src, expected, 93, 211, THRESH_TOZERO_INV);

    UMat usrc(src.size(), src.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat udst(src.size(), src.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    src.copyTo(usrc);
    cv::threshold(usrc, udst, 93, 211, THRESH_TOZERO_INV);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Imgproc_Metal_Threshold, ThresholdRoi)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(48, 64, CV_32FC1);
    Mat base(src.size(), src.type());
    randu(src, -10.0f, 10.0f);
    randu(base, -5.0f, 5.0f);

    Rect roi(9, 7, 31, 23);
    Mat expected = base.clone();
    cv::threshold(src(roi), expected(roi), 2.5, 7.0, THRESH_TRUNC);

    UMat usrc, udst;
    src.copyTo(usrc);
    base.copyTo(udst);
    cv::threshold(usrc(roi), udst(roi), 2.5, 7.0, THRESH_TRUNC);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Imgproc_Metal_Threshold, ThresholdUnsupportedFallback)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(23, 29, CV_16UC1);
    randu(src, 0, 1000);

    Mat expected;
    cv::threshold(src, expected, 127, 255, THRESH_BINARY);

    UMat usrc, udst;
    src.copyTo(usrc);
    cv::threshold(usrc, udst, 127, 255, THRESH_BINARY);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

#endif // HAVE_METAL

}} // namespace
