// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#ifdef HAVE_METAL
#include "opencv2/core/metal.hpp"
#endif

namespace opencv_test { namespace {

#ifdef HAVE_METAL

TEST(Core_Metal_UMat, UploadDownload)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(33, 35, CV_8UC3);
    randu(src, 0, 255);

    UMat u;
    src.copyTo(u);

    Mat dst;
    u.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, src, NORM_INF), 0);
}

TEST(Core_Metal_UMat, MapWriteDownload)
{
    if (!cv::metal::haveMetal())
        return;

    UMat u(32, 32, CV_8UC1);
    {
        Mat mapped = u.getMat(ACCESS_WRITE);
        mapped.setTo(Scalar::all(17));
    }

    Mat dst;
    u.copyTo(dst);

    Mat expected(dst.size(), dst.type(), Scalar::all(17));
    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, DeviceCopy)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(31, 37, CV_32FC2);
    randu(src, -10.0f, 10.0f);

    UMat u1;
    src.copyTo(u1);

    UMat u2;
    u1.copyTo(u2);

    Mat dst;
    u2.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, src, NORM_INF), 0);
}

TEST(Core_Metal_UMat, DeviceMemoryUsageUploadDownload)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(29, 41, CV_8UC4);
    randu(src, 0, 255);

    UMat u(src.size(), src.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    src.copyTo(u);

    Mat dst;
    u.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, src, NORM_INF), 0);
}

TEST(Core_Metal_UMat, DeviceMemoryUsageMapReadWrite)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(32, 32, CV_8UC1);
    randu(src, 0, 254);

    UMat u(src.size(), src.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    src.copyTo(u);

    Mat expected = src + Scalar::all(1);
    {
        Mat mapped = u.getMat(ACCESS_RW);
        mapped += Scalar::all(1);
    }

    Mat dst;
    u.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, DeviceMemoryUsageDeviceCopy)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(31, 37, CV_32FC1);
    randu(src, -10.0f, 10.0f);

    UMat u1(src.size(), src.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    src.copyTo(u1);

    UMat u2(src.size(), src.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    u1.copyTo(u2);

    Mat dst;
    u2.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, src, NORM_INF), 0);
}

TEST(Core_Metal_UMat, RoiUploadDownload)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(48, 64, CV_8UC3, Scalar::all(0));
    Mat patch(17, 23, src.type());
    randu(patch, 0, 255);
    Rect roi(5, 7, patch.cols, patch.rows);
    patch.copyTo(src(roi));

    UMat u(src.size(), src.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    Mat zeros(src.size(), src.type(), Scalar::all(0));
    zeros.copyTo(u);
    patch.copyTo(u(roi));

    Mat dst;
    u.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, src, NORM_INF), 0);
}

TEST(Core_Metal_UMat, RoiDeviceCopy)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(48, 64, CV_8UC1);
    randu(src, 0, 255);

    UMat u1(src.size(), src.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    src.copyTo(u1);

    UMat u2(src.size(), src.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    Mat zeros(src.size(), src.type(), Scalar::all(0));
    zeros.copyTo(u2);

    Rect roi(9, 11, 29, 21);
    u1(roi).copyTo(u2(roi));

    Mat dst;
    u2.copyTo(dst);

    Mat expected(src.size(), src.type(), Scalar::all(0));
    src(roi).copyTo(expected(roi));
    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, MaskedCopyToSingleChannelMask)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(37, 41, CV_8UC3);
    Mat mask(src.size(), CV_8UC1);
    randu(src, 0, 255);
    randu(mask, 0, 2);
    mask *= 255;

    Mat expected;
    src.copyTo(expected, mask);

    UMat usrc, umask, udst;
    src.copyTo(usrc);
    mask.copyTo(umask);
    usrc.copyTo(udst, umask);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, MaskedCopyToChannelMask)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(35, 39, CV_8UC3);
    Mat mask(src.size(), CV_8UC3);
    randu(src, 0, 255);
    randu(mask, 0, 2);
    mask *= 255;

    Mat expected;
    src.copyTo(expected, mask);

    UMat usrc, umask, udst;
    src.copyTo(usrc);
    mask.copyTo(umask);
    usrc.copyTo(udst, umask);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, MaskedCopyToRoiPreservesUnmasked)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(48, 64, CV_8UC3);
    Mat base(src.size(), src.type());
    Mat mask(src.size(), CV_8UC1);
    randu(src, 0, 255);
    randu(base, 0, 255);
    randu(mask, 0, 2);
    mask *= 255;

    Rect roi(7, 9, 31, 23);
    Mat expected = base.clone();
    src(roi).copyTo(expected(roi), mask(roi));

    UMat usrc, umask, udst;
    src.copyTo(usrc);
    mask.copyTo(umask);
    base.copyTo(udst);
    usrc(roi).copyTo(udst(roi), umask(roi));

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, DeviceMemoryUsageMaskedCopyTo)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(29, 43, CV_8UC4);
    Mat base(src.size(), src.type());
    Mat mask(src.size(), CV_8UC1);
    randu(src, 0, 255);
    randu(base, 0, 255);
    randu(mask, 0, 2);
    mask *= 255;

    Mat expected = base.clone();
    src.copyTo(expected, mask);

    UMat usrc(src.size(), src.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat umask(mask.size(), mask.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat udst(base.size(), base.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    src.copyTo(usrc);
    mask.copyTo(umask);
    base.copyTo(udst);
    usrc.copyTo(udst, umask);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

#endif // HAVE_METAL

}} // namespace
