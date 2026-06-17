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

TEST(Core_Metal_UMat, Add8USaturates)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(37, 41, CV_8UC1);
    Mat src2(src1.size(), src1.type());
    randu(src1, 128, 255);
    randu(src2, 128, 255);

    Mat expected;
    cv::add(src1, src2, expected);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::add(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, Add32F)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(35, 39, CV_32FC1);
    Mat src2(src1.size(), src1.type());
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);

    Mat expected;
    cv::add(src1, src2, expected);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::add(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 1e-6);
}

TEST(Core_Metal_UMat, AddChannels)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(29, 43, CV_8UC3);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 255);
    randu(src2, 0, 255);

    Mat expected;
    cv::add(src1, src2, expected);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::add(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);

    src1.create(31, 37, CV_32FC4);
    src2.create(src1.size(), src1.type());
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);
    cv::add(src1, src2, expected);

    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::add(usrc1, usrc2, udst);
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 1e-6);
}

TEST(Core_Metal_UMat, DeviceMemoryUsageAdd)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(31, 37, CV_32FC4);
    Mat src2(src1.size(), src1.type());
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);

    Mat expected;
    cv::add(src1, src2, expected);

    UMat usrc1(src1.size(), src1.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat usrc2(src2.size(), src2.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat udst(src1.size(), src1.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::add(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 1e-6);
}

TEST(Core_Metal_UMat, AddRoi)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(48, 64, CV_8UC4);
    Mat src2(src1.size(), src1.type());
    Mat base(src1.size(), src1.type());
    randu(src1, 0, 255);
    randu(src2, 0, 255);
    randu(base, 0, 255);

    Rect roi(9, 7, 31, 23);
    Mat expected = base.clone();
    cv::add(src1(roi), src2(roi), expected(roi));

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    base.copyTo(udst);
    cv::add(usrc1(roi), usrc2(roi), udst(roi));

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, AddUnsupportedTypeFallback)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(23, 29, CV_16UC1);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 1000);
    randu(src2, 0, 1000);

    Mat expected;
    cv::add(src1, src2, expected);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::add(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, Subtract8USaturates)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(37, 41, CV_8UC1);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 255);
    randu(src2, 0, 255);

    Mat expected;
    cv::subtract(src1, src2, expected);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::subtract(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, Subtract32F)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(35, 39, CV_32FC1);
    Mat src2(src1.size(), src1.type());
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);

    Mat expected;
    cv::subtract(src1, src2, expected);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::subtract(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 1e-6);
}

TEST(Core_Metal_UMat, SubtractChannels)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(29, 43, CV_8UC3);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 255);
    randu(src2, 0, 255);

    Mat expected;
    cv::subtract(src1, src2, expected);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::subtract(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);

    src1.create(31, 37, CV_32FC4);
    src2.create(src1.size(), src1.type());
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);
    cv::subtract(src1, src2, expected);

    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::subtract(usrc1, usrc2, udst);
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 1e-6);
}

TEST(Core_Metal_UMat, DeviceMemoryUsageSubtract)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(31, 37, CV_32FC4);
    Mat src2(src1.size(), src1.type());
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);

    Mat expected;
    cv::subtract(src1, src2, expected);

    UMat usrc1(src1.size(), src1.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat usrc2(src2.size(), src2.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat udst(src1.size(), src1.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::subtract(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 1e-6);
}

TEST(Core_Metal_UMat, SubtractRoi)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(48, 64, CV_8UC4);
    Mat src2(src1.size(), src1.type());
    Mat base(src1.size(), src1.type());
    randu(src1, 0, 255);
    randu(src2, 0, 255);
    randu(base, 0, 255);

    Rect roi(9, 7, 31, 23);
    Mat expected = base.clone();
    cv::subtract(src1(roi), src2(roi), expected(roi));

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    base.copyTo(udst);
    cv::subtract(usrc1(roi), usrc2(roi), udst(roi));

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, SubtractUnsupportedTypeFallback)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(23, 29, CV_16UC1);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 1000);
    randu(src2, 0, 1000);

    Mat expected;
    cv::subtract(src1, src2, expected);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::subtract(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, Multiply8USaturates)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(37, 41, CV_8UC1);
    Mat src2(src1.size(), src1.type());
    randu(src1, 16, 255);
    randu(src2, 16, 255);

    Mat expected;
    cv::multiply(src1, src2, expected);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::multiply(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, Multiply32FScale)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(35, 39, CV_32FC1);
    Mat src2(src1.size(), src1.type());
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);

    Mat expected;
    cv::multiply(src1, src2, expected, 0.125);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::multiply(usrc1, usrc2, udst, 0.125);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 1e-5);
}

TEST(Core_Metal_UMat, MultiplyChannels)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(29, 43, CV_8UC3);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 16);
    randu(src2, 0, 16);

    Mat expected;
    cv::multiply(src1, src2, expected);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::multiply(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);

    src1.create(31, 37, CV_32FC4);
    src2.create(src1.size(), src1.type());
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);
    cv::multiply(src1, src2, expected, 2.0);

    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::multiply(usrc1, usrc2, udst, 2.0);
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 1e-5);
}

TEST(Core_Metal_UMat, DeviceMemoryUsageMultiply)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(31, 37, CV_32FC4);
    Mat src2(src1.size(), src1.type());
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);

    Mat expected;
    cv::multiply(src1, src2, expected, 0.25);

    UMat usrc1(src1.size(), src1.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat usrc2(src2.size(), src2.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat udst(src1.size(), src1.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::multiply(usrc1, usrc2, udst, 0.25);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 1e-5);
}

TEST(Core_Metal_UMat, MultiplyRoi)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(48, 64, CV_8UC4);
    Mat src2(src1.size(), src1.type());
    Mat base(src1.size(), src1.type());
    randu(src1, 0, 16);
    randu(src2, 0, 16);
    randu(base, 0, 255);

    Rect roi(9, 7, 31, 23);
    Mat expected = base.clone();
    cv::multiply(src1(roi), src2(roi), expected(roi));

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    base.copyTo(udst);
    cv::multiply(usrc1(roi), usrc2(roi), udst(roi));

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, MultiplyUnsupportedTypeFallback)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(23, 29, CV_16UC1);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 1000);
    randu(src2, 0, 1000);

    Mat expected;
    cv::multiply(src1, src2, expected);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::multiply(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, BitwiseAndOrXor)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(37, 41, CV_8UC3);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 255);
    randu(src2, 0, 255);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);

    Mat expected, dst;
    cv::bitwise_and(src1, src2, expected);
    cv::bitwise_and(usrc1, usrc2, udst);
    udst.copyTo(dst);
    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);

    cv::bitwise_or(src1, src2, expected);
    cv::bitwise_or(usrc1, usrc2, udst);
    udst.copyTo(dst);
    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);

    cv::bitwise_xor(src1, src2, expected);
    cv::bitwise_xor(usrc1, usrc2, udst);
    udst.copyTo(dst);
    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, BitwiseNot)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(35, 39, CV_8UC4);
    randu(src, 0, 255);

    Mat expected;
    cv::bitwise_not(src, expected);

    UMat usrc, udst;
    src.copyTo(usrc);
    cv::bitwise_not(usrc, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, BitwiseFloatBytes)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(31, 37, CV_32FC2);
    Mat src2(src1.size(), src1.type());
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);

    Mat expected;
    cv::bitwise_xor(src1, src2, expected);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::bitwise_xor(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst.reshape(1), expected.reshape(1), NORM_INF), 0);
}

TEST(Core_Metal_UMat, DeviceMemoryUsageBitwise)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(31, 37, CV_8UC4);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 255);
    randu(src2, 0, 255);

    Mat expected;
    cv::bitwise_and(src1, src2, expected);

    UMat usrc1(src1.size(), src1.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat usrc2(src2.size(), src2.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat udst(src1.size(), src1.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::bitwise_and(usrc1, usrc2, udst);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, BitwiseRoi)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(48, 64, CV_8UC4);
    Mat src2(src1.size(), src1.type());
    Mat base(src1.size(), src1.type());
    randu(src1, 0, 255);
    randu(src2, 0, 255);
    randu(base, 0, 255);

    Rect roi(9, 7, 31, 23);
    Mat expected = base.clone();
    cv::bitwise_or(src1(roi), src2(roi), expected(roi));

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    base.copyTo(udst);
    cv::bitwise_or(usrc1(roi), usrc2(roi), udst(roi));

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, BitwiseMaskFallback)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(29, 43, CV_8UC3);
    Mat src2(src1.size(), src1.type());
    Mat mask(src1.size(), CV_8UC1);
    randu(src1, 0, 255);
    randu(src2, 0, 255);
    randu(mask, 0, 2);
    mask *= 255;

    Mat expected;
    cv::bitwise_xor(src1, src2, expected, mask);

    UMat usrc1, usrc2, umask, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    mask.copyTo(umask);
    cv::bitwise_xor(usrc1, usrc2, udst, umask);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, Compare8UAllOps)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(37, 41, CV_8UC1);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 255);
    randu(src2, 0, 255);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);

    const int ops[] = { CMP_EQ, CMP_GT, CMP_GE, CMP_LT, CMP_LE, CMP_NE };
    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); ++i)
    {
        Mat expected;
        cv::compare(src1, src2, expected, ops[i]);

        cv::compare(usrc1, usrc2, udst, ops[i]);
        Mat dst;
        udst.copyTo(dst);

        EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
    }
}

TEST(Core_Metal_UMat, Compare32FChannels)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(31, 37, CV_32FC4);
    Mat src2(src1.size(), src1.type());
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);

    Mat expected;
    cv::compare(src1, src2, expected, CMP_LE);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::compare(usrc1, usrc2, udst, CMP_LE);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, DeviceMemoryUsageCompare)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(31, 37, CV_8UC4);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 255);
    randu(src2, 0, 255);

    Mat expected;
    cv::compare(src1, src2, expected, CMP_GT);

    UMat usrc1(src1.size(), src1.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat usrc2(src2.size(), src2.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat udst(src1.size(), CV_8UC4, USAGE_ALLOCATE_DEVICE_MEMORY);
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    cv::compare(usrc1, usrc2, udst, CMP_GT);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, CompareRoi)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src1(48, 64, CV_32FC1);
    Mat src2(src1.size(), src1.type());
    Mat base(src1.size(), CV_8UC1);
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);
    randu(base, 0, 255);

    Rect roi(9, 7, 31, 23);
    Mat expected = base.clone();
    cv::compare(src1(roi), src2(roi), expected(roi), CMP_NE);

    UMat usrc1, usrc2, udst;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);
    base.copyTo(udst);
    cv::compare(usrc1(roi), usrc2(roi), udst(roi), CMP_NE);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, CompareScalarFallback)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(29, 43, CV_8UC3);
    randu(src, 0, 255);

    Mat expected;
    cv::compare(src, Scalar(127, 128, 129), expected, CMP_GE);

    UMat usrc, udst;
    src.copyTo(usrc);
    cv::compare(usrc, Scalar(127, 128, 129), udst, CMP_GE);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, ConvertTo8UTo32F)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(37, 41, CV_8UC3);
    randu(src, 0, 255);

    Mat expected;
    src.convertTo(expected, CV_32F, 1.25, -7.0);

    UMat usrc, udst;
    src.copyTo(usrc);
    usrc.convertTo(udst, CV_32F, 1.25, -7.0);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, ConvertTo32FTo8U)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(35, 39, CV_32FC1);
    randu(src, -100.0f, 300.0f);

    Mat expected;
    src.convertTo(expected, CV_8U, 0.75, 12.5);

    UMat usrc, udst;
    src.copyTo(usrc);
    usrc.convertTo(udst, CV_8U, 0.75, 12.5);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, ConvertToSameDepthScale)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(31, 37, CV_32FC4);
    randu(src, -10.0f, 10.0f);

    Mat expected;
    src.convertTo(expected, CV_32F, -2.0, 3.5);

    UMat usrc, udst;
    src.copyTo(usrc);
    usrc.convertTo(udst, CV_32F, -2.0, 3.5);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 1e-6);
}

TEST(Core_Metal_UMat, DeviceMemoryUsageConvertTo)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(31, 37, CV_8UC4);
    randu(src, 0, 255);

    Mat expected;
    src.convertTo(expected, CV_32F, 0.5, 9.0);

    UMat usrc(src.size(), src.type(), USAGE_ALLOCATE_DEVICE_MEMORY);
    UMat udst(src.size(), CV_32FC4, USAGE_ALLOCATE_DEVICE_MEMORY);
    src.copyTo(usrc);
    usrc.convertTo(udst, CV_32F, 0.5, 9.0);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, ConvertToRoi)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(48, 64, CV_8UC1);
    Mat base(src.size(), CV_32FC1);
    randu(src, 0, 255);
    randu(base, -10.0f, 10.0f);

    Rect roi(9, 7, 31, 23);
    Mat expected = base.clone();
    src(roi).convertTo(expected(roi), CV_32F, 2.0, -11.0);

    UMat usrc, udst;
    src.copyTo(usrc);
    base.copyTo(udst);
    usrc(roi).convertTo(udst(roi), CV_32F, 2.0, -11.0);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, ConvertToUnsupportedFallback)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src(23, 29, CV_16UC1);
    randu(src, 0, 1000);

    Mat expected;
    src.convertTo(expected, CV_8U, 0.25, 3.0);

    UMat usrc, udst;
    src.copyTo(usrc);
    usrc.convertTo(udst, CV_8U, 0.25, 3.0);

    Mat dst;
    udst.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, ConvertToInPlace)
{
    if (!cv::metal::haveMetal())
        return;

    Mat src8u(23, 29, CV_8UC3);
    randu(src8u, 0, 255);

    Mat expected32f;
    src8u.convertTo(expected32f, CV_32F, 0.5, -3.0);

    UMat actual32f;
    src8u.copyTo(actual32f);
    actual32f.convertTo(actual32f, CV_32F, 0.5, -3.0);

    Mat dst32f;
    actual32f.copyTo(dst32f);

    EXPECT_LE(cvtest::norm(dst32f, expected32f, NORM_INF), 0);

    Mat src32f(3, 4, CV_32FC1);
    randu(src32f, -1.0f, 1.0f);

    Mat expected64f;
    src32f.convertTo(expected64f, CV_64F);

    UMat actual64f;
    src32f.copyTo(actual64f);
    actual64f.convertTo(actual64f, CV_64F);

    Mat dst64f;
    actual64f.copyTo(dst64f);

    EXPECT_LE(cvtest::norm(dst64f, expected64f, NORM_INF), 0);
}

TEST(Core_Metal_UMat, SetTo8U)
{
    if (!cv::metal::haveMetal())
        return;

    UMat u(31, 37, CV_8UC3);
    u.setTo(Scalar(7, 17, 29));

    Mat dst;
    u.copyTo(dst);

    Mat expected(dst.size(), dst.type(), Scalar(7, 17, 29));
    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, SetTo32F)
{
    if (!cv::metal::haveMetal())
        return;

    UMat u(29, 41, CV_32FC4);
    u.setTo(Scalar(1.25, -2.5, 3.75, -4.5));

    Mat dst;
    u.copyTo(dst);

    Mat expected(dst.size(), dst.type(), Scalar(1.25, -2.5, 3.75, -4.5));
    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, SetToSingleChannelMask)
{
    if (!cv::metal::haveMetal())
        return;

    Mat base(35, 39, CV_8UC3);
    Mat mask(base.size(), CV_8UC1);
    randu(base, 0, 255);
    randu(mask, 0, 2);
    mask *= 255;

    Mat expected = base.clone();
    expected.setTo(Scalar(3, 5, 7), mask);

    UMat u, umask;
    base.copyTo(u);
    mask.copyTo(umask);
    u.setTo(Scalar(3, 5, 7), umask);

    Mat dst;
    u.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, SetToChannelMask)
{
    if (!cv::metal::haveMetal())
        return;

    Mat base(33, 43, CV_8UC3);
    Mat mask(base.size(), CV_8UC3);
    randu(base, 0, 255);
    randu(mask, 0, 2);
    mask *= 255;

    Mat expected = base.clone();
    expected.setTo(Scalar(11, 13, 17), mask);

    UMat u, umask;
    base.copyTo(u);
    mask.copyTo(umask);
    u.setTo(Scalar(11, 13, 17), umask);

    Mat dst;
    u.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, SetToRoi)
{
    if (!cv::metal::haveMetal())
        return;

    Mat base(48, 64, CV_32FC1);
    randu(base, -10.0f, 10.0f);

    Rect roi(9, 11, 29, 23);
    Mat expected = base.clone();
    expected(roi).setTo(Scalar(6.5));

    UMat u;
    base.copyTo(u);
    u(roi).setTo(Scalar(6.5));

    Mat dst;
    u.copyTo(dst);

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, DeviceMemoryUsageSetTo)
{
    if (!cv::metal::haveMetal())
        return;

    UMat u(31, 37, CV_8UC4, USAGE_ALLOCATE_DEVICE_MEMORY);
    u.setTo(Scalar(19, 23, 29, 31));

    Mat dst;
    u.copyTo(dst);

    Mat expected(dst.size(), dst.type(), Scalar(19, 23, 29, 31));
    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, SetToUnsupportedTypeFallback)
{
    if (!cv::metal::haveMetal())
        return;

    UMat u(23, 29, CV_16UC1);
    u.setTo(Scalar(1024));

    Mat dst;
    u.copyTo(dst);

    Mat expected(dst.size(), dst.type(), Scalar(1024));
    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 0);
}

TEST(Core_Metal_UMat, BufferPoolGrowing)
{
    if (!cv::metal::haveMetal())
        return;

    BufferPoolController* controller = UMat::getStdAllocator()->getBufferPoolController("METAL");
    ASSERT_TRUE(controller != NULL);

    size_t oldMaxReservedSize = controller->getMaxReservedSize();
    controller->freeAllReservedBuffers();
    controller->setMaxReservedSize(4 * 1024 * 1024);

    {
        for (int i = 0; i < 16; ++i)
        {
            UMat u(256, 256, CV_8UC4);
            u.setTo(Scalar::all(i));
        }
    }

    EXPECT_GT(controller->getReservedSize(), (size_t)0);
    EXPECT_LE(controller->getReservedSize(), controller->getMaxReservedSize());

    controller->freeAllReservedBuffers();
    EXPECT_EQ(controller->getReservedSize(), (size_t)0);
    controller->setMaxReservedSize(oldMaxReservedSize);
}

#endif // HAVE_METAL

}} // namespace
