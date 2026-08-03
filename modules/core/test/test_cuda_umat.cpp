// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

#ifdef HAVE_CUDA

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/ocl.hpp>

namespace opencv_test { namespace {

static void skipIfNoCudaDevice()
{
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
        throw SkipTestException("No CUDA device available");
}

static UMat makeCudaUMat()
{
    UMat u;
    u.allocator = cv::cuda::getCudaAllocator();
    return u;
}

TEST(Core_CudaUMat, roundtrip_contiguous)
{
    skipIfNoCudaDevice();
    ASSERT_TRUE(cv::cuda::getCudaAllocator() != NULL);

    Mat src(64, 48, CV_32FC1);
    randu(src, 0, 1);

    UMat u = makeCudaUMat();
    src.copyTo(u);
    ASSERT_TRUE(u.u != NULL && u.u->handle != NULL);

    Mat dst;
    u.copyTo(dst);

    EXPECT_EQ(src.size(), dst.size());
    EXPECT_EQ(src.type(), dst.type());
    EXPECT_EQ(0.0, cvtest::norm(src, dst, NORM_INF));
}

TEST(Core_CudaUMat, roundtrip_multichannel)
{
    skipIfNoCudaDevice();

    Mat src(37, 53, CV_8UC3);
    randu(src, 0, 255);

    UMat u = makeCudaUMat();
    src.copyTo(u);

    Mat dst;
    u.copyTo(dst);
    EXPECT_EQ(0.0, cvtest::norm(src, dst, NORM_INF));
}

TEST(Core_CudaUMat, roundtrip_roi_source)
{
    skipIfNoCudaDevice();

    Mat full(100, 100, CV_32FC1);
    randu(full, 0, 1);
    Mat roi = full(Rect(10, 12, 50, 40));
    ASSERT_FALSE(roi.isContinuous());

    UMat u = makeCudaUMat();
    roi.copyTo(u);

    Mat dst;
    u.copyTo(dst);
    EXPECT_EQ(roi.size(), dst.size());
    EXPECT_EQ(0.0, cvtest::norm(roi, dst, NORM_INF));
}

TEST(Core_CudaUMat, download_into_roi)
{
    skipIfNoCudaDevice();

    Mat src(40, 50, CV_32FC1);
    randu(src, 0, 1);

    UMat u = makeCudaUMat();
    src.copyTo(u);

    Mat canvas(80, 90, CV_32FC1, Scalar(0));
    Mat dstRoi = canvas(Rect(5, 7, 50, 40));
    ASSERT_FALSE(dstRoi.isContinuous());
    u.copyTo(dstRoi);

    EXPECT_EQ(0.0, cvtest::norm(src, dstRoi, NORM_INF));
}

TEST(Core_CudaUMat, device_to_device_copy)
{
    skipIfNoCudaDevice();

    Mat src(70, 90, CV_32FC1);
    randu(src, 0, 1);

    UMat a = makeCudaUMat();
    src.copyTo(a);

    UMat b = makeCudaUMat();
    a.copyTo(b);

    Mat dst;
    b.copyTo(dst);
    EXPECT_EQ(0.0, cvtest::norm(src, dst, NORM_INF));
}

// Proves the allocator's handle is a real, kernel-usable CUDA buffer via GpuMat interop.
TEST(Core_CudaUMat, kernel_interop_via_gpumat)
{
    skipIfNoCudaDevice();

    Mat src(32, 40, CV_32FC1, Scalar(1.0f));

    UMat u = makeCudaUMat();
    src.copyTo(u);
    ASSERT_TRUE(u.u != NULL && u.u->handle != NULL);

    cv::cuda::GpuMat g(u.rows, u.cols, u.type(), u.u->handle,
                       (size_t)u.cols * CV_ELEM_SIZE(u.type()));

    Mat viaGpu;
    g.download(viaGpu);
    EXPECT_EQ(0.0, cvtest::norm(src, viaGpu, NORM_INF));

    g.setTo(Scalar(7.0f));
    u.u->markHostCopyObsolete(true);          // device was mutated behind UMat's back
    Mat afterKernel;
    u.copyTo(afterKernel);
    EXPECT_EQ(0.0, cvtest::norm(Mat(src.size(), src.type(), Scalar(7.0f)), afterKernel, NORM_INF));
}

// P1 gives a UMat storage on CUDA, not GPU execution. With OpenCL on, cv:: ops feed the
// CUDA pointer to the driver as a cl_mem and segfault; the tests below force OpenCL off so
// ops resolve through the host getMat() fallback, and pin why the OpenCL path is unusable.

struct NoOpenCL
{
    bool prev;
    NoOpenCL() : prev(cv::ocl::useOpenCL()) { cv::ocl::setUseOpenCL(false); }
    ~NoOpenCL() { cv::ocl::setUseOpenCL(prev); }
};

// A CUDA UMat fails the OpenCL kernel path's isCLBuffer precondition (carries the CUDA allocator).
TEST(Core_CudaUMat, is_not_opencl_buffer)
{
    skipIfNoCudaDevice();
    UMat u = makeCudaUMat();
    Mat(8, 8, CV_32FC1, Scalar(1)).copyTo(u);
    ASSERT_TRUE(u.u != NULL);
    EXPECT_EQ(cv::cuda::getCudaAllocator(), u.u->currAllocator);
#ifdef HAVE_OPENCL
    EXPECT_NE((MatAllocator*)cv::ocl::getOpenCLAllocator(), u.u->currAllocator);
#endif
}

TEST(Core_CudaUMat, op_arithmetic_host_fallback)
{
    skipIfNoCudaDevice();
    NoOpenCL guard;

    Mat a(50, 40, CV_32FC1), b(50, 40, CV_32FC1);
    randu(a, 0, 10); randu(b, 0, 10);

    UMat ua = makeCudaUMat(), ub = makeCudaUMat();
    a.copyTo(ua); b.copyTo(ub);

    UMat usum = makeCudaUMat(), udiff = makeCudaUMat(), uprod = makeCudaUMat();
    cv::add(ua, ub, usum);
    cv::subtract(ua, ub, udiff);
    cv::multiply(ua, ub, uprod);

    Mat sum, diff, prod;
    usum.copyTo(sum); udiff.copyTo(diff); uprod.copyTo(prod);
    EXPECT_LE(cvtest::norm(sum,  a + b,       NORM_INF), 1e-5);
    EXPECT_LE(cvtest::norm(diff, a - b,       NORM_INF), 1e-5);
    EXPECT_LE(cvtest::norm(prod, a.mul(b),    NORM_INF), 1e-4);
}

TEST(Core_CudaUMat, op_convertTo_host_fallback)
{
    skipIfNoCudaDevice();
    NoOpenCL guard;

    Mat a(30, 45, CV_32FC1);
    randu(a, 0, 100);

    UMat ua = makeCudaUMat();
    a.copyTo(ua);
    UMat u8 = makeCudaUMat();
    ua.convertTo(u8, CV_8U, 2.0, 1.0);

    Mat got, expected;
    u8.copyTo(got);
    a.convertTo(expected, CV_8U, 2.0, 1.0);
    EXPECT_EQ(0.0, cvtest::norm(got, expected, NORM_INF));
}

TEST(Core_CudaUMat, op_addWeighted_host_fallback)
{
    skipIfNoCudaDevice();
    NoOpenCL guard;

    Mat a(40, 40, CV_32FC1), b(40, 40, CV_32FC1);
    randu(a, 0, 5); randu(b, 0, 5);

    UMat ua = makeCudaUMat(), ub = makeCudaUMat();
    a.copyTo(ua); b.copyTo(ub);

    UMat ur = makeCudaUMat();
    cv::addWeighted(ua, 2.0, ub, 1.0, 0.0, ur);

    Mat got, expected;
    ur.copyTo(got);
    cv::addWeighted(a, 2.0, b, 1.0, 0.0, expected);
    EXPECT_LE(cvtest::norm(got, expected, NORM_INF), 1e-4);
}

}} // namespace

#endif // HAVE_CUDA
