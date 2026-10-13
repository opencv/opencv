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
    // create() falls back to another allocator when cudaMalloc fails; that handle is not a CUDA pointer
    ASSERT_EQ(cv::cuda::getCudaAllocator(), u.u->currAllocator);

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

// P1 gives a UMat storage on CUDA, not GPU execution. A CUDA UMat has no cl_mem, so the OpenCL
// kernel path declines it and cv:: ops resolve through the host getMat() fallback. The tests
// below cover that fallback with OpenCL off, and with OpenCL on where the kernel path rejects it.

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

// a host UMat copied into a CUDA UMat is uploaded, so the CUDA side never gets a host mirror
TEST(Core_CudaUMat, copy_from_host_umat_does_not_map)
{
    skipIfNoCudaDevice();
    NoOpenCL guard;

    Mat src(32, 24, CV_32FC1);
    randu(src, 0, 1);
    UMat host;
    src.copyTo(host);
    ASSERT_EQ(Mat::getDefaultAllocator(), host.u->currAllocator);

    UMat dev = makeCudaUMat();
    host.copyTo(dev);
    ASSERT_TRUE(dev.u != NULL);
    EXPECT_TRUE(dev.u->data == NULL);

    Mat back;
    dev.copyTo(back);
    EXPECT_EQ(0.0, cvtest::norm(src, back, NORM_INF));
}

// fitSameSize then copyTo uploads into a CUDA UMat with the source layout and no host mirror
TEST(Core_CudaUMat, fitSameSize_copyTo_keeps_layout)
{
    skipIfNoCudaDevice();

    const MatShape blk({1, 2, 4, 4, 8}, DATA_LAYOUT_BLOCK, 13);
    Mat src;
    src.fit(blk, CV_32F);
    randu(src, 0, 10);

    UMat dev = makeCudaUMat();
    dev.fitSameSize(src, src.type());
    src.copyTo(dev);
    ASSERT_TRUE(dev.u != NULL);
    EXPECT_EQ(cv::cuda::getCudaAllocator(), dev.u->currAllocator);
    EXPECT_TRUE(dev.u->data == NULL);
    EXPECT_EQ(DATA_LAYOUT_BLOCK, dev.size.layout);

    Mat back;
    back.fitSameSize(dev, dev.type());
    dev.copyTo(back);
    EXPECT_EQ(DATA_LAYOUT_BLOCK, back.size.layout);
    EXPECT_EQ(0.0, cvtest::norm(src, back, NORM_INF));
}

#ifdef HAVE_OPENCL
// turns OpenCL on for one test and restores the previous state afterwards
struct WithOpenCL
{
    bool prev;
    WithOpenCL() : prev(cv::ocl::useOpenCL()) { cv::ocl::setUseOpenCL(true); }
    ~WithOpenCL() { cv::ocl::setUseOpenCL(prev); }
};

// Kernel::set must decline a CUDA UMat instead of passing its device pointer to clSetKernelArg
TEST(Core_CudaUMat, opencl_kernel_declines_cuda_umat)
{
    skipIfNoCudaDevice();
    WithOpenCL guard;
    if (!cv::ocl::useOpenCL())
        throw SkipTestException("OpenCL is not available");

    UMat u = makeCudaUMat();
    Mat(8, 8, CV_32FC1, Scalar(1)).copyTo(u);

    ocl::Kernel k("touch", ocl::ProgramSource("__kernel void touch(__global const float* a) {}"));
    ASSERT_FALSE(k.empty());
    EXPECT_EQ(-1, k.set(0, ocl::KernelArg::PtrReadOnly(u)));
    EXPECT_TRUE(k.empty());
}

// with OpenCL on, ops reach the kernel path, are declined there and finish on the CPU
TEST(Core_CudaUMat, op_opencl_on_falls_back)
{
    skipIfNoCudaDevice();
    WithOpenCL guard;
    if (!cv::ocl::useOpenCL())
        throw SkipTestException("OpenCL is not available");

    Mat a(50, 40, CV_32FC1), b(50, 40, CV_32FC1);
    randu(a, 0, 10); randu(b, 0, 10);

    UMat ua = makeCudaUMat(), ub = makeCudaUMat();
    a.copyTo(ua); b.copyTo(ub);

    UMat usum = makeCudaUMat(), uw = makeCudaUMat(), u8 = makeCudaUMat();
    cv::add(ua, ub, usum);
    cv::addWeighted(ua, 2.0, ub, 1.0, 0.0, uw);
    ua.convertTo(u8, CV_8U, 2.0, 1.0);

    Mat sum, w, c8, expectedW, expected8;
    usum.copyTo(sum); uw.copyTo(w); u8.copyTo(c8);
    cv::addWeighted(a, 2.0, b, 1.0, 0.0, expectedW);
    a.convertTo(expected8, CV_8U, 2.0, 1.0);
    EXPECT_LE(cvtest::norm(sum, a + b, NORM_INF), 1e-5);
    EXPECT_LE(cvtest::norm(w, expectedW, NORM_INF), 1e-4);
    EXPECT_EQ(0.0, cvtest::norm(c8, expected8, NORM_INF));
}

// device to device across allocators (CUDA -> OpenCL -> CUDA) goes through the host and keeps the layout
TEST(Core_CudaUMat, fitSameSize_copyTo_cuda_opencl_cuda)
{
    skipIfNoCudaDevice();
    WithOpenCL guard;
    if (!cv::ocl::useOpenCL())
        throw SkipTestException("OpenCL is not available");

    const MatShape blk({1, 2, 4, 4, 8}, DATA_LAYOUT_BLOCK, 13);
    Mat src;
    src.fit(blk, CV_32F);
    randu(src, 0, 10);

    UMat cu = makeCudaUMat();
    cu.fitSameSize(src, src.type());
    src.copyTo(cu);
    ASSERT_EQ(cv::cuda::getCudaAllocator(), cu.u->currAllocator);

    UMat cl;
    cl.fitSameSize(cu, cu.type());
    cu.copyTo(cl);
    EXPECT_EQ(cv::ocl::getOpenCLAllocator(), cl.u->currAllocator);

    UMat cu2 = makeCudaUMat();
    cu2.fitSameSize(cl, cl.type());
    cl.copyTo(cu2);
    ASSERT_EQ(cv::cuda::getCudaAllocator(), cu2.u->currAllocator);
    EXPECT_TRUE(cu2.u->data == NULL);

    Mat back;
    back.fitSameSize(cu2, cu2.type());
    cu2.copyTo(back);
    EXPECT_EQ(DATA_LAYOUT_BLOCK, back.size.layout);
    EXPECT_EQ(0.0, cvtest::norm(src, back, NORM_INF));
}
#endif // HAVE_OPENCL

}} // namespace

#endif // HAVE_CUDA
