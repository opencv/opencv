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

TEST(Core_CudaUMat, user_host_data_coherence)
{
    skipIfNoCudaDevice();
    MatAllocator* cudaAlloc = cv::cuda::getCudaAllocator();
    ASSERT_TRUE(cudaAlloc != NULL);

    // Build a UMat backed by caller-owned host data0 via CudaUMatAllocator::allocate
    Mat host(32, 48, CV_32FC1);
    randu(host, -10.f, 10.f);
    const int sizes[2] = { host.rows, host.cols };
    size_t steps[2] = { CV_AUTOSTEP, CV_AUTOSTEP };

    UMat u;
    u.allocator = cudaAlloc;
    u.flags = UMat::MAGIC_VAL | UMat::CONTINUOUS_FLAG | CV_32FC1;
    u.dims = 2;
    u.size = MatShape(2, sizes);
    u.rows = host.rows;
    u.cols = host.cols;
    u.u = cudaAlloc->allocate(2, sizes, CV_32FC1, host.data, steps, ACCESS_RW, USAGE_DEFAULT);
    u.step[0] = steps[0];
    u.step[1] = steps[1];
    u.addref();

    // 1. download() while deviceCopyObsolete() is true must read data0, not uninitialized VRAM
    Mat downloaded;
    u.copyTo(downloaded);
    EXPECT_EQ(0.0, cvtest::norm(host, downloaded, NORM_INF));

    // 2. Same-allocator copy() while src->deviceCopyObsolete() is true must sync data0 first
    UMat uCopy = makeCudaUMat();
    u.copyTo(uCopy);
    Mat fromCopy;
    uCopy.copyTo(fromCopy);
    EXPECT_EQ(0.0, cvtest::norm(host, fromCopy, NORM_INF));

    // 3. Partial-ROI upload() onto a data0-backed UMat must preserve unwritten pixels outside ROI
    Mat host2 = host.clone();
    size_t steps2[2] = { CV_AUTOSTEP, CV_AUTOSTEP };
    UMat uPartial;
    uPartial.allocator = cudaAlloc;
    uPartial.flags = UMat::MAGIC_VAL | UMat::CONTINUOUS_FLAG | CV_32FC1;
    uPartial.dims = 2;
    uPartial.size = MatShape(2, sizes);
    uPartial.rows = host2.rows;
    uPartial.cols = host2.cols;
    uPartial.u = cudaAlloc->allocate(2, sizes, CV_32FC1, host2.data, steps2, ACCESS_RW, USAGE_DEFAULT);
    uPartial.step[0] = steps2[0];
    uPartial.step[1] = steps2[1];
    uPartial.addref();

    Rect patchRect(4, 5, 12, 10);
    Mat patch(patchRect.size(), CV_32FC1);
    randu(patch, 50.f, 60.f);
    Mat expected = host2.clone();
    patch.copyTo(expected(patchRect));
    patch.copyTo(uPartial(patchRect));

    Mat afterPartial;
    uPartial.copyTo(afterPartial);
    EXPECT_EQ(0.0, cvtest::norm(expected, afterPartial, NORM_INF));
}

TEST(Core_CudaUMat, zero_dim_and_adopt_umatdata)
{
    skipIfNoCudaDevice();
    MatAllocator* cudaAlloc = cv::cuda::getCudaAllocator();
    ASSERT_TRUE(cudaAlloc != NULL);

    // 0-D allocation must not underflow step[-1]
    size_t sentinel[3] = { 0xDEADBEEFu, 0xCAFEBABEu, 0x12345678u };
    UMatData* u0 = cudaAlloc->allocate(0, NULL, CV_32FC1, NULL, &sentinel[1], ACCESS_RW, USAGE_DEFAULT);
    ASSERT_TRUE(u0 != NULL);
    EXPECT_EQ(size_t(0xDEADBEEFu), sentinel[0]);
    EXPECT_EQ(sizeof(float), u0->size);
    cudaAlloc->deallocate(u0);

    // allocate(UMatData*) on a host-originated UMatData must take currAllocator ownership
    const int sz[2] = { 8, 12 };
    size_t st[2] = { CV_AUTOSTEP, CV_AUTOSTEP };
    UMatData* uh = Mat::getStdAllocator()->allocate(2, sz, CV_32FC1, NULL, st, ACCESS_RW, USAGE_DEFAULT);
    ASSERT_TRUE(uh != NULL);
    ASSERT_TRUE(cudaAlloc->allocate(uh, ACCESS_RW, USAGE_DEFAULT));
    EXPECT_EQ(cudaAlloc, uh->currAllocator);
    EXPECT_TRUE(uh->handle != NULL);
    cudaAlloc->deallocate(uh);
}

TEST(Core_CudaUMat, roi_to_roi_2d_copy)
{
    skipIfNoCudaDevice();

    Mat srcFull(96, 112, CV_32FC3), dstFull(96, 112, CV_32FC3);
    randu(srcFull, -5.f, 5.f);
    randu(dstFull, 10.f, 20.f);

    UMat uSrc = makeCudaUMat(), uDst = makeCudaUMat();
    srcFull.copyTo(uSrc);
    dstFull.copyTo(uDst);

    Rect srcRoi(7, 9, 45, 37), dstRoi(13, 19, 45, 37);
    uSrc(srcRoi).copyTo(uDst(dstRoi));

    Mat expected = dstFull.clone();
    srcFull(srcRoi).copyTo(expected(dstRoi));

    Mat got;
    uDst.copyTo(got);
    EXPECT_EQ(0.0, cvtest::norm(expected, got, NORM_INF));
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

#ifdef HAVE_OPENCL
// turns OpenCL on for one test and restores the previous state afterwards
struct WithOpenCL
{
    bool prev;
    WithOpenCL() : prev(cv::ocl::useOpenCL()) { cv::ocl::setUseOpenCL(true); }
    ~WithOpenCL() { cv::ocl::setUseOpenCL(prev); }
};

// with OpenCL on, the OpenCL kernel path declines CUDA UMats and the op finishes on the CPU
TEST(Core_CudaUMat, op_opencl_on_falls_back)
{
    skipIfNoCudaDevice();
    WithOpenCL guard;
    if (!cv::ocl::useOpenCL())
        throw SkipTestException("OpenCL is not available");

    Mat a(50, 40, CV_32FC1), b(50, 40, CV_32FC1);
    randu(a, 0, 10); randu(b, 0, 10);

    UMat ua = makeCudaUMat(), ub = makeCudaUMat(), usum = makeCudaUMat();
    a.copyTo(ua); b.copyTo(ub);
    cv::add(ua, ub, usum);

    Mat sum;
    usum.copyTo(sum);
    EXPECT_LE(cvtest::norm(sum, a + b, NORM_INF), 1e-5);
}

// device to device across allocators (CUDA -> OpenCL -> CUDA) goes through a host copy
TEST(Core_CudaUMat, copy_cuda_opencl_cuda)
{
    skipIfNoCudaDevice();
    WithOpenCL guard;
    if (!cv::ocl::useOpenCL())
        throw SkipTestException("OpenCL is not available");

    Mat src(30, 20, CV_32FC1);
    randu(src, 0, 10);

    UMat cu = makeCudaUMat(), cl, cu2 = makeCudaUMat();
    src.copyTo(cu);
    cu.copyTo(cl);
    EXPECT_EQ(cv::ocl::getOpenCLAllocator(), cl.u->currAllocator);
    cl.copyTo(cu2);

    Mat back;
    cu2.copyTo(back);
    EXPECT_EQ(0.0, cvtest::norm(src, back, NORM_INF));
}
#endif // HAVE_OPENCL

}} // namespace

#endif // HAVE_CUDA
