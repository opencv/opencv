// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "opencv2/core/utils/host_device_buffer.private.hpp"
#include "opencv2/core/ocl.hpp"
#ifdef HAVE_CUDA
#include "opencv2/core/cuda.hpp"
#endif

namespace opencv_test { namespace {

using cv::utils::HostDeviceBuffer;

// sets ocl::useOpenCL() for one test and restores it afterwards
struct OpenCLSwitch
{
    bool prev;
    explicit OpenCLSwitch(bool on) : prev(cv::ocl::useOpenCL()) { cv::ocl::setUseOpenCL(on); }
    ~OpenCLSwitch() { cv::ocl::setUseOpenCL(prev); }
};

static Mat randomMat(const MatShape& shape, int type)
{
    Mat m;
    m.fit(shape, type);
    randu(m, 0, 10);
    return m;
}

TEST(Core_HostDeviceBuffer, host_only)
{
    const MatShape shape({3, 4, 5});
    const Mat src = randomMat(shape, CV_32F);

    HostDeviceBuffer buf;
    ASSERT_EQ(HostDeviceBuffer::HOST, buf.fallbackForWrite(shape, CV_32F, HostDeviceBuffer::HOST));
    src.copyTo(buf.hostForWrite(shape, CV_32F));
    buf.commit(HostDeviceBuffer::HOST);

    EXPECT_EQ(HostDeviceBuffer::HOST, buf.fallbackForRead(HostDeviceBuffer::HOST));
    EXPECT_EQ(0, cvtest::norm(src, buf.hostForRead(), NORM_INF));
}

TEST(Core_HostDeviceBuffer, opencl_off_falls_back_to_host)
{
    OpenCLSwitch noOpenCL(false);
    const MatShape shape({2, 8});

    HostDeviceBuffer buf;
    EXPECT_TRUE(buf.oclForWrite(shape, CV_8U) == NULL);
    EXPECT_EQ(HostDeviceBuffer::HOST, buf.fallbackForWrite(shape, CV_8U, HostDeviceBuffer::OPENCL));

    buf.hostForWrite(shape, CV_8U).setTo(Scalar::all(3));
    buf.commit(HostDeviceBuffer::HOST);
    EXPECT_EQ(HostDeviceBuffer::HOST, buf.fallbackForRead(HostDeviceBuffer::OPENCL));
}

TEST(Core_HostDeviceBuffer, opencl_roundtrip)
{
    OpenCLSwitch withOpenCL(true);
    if (!cv::ocl::useOpenCL())
        throw SkipTestException("OpenCL is not available");

    const MatShape shape({1, 2, 4, 4, 8}, DATA_LAYOUT_BLOCK, 13);
    const Mat src = randomMat(shape, CV_32F);

    HostDeviceBuffer buf;
    src.copyTo(buf.hostForWrite(shape, CV_32F));
    buf.commit(HostDeviceBuffer::HOST);

    ASSERT_EQ(HostDeviceBuffer::OPENCL, buf.fallbackForRead(HostDeviceBuffer::OPENCL));
    UMat* dev = buf.oclForRead();
    ASSERT_TRUE(dev != NULL && dev->u != NULL);
    EXPECT_EQ(cv::ocl::getOpenCLAllocator(), dev->u->currAllocator);
    EXPECT_EQ(DATA_LAYOUT_BLOCK, dev->size.layout);
    EXPECT_EQ(13, dev->size.C);
    EXPECT_EQ(HostDeviceBuffer::HOST | HostDeviceBuffer::OPENCL, buf.valid());

    // a device write makes the host copy stale until it is read back
    dev->setTo(Scalar::all(7));
    buf.commit(HostDeviceBuffer::OPENCL);
    const Mat expected(src.dims, src.size.p, CV_32F, Scalar::all(7));
    EXPECT_EQ(0, cvtest::norm(expected, buf.hostForRead(), NORM_INF));
    EXPECT_EQ(DATA_LAYOUT_BLOCK, buf.hostForRead().size.layout);
}

TEST(Core_HostDeviceBuffer, reuses_storage)
{
    HostDeviceBuffer buf;
    const uchar* data = buf.hostForWrite(MatShape({64, 64}), CV_32F).data;
    EXPECT_EQ(data, buf.hostForWrite(MatShape({16, 16, 4}), CV_32F).data);
    EXPECT_EQ(data, buf.hostForWrite(MatShape({64, 64}), CV_32F).data);
}

TEST(Core_HostDeviceBuffer, read_after_reset_throws)
{
    HostDeviceBuffer buf;
    buf.hostForWrite(MatShape({4}), CV_32F).setTo(Scalar::all(1));
    buf.commit(HostDeviceBuffer::HOST);
    buf.reset();
    EXPECT_EQ(0, buf.valid());
    EXPECT_ANY_THROW(buf.hostForRead());
}

TEST(Core_HostDeviceBuffer, move_only)
{
    static_assert(!std::is_copy_constructible<HostDeviceBuffer>::value, "HostDeviceBuffer must not be copyable");
    static_assert(!std::is_copy_assignable<HostDeviceBuffer>::value, "HostDeviceBuffer must not be copyable");

    HostDeviceBuffer a;
    a.hostForWrite(MatShape({8}), CV_32F).setTo(Scalar::all(5));
    a.commit(HostDeviceBuffer::HOST);

    HostDeviceBuffer b(std::move(a));
    EXPECT_EQ(0, a.valid());
    EXPECT_EQ(HostDeviceBuffer::HOST, b.valid());
    EXPECT_EQ(5.f, b.hostForRead().at<float>(0));

    HostDeviceBuffer c;
    c = std::move(b);
    EXPECT_EQ(0, b.valid());
    EXPECT_EQ(5.f, c.hostForRead().at<float>(0));
}

TEST(Core_HostDeviceBuffer, write_slot_is_not_current_until_commit)
{
    HostDeviceBuffer buf;
    buf.hostForWrite(MatShape({64, 64}), CV_32F).setTo(Scalar::all(1));
    buf.commit(HostDeviceBuffer::HOST);
    ASSERT_EQ(HostDeviceBuffer::HOST, buf.valid());

    buf.hostForWrite(MatShape({16, 16, 4}), CV_32F);
    EXPECT_EQ(0, buf.valid());
    EXPECT_ANY_THROW(buf.hostForRead());
}

TEST(Core_HostDeviceBuffer, unusable_cuda_keeps_opencl_value_without_download)
{
#ifdef HAVE_CUDA
    if (cv::cuda::getCudaEnabledDeviceCount() > 0)
        throw SkipTestException("Needs a machine where CUDA is not usable");
#endif
    OpenCLSwitch withOpenCL(true);
    if (!cv::ocl::useOpenCL())
        throw SkipTestException("OpenCL is not available");

    HostDeviceBuffer buf;
    UMat* dev = buf.oclForWrite(MatShape({4, 4}), CV_32F);
    ASSERT_TRUE(dev != NULL);
    dev->setTo(Scalar::all(2));
    buf.commit(HostDeviceBuffer::OPENCL);

    EXPECT_TRUE(buf.cudaForRead() == NULL);
    EXPECT_EQ(HostDeviceBuffer::OPENCL, buf.fallbackForRead());
    EXPECT_EQ(HostDeviceBuffer::OPENCL, buf.valid());  // no host copy was made
}

#ifdef HAVE_CUDA
TEST(Core_HostDeviceBuffer, cuda_roundtrip)
{
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
        throw SkipTestException("No CUDA device available");

    const MatShape shape({2, 3, 4, 5});
    const Mat src = randomMat(shape, CV_32F);

    HostDeviceBuffer buf;
    src.copyTo(buf.hostForWrite(shape, CV_32F));
    buf.commit(HostDeviceBuffer::HOST);

    ASSERT_EQ(HostDeviceBuffer::CUDA, buf.fallbackForRead());
    UMat* dev = buf.cudaForRead();
    ASSERT_TRUE(dev != NULL && dev->u != NULL);
    EXPECT_EQ(cv::cuda::getCudaAllocator(), dev->u->currAllocator);
    EXPECT_TRUE(dev->u->data == NULL);  // uploaded, never mapped
    Mat uploaded;
    dev->copyTo(uploaded);
    EXPECT_EQ(0, cvtest::norm(src, uploaded, NORM_INF));

    // a CUDA producer writes a new value, so the next host read must download it
    UMat* out = buf.cudaForWrite(shape, CV_32F);
    ASSERT_TRUE(out != NULL && out->u != NULL);
    cv::cuda::GpuMat g(1, (int)out->total(), CV_32F, out->u->handle);
    g.setTo(Scalar::all(7));
    buf.commit(HostDeviceBuffer::CUDA);
    const Mat expected(src.dims, src.size.p, CV_32F, Scalar::all(7));
    EXPECT_EQ(0, cvtest::norm(expected, buf.hostForRead(), NORM_INF));
}
#endif

}} // namespace
