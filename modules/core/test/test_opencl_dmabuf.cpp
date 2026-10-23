// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#if defined(__linux__)
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace opencv_test {
namespace ocl {

TEST(OCL_DmaBuf, InvalidDescriptor)
{
    EXPECT_THROW(cv::ocl::createUMatFromDmaBuf(-1, 4096, 64, 64, 64, CV_8UC1), cv::Exception);
}

TEST(OCL_DmaBuf, RejectsRegularUMat)
{
    UMat ordinary(16, 16, CV_8UC1);
    EXPECT_THROW(cv::ocl::acquireExternalMemory(ordinary), cv::Exception);
    EXPECT_THROW(cv::ocl::releaseExternalMemory(ordinary), cv::Exception);
}

#if defined(__linux__)
#ifdef HAVE_OPENCL
namespace {

struct DmaHeapAllocationData
{
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};

struct DmaBufSync
{
    uint64_t flags;
};

#ifndef DMA_HEAP_IOC_MAGIC
#define DMA_HEAP_IOC_MAGIC 'H'
#endif
#ifndef DMA_HEAP_IOCTL_ALLOC
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct DmaHeapAllocationData)
#endif
#ifndef DMA_BUF_BASE
#define DMA_BUF_BASE 'b'
#endif
#ifndef DMA_BUF_IOCTL_SYNC
#define DMA_BUF_IOCTL_SYNC _IOW(DMA_BUF_BASE, 0, struct DmaBufSync)
#endif

static const uint64_t DMA_BUF_SYNC_READ = (1ull << 0);
static const uint64_t DMA_BUF_SYNC_WRITE = (2ull << 0);
static const uint64_t DMA_BUF_SYNC_START = (0ull << 2);
static const uint64_t DMA_BUF_SYNC_END = (1ull << 2);

static int allocateDmaBuf(size_t size)
{
    static const char* const heaps[] = {
        "/dev/dma_heap/system",
        "/dev/dma_heap/system-uncached",
        "/dev/dma_heap/reserved"
    };
    for (size_t i = 0; i < sizeof(heaps) / sizeof(heaps[0]); ++i)
    {
        int heap = ::open(heaps[i], O_RDWR | O_CLOEXEC);
        if (heap < 0)
            continue;

        DmaHeapAllocationData data = {};
        data.len = size;
        data.fd_flags = O_RDWR | O_CLOEXEC;
        const int status = ::ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &data);
        ::close(heap);
        if (status == 0)
            return static_cast<int>(data.fd);
    }
    return -1;
}

static bool syncDmaBuf(int fd, uint64_t flags)
{
    DmaBufSync sync = { flags };
    return ::ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) == 0;
}

} // anonymous namespace

TEST(OCL_DmaBuf, ZeroCopyExternalMemory)
{
    cv::ocl::setUseOpenCL(true);
    if (!cv::ocl::useOpenCL())
        throw SkipTestException("OpenCL is not available / disabled");

    const cv::ocl::Device& device = cv::ocl::Device::getDefault();
    if (!device.isExtensionSupported("cl_khr_external_memory") ||
        !device.isExtensionSupported("cl_khr_external_memory_dma_buf"))
    {
        throw SkipTestException("OpenCL DMA-BUF external-memory extensions are not available");
    }

    const int rows = 64;
    const int cols = 64;
    const size_t step = static_cast<size_t>(cols);
    const size_t size = static_cast<size_t>(rows) * step;

    const int fd = allocateDmaBuf(size);
    if (fd < 0)
        throw SkipTestException("Linux DMA heap is not available");

    void* mapping = ::mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED)
    {
        ::close(fd);
        throw SkipTestException("DMA-BUF mmap is not available");
    }

    ASSERT_TRUE(syncDmaBuf(fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE));
    std::memset(mapping, 7, size);
    ASSERT_TRUE(syncDmaBuf(fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE));

    UMat umat = cv::ocl::createUMatFromDmaBuf(fd, size, step, rows, cols, CV_8UC1);
    EXPECT_GE(::fcntl(fd, F_GETFD), 0);  // OpenCV must not consume the caller's fd.

    cv::ocl::acquireExternalMemory(umat);

    const char* kernelCode =
            "__kernel void add_value(__global uchar* data, int n, int value) {"
            "  int i = (int)get_global_id(0);"
            "  if (i < n) data[i] = (uchar)(data[i] + value);"
            "}";
    cv::ocl::ProgramSource source("opencv_dmabuf_test", "add_value", kernelCode, "");
    cv::String buildError;
    cv::ocl::Kernel kernel("add_value", source, cv::String(), &buildError);
    ASSERT_FALSE(kernel.empty()) << buildError;
    kernel.args(cv::ocl::KernelArg::PtrReadWrite(umat), static_cast<int>(size), 3);
    size_t globalSize[] = { size };
    ASSERT_TRUE(kernel.run_(1, globalSize, NULL, false));

    cv::ocl::releaseExternalMemory(umat);

    ASSERT_TRUE(syncDmaBuf(fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ));
    const uchar* data = static_cast<const uchar*>(mapping);
    for (size_t i = 0; i < size; i += 257)
        EXPECT_EQ((uchar)10, data[i]);
    EXPECT_EQ((uchar)10, data[size - 1]);
    ASSERT_TRUE(syncDmaBuf(fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ));

    ::munmap(mapping, size);
    ::close(fd);
}
#endif // HAVE_OPENCL
#endif // __linux__

} // namespace ocl
} // namespace opencv_test
