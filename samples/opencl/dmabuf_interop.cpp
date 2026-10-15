// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>

#include <iostream>

#if defined(__linux__)
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

struct DmaHeapAllocationData
{
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};

#ifndef DMA_HEAP_IOC_MAGIC
#define DMA_HEAP_IOC_MAGIC 'H'
#endif
#ifndef DMA_HEAP_IOCTL_ALLOC
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct DmaHeapAllocationData)
#endif

static int allocateDmaBuf(size_t size)
{
    int heap = ::open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    if (heap < 0)
        return -1;
    DmaHeapAllocationData data = {};
    data.len = size;
    data.fd_flags = O_RDWR | O_CLOEXEC;
    const int status = ::ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &data);
    ::close(heap);
    return status == 0 ? static_cast<int>(data.fd) : -1;
}

} // anonymous namespace
#endif

int main()
{
#if !defined(__linux__)
    std::cout << "DMA-BUF interop is available on Linux only." << std::endl;
    return 0;
#else
    cv::ocl::setUseOpenCL(true);
    if (!cv::ocl::useOpenCL())
    {
        std::cerr << "OpenCL is not available." << std::endl;
        return 1;
    }

    const int rows = 480;
    const int cols = 640;
    const size_t step = static_cast<size_t>(cols);
    const size_t size = static_cast<size_t>(rows) * step;

    const int fd = allocateDmaBuf(size);
    if (fd < 0)
    {
        std::cerr << "Unable to allocate /dev/dma_heap/system." << std::endl;
        return 1;
    }

    // Import once. In a camera pipeline, do this once for every DMA-BUF in the ring.
    cv::UMat frame = cv::ocl::createUMatFromDmaBuf(fd, size, step, rows, cols, CV_8UC1);

    // The producer must be done with the buffer before acquireExternalMemory().
    cv::ocl::acquireExternalMemory(frame);
    cv::add(frame, cv::Scalar::all(1), frame);

    // This waits until ownership has been returned to the external API.
    cv::ocl::releaseExternalMemory(frame);

    // The original descriptor is still owned by the caller.
    ::close(fd);
    std::cout << "DMA-BUF processed without an intermediate copy." << std::endl;
    return 0;
#endif
}
