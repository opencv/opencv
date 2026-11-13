// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_SRC_ACCEL_IMPL_HPP
#define OPENCV_CORE_SRC_ACCEL_IMPL_HPP

#include "opencv2/core/private/accel_backend.hpp"

namespace cv { namespace accel {

namespace detail { class BackendAllocator; }

#ifdef HAVE_CUDA
namespace cuda_impl {
//! the built-in CUDA backend (backend_cuda.cpp)
Ptr<Backend> createCudaBackend();
}
#endif

struct Device::Impl
{
    Impl(const Ptr<Backend>& b, int i, int id_, detail::BackendAllocator* a);

    Ptr<Backend> backend;
    int index;                              // within the backend
    int id;                                 // among all devices: where each thread keeps its stream
    String name;                            // "cuda:0"
    detail::BackendAllocator* allocator;    // the backend's, shared by its devices
};

namespace detail {

/** @brief The MatAllocator of one backend, shared by its devices. Each buffer keeps its device
in UMatData::allocatorContext. Memory the CPU cannot address gets a host copy on first CPU access. */
class BackendAllocator CV_FINAL : public MatAllocator
{
public:
    //! a buffer on @p dev of @p sizes and @p type, with the steps of a continuous one in @p step
    UMatData* allocate(Device::Impl* dev, int dims, const int* sizes, int type, size_t* step) const;

    //! refuses: device data is created with UMat::createOn(), which names the device
    UMatData* allocate(int dims, const int* sizes, int type, void* data, size_t* step,
                       AccessFlag flags, UMatUsageFlags usageFlags) const CV_OVERRIDE;
    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const CV_OVERRIDE;
    void deallocate(UMatData* u) const CV_OVERRIDE;
    void map(UMatData* u, AccessFlag accessFlags) const CV_OVERRIDE;
    void unmap(UMatData* u) const CV_OVERRIDE;
    void download(UMatData* u, void* dst, int dims, const size_t sz[],
                  const size_t srcofs[], const size_t srcstep[],
                  const size_t dststep[]) const CV_OVERRIDE;
    void upload(UMatData* u, const void* src, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[],
                const size_t srcstep[]) const CV_OVERRIDE;
    void copy(UMatData* srcdata, UMatData* dstdata, int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dstofs[], const size_t dststep[], bool sync) const CV_OVERRIDE;
};

bool isBackendAllocator(const MatAllocator* a);
//! the calling thread's stream on @p dev, created on first use
Stream& threadStream(Device::Impl* dev);

}}} // namespace cv::accel::detail

#endif // OPENCV_CORE_SRC_ACCEL_IMPL_HPP
