// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "opencv2/core/utils/host_device_buffer.private.hpp"
#include "opencv2/core/utils/logger.hpp"

namespace cv { namespace utils {

// allocator for a device location, or NULL when the build or the machine doesn't have that backend
static MatAllocator* deviceAllocator(HostDeviceBuffer::Location where)
{
#ifdef HAVE_CUDA
    if (where == HostDeviceBuffer::CUDA && cuda::getCudaEnabledDeviceCount() > 0)
        return cuda::getCudaAllocator();
#endif
#ifdef HAVE_OPENCL
    if (where == HostDeviceBuffer::OPENCL && ocl::useOpenCL())
        return ocl::getOpenCLAllocator();
#endif
    CV_UNUSED(where);
    return NULL;
}

static const char* locationName(HostDeviceBuffer::Location where)
{
    return where == HostDeviceBuffer::CUDA ? "CUDA" : where == HostDeviceBuffer::OPENCL ? "OpenCL" : "host";
}

HostDeviceBuffer::HostDeviceBuffer() : valid_(NONE), failed_(NONE) {}

HostDeviceBuffer::HostDeviceBuffer(HostDeviceBuffer&& other) CV_NOEXCEPT
    : host_(std::move(other.host_)), cuda_(std::move(other.cuda_)), ocl_(std::move(other.ocl_)),
      valid_(other.valid_), failed_(other.failed_)
{
    other.valid_ = other.failed_ = NONE;
}

HostDeviceBuffer& HostDeviceBuffer::operator=(HostDeviceBuffer&& other) CV_NOEXCEPT
{
    if (this != &other)
    {
        host_ = std::move(other.host_);
        cuda_ = std::move(other.cuda_);
        ocl_ = std::move(other.ocl_);
        valid_ = other.valid_;
        failed_ = other.failed_;
        other.valid_ = other.failed_ = NONE;
    }
    return *this;
}

Mat& HostDeviceBuffer::hostForWrite(const MatShape& shape, int type)
{
    valid_ &= ~HOST;  // stale until commit(HOST)
    host_.fit(shape, type);
    return host_;
}

const Mat& HostDeviceBuffer::hostForRead()
{
    if (!(valid_ & HOST))
    {
        CV_Assert(valid_ != NONE && "HostDeviceBuffer: there is no value to read");
        const UMat& src = (valid_ & CUDA) ? cuda_ : ocl_;
        host_.fitSameSize(src, src.type());
        src.copyTo(host_);
        valid_ |= HOST;
    }
    return host_;
}

UMat* HostDeviceBuffer::deviceForWrite(Location where, const MatShape& shape, int type)
{
    if (failed_ & where)
        return NULL;
    UMat& slot = where == CUDA ? cuda_ : ocl_;
    try
    {
        // the device query itself can throw (e.g. CUDA init failure), so it counts as a backend failure too
        MatAllocator* a = deviceAllocator(where);
        if (!a)
            return NULL;
        slot.allocator = a;
        slot.fit(shape, type);
        // the OpenCL allocator quietly hands back host memory when it can't allocate
        if (slot.u && slot.u->currAllocator != a)
            CV_Error(Error::StsNoMem, "device allocation returned host memory");
        valid_ &= ~where;  // stale until commit(where)
        return &slot;
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "HostDeviceBuffer: " << locationName(where) << " is not usable, falling back: " << e.what());
        failed_ |= where;
        valid_ &= ~where;
        slot.release();
        return NULL;
    }
}

UMat* HostDeviceBuffer::deviceForRead(Location where)
{
    if (valid_ & where)
        return where == CUDA ? &cuda_ : &ocl_;
    CV_Assert(valid_ != NONE && "HostDeviceBuffer: there is no value to read");
    // take shape/type from the current copy so an unusable backend costs no download
    const MatShape shape = (valid_ & HOST) ? host_.size : ((valid_ & CUDA) ? cuda_.size : ocl_.size);
    const int type = (valid_ & HOST) ? host_.type() : ((valid_ & CUDA) ? cuda_.type() : ocl_.type());
    UMat* slot = deviceForWrite(where, shape, type);
    if (!slot)
        return NULL;
    const Mat& src = hostForRead();
    try
    {
        src.copyTo(*slot);
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "HostDeviceBuffer: upload to " << locationName(where) << " failed, falling back: " << e.what());
        failed_ |= where;
        slot->release();
        return NULL;
    }
    valid_ |= where;
    return slot;
}

UMat* HostDeviceBuffer::cudaForWrite(const MatShape& shape, int type) { return deviceForWrite(CUDA, shape, type); }
UMat* HostDeviceBuffer::cudaForRead() { return deviceForRead(CUDA); }
UMat* HostDeviceBuffer::oclForWrite(const MatShape& shape, int type) { return deviceForWrite(OPENCL, shape, type); }
UMat* HostDeviceBuffer::oclForRead() { return deviceForRead(OPENCL); }

HostDeviceBuffer::Location HostDeviceBuffer::fallbackForRead(Location first)
{
    CV_Assert(first == CUDA || first == OPENCL || first == HOST);
    if (first == CUDA && cudaForRead())
        return CUDA;
    if (first != HOST && oclForRead())
        return OPENCL;
    hostForRead();
    return HOST;
}

HostDeviceBuffer::Location HostDeviceBuffer::fallbackForWrite(const MatShape& shape, int type, Location first)
{
    CV_Assert(first == CUDA || first == OPENCL || first == HOST);
    if (first == CUDA && cudaForWrite(shape, type))
        return CUDA;
    if (first != HOST && oclForWrite(shape, type))
        return OPENCL;
    hostForWrite(shape, type);
    return HOST;
}

void HostDeviceBuffer::commit(Location where)
{
    CV_Assert(where == HOST || where == CUDA || where == OPENCL);
    valid_ = where;
}

void HostDeviceBuffer::reset()
{
    valid_ = NONE;
}

void HostDeviceBuffer::release()
{
    host_.release();
    cuda_.release();
    ocl_.release();
    valid_ = failed_ = NONE;
}

}} // cv::utils::
