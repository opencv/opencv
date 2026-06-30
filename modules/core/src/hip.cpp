// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#define OPENCV_CORE_HIP_IMPL
#include "precomp.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/core/hip.hpp"
#include "opencv2/core/private/hip_stubs.hpp"
#include "umatrix.hpp"

#ifdef HAVE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "opencv2/core/hipdev.hpp"
#endif

using namespace cv;
using namespace cv::hip;

#ifdef HAVE_HIP

#define hipSafeCall(expr) CV_HIP_SAFE_CALL(expr)

#endif

// HipAllocator

#ifdef HAVE_HIP
namespace {

class HipAllocator CV_FINAL : public MatAllocator
{
    // HIP allocation failed: fall back down the chain HIP -> OpenCL -> CPU (straight to CPU if OpenCL isn't built in).
    static UMatData* fallbackAllocate(int dims, const int* sizes, int type,
                                      void* data, size_t* step,
                                      AccessFlag flags, UMatUsageFlags usageFlags)
    {
#ifdef HAVE_OPENCL
        return ocl::getOpenCLAllocator()->allocate(dims, sizes, type, data, step, flags, usageFlags);
#else
        return Mat::getStdAllocator()->allocate(dims, sizes, type, data, step, flags, usageFlags);
#endif
    }

public:
    UMatData* allocate(int dims, const int* sizes, int type,
                       void* data, size_t* step,
                       AccessFlag flags, UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        CV_UNUSED(flags); CV_UNUSED(usageFlags);
        CV_Assert(data == nullptr);

        size_t total = CV_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; i--) {
            if (step) step[i] = total;
            total *= sizes[i];
        }

        void* devicePtr = nullptr;
        if (hipMalloc(&devicePtr, total) != hipSuccess || !devicePtr) {
            (void)hipGetLastError();  // clear the sticky error before using another backend
            return fallbackAllocate(dims, sizes, type, data, step, flags, usageFlags);
        }

        UMatData* u = new UMatData(this);
        u->data     = nullptr;
        u->origdata = nullptr;
        u->handle   = devicePtr;
        u->size     = total;
        u->flags    = UMatData::COPY_ON_MAP;
        u->markHostCopyObsolete(true);
        return u;
    }

    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        CV_UNUSED(usageFlags);
        if (!u) return false;

        UMatDataAutoLock lock(u);

        if (u->handle == nullptr) {
            CV_Assert(u->origdata != nullptr);
            void* devicePtr = nullptr;
            if (hipMalloc(&devicePtr, u->size) != hipSuccess || !devicePtr) {
                (void)hipGetLastError();  // clear the sticky error
#ifdef HAVE_OPENCL
                // OpenCL's UMatData allocate has no disabled-guard, so delegate only when OpenCL is usable.
                if (ocl::useOpenCL())
                    return ocl::getOpenCLAllocator()->allocate(u, accessFlags, usageFlags);
#endif
                return false;
            }
            u->handle = devicePtr;
            if (u->origdata) {
                hipSafeCall(hipMemcpy(u->handle, u->origdata, u->size, hipMemcpyHostToDevice));
                u->markHostCopyObsolete(false);
                u->markDeviceCopyObsolete(false);
            }
        }
        // Claim the UMatData (e.g. from Mat::getUMat) as HIP-resident, mirroring the
        // OpenCL allocator so isHipUMat() holds; deallocate() restores prevAllocator.
        if (u->currAllocator != this) {
            u->prevAllocator = u->currAllocator;
            u->currAllocator = this;
        }
        if (!!(accessFlags & ACCESS_WRITE))
            u->markHostCopyObsolete(true);
        return true;
    }

    void deallocate(UMatData* u) const CV_OVERRIDE
    {
        if (!u) return;
        CV_Assert(u->urefcount == 0 && u->refcount == 0);

        // Borrowed from another allocator (Mat::getUMat): free the device handle,
        // restore the original allocator + host buffer, and let it finish cleanup.
        if (u->prevAllocator) {
            if (u->handle) {
                hipSafeCall(hipFree(u->handle));
                u->handle = nullptr;
            }
            u->markDeviceCopyObsolete(true);
            u->currAllocator = u->prevAllocator;
            u->prevAllocator = nullptr;
            if (u->data && u->copyOnMap() && u->data != u->origdata)
                fastFree(u->data);
            u->data = u->origdata;
            u->currAllocator->deallocate(u);
            return;
        }

        if (u->handle) {
            hipSafeCall(hipFree(u->handle));
            u->handle = nullptr;
        }
        if (u->data && !(u->flags & UMatData::USER_ALLOCATED)) {
            fastFree(u->data);
            u->data = nullptr;
        }
        delete u;
    }

    void map(UMatData* u, AccessFlag accessFlags) const CV_OVERRIDE
    {
        if (!u) return;
        UMatDataAutoLock lock(u);
        if (u->hostCopyObsolete() && u->handle) {
            if (!u->data)
                u->data = u->origdata = (uchar*)fastMalloc(u->size);
            hipSafeCall(hipMemcpy(u->data, u->handle, u->size, hipMemcpyDeviceToHost));
            u->markHostCopyObsolete(false);
        }
        if (accessFlags & ACCESS_WRITE)
            u->markDeviceCopyObsolete(true);
    }

    void unmap(UMatData* u) const CV_OVERRIDE
    {
        if (!u) return;
        UMatDataAutoLock lock(u);
        if (u->deviceCopyObsolete() && u->handle && u->data) {
            hipSafeCall(hipMemcpy(u->handle, u->data, u->size, hipMemcpyHostToDevice));
            u->markDeviceCopyObsolete(false);
            u->markHostCopyObsolete(true);
        }
        if (u->data && !(u->flags & UMatData::USER_ALLOCATED)) {
            fastFree(u->data);
            u->data     = nullptr;
            u->origdata = nullptr;
            u->markHostCopyObsolete(true);  // host buffer gone; device is authoritative
        }
    }

    void download(UMatData* u, void* dst, int dims, const size_t sz[],
                  const size_t srcofs[], const size_t srcstep[],
                  const size_t dststep[]) const CV_OVERRIDE
    {
        if (!u || !u->handle) return;
        if (dims <= 2) {
            const uchar* src = (const uchar*)u->handle;
            if (dims == 2)
                src += srcofs[0] * srcstep[0] + srcofs[1];
            hipSafeCall(hipMemcpy2D(dst, dststep[0],
                                    src, srcstep[0],
                                    sz[dims - 1], sz[0],
                                    hipMemcpyDeviceToHost));
        } else {
            hipSafeCall(hipMemcpy(dst, u->handle, u->size, hipMemcpyDeviceToHost));
        }
    }

    void upload(UMatData* u, const void* src, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[],
                const size_t srcstep[]) const CV_OVERRIDE
    {
        if (!u || !u->handle) return;
        if (dims <= 2) {
            uchar* dst = (uchar*)u->handle;
            if (dims == 2)
                dst += dstofs[0] * dststep[0] + dstofs[1];
            hipSafeCall(hipMemcpy2D(dst, dststep[0],
                                    src, srcstep[0],
                                    sz[dims - 1], sz[0],
                                    hipMemcpyHostToDevice));
        } else {
            hipSafeCall(hipMemcpy(u->handle, src, u->size, hipMemcpyHostToDevice));
        }
        // upload writes only the device buffer; host copy is now stale (matches OpenCLAllocator::upload).
        u->markHostCopyObsolete(true);
        u->markDeviceCopyObsolete(false);
    }

    void copy(UMatData* srcdata, UMatData* dstdata, int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dstofs[], const size_t dststep[], bool sync) const CV_OVERRIDE
    {
        if (!srcdata || !dstdata) return;

        const bool srcOnDevice = (srcdata->handle != nullptr);
        const bool dstOnDevice = (dstdata->handle != nullptr);

        hipMemcpyKind kind;
        const void* rawSrc = nullptr;
        void*       rawDst = nullptr;

        if (srcOnDevice && dstOnDevice) {
            kind = hipMemcpyDeviceToDevice;
            rawSrc = srcdata->handle;
            rawDst = dstdata->handle;
        } else if (srcOnDevice) {
            kind = hipMemcpyDeviceToHost;
            rawSrc = srcdata->handle;
            rawDst = dstdata->data;
        } else if (dstOnDevice) {
            kind = hipMemcpyHostToDevice;
            rawSrc = srcdata->data;
            rawDst = dstdata->handle;
        } else {
            return; // both CPU - generic Mat path handles this
        }

        if (!rawSrc || !rawDst) return;

        if (dims <= 2) {
            const uchar* src = (const uchar*)rawSrc;
            uchar*       dst = (uchar*)rawDst;
            if (dims == 2) {
                src += srcofs[0] * srcstep[0] + srcofs[1];
                dst += dstofs[0] * dststep[0] + dstofs[1];
            }
            if (sync || kind != hipMemcpyDeviceToDevice)
                hipSafeCall(hipMemcpy2D(dst, dststep[0], src, srcstep[0],
                                        sz[dims - 1], sz[0], kind));
            else
                hipSafeCall(hipMemcpy2DAsync(dst, dststep[0], src, srcstep[0],
                                             sz[dims - 1], sz[0], kind, 0));
        } else {
            hipSafeCall(hipMemcpy(rawDst, rawSrc, srcdata->size, kind));
        }

        dstdata->markHostCopyObsolete(dstOnDevice);
        dstdata->markDeviceCopyObsolete(!dstOnDevice);
    }
};

HipAllocator hipAllocatorInstance;

} // anonymous namespace

namespace cv { namespace hip {

bool isHipUMat(InputArray a)
{
    if (!a.isUMat()) return false;
    UMat u = a.getUMat();
    return u.u != nullptr && u.u->currAllocator == getHipAllocator();
}

}} // cv::hip


namespace cv { namespace hip {

static bool g_useHip = true;

CV_EXPORTS_W bool useHip()
{
    if (!g_useHip) return false;
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n == 0) {
        g_useHip = false;
        return false;
    }
    // Mat::copyTo/setTo fire CV_OCL_RUN before the HIP currAllocator check, which would feed a HIP buffer to OpenCL.
    cv::ocl::setUseOpenCL(false);
    return true;
}

CV_EXPORTS MatAllocator* getHipAllocator()
{
    return &hipAllocatorInstance;
}

}} // cv::hip
#endif

#ifndef HAVE_HIP
namespace cv { namespace hip {
bool isHipUMat(InputArray) { return false; }
bool useHip() { return false; }
MatAllocator* getHipAllocator() { return nullptr; }
}} // cv::hip
#endif

// Device management

int cv::hip::getHipEnabledDeviceCount()
{
#ifndef HAVE_HIP
    return 0;
#else
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err == hipErrorNoDevice)           return 0;
    if (err == hipErrorInsufficientDriver) return -1;
    hipSafeCall(err);
    return count;
#endif
}

void cv::hip::setDevice(int device)
{
#ifndef HAVE_HIP
    CV_UNUSED(device); throw_no_hip();
#else
    hipSafeCall(hipSetDevice(device));
#endif
}

int cv::hip::getDevice()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    int device = 0;
    hipSafeCall(hipGetDevice(&device));
    return device;
#endif
}

void cv::hip::resetDevice()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipSafeCall(hipDeviceReset());
#endif
}

bool cv::hip::deviceSupports(FeatureSet feature_set)
{
#ifndef HAVE_HIP
    CV_UNUSED(feature_set); return false;
#else
    if (getHipEnabledDeviceCount() <= 0) return false;
    hipDeviceProp_t prop;
    hipSafeCall(hipGetDeviceProperties(&prop, getDevice()));
    switch (feature_set) {
        case GLOBAL_ATOMICS:         return prop.arch.hasGlobalInt32Atomics != 0;
        case SHARED_ATOMICS:         return prop.arch.hasSharedInt32Atomics != 0;
        case NATIVE_DOUBLE:          return prop.arch.hasDoubles != 0;
        case WARP_SHUFFLE_FUNCTIONS: return prop.arch.hasWarpShuffle != 0;
        case DYNAMIC_PARALLELISM:    return prop.arch.hasDynamicParallelism != 0;
    }
    return false;
#endif
}

// TargetArchs

bool cv::hip::TargetArchs::builtWith(FeatureSet feature_set) { return deviceSupports(feature_set); }

bool cv::hip::TargetArchs::has(int major, int minor)
{
#ifndef HAVE_HIP
    CV_UNUSED(major); CV_UNUSED(minor); return false;
#else
    int count = getHipEnabledDeviceCount();
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t p; hipSafeCall(hipGetDeviceProperties(&p, i));
        if (p.major == major && p.minor == minor) return true;
    }
    return false;
#endif
}

bool cv::hip::TargetArchs::hasBin(int major, int minor) { return has(major, minor); }

bool cv::hip::TargetArchs::hasEqualOrGreater(int major, int minor)
{
#ifndef HAVE_HIP
    CV_UNUSED(major); CV_UNUSED(minor); return false;
#else
    int count = getHipEnabledDeviceCount(), ref = major * 10 + minor;
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t p; hipSafeCall(hipGetDeviceProperties(&p, i));
        if (p.major * 10 + p.minor >= ref) return true;
    }
    return false;
#endif
}

bool cv::hip::TargetArchs::hasEqualOrGreaterBin(int major, int minor)
{
    return hasEqualOrGreater(major, minor);
}

// DeviceInfo

#ifdef HAVE_HIP
static hipDeviceProp_t getDeviceProp(int id)
{
    hipDeviceProp_t p;
    hipSafeCall(hipGetDeviceProperties(&p, id));
    return p;
}
#endif

cv::hip::DeviceInfo::DeviceInfo()
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipSafeCall(hipGetDevice(&device_id_));
#endif
}

cv::hip::DeviceInfo::DeviceInfo(int device_id) : device_id_(device_id)
{
#ifndef HAVE_HIP
    CV_UNUSED(device_id); throw_no_hip();
#else
    CV_Assert(device_id_ >= 0 && device_id_ < getHipEnabledDeviceCount());
#endif
}

int cv::hip::DeviceInfo::deviceID() const { return device_id_; }

const char* cv::hip::DeviceInfo::name() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    static thread_local char buf[256];
    std::strncpy(buf, getDeviceProp(device_id_).name, 255);
    return buf;
#endif
}

#ifdef HAVE_HIP
#define DEVINFO_PROP(rettype, method, field) \
rettype cv::hip::DeviceInfo::method() const { \
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip(); \
    return (rettype)getDeviceProp(device_id_).field; \
}
#else
#define DEVINFO_PROP(rettype, method, field) \
rettype cv::hip::DeviceInfo::method() const { throw_no_hip(); }
#endif

DEVINFO_PROP(size_t, totalGlobalMem,           totalGlobalMem)
DEVINFO_PROP(size_t, sharedMemPerBlock,        sharedMemPerBlock)
DEVINFO_PROP(int,    regsPerBlock,             regsPerBlock)
DEVINFO_PROP(int,    warpSize,                 warpSize)
DEVINFO_PROP(size_t, memPitch,                 memPitch)
DEVINFO_PROP(int,    maxThreadsPerBlock,       maxThreadsPerBlock)
DEVINFO_PROP(int,    clockRate,                clockRate)
DEVINFO_PROP(size_t, totalConstMem,            totalConstMem)
DEVINFO_PROP(int,    majorVersion,             major)
DEVINFO_PROP(int,    minorVersion,             minor)
DEVINFO_PROP(size_t, textureAlignment,         textureAlignment)
DEVINFO_PROP(size_t, texturePitchAlignment,    texturePitchAlignment)
DEVINFO_PROP(int,    multiProcessorCount,      multiProcessorCount)
DEVINFO_PROP(int,    memoryClockRate,          memoryClockRate)
DEVINFO_PROP(int,    memoryBusWidth,           memoryBusWidth)
DEVINFO_PROP(int,    l2CacheSize,              l2CacheSize)
DEVINFO_PROP(int,    maxThreadsPerMultiProcessor, maxThreadsPerMultiProcessor)
DEVINFO_PROP(int,    maxTexture1D,             maxTexture1D)
DEVINFO_PROP(int,    maxTexture1DLinear,        maxTexture1DLinear)
DEVINFO_PROP(int,    pciBusID,                 pciBusID)
DEVINFO_PROP(int,    pciDeviceID,              pciDeviceID)
DEVINFO_PROP(int,    pciDomainID,              pciDomainID)
DEVINFO_PROP(int,    asicRevision,             asicRevision)

bool cv::hip::DeviceInfo::kernelExecTimeoutEnabled() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).kernelExecTimeoutEnabled != 0;
#endif
}
bool cv::hip::DeviceInfo::integrated() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).integrated != 0;
#endif
}
bool cv::hip::DeviceInfo::canMapHostMemory() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).canMapHostMemory != 0;
#endif
}
bool cv::hip::DeviceInfo::concurrentKernels() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).concurrentKernels != 0;
#endif
}
bool cv::hip::DeviceInfo::ECCEnabled() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).ECCEnabled != 0;
#endif
}
bool cv::hip::DeviceInfo::tccDriver() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).tccDriver != 0;
#endif
}
bool cv::hip::DeviceInfo::cooperativeLaunch() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).cooperativeLaunch != 0;
#endif
}
bool cv::hip::DeviceInfo::isLargeBar() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    if (getHipEnabledDeviceCount() <= 0) throw_no_hip();
    return getDeviceProp(device_id_).isLargeBar != 0;
#endif
}

DeviceInfo::ComputeMode cv::hip::DeviceInfo::computeMode() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    switch (getDeviceProp(device_id_).computeMode) {
        case 0: return ComputeModeDefault;
        case 1: return ComputeModeExclusive;
        case 2: return ComputeModeProhibited;
        case 3: return ComputeModeExclusiveProcess;
        default: return ComputeModeDefault;
    }
#endif
}

Vec3i cv::hip::DeviceInfo::maxThreadsDim() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipDeviceProp_t p = getDeviceProp(device_id_);
    return Vec3i(p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
#endif
}

Vec3i cv::hip::DeviceInfo::maxGridSize() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipDeviceProp_t p = getDeviceProp(device_id_);
    return Vec3i(p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
#endif
}

Vec2i cv::hip::DeviceInfo::maxTexture2D() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipDeviceProp_t p = getDeviceProp(device_id_);
    return Vec2i(p.maxTexture2D[0], p.maxTexture2D[1]);
#endif
}

Vec3i cv::hip::DeviceInfo::maxTexture3D() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    hipDeviceProp_t p = getDeviceProp(device_id_);
    return Vec3i(p.maxTexture3D[0], p.maxTexture3D[1], p.maxTexture3D[2]);
#endif
}

cv::String cv::hip::DeviceInfo::gcnArchName() const
{
#ifndef HAVE_HIP
    throw_no_hip();
#else
    static thread_local char buf[256];
    std::strncpy(buf, getDeviceProp(device_id_).gcnArchName, 255);
    return String(buf);
#endif
}

void cv::hip::DeviceInfo::queryMemory(size_t& totalMemory, size_t& freeMemory) const
{
#ifndef HAVE_HIP
    CV_UNUSED(totalMemory); CV_UNUSED(freeMemory); throw_no_hip();
#else
    int prev = getDevice();
    if (prev != device_id_) setDevice(device_id_);
    hipSafeCall(hipMemGetInfo(&freeMemory, &totalMemory));
    if (prev != device_id_) setDevice(prev);
#endif
}

size_t cv::hip::DeviceInfo::freeMemory()  const { size_t t = 0, f = 0; queryMemory(t, f); return f; }
size_t cv::hip::DeviceInfo::totalMemory() const { size_t t = 0, f = 0; queryMemory(t, f); return t; }

bool cv::hip::DeviceInfo::supports(FeatureSet feature_set) const
{
#ifndef HAVE_HIP
    CV_UNUSED(feature_set); return false;
#else
    hipDeviceProp_t p = getDeviceProp(device_id_);
    switch (feature_set) {
        case GLOBAL_ATOMICS:         return p.arch.hasGlobalInt32Atomics != 0;
        case SHARED_ATOMICS:         return p.arch.hasSharedInt32Atomics != 0;
        case NATIVE_DOUBLE:          return p.arch.hasDoubles != 0;
        case WARP_SHUFFLE_FUNCTIONS: return p.arch.hasWarpShuffle != 0;
        case DYNAMIC_PARALLELISM:    return p.arch.hasDynamicParallelism != 0;
    }
    return false;
#endif
}

bool cv::hip::DeviceInfo::isCompatible() const
{
#ifndef HAVE_HIP
    return false;
#else
    return getHipEnabledDeviceCount() > 0 && device_id_ < getHipEnabledDeviceCount();
#endif
}

// Print functions

void cv::hip::printHipDeviceInfo(int device)
{
#ifndef HAVE_HIP
    CV_UNUSED(device); throw_no_hip();
#else
    hipDeviceProp_t p;
    hipSafeCall(hipGetDeviceProperties(&p, device));
    CV_LOG_INFO(NULL, cv::format("Device %d: \"%s\"", device, p.name));
    CV_LOG_INFO(NULL, cv::format("  HIP Compute Capability:              %d.%d", p.major, p.minor));
    CV_LOG_INFO(NULL, cv::format("  GCN Architecture:                    %s", p.gcnArchName));
    CV_LOG_INFO(NULL, cv::format("  Total global memory:                 %.0f MB", (double)p.totalGlobalMem / (1 << 20)));
    CV_LOG_INFO(NULL, cv::format("  Shared memory per block:             %zu bytes", p.sharedMemPerBlock));
    CV_LOG_INFO(NULL, cv::format("  Registers per block:                 %d", p.regsPerBlock));
    CV_LOG_INFO(NULL, cv::format("  Warp size:                           %d", p.warpSize));
    CV_LOG_INFO(NULL, cv::format("  Max threads per block:               %d", p.maxThreadsPerBlock));
    CV_LOG_INFO(NULL, cv::format("  Max block dimensions:                [%d, %d, %d]",
                                 p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]));
    CV_LOG_INFO(NULL, cv::format("  Max grid dimensions:                 [%d, %d, %d]",
                                 p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]));
    CV_LOG_INFO(NULL, cv::format("  Clock rate:                          %.2f GHz", p.clockRate * 1e-6));
    CV_LOG_INFO(NULL, cv::format("  Memory clock rate:                   %.2f GHz", p.memoryClockRate * 1e-6));
    CV_LOG_INFO(NULL, cv::format("  Memory bus width:                    %d-bit", p.memoryBusWidth));
    CV_LOG_INFO(NULL, cv::format("  L2 cache size:                       %d bytes", p.l2CacheSize));
    CV_LOG_INFO(NULL, cv::format("  Multiprocessors:                     %d", p.multiProcessorCount));
    CV_LOG_INFO(NULL, cv::format("  Max threads per multiprocessor:      %d", p.maxThreadsPerMultiProcessor));
    CV_LOG_INFO(NULL, cv::format("  Concurrent kernels:                  %s", p.concurrentKernels ? "Yes" : "No"));
    CV_LOG_INFO(NULL, cv::format("  ECC enabled:                         %s", p.ECCEnabled ? "Yes" : "No"));
    CV_LOG_INFO(NULL, cv::format("  Cooperative launch:                  %s", p.cooperativeLaunch ? "Yes" : "No"));
    CV_LOG_INFO(NULL, cv::format("  Large bar:                           %s", p.isLargeBar ? "Yes" : "No"));
    CV_LOG_INFO(NULL, cv::format("  PCI Bus/Device/Domain:               %d/%d/%d",
                                 p.pciBusID, p.pciDeviceID, p.pciDomainID));
#endif
}
