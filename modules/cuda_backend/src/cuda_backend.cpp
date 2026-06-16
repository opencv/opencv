// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
// GPU HAL CUDA backend plugin: cv::hal::Backend + a CudaAllocator that keeps
// UMat results resident in VRAM. Loaded at runtime by hal_backend.cpp via dlopen.

#include "opencv2/core.hpp"
#include "opencv2/core/hal/backend.hpp"
#include "opencv2/core/hal/backend_registry.hpp"
#include "opencv2/core/cuda.hpp"

#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef HAVE_OPENCV_CUDAWARPING
#include "opencv2/cudawarping.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAARITHM
#include "opencv2/cudaarithm.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAFILTERS
#include "opencv2/cudafilters.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAIMGPROC
#include "opencv2/cudaimgproc.hpp"
#endif

namespace cv { namespace hal {

class CudaBackend;                       // fwd
static Backend* getCudaBackendInstance();

// Device memory pool: reuse VRAM blocks by size (cudaMalloc/Free are expensive
// and sync the device). Buffers are kept for the plugin's lifetime.
namespace {
struct DevicePool {
    std::mutex mtx;
    std::unordered_map<size_t, std::vector<void*> > freelist;
};
DevicePool& devicePool() { static DevicePool p; return p; }

void* poolAlloc(size_t sz)
{
    DevicePool& p = devicePool();
    std::lock_guard<std::mutex> lk(p.mtx);
    std::vector<void*>& v = p.freelist[sz];
    if (!v.empty()) { void* d = v.back(); v.pop_back(); return d; }
    void* d = nullptr;
    CV_Assert(cudaMalloc(&d, sz) == cudaSuccess);
    return d;
}
void poolFree(size_t sz, void* d)
{
    if (!d) return;
    DevicePool& p = devicePool();
    std::lock_guard<std::mutex> lk(p.mtx);
    p.freelist[sz].push_back(d);
}
} // anonymous namespace

// MatAllocator that puts UMat memory in VRAM, host copy on demand via map().
// UMatData convention: handle = device ptr, data = host shadow (0 until used).
class CudaAllocator CV_FINAL : public MatAllocator
{
public:
    UMatData* allocate(int dims, const int* sizes, int type,
                       void* data, size_t* step,
                       AccessFlag /*flags*/,
                       UMatUsageFlags /*usageFlags*/) const CV_OVERRIDE
    {
        size_t esz = CV_ELEM_SIZE(type);
        int rows = dims >= 1 ? sizes[0] : 1;
        int cols = dims >= 2 ? sizes[1] : 1;
        if (step)
        {
            step[dims - 1] = esz;            // last dim must equal elemSize
            if (dims >= 2) step[0] = (size_t)cols * esz;
        }
        size_t total = (size_t)rows * (size_t)cols * esz;

        void* dev = data ? data : poolAlloc(total);

        UMatData* u = new UMatData(this);
        u->data      = 0;
        u->origdata  = 0;
        u->handle    = dev;
        u->size      = total;
        u->flags     = data ? UMatData::USER_ALLOCATED
                            : static_cast<UMatData::MemoryFlag>(0);
        u->markHostCopyObsolete(true);
        u->gpuBackend = getCudaBackendInstance();
        return u;
    }

    bool allocate(UMatData* u, AccessFlag /*accessFlags*/,
                  UMatUsageFlags /*usageFlags*/) const CV_OVERRIDE
    {
        return u != nullptr;
    }

    void deallocate(UMatData* u) const CV_OVERRIDE
    {
        if (!u) return;
        if (u->handle && !(u->flags & UMatData::USER_ALLOCATED))
            poolFree(u->size, u->handle);
        if (u->data)
            fastFree(u->data);
        delete u;
    }

    void map(UMatData* u, AccessFlag accessFlags) const CV_OVERRIDE
    {
        if (!u) return;
        if (u->data == 0)
            u->data = (uchar*)fastMalloc(u->size);
        if ((accessFlags & ACCESS_READ) && u->hostCopyObsolete())
        {
            CV_Assert(cudaMemcpy(u->data, u->handle, u->size,
                                 cudaMemcpyDeviceToHost) == cudaSuccess);
            u->markHostCopyObsolete(false);
        }
        if (accessFlags & ACCESS_WRITE)
            u->markDeviceCopyObsolete(true);
    }

    void unmap(UMatData* u) const CV_OVERRIDE
    {
        if (!u) return;
        if (u->deviceCopyObsolete() && u->data && u->handle)
        {
            CV_Assert(cudaMemcpy(u->handle, u->data, u->size,
                                 cudaMemcpyHostToDevice) == cudaSuccess);
            u->markDeviceCopyObsolete(false);
        }
    }
};

static CudaAllocator* getCudaAllocator()
{
    static CudaAllocator alloc;
    return &alloc;
}

// View a resident UMat's device memory as a GpuMat (zero copy).
static cuda::GpuMat extractGpuMat(const UMat& u)
{
    CV_Assert(u.u != nullptr && u.u->handle != nullptr);
    return cuda::GpuMat(u.rows, u.cols, u.type(),
                        u.u->handle, u.step[0]);
}

// Allocate a VRAM UMat (returned via 'out') and view it as a GpuMat written in-place.
static cuda::GpuMat makeResidentOutput(UMat& out, int rows, int cols, int type)
{
    out.allocator = getCudaAllocator();
    out.create(rows, cols, type);
    out.u->markHostCopyObsolete(true);
    return extractGpuMat(out);
}

// One typed method per op. Returns false (fall through to CPU) when the source
// isn't a resident CUDA UMat or the needed contrib module isn't built.
class CudaBackend CV_FINAL : public Backend
{
public:
    bool resize(InputArray src, OutputArray dst, Size dsize,
                double inv_scale_x, double inv_scale_y, int interpolation) CV_OVERRIDE
    {
#ifdef HAVE_OPENCV_CUDAWARPING
        UMat su = src.getUMat();
        if (!su.u || !su.u->handle) return false;
        cuda::GpuMat gsrc = extractGpuMat(su);
        Size ds = dsize;
        if (ds.empty())
            ds = Size(saturate_cast<int>(gsrc.cols * inv_scale_x),
                      saturate_cast<int>(gsrc.rows * inv_scale_y));
        UMat out;
        cuda::GpuMat gdst = makeResidentOutput(out, ds.height, ds.width, su.type());
        cuda::resize(gsrc, gdst, ds, inv_scale_x, inv_scale_y, interpolation);
        dst.assign(out);
        return true;
#else
        (void)src; (void)dst; (void)dsize;
        (void)inv_scale_x; (void)inv_scale_y; (void)interpolation;
        return false;
#endif
    }

    bool gaussianBlur(InputArray src, OutputArray dst, Size ksize,
                      double sigma1, double sigma2) CV_OVERRIDE
    {
#ifdef HAVE_OPENCV_CUDAFILTERS
        if (ksize.width <= 1 || ksize.height <= 1) return false;  // 1x1 -> CPU
        UMat su = src.getUMat();
        if (!su.u || !su.u->handle) return false;
        cuda::GpuMat gsrc = extractGpuMat(su);
        UMat out;
        cuda::GpuMat gdst = makeResidentOutput(out, gsrc.rows, gsrc.cols, su.type());
        // Cache the filter — createGaussianFilter is expensive, config recurs.
        static cv::Ptr<cuda::Filter> cached;
        static int    cT = -1, cKw = -1, cKh = -1;
        static double cS1 = -1, cS2 = -1;
        int t = su.type();
        if (cached.empty() || cT != t || cKw != ksize.width ||
            cKh != ksize.height || cS1 != sigma1 || cS2 != sigma2)
        {
            cached = cuda::createGaussianFilter(t, t, ksize, sigma1, sigma2);
            cT = t; cKw = ksize.width; cKh = ksize.height;
            cS1 = sigma1; cS2 = sigma2;
        }
        cached->apply(gsrc, gdst);
        dst.assign(out);
        return true;
#else
        (void)src; (void)dst; (void)ksize; (void)sigma1; (void)sigma2;
        return false;
#endif
    }

    bool cvtColor(InputArray src, OutputArray dst, int code, int dcn) CV_OVERRIDE
    {
#ifdef HAVE_OPENCV_CUDAIMGPROC
        UMat su = src.getUMat();
        if (!su.u || !su.u->handle) return false;
        cuda::GpuMat gsrc = extractGpuMat(su);
        int outcn = dcn;
        if (outcn <= 0)
            outcn = (code == COLOR_BGR2GRAY || code == COLOR_RGB2GRAY)
                        ? 1 : gsrc.channels();
        int outType = CV_MAKETYPE(su.depth(), outcn);
        UMat out;
        cuda::GpuMat gdst = makeResidentOutput(out, gsrc.rows, gsrc.cols, outType);
        cuda::cvtColor(gsrc, gdst, code, outcn);
        dst.assign(out);
        return true;
#else
        (void)src; (void)dst; (void)code; (void)dcn;
        return false;
#endif
    }

    bool threshold(InputArray src, OutputArray dst, double thresh,
                   double maxval, int type) CV_OVERRIDE
    {
#ifdef HAVE_OPENCV_CUDAARITHM
        UMat su = src.getUMat();
        if (!su.u || !su.u->handle) return false;
        cuda::GpuMat gsrc = extractGpuMat(su);
        UMat out;
        cuda::GpuMat gdst = makeResidentOutput(out, gsrc.rows, gsrc.cols, su.type());
        cuda::threshold(gsrc, gdst, thresh, maxval, type);
        dst.assign(out);
        return true;
#else
        (void)src; (void)dst; (void)thresh; (void)maxval; (void)type;
        return false;
#endif
    }

    MatAllocator* allocator() const CV_OVERRIDE { return getCudaAllocator(); }
};

static Backend* getCudaBackendInstance()
{
    static CudaBackend backend;
    return &backend;
}

}} // cv::hal

// Factory — dlopen entry point; extern "C" so dlsym finds an unmangled name.
extern "C" CV_EXPORTS cv::hal::Backend* cv_hal_createCudaBackend();

cv::hal::Backend* cv_hal_createCudaBackend()
{
    return cv::hal::getCudaBackendInstance();
}
