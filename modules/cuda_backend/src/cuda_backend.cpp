// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
// =============================================================
// cuda_backend.cpp
// GPU HAL CUDA backend plugin.
//
// Implements cv::hal::Backend for CUDA, plus a CudaAllocator so
// that UMat results stay RESIDENT in GPU VRAM between operations.
// A chain like resize -> GaussianBlur -> cvtColor crosses the PCIe
// bus only twice (one upload, one download) — the intermediates
// never leave the device.
//
// Loaded at runtime by hal_backend.cpp via dlopen.
// User code never calls this file directly.
// =============================================================

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

// =============================================================
// Device memory pool — reuse VRAM blocks by exact size instead of
// cudaMalloc/cudaFree per call. cudaMalloc/cudaFree are expensive
// (~100us, implicit device sync); reuse makes per-op alloc ~free.
// Buffers are kept for the plugin's lifetime (never cudaFree'd).
// =============================================================
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

// =============================================================
// CudaAllocator — a cv::MatAllocator that puts UMat memory in
// GPU VRAM and downloads to the host only on demand (map()).
//
// Field convention on UMatData:
//   handle = CUDA device pointer (the VRAM)
//   data   = host shadow pointer (0 until first CPU access)
//   flags  = HOST/DEVICE_COPY_OBSOLETE track which side is current
// =============================================================
class CudaAllocator CV_FINAL : public MatAllocator
{
public:
    // create() path: data==0, we cudaMalloc fresh VRAM.
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

        void* dev = nullptr;
        if (data)
            dev = data;                      // wrap externally-owned device mem
        else
            dev = poolAlloc(total);          // pooled VRAM (reused, not malloc'd)

        UMatData* u = new UMatData(this);
        u->data      = 0;                    // no host copy yet
        u->origdata  = 0;
        u->handle    = dev;                  // device pointer lives here
        u->size      = total;
        u->flags     = data ? UMatData::USER_ALLOCATED
                            : static_cast<UMatData::MemoryFlag>(0);
        u->markHostCopyObsolete(true);       // device is the source of truth
        u->gpuBackend = getCudaBackendInstance();
        return u;
    }

    bool allocate(UMatData* u, AccessFlag /*accessFlags*/,
                  UMatUsageFlags /*usageFlags*/) const CV_OVERRIDE
    {
        // header already carries our device memory
        return u != nullptr;
    }

    void deallocate(UMatData* u) const CV_OVERRIDE
    {
        if (!u) return;
        if (u->handle && !(u->flags & UMatData::USER_ALLOCATED))
            poolFree(u->size, u->handle);    // return VRAM to pool (no cudaFree)
        if (u->data)
            fastFree(u->data);
        delete u;
    }

    // CPU wants to touch the data — make a host copy available.
    void map(UMatData* u, AccessFlag accessFlags) const CV_OVERRIDE
    {
        if (!u) return;
        if (u->data == 0)
            u->data = (uchar*)fastMalloc(u->size);   // host shadow
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

// =============================================================
// extractGpuMat — view a resident UMat's device memory as a
// GpuMat (zero copy). Reads the device pointer from u->handle.
// =============================================================
static cuda::GpuMat extractGpuMat(const UMat& u)
{
    CV_Assert(u.u != nullptr && u.u->handle != nullptr);
    return cuda::GpuMat(u.rows, u.cols, u.type(),
                        u.u->handle, u.step[0]);
}

// =============================================================
// makeResidentOutput — allocate a UMat in VRAM with the given
// geometry, return a GpuMat view of it for the kernel to write
// into in-place. The UMat is returned via 'out'.
// =============================================================
static cuda::GpuMat makeResidentOutput(UMat& out, int rows, int cols, int type)
{
    out.allocator = getCudaAllocator();
    out.create(rows, cols, type);            // cudaMalloc via our allocator
    out.u->markHostCopyObsolete(true);       // device will hold the result
    return extractGpuMat(out);
}

// =============================================================
// CudaBackend
// =============================================================
class CudaBackend CV_FINAL : public Backend
{
public:
    // Each op is its own typed method. Returns false (fall through to CPU)
    // when the source isn't a resident CUDA UMat or the contrib module that
    // provides the kernel isn't built.

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
        // Cache the filter — createGaussianFilter is expensive and the same
        // (type, ksize, sigma) recurs across frames.
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

// =============================================================
// Factory — dlopen entry point. extern "C" => no name mangling.
// =============================================================
extern "C" CV_EXPORTS cv::hal::Backend* cv_hal_createCudaBackend();

cv::hal::Backend* cv_hal_createCudaBackend()
{
    return cv::hal::getCudaBackendInstance();
}
