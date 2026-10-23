// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: the CUDA backend.
//
// Discrete GPUs: device memory from the stream-ordered pool (cudaMallocAsync), which keeps freed
// memory for reuse; the CPU works on a pinned copy, so uploads from it run asynchronously.
// Integrated GPUs from Xavier on (compute capability 7.2): pinned memory mapped into the GPU,
// which the CPU and the GPU both address and which is I/O coherent, so nothing is ever copied.

#include "../precomp.hpp"
#include "accel_impl.hpp"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include "../cuda/accel_kernels.hpp"
#include "opencv2/core/utils/logger.hpp"

#include <memory>

namespace cv { namespace accel { namespace impl {

namespace {

void check(cudaError_t e, const char* what)
{
    if (e == cudaSuccess)
        return;
    cudaGetLastError();     // a non-sticky error must not reach the thread's next call
    CV_Error_(Error::GpuApiCallError, ("accel(cuda): %s failed: %s", what, cudaGetErrorString(e)));
}

//! makes @p device current for the calling thread while it lives
class DeviceGuard
{
public:
    explicit DeviceGuard(int device) : prev_(-1)
    {
        int current = -1;
        check(cudaGetDevice(&current), "cudaGetDevice");
        if (current != device)
        {
            check(cudaSetDevice(device), "cudaSetDevice");
            prev_ = current;
        }
    }
    ~DeviceGuard()
    {
        if (prev_ >= 0)
            cudaSetDevice(prev_);
    }
private:
    int prev_;
};

class CudaEvent CV_FINAL : public Event
{
public:
    explicit CudaEvent(cudaEvent_t e) : e_(e) {}
    ~CudaEvent() { cudaEventDestroy(e_); }

    bool query() CV_OVERRIDE
    {
        const cudaError_t r = cudaEventQuery(e_);
        if (r == cudaErrorNotReady)
        {
            cudaGetLastError();
            return false;
        }
        check(r, "cudaEventQuery");
        return true;
    }

    void wait() CV_OVERRIDE
    {
        check(cudaEventSynchronize(e_), "cudaEventSynchronize");
    }

    cudaEvent_t e_;
};

class CudaStream CV_FINAL : public Stream
{
public:
    explicit CudaStream(int device) : device_(device), s_(0)
    {
        DeviceGuard g(device);
        check(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }
    ~CudaStream() { cudaStreamDestroy(s_); }   // queued work still completes

    Ptr<Event> record() CV_OVERRIDE
    {
        DeviceGuard g(device_);
        cudaEvent_t e = 0;
        check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreateWithFlags");
        const cudaError_t r = cudaEventRecord(e, s_);
        if (r != cudaSuccess)
        {
            cudaEventDestroy(e);
            check(r, "cudaEventRecord");
        }
        return makePtr<CudaEvent>(e);
    }

    bool waitEvent(const Ptr<Event>& event) CV_OVERRIDE
    {
        CudaEvent* ce = dynamic_cast<CudaEvent*>(event.get());
        if (!ce)
            return false;
        DeviceGuard g(device_);
        check(cudaStreamWaitEvent(s_, ce->e_, 0), "cudaStreamWaitEvent");  // also across CUDA devices
        return true;
    }

    void synchronize() CV_OVERRIDE
    {
        check(cudaStreamSynchronize(s_), "cudaStreamSynchronize");
    }

    const int device_;
    cudaStream_t s_;
};

CudaStream& cudaStreamOf(Stream& s)
{
    CudaStream* cs = dynamic_cast<CudaStream*>(&s);
    CV_Assert(cs && "accel(cuda): the stream belongs to another backend");
    return *cs;
}

class CudaCoreOps CV_FINAL : public CoreOps
{
public:
    bool flip(const UMat& src, UMat& dst, int flipCode, Stream& stream) CV_OVERRIDE
    {
        const int esz = (int)src.elemSize();
        if (src.dims != 2 || !cuda_kernels::supportedElementSize(esz))
            return false;
        CudaStream& s = cudaStreamOf(stream);
        DeviceGuard g(s.device_);
        check(cuda_kernels::flip((const unsigned char*)devicePtr(src), src.step[0],
                                 (unsigned char*)devicePtr(dst), dst.step[0],
                                 src.rows, src.cols, esz, flipCode, s.s_), "flip");
        return true;
    }

    bool transpose(const UMat& src, UMat& dst, Stream& stream) CV_OVERRIDE
    {
        const int esz = (int)src.elemSize();
        if (src.dims != 2 || !cuda_kernels::supportedElementSize(esz))
            return false;
        CudaStream& s = cudaStreamOf(stream);
        DeviceGuard g(s.device_);
        check(cuda_kernels::transpose((const unsigned char*)devicePtr(src), src.step[0],
                                      (unsigned char*)devicePtr(dst), dst.step[0],
                                      src.rows, src.cols, esz, s.s_), "transpose");
        return true;
    }
};

class CudaBackend CV_FINAL : public Backend
{
public:
    CudaBackend()
    {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess)
        {
            cudaGetLastError();     // no driver or no device
            n = 0;
        }
        for (int i = 0; i < n; i++)
        {
            std::unique_ptr<Dev> d(new Dev());
            check(cudaGetDeviceProperties(&d->prop, i), "cudaGetDeviceProperties");
            const int cc = d->prop.major * 10 + d->prop.minor;
            // before Xavier, pinned memory is uncached for the CPU: copies are faster there
            d->shared = d->prop.integrated && d->prop.canMapHostMemory && cc >= 72;
            int pools = 0;
            if (cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, i) != cudaSuccess)
            {
                cudaGetLastError();
                pools = 0;
            }
            d->pools = pools != 0 && !d->shared;
            devs_.push_back(std::move(d));
        }
    }

    String name() const CV_OVERRIDE { return "cuda"; }
    int deviceCount() const CV_OVERRIDE { return (int)devs_.size(); }
    String deviceDescription(int device) const CV_OVERRIDE { return devs_[device]->prop.name; }
    bool sharesHostMemory(int device) const CV_OVERRIDE { return devs_[device]->shared; }

    void* allocate(int device, size_t size) CV_OVERRIDE
    {
        Dev& d = *devs_[device];
        DeviceGuard g(device);
        void* p = 0;
        cudaError_t r;
        if (d.shared)
            r = cudaHostAlloc(&p, size, cudaHostAllocMapped | cudaHostAllocPortable);  // one address for CPU and GPU (UVA)
        else if (d.pools)
        {
            cudaStream_t s = allocStream(device);
            r = cudaMallocAsync(&p, size, s);
            if (r == cudaSuccess)
                r = cudaStreamSynchronize(s);   // from now on usable on every stream
        }
        else
            r = cudaMalloc(&p, size);
        if (r == cudaErrorMemoryAllocation)
        {
            cudaGetLastError();
            return 0;
        }
        check(r, "allocate");
        return p;
    }

    void deallocate(int device, void* ptr) CV_OVERRIDE
    {
        // called while UMats are released: report failures, never throw
        try
        {
            Dev& d = *devs_[device];
            DeviceGuard g(device);
            const cudaError_t r = d.shared ? cudaFreeHost(ptr)
                                : d.pools ? cudaFreeAsync(ptr, allocStream(device))
                                : cudaFree(ptr);
            if (r != cudaSuccess)
            {
                cudaGetLastError();
                CV_LOG_WARNING(NULL, "accel(cuda): freeing device memory failed: " << cudaGetErrorString(r));
            }
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_WARNING(NULL, "accel(cuda): freeing device memory failed: " << e.what());
        }
    }

    void* allocateHost(int device, size_t size) CV_OVERRIDE
    {
        DeviceGuard g(device);
        void* p = 0;
        if (cudaHostAlloc(&p, size, cudaHostAllocPortable) != cudaSuccess)
        {
            cudaGetLastError();
            return 0;
        }
        return p;
    }

    void deallocateHost(int, void* ptr) CV_OVERRIDE
    {
        if (cudaFreeHost(ptr) != cudaSuccess)
            cudaGetLastError();
    }

    void copy(int device, CopyKind kind, void* dst, size_t dstStep, const void* src, size_t srcStep,
              size_t widthBytes, size_t rows, Stream& stream, bool hostStable) CV_OVERRIDE
    {
        CudaStream& s = cudaStreamOf(stream);
        DeviceGuard g(device);
        check(cudaMemcpy2DAsync(dst, dstStep, src, srcStep, widthBytes, rows, cudaMemcpyDefault, s.s_),
              "cudaMemcpy2DAsync");
        if (kind == COPY_HOST_TO_DEVICE && !hostStable)
        {
            // the caller's memory may change once this returns, and pinned memory is read asynchronously
            Ptr<Event> e = s.record();
            e->wait();
        }
    }

    Ptr<Stream> createStream(int device) CV_OVERRIDE
    {
        return makePtr<CudaStream>(device);
    }

    Ops* ops(const String& module) CV_OVERRIDE
    {
        return module == CoreOps::module() ? &core_ : 0;
    }

private:
    struct Dev
    {
        Dev() : shared(false), pools(false), allocStream_(0) {}
        cudaDeviceProp prop;
        bool shared;
        bool pools;
        std::once_flag once;
        cudaStream_t allocStream_;
    };

    //! where pool memory is allocated and freed; created with the device's context on first use
    cudaStream_t allocStream(int device)
    {
        Dev& d = *devs_[device];
        std::call_once(d.once, [&]() {
            DeviceGuard g(device);
            cudaMemPool_t pool;
            if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess)
            {
                // keep freed memory in the pool instead of returning it to the driver at each synchronization
                uint64_t threshold = UINT64_MAX;
                if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold) != cudaSuccess)
                    cudaGetLastError();
            }
            else
                cudaGetLastError();
            check(cudaStreamCreateWithFlags(&d.allocStream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        });
        return d.allocStream_;
    }

    std::vector<std::unique_ptr<Dev> > devs_;
    CudaCoreOps core_;
};

} // namespace

Ptr<Backend> createCudaBackend()
{
    return makePtr<CudaBackend>();
}

}}} // namespace cv::accel::impl

#endif // HAVE_CUDA
