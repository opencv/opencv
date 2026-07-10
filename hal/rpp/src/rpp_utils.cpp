/**
 * simoncatbot-opencv RPP HAL - Unified Utilities Implementation
 */

#include "rpp_hal_utils.hpp"
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <unistd.h>

// RPP transitively includes hip_runtime_api.h, so hip symbols are available
// when RPP_BACKEND_HIP is defined. Always include the API header for
// type/function declarations; the actual HIP driver code only runs under
// the guard.
#include <hip/hip_runtime_api.h>

#ifdef RPP_BACKEND_HIP
#include <rpp/rppt_tensor_bitwise_operations.h>
#endif

namespace cv { namespace hal { namespace rpp {

namespace {

// RPP 3.x writes internal HIP errors to stderr. Redirect it to /dev/null
// transiently so user applications don't see benign messages.
class StderrSilencer {
public:
    StderrSilencer() {
        old_ = dup(STDERR_FILENO);
        if (old_ >= 0) {
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) {
                dup2(devnull, STDERR_FILENO);
                close(devnull);
            }
        }
    }
    ~StderrSilencer() {
        if (old_ >= 0) {
            dup2(old_, STDERR_FILENO);
            close(old_);
        }
    }
private:
    int old_;
};

inline bool checkHip(hipError_t err) {
    return err == hipSuccess;
}

// Simple device buffer pool keyed by size. Keeps a few freed buffers alive
// so that short chains of RPP ops (resize -> flip -> boxFilter) reuse
// device memory instead of paying hipMalloc/hipFree on every HAL call.
class DeviceBufferPool {
public:
    static DeviceBufferPool& instance() {
        static DeviceBufferPool pool;
        return pool;
    }

    void* acquire(size_t bytes) {
        if (!enabled_) return nullptr;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = buffers_.lower_bound(bytes);
        while (it != buffers_.end()) {
            if (it->first < bytes * 2) {
                void* p = it->second;
                buffers_.erase(it);
                return p;
            }
            ++it;
        }
        return nullptr;
    }

    void release(void* p, size_t bytes) {
        if (!enabled_ || !p) {
            if (p) (void)hipFree(p);
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffers_.size() >= maxSize_) {
            // evict smallest
            auto it = buffers_.begin();
            (void)hipFree(it->second);
            buffers_.erase(it);
        }
        buffers_.emplace(bytes, p);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& kv : buffers_) {
            (void)hipFree(kv.second);
        }
        buffers_.clear();
    }

    void setEnabled(bool enabled) {
        if (!enabled) {
            clear();
        }
        enabled_ = enabled;
    }

    bool enabled() const { return enabled_; }

private:
    DeviceBufferPool() : enabled_(true), maxSize_(16) {}
    ~DeviceBufferPool() { clear(); }

    bool enabled_;
    size_t maxSize_;
    std::mutex mutex_;
    std::multimap<size_t, void*> buffers_;
};

} // namespace

// =========================================================================
// Pool public API
// =========================================================================

void releaseHipPool() {
    #ifdef RPP_BACKEND_HIP
    DeviceBufferPool::instance().clear();
    #endif
}

void setHipPoolingEnabled(bool enabled) {
    #ifdef RPP_BACKEND_HIP
    DeviceBufferPool::instance().setEnabled(enabled);
    #else
    (void)enabled;
    #endif
}

bool isHipPoolingEnabled() {
    #ifdef RPP_BACKEND_HIP
    return DeviceBufferPool::instance().enabled();
    #else
    return false;
    #endif
}

// =========================================================================
// Availability checks
// =========================================================================

bool isRppGpuAvailable() {
    #ifdef RPP_BACKEND_HIP
    const char* force = getenv("OPENCV_RPP_FORCE_GPU");
    if (force && (strcmp(force, "1") == 0 || strcmp(force, "yes") == 0 || strcmp(force, "true") == 0)) {
        int deviceCount = 0;
        return (hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0);
    }

    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess || deviceCount <= 0) {
        return false;
    }

    // RPP 3.x HIP backend on some systems leaves a sticky async
    // "illegal memory access" error after the first operation, even though
    // the output is correct. Probe with a tiny real operation so that, if
    // this happens, we report the GPU path as unavailable and the HAL falls
    // back to the RPP HOST path or OpenCV native.
    static bool probed = false;
    static bool usable = false;
    if (probed) {
        return usable;
    }
    probed = true;

    StderrSilencer silence;
    (void)silence;

    rppHandle_t handle = nullptr;
    rppStatus_t status = rppCreate(&handle, 1, 0, nullptr, RPP_HIP_BACKEND);
    if (status == rppStatusSuccess && handle) {
        void* d_a = nullptr;
        void* d_b = nullptr;
        void* d_d = nullptr;
        bool ok = (hipMalloc(&d_a, 1) == hipSuccess) &&
                  (hipMalloc(&d_b, 1) == hipSuccess) &&
                  (hipMalloc(&d_d, 1) == hipSuccess);
        if (ok) {
            RpptDesc desc{};
            desc.numDims = 4; desc.dataType = U8;
            desc.n = 1; desc.c = 1; desc.h = 1; desc.w = 1;
            desc.layout = NHWC;
            desc.strides.wStride = 1; desc.strides.hStride = 1;
            desc.strides.cStride = 1; desc.strides.nStride = 1;
            RpptROI roi{};
            roi.xywhROI.xy.x = 0; roi.xywhROI.xy.y = 0;
            roi.xywhROI.roiWidth = 1; roi.xywhROI.roiHeight = 1;

            (void)rppt_bitwise_and(d_a, d_b, &desc, d_d, &desc, &roi, XYWH, handle, RPP_HIP_BACKEND);
            (void)hipDeviceSynchronize();
            hipError_t last = hipGetLastError();
            usable = (last == hipSuccess);
        }
        (void)hipFree(d_a); (void)hipFree(d_b); (void)hipFree(d_d);
        (void)hipGetLastError();
    }
    return usable;
    #else
    return false;
    #endif
}

bool isRppCpuAvailable() {
    StderrSilencer silence;
    (void)silence;
    rppHandle_t handle;
    rppStatus_t status = rppCreate(&handle, 1, 0, nullptr, RPP_HOST_BACKEND);
    if (status != rppStatusSuccess) {
        return false;
    }
    rppDestroy(handle, RPP_HOST_BACKEND);
    return true;
}

bool isRppAvailable() {
    return isRppGpuAvailable() || isRppCpuAvailable();
}

// =========================================================================
// Descriptor builders
// =========================================================================

void buildRppDescNHWC(RpptDesc& desc, int width, int height, int channels, int depth) {
    std::memset(&desc, 0, sizeof(RpptDesc));
    desc.numDims = 4;
    desc.offsetInBytes = 0;
    desc.dataType = cvDepthToRppDataType(depth);
    desc.n = 1;
    desc.c = channels;
    desc.h = height;
    desc.w = width;
    desc.layout = NHWC;
    desc.strides.wStride = channels;
    desc.strides.hStride = width * channels;
    desc.strides.cStride = 1;
    desc.strides.nStride = desc.strides.hStride * height;
}

void buildFullRoi(RpptROI& roi, int width, int height) {
    roi.xywhROI.xy.x = 0;
    roi.xywhROI.xy.y = 0;
    roi.xywhROI.roiWidth  = width;
    roi.xywhROI.roiHeight = height;
}

// =========================================================================
// Data type conversion
// =========================================================================

RpptDataType cvDepthToRppDataType(int cvDepth) {
    switch (cvDepth) {
        case CV_8U:  return U8;
        case CV_8S:  return I8;
        case CV_16U: return U8;   // Not available in RPP, map to closest
        case CV_16S: return I16;
        case CV_32S: return I16;  // Not available in RPP, map to closest
        case CV_32F: return F32;
        case CV_64F: return F32;  // Not available in RPP, map to closest
        default:     return U8;
    }
}

// =========================================================================
// Memory helpers for GPU path
// =========================================================================

#ifdef RPP_BACKEND_HIP

bool uploadRawToHip(const void* host_ptr, size_t step, int w, int h, int depth, int cn, void** out_dev_ptr) {
    const size_t elemSize = static_cast<size_t>(CV_ELEM_SIZE1(depth));
    const size_t rowBytes = static_cast<size_t>(w) * static_cast<size_t>(cn) * elemSize;
    size_t totalBytes = rowBytes * h;

    // Try pool first.
    void* devPtr = DeviceBufferPool::instance().acquire(totalBytes);
    if (!devPtr) {
        if (!checkHip(hipMalloc(&devPtr, totalBytes)) || devPtr == nullptr) {
            return false;
        }
    }

    const uchar* src = static_cast<const uchar*>(host_ptr);
    uchar* dst = static_cast<uchar*>(devPtr);
    for (int row = 0; row < h; ++row) {
        if (!checkHip(hipMemcpy(dst + row * rowBytes, src + row * step, rowBytes, hipMemcpyHostToDevice))) {
            DeviceBufferPool::instance().release(devPtr, totalBytes);
            return false;
        }
    }

    *out_dev_ptr = devPtr;
    return true;
}

bool downloadRawFromHip(void* dev_ptr, void* host_ptr, size_t step, int w, int h, int depth, int cn) {
    const size_t elemSize = static_cast<size_t>(CV_ELEM_SIZE1(depth));
    const size_t rowBytes = static_cast<size_t>(w) * static_cast<size_t>(cn) * elemSize;

    uchar* dst = static_cast<uchar*>(host_ptr);
    uchar* src = static_cast<uchar*>(dev_ptr);
    for (int row = 0; row < h; ++row) {
        if (!checkHip(hipMemcpy(dst + row * step, src + row * rowBytes, rowBytes, hipMemcpyDeviceToHost))) {
            return false;
        }
    }
    return true;
}

void freeHipPtr(void* devPtr) {
    if (devPtr) {
        // We don't know the original size here; release with zero and the
        // pool will still accept it. In a fully optimized version the caller
        // would pass the size.
        DeviceBufferPool::instance().release(devPtr, 0);
    }
}

#else // !RPP_BACKEND_HIP

bool uploadRawToHip(const void*, size_t, int, int, int, int, void**) {
    return false;
}

bool downloadRawFromHip(void*, void*, size_t, int, int, int, int) {
    return false;
}

void freeHipPtr(void*) {}

#endif // RPP_BACKEND_HIP

// =========================================================================
// RPP Handle helpers
// =========================================================================

rppHandle_t createRppGpuHandle(size_t /*batchSize*/) {
    #ifdef RPP_BACKEND_HIP
    // RPP 3.x HIP backend's handle creation and first kernel launch can leave
    // a sticky async "illegal memory access" error that later RPP calls
    // report as hipInit failures, even though each individual operation
    // produces the correct result. Reuse a single per-thread handle and
    // clear the sticky error to keep the context usable.
    static thread_local rppHandle_t s_handle = nullptr;
    if (s_handle) {
        return s_handle;
    }

    // Clear any sticky HIP error left by previous RPP operations or context state.
    (void)hipGetLastError();

    StderrSilencer silence;
    (void)silence;

    rppStatus_t status = rppCreate(&s_handle, 1, 0, nullptr, RPP_HIP_BACKEND);

    // Discard any benign sticky HIP error from rppCreate internals.
    (void)hipGetLastError();

    if (status == rppStatusSuccess) {
        return s_handle;
    }
    #endif
    return nullptr;
}

rppHandle_t createRppCpuHandle(size_t batchSize, Rpp32u numThreads) {
    StderrSilencer silence;
    (void)silence;
    (void)batchSize; (void)numThreads;
    rppHandle_t handle = nullptr;
    rppStatus_t status = rppCreate(&handle, 1, 0, nullptr, RPP_HOST_BACKEND);
    if (status == rppStatusSuccess) {
        return handle;
    }
    return nullptr;
}

void destroyRppGpuHandle(rppHandle_t handle) {
    if (handle) {
        // WORKAROUND: RPP 3.x HIP backend has an internal hipFree bug
        // in its scratchBufferHip cleanup path. Calling rppDestroy with
        // RPP_HIP_BACKEND triggers "an illegal memory access was encountered"
        // and leaves the HIP device context poisoned for the next rppCreate.
        // Skip destroy to avoid the crash and context corruption.
        // This leaks the handle, but RPP's internal scratch buffer lifetime
        // is effectively process-scoped anyway.
        (void)handle;
        // Clear any sticky error left by the skipped destroy / prior work.
        (void)hipGetLastError();
    }
}

void destroyRppCpuHandle(rppHandle_t handle) {
    if (handle) {
        StderrSilencer silence;
        (void)silence;
        rppDestroy(handle, RPP_HOST_BACKEND);
    }
}

}}} // namespace cv::hal::rpp
