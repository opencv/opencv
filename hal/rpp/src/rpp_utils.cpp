/**
 * simoncatbot-opencv RPP HAL - Unified Utilities Implementation
 */

#include "rpp_hal_utils.hpp"
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

// Only include HIP headers for GPU path
#ifdef RPP_BACKEND_HIP
#include <hip/hip_runtime.h>
#endif

namespace cv { namespace hal { namespace rpp {

// =========================================================================
// Availability checks
// =========================================================================

bool isRppGpuAvailable() {
    #ifdef RPP_BACKEND_HIP
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    return (err == hipSuccess && deviceCount > 0);
    #else
    return false;
    #endif
}

bool isRppCpuAvailable() {
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
    size_t elemSize = CV_ELEM_SIZE(depth);
    size_t rowBytes = w * cn * elemSize;
    size_t totalBytes = rowBytes * h;

    void* devPtr = nullptr;
    hipError_t err = hipMalloc(&devPtr, totalBytes);
    if (err != hipSuccess || devPtr == nullptr) {
        return false;
    }

    const uchar* src = static_cast<const uchar*>(host_ptr);
    uchar* dst = static_cast<uchar*>(devPtr);
    for (int row = 0; row < h; ++row) {
        hipMemcpy(dst + row * rowBytes, src + row * step, rowBytes, hipMemcpyHostToDevice);
    }

    *out_dev_ptr = devPtr;
    return true;
}

bool downloadRawFromHip(void* dev_ptr, void* host_ptr, size_t step, int w, int h, int depth, int cn) {
    size_t elemSize = CV_ELEM_SIZE(depth);
    size_t rowBytes = w * cn * elemSize;

    uchar* dst = static_cast<uchar*>(host_ptr);
    uchar* src = static_cast<uchar*>(dev_ptr);
    for (int row = 0; row < h; ++row) {
        hipMemcpy(dst + row * step, src + row * rowBytes, rowBytes, hipMemcpyDeviceToHost);
    }
    return true;
}

void freeHipPtr(void* devPtr) {
    if (devPtr) {
        hipFree(devPtr);
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

rppHandle_t createRppGpuHandle(size_t batchSize) {
    #ifdef RPP_BACKEND_HIP
    int old_stderr = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
        dup2(devnull, STDERR_FILENO);
        close(devnull);
    }
    
    rppHandle_t handle;
    rppStatus_t status = rppCreate(&handle, batchSize, 0, nullptr, RPP_HIP_BACKEND);
    
    if (old_stderr >= 0) {
        dup2(old_stderr, STDERR_FILENO);
        close(old_stderr);
    }
    
    if (status == rppStatusSuccess) {
        return handle;
    }
    #endif
    return nullptr;
}

rppHandle_t createRppCpuHandle(size_t batchSize, Rpp32u numThreads) {
    rppHandle_t handle;
    rppStatus_t status = rppCreate(&handle, batchSize, numThreads, nullptr, RPP_HOST_BACKEND);
    if (status == rppStatusSuccess) {
        return handle;
    }
    return nullptr;
}

void destroyRppGpuHandle(rppHandle_t handle) {
    if (handle) {
        // WORKAROUND: RPP has internal hipFree bug on destroy.
        // Skip destroy to avoid crash at exit.
        // Leaks the handle, but acceptable for now.
        (void)handle;
    }
}

void destroyRppCpuHandle(rppHandle_t handle) {
    if (handle) {
        rppDestroy(handle, RPP_HOST_BACKEND);
    }
}

}}} // namespace cv::hal::rpp
