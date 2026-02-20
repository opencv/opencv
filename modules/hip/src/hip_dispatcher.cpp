#include "../include/opencv2/hip/hip_dispatcher.hpp"
#include <hip/hip_runtime.h>
#include <sstream>

namespace cv {
namespace hip {

// =====================================================================
// HIPException implementation
// =====================================================================

HIPException::HIPException(hipError_t error, const char* func, const char* file, int line)
    : cv::Exception(0, cv::format("HIP error: %s", hipGetErrorString(error)), func, file, line),
      hip_error(error) {}

// =====================================================================
// GPUMemory implementation
// =====================================================================

GPUMemory::~GPUMemory() {
    if (ptr_ != nullptr) {
        deallocate();
    }
}

void GPUMemory::allocate(size_t size) {
    if (ptr_ != nullptr) {
        deallocate();
    }
    if (size == 0) return;
    
    HIP_CHECK(hipMalloc(&ptr_, size));
    size_ = size;
    HIP_CHECK(hipGetDevice(&device_));
}

void GPUMemory::deallocate() {
    if (ptr_ == nullptr) return;
    hipFree(ptr_);
    ptr_ = nullptr;
    size_ = 0;
}

void GPUMemory::upload(const void* host_ptr, size_t size) {
    if (size == 0) return;
    CV_Assert(host_ptr != nullptr);
    
    if (size > size_) {
        allocate(size);
    }
    
    HIP_CHECK(hipMemcpyHtoDAsync(ptr_, (void*)host_ptr, size, hipStreamDefault));
    HIP_CHECK(hipStreamSynchronize(hipStreamDefault));
}

void GPUMemory::download(void* host_ptr, size_t size) {
    if (size == 0 || ptr_ == nullptr) return;
    CV_Assert(host_ptr != nullptr);
    CV_Assert(size <= size_);
    
    HIP_CHECK(hipMemcpyDtoHAsync(host_ptr, ptr_, size, hipStreamDefault));
    HIP_CHECK(hipStreamSynchronize(hipStreamDefault));
}

// =====================================================================
// GPUDevice implementation
// =====================================================================

int GPUDevice::getDeviceCount() {
    int count = 0;
    hipGetDeviceCount(&count);
    return count;
}

void GPUDevice::selectDevice(int device) {
    CV_Assert(device >= 0 && device < getDeviceCount());
    HIP_CHECK(hipSetDevice(device));
}

int GPUDevice::getCurrentDevice() {
    int device = -1;
    HIP_CHECK(hipGetDevice(&device));
    return device;
}

void GPUDevice::synchronize() {
    HIP_CHECK(hipDeviceSynchronize());
}

size_t GPUDevice::getFreeMemory() {
    size_t free = 0, total = 0;
    HIP_CHECK(hipMemGetInfo(&free, &total));
    return free;
}

size_t GPUDevice::getTotalMemory() {
    size_t free = 0, total = 0;
    HIP_CHECK(hipMemGetInfo(&free, &total));
    return total;
}

// =====================================================================
// Global GPU Configuration
// =====================================================================

static GPUConfig g_gpu_config;

GPUConfig& getGPUConfig() {
    return g_gpu_config;
}

bool isGPUAvailable() {
    return GPUDevice::hasGPU() && getGPUConfig().enabled;
}

bool shouldUseGPU(size_t image_size, float flops_per_element) {
    if (!isGPUAvailable()) return false;
    
    const auto& config = getGPUConfig();
    
    // Check if image is large enough to justify transfer overhead
    if (image_size < config.min_image_size_bytes) {
        return false;
    }
    
    // Check if operation is compute-dense enough
    if (flops_per_element < config.min_flops_per_element) {
        return false;
    }
    
    // Check if GPU has enough memory
    if (GPUDevice::getFreeMemory() < image_size * 3) {  // 3x for safety margin
        return false;
    }
    
    return true;
}

void setGPUEnabled(bool enable) {
    getGPUConfig().enabled = enable;
}

} // namespace hip
} // namespace cv
