#ifndef OPENCV_HIP_DISPATCHER_HPP
#define OPENCV_HIP_DISPATCHER_HPP

#include <opencv2/core.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_common.h>

namespace cv {
namespace hip {

/**
 * @brief Error handling for HIP operations
 */
class HIPException : public cv::Exception {
public:
    HIPException(hipError_t error, const char* func, const char* file, int line);
    hipError_t hip_error;
};

#define HIP_CHECK(error) \
    if((error) != hipSuccess) { \
        throw cv::hip::HIPException((error), #error, __FILE__, __LINE__); \
    }

/**
 * @brief GPU memory management utilities
 */
class CV_EXPORTS GPUMemory {
public:
    GPUMemory() : ptr_(nullptr), size_(0), device_(-1) {}
    ~GPUMemory();
    
    void allocate(size_t size);
    void deallocate();
    void upload(const void* host_ptr, size_t size);
    void download(void* host_ptr, size_t size);
    
    void* get() { return ptr_; }
    const void* get() const { return ptr_; }
    size_t size() const { return size_; }
    bool empty() const { return ptr_ == nullptr; }
    
private:
    void* ptr_;
    size_t size_;
    int device_;
};

/**
 * @brief Device information and capabilities
 */
class CV_EXPORTS GPUDevice {
public:
    static int getDeviceCount();
    static void selectDevice(int device);
    static int getCurrentDevice();
    static void synchronize();
    static size_t getFreeMemory();
    static size_t getTotalMemory();
    
    static bool hasGPU() { return getDeviceCount() > 0; }
};

/**
 * @brief CPU-GPU dispatcher with automatic device selection
 */
template<typename Func>
class Dispatcher {
public:
    Dispatcher(const Func& gpu_func, const Func& cpu_func, size_t image_size, float flops_per_element = 10.0f)
        : gpu_func_(gpu_func), cpu_func_(cpu_func), image_size_(image_size), flops_(flops_per_element) {}
    
    template<typename... Args>
    void operator()(Args&&... args) const {
        if (shouldUseGPU(image_size_, flops_) && GPUDevice::hasGPU()) {
            try {
                gpu_func_(std::forward<Args>(args)...);
                return;
            } catch (const HIPException& e) {
                if (!getGPUConfig().fallback_to_cpu) throw;
                CV_LOG_WARNING(NULL, "GPU operation failed, falling back to CPU: " << e.what());
            }
        }
        cpu_func_(std::forward<Args>(args)...);
    }
    
private:
    Func gpu_func_;
    Func cpu_func_;
    size_t image_size_;
    float flops_;
};

/**
 * @brief Helper to create a dispatcher
 */
template<typename Func1, typename Func2>
Dispatcher<Func1> makeDispatcher(const Func1& gpu_func, const Func2& cpu_func, 
                                  size_t image_size, float flops_per_element = 10.0f) {
    return Dispatcher<Func1>(gpu_func, cpu_func, image_size, flops_per_element);
}

} // namespace hip
} // namespace cv

#endif // OPENCV_HIP_DISPATCHER_HPP
