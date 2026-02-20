#include "../include/opencv2/hip/hip_config.hpp"

namespace cv {
namespace hip {

static GPUConfig g_config;

GPUConfig& getGPUConfig() {
    return g_config;
}

bool isGPUAvailable() {
    // Will be implemented in hip_dispatcher.cpp
    return false;
}

bool shouldUseGPU(size_t image_size, float flops_per_element) {
    // Will be implemented in hip_dispatcher.cpp
    return false;
}

void setGPUEnabled(bool enable) {
    // Will be implemented in hip_dispatcher.cpp
}

} // namespace hip
} // namespace cv
