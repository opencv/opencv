#ifndef OPENCV_HIP_CONFIG_HPP
#define OPENCV_HIP_CONFIG_HPP

#include <opencv2/core/base.hpp>

namespace cv {
namespace hip {

/**
 * @brief Configuration for GPU acceleration thresholds and parameters
 */
struct GPUConfig {
    // Minimum image size (in bytes) to use GPU
    size_t min_image_size_bytes = 1024 * 1024; // 1 MB default
    
    // Minimum compute density (FLOPs per element) to justify GPU
    float min_flops_per_element = 10.0f;
    
    // Enable GPU acceleration globally
    bool enabled = true;
    
    // Fallback to CPU if GPU fails
    bool fallback_to_cpu = true;
    
    // Print performance metrics
    bool verbose = false;
};

/**
 * @brief Get the global GPU configuration
 */
CV_EXPORTS GPUConfig& getGPUConfig();

/**
 * @brief Check if GPU acceleration is available and viable
 */
CV_EXPORTS bool isGPUAvailable();

/**
 * @brief Check if a specific operation should use GPU
 * @param image_size Size of input image in bytes
 * @param flops_per_element Approximate floating-point operations per element
 */
CV_EXPORTS bool shouldUseGPU(size_t image_size, float flops_per_element = 10.0f);

/**
 * @brief Set GPU enabled/disabled
 */
CV_EXPORTS void setGPUEnabled(bool enable);

} // namespace hip
} // namespace cv

#endif // OPENCV_HIP_CONFIG_HPP
