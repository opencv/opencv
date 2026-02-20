/**
 * @brief HIP Threads Gaussian Blur Demonstration
 * 
 * This example shows how to use GPU-accelerated Gaussian Blur
 * with automatic CPU/GPU dispatch based on image size.
 * 
 * Build:
 *   g++ hip_gaussian_blur_demo.cpp -o demo \
 *     -I/path/to/opencv/include \
 *     -L/path/to/opencv/lib -lopencv_core -lopencv_imgproc -lopencv_hip
 * 
 * Run:
 *   ./demo input.jpg
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/hip/hip_kernels.hpp>
#include <opencv2/hip/hip_config.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace cv::hip;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>\n";
        return 1;
    }
    
    // Read image
    Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << argv[1] << "\n";
        return 1;
    }
    
    std::cout << "=== OpenCV HIP Threads Gaussian Blur Demo ===\n\n";
    std::cout << "Input image: " << image.cols << "x" << image.rows 
              << " (" << (image.total() * image.elemSize() / (1024*1024)) 
              << " MB)\n\n";
    
    // Check GPU availability
    if (!isGPUAvailable()) {
        std::cout << "⚠️  GPU not available. Using CPU only.\n";
    } else {
        std::cout << "✓ GPU is available and enabled\n";
        std::cout << "  Free memory: " 
                  << (GPUDevice::getFreeMemory() / (1024*1024)) << " MB\n";
    }
    
    // Get configuration
    auto& config = getGPUConfig();
    std::cout << "\nGPU Configuration:\n"
              << "  Min image size: " << (config.min_image_size_bytes / (1024*1024)) 
              << " MB\n"
              << "  Min compute density: " << config.min_flops_per_element 
              << " FLOPs/element\n"
              << "  Fallback to CPU: " << (config.fallback_to_cpu ? "Yes" : "No") 
              << "\n\n";
    
    // Check if this operation should use GPU
    size_t img_size = image.total() * image.elemSize();
    float flops = 5.0f * 5.0f;  // 5x5 kernel = 25 FLOPs
    bool should_use_gpu = shouldUseGPU(img_size, flops);
    
    std::cout << "Dispatch Decision:\n"
              << "  Image size: " << (img_size / (1024*1024)) << " MB\n"
              << "  Compute density: " << flops << " FLOPs/element\n"
              << "  Use GPU: " << (should_use_gpu ? "Yes" : "No") << "\n\n";
    
    // Apply different kernel sizes
    std::vector<Size> kernel_sizes = {{3, 3}, {5, 5}, {11, 11}, {21, 21}};
    
    for (auto ksize : kernel_sizes) {
        std::cout << "Processing with " << ksize.width << "x" << ksize.height 
                  << " kernel...\n";
        
        Mat result_gpu, result_cpu;
        
        // GPU version (with automatic fallback)
        auto start = Clock::now();
        try {
            gaussianBlur_gpu(image, result_gpu, ksize, 1.0);
        } catch (const std::exception& e) {
            std::cerr << "GPU error: " << e.what() << "\n";
            result_gpu = Mat();
        }
        auto gpu_time = Clock::now() - start;
        
        // CPU version
        start = Clock::now();
        cv::GaussianBlur(image, result_cpu, ksize, 1.0);
        auto cpu_time = Clock::now() - start;
        
        std::cout << "  GPU time: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>
                     (gpu_time).count() << " ms\n";
        std::cout << "  CPU time: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>
                     (cpu_time).count() << " ms\n";
        
        if (!result_gpu.empty() && !result_cpu.empty()) {
            double diff = cv::norm(result_gpu, result_cpu, cv::NORM_L2) / 
                         (result_gpu.total() * result_gpu.channels());
            std::cout << "  Avg difference: " << diff << " (max 1.0 is acceptable)\n";
        }
        
        std::cout << "\n";
    }
    
    // Configuration modification example
    std::cout << "=== Configuration Modification Example ===\n\n";
    
    std::cout << "Original threshold: " << (config.min_image_size_bytes / (1024*1024)) 
              << " MB\n";
    
    // Lower threshold to force GPU for smaller images
    config.min_image_size_bytes = 512 * 1024;  // 512 KB
    std::cout << "Modified threshold: " << (config.min_image_size_bytes / (1024*1024)) 
              << " MB\n";
    
    bool use_gpu_now = shouldUseGPU(img_size, flops);
    std::cout << "Use GPU now: " << (use_gpu_now ? "Yes" : "No") << "\n\n";
    
    // Demonstrate fallback
    std::cout << "=== GPU Fallback Test ===\n";
    std::cout << "Fallback enabled: " << (config.fallback_to_cpu ? "Yes" : "No") 
              << "\n";
    std::cout << "If GPU operation fails, it will automatically use CPU\n";
    
    Mat result;
    gaussianBlur_gpu(image, result, {5, 5}, 1.0);
    std::cout << "Operation completed successfully\n";
    
    return 0;
}
