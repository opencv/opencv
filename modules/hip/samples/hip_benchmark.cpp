/**
 * @brief HIP Threads GPU vs CPU Performance Benchmark
 * 
 * Measures performance of GPU-accelerated operations across
 * different image sizes to determine break-even points.
 * 
 * Build:
 *   g++ hip_benchmark.cpp -o benchmark \
 *     -I/path/to/opencv/include \
 *     -L/path/to/opencv/lib -lopencv_core -lopencv_imgproc -lopencv_hip \
 *     -O3
 * 
 * Run:
 *   ./benchmark
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/hip/hip_kernels.hpp>
#include <opencv2/hip/hip_config.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace cv;
using namespace cv::hip;
using Clock = std::chrono::high_resolution_clock;
using ms = std::chrono::milliseconds;

struct BenchmarkResult {
    int size_mb;
    double gpu_ms;
    double cpu_ms;
    double speedup;
};

template<typename Func>
double measureTime(Func&& func, int iterations = 1) {
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto elapsed = Clock::now() - start;
    return std::chrono::duration_cast<ms>(elapsed).count();
}

void benchmarkGaussianBlur() {
    std::cout << "\n╔════════════════════════════════════════════════════╗\n";
    std::cout << "║   GPU vs CPU Gaussian Blur (5×5 kernel)           ║\n";
    std::cout << "╚════════════════════════════════════════════════════╝\n\n";
    
    std::vector<std::pair<int, int>> image_sizes = {
        {320, 240},      // 0.3 MB
        {640, 480},      // 1.2 MB
        {1280, 960},     // 4.7 MB
        {1920, 1080},    // 10.6 MB
        {2560, 1440},    // 18.9 MB
        {3840, 2160},    // 42.6 MB
    };
    
    std::cout << std::right
              << std::setw(12) << "Image Size"
              << std::setw(12) << "GPU (ms)"
              << std::setw(12) << "CPU (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "GPU Better?"
              << "\n";
    std::cout << std::string(63, '─') << "\n";
    
    for (auto [width, height] : image_sizes) {
        Mat image(height, width, CV_8UC3);
        cv::randu(image, 0, 256);
        
        Mat result;
        
        // GPU benchmark
        double gpu_time = measureTime([&]() {
            gaussianBlur_gpu(image, result, Size(5, 5), 1.0);
        }, 5);
        
        // CPU benchmark
        double cpu_time = measureTime([&]() {
            cv::GaussianBlur(image, result, Size(5, 5), 1.0);
        }, 5);
        
        double speedup = cpu_time / gpu_time;
        bool gpu_better = gpu_time < cpu_time;
        
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(8) << width << "x" 
                  << std::setw(3) << height
                  << std::setw(12) << (gpu_time / 5.0)
                  << std::setw(12) << (cpu_time / 5.0)
                  << std::setw(12) << speedup << "x"
                  << std::setw(15) << (gpu_better ? "✓ YES" : "✗ NO")
                  << "\n";
    }
}

void benchmarkResize() {
    std::cout << "\n╔════════════════════════════════════════════════════╗\n";
    std::cout << "║   GPU vs CPU Image Resize (Bilinear)              ║\n";
    std::cout << "╚════════════════════════════════════════════════════╝\n\n";
    
    std::vector<std::pair<std::pair<int,int>, std::pair<int,int>>> tests = {
        {{640, 480}, {320, 240}},       // Downscale
        {{1920, 1080}, {960, 540}},     // Downscale
        {{320, 240}, {640, 480}},       // Upscale
        {{960, 540}, {1920, 1080}},     // Upscale
    };
    
    std::cout << std::right
              << std::setw(20) << "Scale Operation"
              << std::setw(12) << "GPU (ms)"
              << std::setw(12) << "CPU (ms)"
              << std::setw(12) << "Speedup"
              << "\n";
    std::cout << std::string(56, '─') << "\n";
    
    for (auto [src_size, dst_size] : tests) {
        Mat src(src_size.second, src_size.first, CV_8UC3);
        cv::randu(src, 0, 256);
        
        Mat result;
        
        // GPU benchmark
        double gpu_time = measureTime([&]() {
            resize_gpu(src, result, Size(dst_size.first, dst_size.second));
        }, 5);
        
        // CPU benchmark
        double cpu_time = measureTime([&]() {
            cv::resize(src, result, Size(dst_size.first, dst_size.second));
        }, 5);
        
        double speedup = cpu_time / gpu_time;
        
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(9) << src_size.first << "x"
                  << std::setw(3) << src_size.second
                  << " → "
                  << std::setw(7) << dst_size.first << "x"
                  << std::setw(3) << dst_size.second
                  << std::setw(12) << (gpu_time / 5.0)
                  << std::setw(12) << (cpu_time / 5.0)
                  << std::setw(12) << speedup << "x"
                  << "\n";
    }
}

void benchmarkColorConvert() {
    std::cout << "\n╔════════════════════════════════════════════════════╗\n";
    std::cout << "║   GPU vs CPU Color Conversion (BGR → Gray)        ║\n";
    std::cout << "╚════════════════════════════════════════════════════╝\n\n";
    
    std::vector<std::pair<int, int>> image_sizes = {
        {640, 480},      // 1.2 MB
        {1280, 960},     // 4.7 MB
        {1920, 1080},    // 10.6 MB
        {2560, 1440},    // 18.9 MB
    };
    
    std::cout << std::right
              << std::setw(12) << "Image Size"
              << std::setw(12) << "GPU (ms)"
              << std::setw(12) << "CPU (ms)"
              << std::setw(12) << "Speedup"
              << "\n";
    std::cout << std::string(48, '─') << "\n";
    
    for (auto [width, height] : image_sizes) {
        Mat image(height, width, CV_8UC3);
        cv::randu(image, 0, 256);
        
        Mat result;
        
        // GPU benchmark
        double gpu_time = measureTime([&]() {
            cvtColor_gpu(image, result, cv::COLOR_BGR2GRAY);
        }, 5);
        
        // CPU benchmark
        double cpu_time = measureTime([&]() {
            cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);
        }, 5);
        
        double speedup = cpu_time / gpu_time;
        
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(8) << width << "x" 
                  << std::setw(3) << height
                  << std::setw(12) << (gpu_time / 5.0)
                  << std::setw(12) << (cpu_time / 5.0)
                  << std::setw(12) << speedup << "x"
                  << "\n";
    }
}

int main() {
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "      OpenCV HIP Threads Performance Benchmark Suite       \n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    
    // Check GPU status
    std::cout << "\nGPU Status:\n";
    if (isGPUAvailable()) {
        std::cout << "  ✓ GPU is available and enabled\n";
        std::cout << "  Free memory: " << (GPUDevice::getFreeMemory() / (1024*1024)) 
                  << " MB\n";
    } else {
        std::cout << "  ✗ GPU not available - using CPU only\n";
    }
    
    benchmarkGaussianBlur();
    benchmarkResize();
    benchmarkColorConvert();
    
    std::cout << "\n════════════════════════════════════════════════════════════\n";
    std::cout << "                    Benchmark Complete                    \n";
    std::cout << "════════════════════════════════════════════════════════════\n\n";
    
    return 0;
}
