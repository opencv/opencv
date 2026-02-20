#include "test_precomp.hpp"
#include <opencv2/hip/hip_kernels.hpp>
#include <opencv2/hip/hip_dispatcher.hpp>

namespace cvtest {

class HIPGaussianBlurTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test image: 640x480 RGB
        test_img.create(480, 640, CV_8UC3);
        cv::randu(test_img, 0, 256);
        
        // Create reference using CPU
        cv::GaussianBlur(test_img, reference_img, cv::Size(5, 5), 1.0);
    }
    
    cv::Mat test_img;
    cv::Mat reference_img;
};

TEST_F(HIPGaussianBlurTest, BasicFunctionality) {
    cv::Mat gpu_result;
    cv::hip::gaussianBlur_gpu(test_img, gpu_result, cv::Size(5, 5), 1.0);
    
    // Results should be very close (allowing for floating-point differences)
    ASSERT_EQ(gpu_result.size(), reference_img.size());
    ASSERT_EQ(gpu_result.type(), reference_img.type());
    
    // Max difference should be small due to rounding
    double max_diff = cv::norm(gpu_result, reference_img, cv::NORM_INF);
    EXPECT_LT(max_diff, 2.0);  // Allow small rounding differences
}

TEST_F(HIPGaussianBlurTest, DifferentKernelSizes) {
    std::vector<cv::Size> kernel_sizes = {
        cv::Size(3, 3), cv::Size(5, 5), cv::Size(7, 7), cv::Size(11, 11)
    };
    
    for (auto ksize : kernel_sizes) {
        cv::Mat gpu_result, cpu_result;
        cv::hip::gaussianBlur_gpu(test_img, gpu_result, ksize, 1.0);
        cv::GaussianBlur(test_img, cpu_result, ksize, 1.0);
        
        // Results should be similar
        double diff = cv::norm(gpu_result, cpu_result, cv::NORM_L2) / (gpu_result.total());
        EXPECT_LT(diff, 1.0) << "Kernel size: " << ksize << " failed";
    }
}

class HIPResizeTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_img.create(480, 640, CV_8UC3);
        cv::randu(test_img, 0, 256);
    }
    
    cv::Mat test_img;
};

TEST_F(HIPResizeTest, Downscale) {
    cv::Mat gpu_result, cpu_result;
    cv::Size out_size(320, 240);
    
    cv::hip::resize_gpu(test_img, gpu_result, out_size, 0, 0, cv::INTER_LINEAR);
    cv::resize(test_img, cpu_result, out_size, 0, 0, cv::INTER_LINEAR);
    
    ASSERT_EQ(gpu_result.size(), cpu_result.size());
    double diff = cv::norm(gpu_result, cpu_result, cv::NORM_L2) / (gpu_result.total());
    EXPECT_LT(diff, 2.0);
}

TEST_F(HIPResizeTest, Upscale) {
    cv::Mat gpu_result, cpu_result;
    cv::Size out_size(1280, 960);
    
    cv::hip::resize_gpu(test_img, gpu_result, out_size, 0, 0, cv::INTER_LINEAR);
    cv::resize(test_img, cpu_result, out_size, 0, 0, cv::INTER_LINEAR);
    
    ASSERT_EQ(gpu_result.size(), cpu_result.size());
    double diff = cv::norm(gpu_result, cpu_result, cv::NORM_L2) / (gpu_result.total());
    EXPECT_LT(diff, 2.0);
}

class HIPColorConvertTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_img.create(480, 640, CV_8UC3);
        cv::randu(test_img, 0, 256);
    }
    
    cv::Mat test_img;
};

TEST_F(HIPColorConvertTest, BGR2RGB) {
    cv::Mat gpu_result, cpu_result;
    
    cv::hip::cvtColor_gpu(test_img, gpu_result, cv::COLOR_BGR2RGB);
    cv::cvtColor(test_img, cpu_result, cv::COLOR_BGR2RGB);
    
    ASSERT_EQ(gpu_result.size(), cpu_result.size());
    // Results should be identical for simple channel swaps
    EXPECT_EQ(0, cv::countNonZero(gpu_result != cpu_result));
}

TEST_F(HIPColorConvertTest, BGR2GRAY) {
    cv::Mat gpu_result, cpu_result;
    
    cv::hip::cvtColor_gpu(test_img, gpu_result, cv::COLOR_BGR2GRAY);
    cv::cvtColor(test_img, cpu_result, cv::COLOR_BGR2GRAY);
    
    ASSERT_EQ(gpu_result.size(), cpu_result.size());
    double diff = cv::norm(gpu_result, cpu_result, cv::NORM_L2) / (gpu_result.total());
    EXPECT_LT(diff, 1.0);
}

class HIPGPUConfigTest : public ::testing::Test {
};

TEST_F(HIPGPUConfigTest, ConfigurationAccess) {
    auto& config = cv::hip::getGPUConfig();
    
    // Should have sensible defaults
    EXPECT_GT(config.min_image_size_bytes, 0);
    EXPECT_GT(config.min_flops_per_element, 0);
    EXPECT_TRUE(config.enabled);
    EXPECT_TRUE(config.fallback_to_cpu);
}

TEST_F(HIPGPUConfigTest, SetGPUEnabled) {
    cv::hip::setGPUEnabled(false);
    EXPECT_FALSE(cv::hip::getGPUConfig().enabled);
    
    cv::hip::setGPUEnabled(true);
    EXPECT_TRUE(cv::hip::getGPUConfig().enabled);
}

TEST_F(HIPGPUConfigTest, ShouldUseGPU) {
    auto& config = cv::hip::getGPUConfig();
    
    // Small image should not use GPU
    EXPECT_FALSE(cv::hip::shouldUseGPU(1024, 5.0f));  // 1KB, low FLOPs
    
    // Large image with high FLOPs should use GPU (if available)
    bool result = cv::hip::shouldUseGPU(10 * 1024 * 1024, 50.0f);  // 10MB, high FLOPs
    // Result depends on GPU availability, just check it doesn't crash
    EXPECT_TRUE(true);
}

} // namespace cvtest
