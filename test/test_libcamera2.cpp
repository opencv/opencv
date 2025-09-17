#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <sstream>
#include <thread>
#include <map>
#include <fstream>

// 测试结果结构
struct TestResult {
    std::string test_name;
    bool passed;
    std::string details;
    double measured_value = -1;
};

// 图像质量分析
struct ImageQuality {
    double mean_brightness;
    double std_brightness;
    double mean_contrast;
    cv::Size resolution;
};

class LibcameraDetailedTester {
private:
    cv::VideoCapture cap;
    std::vector<TestResult> results;
    int test_counter = 0;
    std::string result_dir = "test_results";  // 默认目录
    
    void logTest(const std::string& name, bool passed, const std::string& details = "", double value = -1) {
        TestResult result;
        result.test_name = name;
        result.passed = passed;
        result.details = details;
        result.measured_value = value;
        results.push_back(result);
        
        std::cout << "[" << std::setw(2) << ++test_counter << "] " 
                  << (passed ? "✅" : "❌") << " " 
                  << std::setw(30) << std::left << name 
                  << " | " << details;
        if (value >= 0) std::cout << " (value: " << value << ")";
        std::cout << std::endl;
    }
    
    ImageQuality analyzeImage(const cv::Mat& frame) {
        ImageQuality quality;
        quality.resolution = frame.size();
        
        // 转换为灰度图分析亮度
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        quality.mean_brightness = mean[0];
        quality.std_brightness = stddev[0];
        
        // 计算对比度
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::Scalar contrast_scalar = cv::mean(cv::abs(laplacian));
        quality.mean_contrast = contrast_scalar[0];
        
        return quality;
    }
    
    bool captureTestFrames(const std::string& test_name, int num_frames = 5) {
        std::vector<cv::Mat> frames;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_frames; i++) {
            cv::Mat frame;
            cap >> frame;
            
            if (frame.empty()) {
                logTest(test_name + " - Frame Capture", false, "Frame " + std::to_string(i) + " is empty");
                return false;
            }
            
            frames.push_back(frame.clone());
            
            // 保存样本帧
            if (i == 0 || i == num_frames-1) {
                std::string filename = result_dir + "/test_" + test_name + "_frame_" + std::to_string(i) + ".jpg";
                cv::imwrite(filename, frame);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double actual_fps = (num_frames * 1000.0) / duration.count();
        
        // 分析最后一帧的质量
        ImageQuality quality = analyzeImage(frames.back());
        
        logTest(test_name + " - Frame Capture", true, 
                "Resolution: " + std::to_string(quality.resolution.width) + "x" + std::to_string(quality.resolution.height) +
                ", Brightness: " + std::to_string((int)quality.mean_brightness) +
                ", Contrast: " + std::to_string((int)quality.mean_contrast), actual_fps);
        
        return true;
    }
    
public:
    // 构造函数，设置结果目录
    LibcameraDetailedTester(const std::string& output_dir = "test_results") : result_dir(output_dir) {}
    
    bool initialize() {
        cap.open(0);
        if (!cap.isOpened()) {
            logTest("Camera Initialization", false, "Failed to open camera");
            return false;
        }
        
        std::string backend = cap.getBackendName();
        bool is_libcamera = (backend == "LIBCAMERA");
        logTest("Backend Detection", is_libcamera, "Using: " + backend);
        
        return is_libcamera;
    }
    
    void testResolutions() {
        std::cout << "\n🔧 Testing Resolution Settings...\n";
        
        std::vector<std::pair<int, int>> resolutions = {
            {640, 480},
            {1280, 720},
            {1920, 1080}
        };
        
        for (auto res : resolutions) {
            bool width_set = cap.set(cv::CAP_PROP_FRAME_WIDTH, res.first);
            bool height_set = cap.set(cv::CAP_PROP_FRAME_HEIGHT, res.second);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 等待配置生效
            
            double actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            double actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            
            std::string test_name = "Resolution_" + std::to_string(res.first) + "x" + std::to_string(res.second);
            std::string details = "Set: " + std::to_string(res.first) + "x" + std::to_string(res.second) + 
                                " | Got: " + std::to_string((int)actual_width) + "x" + std::to_string((int)actual_height);
            
            logTest(test_name + " - Setting", width_set && height_set, details);
            
            if (width_set && height_set) {
                captureTestFrames(test_name);
            }
        }
    }
    
    void testFramerates() {
        std::cout << "\n🎬 Testing Framerate Settings...\n";
        
        // 先设置一个标准分辨率
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        std::vector<int> framerates = {15, 30, 60};
        
        for (int fps : framerates) {
            bool fps_set = cap.set(cv::CAP_PROP_FPS, fps);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            double actual_fps = cap.get(cv::CAP_PROP_FPS);
            
            std::string test_name = "Framerate_" + std::to_string(fps) + "fps";
            std::string details = "Set: " + std::to_string(fps) + " | Got: " + std::to_string((int)actual_fps);
            
            logTest(test_name + " - Setting", fps_set, details);
            
            if (fps_set) {
                captureTestFrames(test_name);
            }
        }
    }
    
    void testImageQualitySettings() {
        std::cout << "\n🔆 Testing Image Quality Settings...\n";
        
        // 设置标准配置
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        std::map<std::string, std::vector<double>> quality_settings = {
            {"Brightness", {0.0, 0.5, 1.0}},
            {"Contrast", {0.5, 1.0, 1.5}},
            {"Saturation", {0.5, 1.0, 1.5}},
            {"Sharpness", {0.0, 0.5, 1.0}}
        };
        
        std::map<std::string, int> property_map = {
            {"Brightness", cv::CAP_PROP_BRIGHTNESS},
            {"Contrast", cv::CAP_PROP_CONTRAST},
            {"Saturation", cv::CAP_PROP_SATURATION},
            {"Sharpness", cv::CAP_PROP_SHARPNESS}
        };
        
        for (auto& setting : quality_settings) {
            std::string property_name = setting.first;
            int property_id = property_map[property_name];
            
            for (double value : setting.second) {
                bool prop_set = cap.set(property_id, value);
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                
                double actual_value = cap.get(property_id);
                
                std::string test_name = property_name + "_" + std::to_string(value);
                std::string details = "Set: " + std::to_string(value) + " | Got: " + std::to_string(actual_value);
                
                logTest(test_name + " - Setting", prop_set, details);
                
                if (prop_set) {
                    captureTestFrames(test_name, 3);
                }
            }
        }
    }
    
    void testExposureControl() {
        std::cout << "\n🌟 Testing Exposure Control...\n";
        
        // 测试自动曝光
        bool auto_exp_on = cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
        logTest("Auto Exposure ON", auto_exp_on, "Auto exposure enabled");
        if (auto_exp_on) captureTestFrames("AutoExposure_ON", 3);
        
        bool auto_exp_off = cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
        logTest("Auto Exposure OFF", auto_exp_off, "Auto exposure disabled");
        
        // 测试手动曝光
        if (auto_exp_off) {
            std::vector<double> exposure_values = {0.2, 0.5, 0.8};
            for (double exp : exposure_values) {
                bool exp_set = cap.set(cv::CAP_PROP_EXPOSURE, exp);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                
                double actual_exp = cap.get(cv::CAP_PROP_EXPOSURE);
                
                std::string test_name = "Manual_Exposure_" + std::to_string(exp);
                std::string details = "Set: " + std::to_string(exp) + " | Got: " + std::to_string(actual_exp);
                
                logTest(test_name, exp_set, details);
                if (exp_set) captureTestFrames(test_name, 3);
            }
        }
    }
    
    void testWhiteBalance() {
        std::cout << "\n🌡️ Testing White Balance...\n";
        
        // 测试自动白平衡
        bool auto_wb_on = cap.set(cv::CAP_PROP_AUTO_WB, 1);
        logTest("Auto White Balance ON", auto_wb_on, "Auto WB enabled");
        if (auto_wb_on) captureTestFrames("AutoWB_ON", 3);
        
        bool auto_wb_off = cap.set(cv::CAP_PROP_AUTO_WB, 0);
        logTest("Auto White Balance OFF", auto_wb_off, "Auto WB disabled");
        
        // 测试色温设置
        if (auto_wb_off) {
            std::vector<int> wb_temps = {3000, 5000, 7000};
            for (int temp : wb_temps) {
                bool temp_set = cap.set(cv::CAP_PROP_WB_TEMPERATURE, temp);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                
                double actual_temp = cap.get(cv::CAP_PROP_WB_TEMPERATURE);
                
                std::string test_name = "WB_Temperature_" + std::to_string(temp) + "K";
                std::string details = "Set: " + std::to_string(temp) + "K | Got: " + std::to_string((int)actual_temp) + "K";
                
                logTest(test_name, temp_set, details);
                if (temp_set) captureTestFrames(test_name, 3);
            }
        }
    }
    
    void testROI() {
        std::cout << "\n📐 Testing ROI (Region of Interest)...\n";
        
        // 设置标准配置
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        std::vector<std::tuple<double, double, double, double>> roi_settings = {
            {0.0, 0.0, 1.0, 1.0},    // 全画面
            {0.25, 0.25, 0.5, 0.5},  // 中央1/4
            {0.0, 0.0, 0.5, 0.5}     // 左上1/4
        };
        
        for (auto roi : roi_settings) {
            double x = std::get<0>(roi);
            double y = std::get<1>(roi);
            double w = std::get<2>(roi);
            double h = std::get<3>(roi);
            
            bool roi_x_set = cap.set(cv::CAP_PROP_XI_AEAG_ROI_OFFSET_X, x);
            bool roi_y_set = cap.set(cv::CAP_PROP_XI_AEAG_ROI_OFFSET_Y, y);
            bool roi_w_set = cap.set(cv::CAP_PROP_XI_AEAG_ROI_WIDTH, w);
            bool roi_h_set = cap.set(cv::CAP_PROP_XI_AEAG_ROI_HEIGHT, h);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            std::string test_name = "ROI_" + std::to_string((int)(x*100)) + "_" + std::to_string((int)(y*100)) + 
                                   "_" + std::to_string((int)(w*100)) + "_" + std::to_string((int)(h*100));
            std::string details = "X:" + std::to_string(x) + " Y:" + std::to_string(y) + 
                                 " W:" + std::to_string(w) + " H:" + std::to_string(h);
            
            bool roi_set = roi_x_set && roi_y_set && roi_w_set && roi_h_set;
            logTest(test_name, roi_set, details);
            
            if (roi_set) {
                captureTestFrames(test_name, 3);
            }
        }
    }
    
    void performanceTest() {
        std::cout << "\n⚡ Performance & Stability Test...\n";
        
        // 设置高分辨率高帧率
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        int total_frames = 100;
        int captured_frames = 0;
        int empty_frames = 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < total_frames; i++) {
            cv::Mat frame;
            cap >> frame;
            
            if (frame.empty()) {
                empty_frames++;
            } else {
                captured_frames++;
                
                // 保存一些样本帧
                if (i % 20 == 0) {
                    std::string filename = result_dir + "/performance_frame_" + std::to_string(i) + ".jpg";
                    cv::imwrite(filename, frame);
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double actual_fps = (captured_frames * 1000.0) / duration.count();
        double success_rate = (double)captured_frames / total_frames * 100.0;
        
        std::string details = "Captured: " + std::to_string(captured_frames) + "/" + std::to_string(total_frames) + 
                             " (Success: " + std::to_string((int)success_rate) + "%) | " +
                             "Empty: " + std::to_string(empty_frames);
        
        logTest("Performance Test", success_rate >= 95.0, details, actual_fps);
    }
    
    void generateReport() {
        std::cout << "\n📊 Test Summary Report\n";
        std::cout << "========================\n";
        
        int passed = 0, total = 0;
        for (const auto& result : results) {
            total++;
            if (result.passed) passed++;
        }
        
        std::cout << "Overall Result: " << passed << "/" << total << " tests passed ("
                  << std::fixed << std::setprecision(1) << (100.0 * passed / total) << "%)\n\n";
        
        // 保存详细报告到文件
        std::ofstream report(result_dir + "/libcamera_test_report.txt");
        report << "LibCamera Detailed Test Report\n";
        report << "Generated: " << std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch()).count() << "\n\n";
        
        for (const auto& result : results) {
            report << (result.passed ? "[PASS]" : "[FAIL]") << " " 
                   << result.test_name << ": " << result.details;
            if (result.measured_value >= 0) {
                report << " (Measured: " << result.measured_value << ")";
            }
            report << "\n";
        }
        
        report.close();
        std::cout << "📄 Detailed report saved to: " << result_dir << "/libcamera_test_report.txt\n";
    }
    
    void runAllTests() {
        std::cout << "🚀 Starting LibCamera Detailed Testing...\n";
        std::cout << "==========================================\n";
        
        // 创建结果目录
        std::string mkdir_cmd = "mkdir -p " + result_dir;
        system(mkdir_cmd.c_str());
        std::cout << "📁 Results will be saved to: " << result_dir << "/\n\n";
        
        if (!initialize()) {
            std::cout << "❌ Initialization failed. Exiting.\n";
            return;
        }
        
        testResolutions();
        testFramerates();
        testImageQualitySettings();
        testExposureControl();
        testWhiteBalance();
        testROI();
        performanceTest();
        
        cap.release();
        generateReport();
        
        std::cout << "\n🎉 Testing completed!\n";
    }
};

int main() {
    // 设置测试结果目录
    std::string test_result_dir = "libcamera_test_results";
    
    std::cout << "📋 LibCamera Testing Suite\n";
    std::cout << "==========================\n";
    std::cout << "Test results will be saved to: " << test_result_dir << "/\n\n";
    
    LibcameraDetailedTester tester(test_result_dir);
    tester.runAllTests();
    return 0;
}