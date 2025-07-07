#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "=== OpenCV VideoIO Backend Detection ===" << std::endl;
    
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "Build information:" << std::endl;
    std::cout << cv::getBuildInformation() << std::endl;
    
    cv::VideoCapture cap;
    
    std::cout << "\n=== Testing LIBCAMERA Backend ===" << std::endl;
    bool libcamera_available = cap.open(0, cv::CAP_LIBCAMERA);
    if (libcamera_available) {
        std::cout << "✅ LIBCAMERA backend is available and working" << std::endl;
        std::cout << "Backend name: " << cap.getBackendName() << std::endl;

        double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        
        std::cout << "Default resolution: " << width << "x" << height << std::endl;
        std::cout << "Default FPS: " << fps << std::endl;
        
        cap.release();
    } else {
        std::cout << "❌ LIBCAMERA backend is NOT available" << std::endl;
    }
    
    std::cout << "\n=== Testing Default Backend ===" << std::endl;
    bool default_available = cap.open(0);
    if (default_available) {
        std::cout << "✅ Default backend is working" << std::endl;
        std::cout << "Backend name: " << cap.getBackendName() << std::endl;
        cv::Mat dummy_frame;
        for (int i = 0; i < 30; ++i) {
            cap.read(dummy_frame);
        }
        
        cv::Mat frame;
        cap >> frame;
        if (!frame.empty()) {
            cv::imwrite("test_frame.jpg", frame);
            std::cout << "Captured a test frame and saved as test_frame.jpg" << std::endl;
        } else {
            std::cout << "❌ Failed to capture a frame" << std::endl;
        }

        cap.release();
    } else {
        std::cout << "❌ No working backend found" << std::endl;
    }
    
    return 0;
}
