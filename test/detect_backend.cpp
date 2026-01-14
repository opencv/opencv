// Running method:
// g++ -std=c++17 -o detect_backend detect_backend.cpp $(pkg-config --cflags --libs opencv4) && ./detect_backend

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <thread>   
#include <chrono>   

int main() {
    std::cout << "=== Direct Test of OpenCV LIBCAMERA Backend ===" << std::endl;

    // --- Step 1: Open ---
    std::cout << "\nAttempting to open camera with cv::CAP_LIBCAMERA..." << std::endl;
    cv::VideoCapture cap(0, cv::CAP_LIBCAMERA);

    // --- Step 2: Check if the camera was opened successfully ---
    if (!cap.isOpened()) {
        std::cout << "❌ FAILED: cap.isOpened() returned false." << std::endl;
        std::cout << "   This means OpenCV could not open the camera using the LIBCAMERA backend." << std::endl;
        std::cout << "   Possible reasons: Plugin not found, camera not connected, permissions error, etc." << std::endl;
        return -1;
    }

    // --- Step 3: Check the actual backend used ---
    std::string backend_name = cap.getBackendName();
    std::cout << "✅ SUCCESS: cap.isOpened() returned true." << std::endl;
    std::cout << "   Actual backend in use: " << backend_name << std::endl;

    if (backend_name == "LIBCAMERA") {
        std::cout << "   ✅ VERIFIED: The LIBCAMERA backend is confirmed to be working!" << std::endl;
    } else {
        std::cout << "   ⚠️ WARNING: A camera was opened, but it used the '" << backend_name << "' backend, not LIBCAMERA." << std::endl;
        std::cout << "   This means the LIBCAMERA plugin is likely not working correctly or has a lower priority." << std::endl;
        cap.release();
        return -1;
    }

    std::cout << "\nAttempting to capture a single frame..." << std::endl;
    cv::Mat frame;

    // 热身用，让自动曝光正常运行
    // Warming up to allow auto-exposure to stabilize
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); 

    if (cap.read(frame) && !frame.empty()) {
        std::cout << "✅ SUCCESS: A frame was captured." << std::endl;
        std::cout << "   Frame size: " << frame.cols << "x" << frame.rows << std::endl;
        cv::imwrite("libcamera_test_frame.jpg", frame);
        std::cout << "   Frame saved as 'libcamera_test_frame.jpg'." << std::endl;
    } else {
        std::cout << "❌ FAILED: Could not read a valid frame from the camera." << std::endl;
        cap.release();
        return -1;
    }

    cap.release();
    return 0;
}