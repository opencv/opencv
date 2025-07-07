#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

int main()
{
    std::cout << "Testing OpenCV with Libcamera Backend\n";
    std::cout << "======================================\n\n";

    // --- 步骤 1: 尝试使用 Libcamera 后端打开摄像头 ---
    // 这是请求使用特定后端的标准公共 API
    std::cout << "Attempting to open camera with Libcamera backend (cv::CAP_LIBCAMERA)...\n";
    cv::VideoCapture cap(0, cv::CAP_LIBCAMERA);

    // --- 步骤 2: 检查摄像头是否成功打开 ---
    if (!cap.isOpened()) {
        std::cout << "\nERROR: Failed to open camera using cv::CAP_LIBCAMERA.\n";
        std::cout << "This could mean:\n";
        std::cout << "  1. No camera is connected or detected by the system.\n";
        std::cout << "  2. The Libcamera plugin failed to load or initialize.\n";
        std::cout << "     (Check if libopencv_videoio_libcamera.so is in /usr/local/lib)\n";
        std::cout << "  3. The camera is in use by another application.\n";
        std::cout << "  4. Permission issues (e.g., user not in 'video' group).\n";
        std::cout << "  5. Libcamera itself is not configured correctly.\n";
        return -1;
    }
    std::cout << "Camera opened successfully!\n\n";

    // --- 步骤 3: 验证实际使用的后端 ---
    // 这是确认哪个后端被实际使用的标准公共 API
    std::cout << "Verifying backend used...\n";
    std::string backend_name = cap.getBackendName();
    std::cout << "Actual backend used: " << backend_name << "\n";

    if (backend_name != "LIBCAMERA") {
        std::cout << "\nWARNING: OpenCV did not use the LIBCAMERA backend as requested.\n";
        std::cout << "It fell back to another backend (" << backend_name << ").\n";
        std::cout << "This means the Libcamera plugin is likely not working correctly, even though a camera was found.\n";
        // 你可以选择在这里退出，或者继续测试，但要知道测试的不是 Libcamera
        // return -1;
    } else {
        std::cout << "Success! The LIBCAMERA backend is active.\n\n";
    }

    // --- 步骤 4: 获取摄像头属性 ---
    double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "Camera properties reported by backend:\n";
    std::cout << "  Resolution: " << width << "x" << height << "\n";
    std::cout << "  FPS: " << fps << "\n\n";

    // --- 步骤 5: 尝试捕获一些帧 ---
    std::cout << "Capturing test frames...\n";
    cv::Mat frame;
    int frame_count = 0;
    const int num_frames_to_capture = 10;

    for (int i = 0; i < num_frames_to_capture; ++i) {
        if (cap.read(frame)) {
            frame_count++;
            std::cout << "  Frame " << i + 1 << ": OK (" << frame.cols << "x" << frame.rows
                      << ", type: " << frame.type() << ")\n";
        } else {
            std::cout << "  Frame " << i + 1 << ": FAILED\n";
        }
    }

    // --- 步骤 6: 报告最终结果 ---
    std::cout << "\nSuccessfully captured " << frame_count << "/" << num_frames_to_capture << " frames.\n";

    cap.release(); // 释放摄像头

    if (frame_count > 0 && backend_name == "LIBCAMERA") {
        std::cout << "\nLibcamera plugin test: SUCCESS\n";
        return 0;
    } else {
        std::cout << "\nLibcamera plugin test: FAILED\n";
        if (backend_name != "LIBCAMERA") {
            std::cout << "Reason: Libcamera backend was not used.\n";
        } else if (frame_count == 0) {
            std::cout << "Reason: Failed to capture any frames, even though the camera was opened.\n";
        }
        return -1;
    }
}