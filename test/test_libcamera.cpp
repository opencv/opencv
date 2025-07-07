#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <thread>
#include <sstream>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

int main() {
    struct stat st = {0};
    if (stat("./frame", &st) == -1) {
        mkdir("./frame", 0755);
        std::cout << "Created frame directory" << std::endl;
    }

    cv::VideoCapture cap;
    // cap.open(0);  
    cap.open(0 , cv::CAP_LIBCAMERA); 
    
    if (!cap.isOpened()) {
        std::cerr << "cap initialization failed" << std::endl;
        return -1;
    } else {
        std::string backend = cap.getBackendName();
        std::cout << "Using backend: " << backend << std::endl;
        
        if (backend != "LIBCAMERA") {
            std::cout << "Warning: Not using LIBCAMERA backend!" << std::endl;
        }
    }
    
    bool width_set = cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    bool height_set = cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 关键延迟
    
    bool fps_set = cap.set(cv::CAP_PROP_FPS, 30);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(300)); // 再次延迟
    
    // cap.set(cv::CAP_PROP_SHARPNESS, 0.5);
    // cap.set(cv::CAP_PROP_CONTRAST, 0.5);
    // cap.set(cv::CAP_PROP_BRIGHTNESS, 0.8);
    // cap.set(cv::CAP_PROP_EXPOSURE,0.6);

    std::cerr << "capture started!!!" << std::endl;
    int frame_count = 0;

    // 重要：更长的等待时间让摄像头稳定
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    while (true) {
        cv::Mat frame;
        
        cap >> frame;
        
        if (frame_count > 20){
            break;
        }

        if (frame.empty()) {
            std::cerr << "Error: No frame captured" << std::endl;
            break;
        } else {
            if (frame_count % 5 == 0) {  // 每5帧保存一次
                std::string filename = "./frame/frame_" + std::to_string(frame_count) + ".jpg";
                bool saved = cv::imwrite(filename, frame);
                if (saved) {
                    std::cout << "✅ Saved frame: " << filename << std::endl;
                } else {
                    std::cout << "❌ Failed to save: " << filename << std::endl;
                }
            }
            
            std::cerr << "Success - frame " << frame_count << " processed." << std::endl;
            frame_count++;

            // cv::imshow("Video", frame);
        }
        
        // if (cv::waitKey(30) >= 0) break;  // 移除waitKey以避免GTK依赖
    }

    cap.release();
    // cv::destroyAllWindows();  // 移除以避免GTK依赖
    return 0;
}
