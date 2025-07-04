#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <sstream>
#include <unistd.h>


int main() {
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "cap initialization failed" << std::endl;
        return -1;
    } else {
        std::ostringstream backEndName;
        backEndName << cap.getBackendName();
        std::cout << "Using backend: " << backEndName.str() << std::endl;
    }
    // all sets have default values, no need to explicitly specify
    // here just to test setProperty()
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FRAME_WIDTH,200);
    cap.set(cv::CAP_PROP_SHARPNESS, 0.5);
    cap.set(cv::CAP_PROP_CONTRAST, 0.5);
    cap.set(cv::CAP_PROP_BRIGHTNESS, 0.8);
    cap.set(cv::CAP_PROP_EXPOSURE,0.6);

    std::cerr << "capture started!!!" << std::endl;
    int frame_count = 0;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame_count > 20){
            break;
        }
        // check if the frame is empty
        if (frame.empty()) {
            std::cerr << "Error: No frame captured" << std::endl;
            break;
        } else {
            std::cerr << "Success." << std::endl;
            std::ostringstream filename;
            filename << "./frame_" << frame_count%20 << ".jpg";
            cv::imwrite(filename.str(), frame);
            frame_count++;

            // cv::imshow("Video", frame);
        }
        
        // if (cv::waitKey(30) >= 0) break;  // 移除waitKey以避免GTK依赖
    }

    cap.release();
    // cv::destroyAllWindows();  // 移除以避免GTK依赖
    return 0;
}