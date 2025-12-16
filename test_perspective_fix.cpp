// Test case to reproduce the perspectiveTransform bug
#include <opencv2/core.hpp>
#include <iostream>

int main() {
    // Create a simple 2D point
    cv::Mat src(1, 1, CV_32FC2);
    src.at<cv::Vec2f>(0, 0) = cv::Vec2f(10.0f, 20.0f);
    
    // Create a simple perspective transform matrix (3x3)
    cv::Mat transformMatrix = (cv::Mat_<double>(3, 3) << 
        1.0, 0.0, 5.0,
        0.0, 1.0, 10.0,
        0.0, 0.0, 2.0);
    
    cv::Mat dst(1, 1, CV_32FC2);
    
    // Apply perspective transform
    cv::perspectiveTransform(src, dst, transformMatrix);
    
    // Expected result: (10*1 + 20*0 + 5) / (10*0 + 20*0 + 2) = 15 / 2 = 7.5
    //                  (10*0 + 20*1 + 10) / (10*0 + 20*0 + 2) = 30 / 2 = 15.0
    cv::Vec2f result = dst.at<cv::Vec2f>(0, 0);
    
    std::cout << "Input: (" << src.at<cv::Vec2f>(0, 0)[0] << ", " 
              << src.at<cv::Vec2f>(0, 0)[1] << ")" << std::endl;
    std::cout << "Output: (" << result[0] << ", " << result[1] << ")" << std::endl;
    std::cout << "Expected: (7.5, 15.0)" << std::endl;
    
    // Check if result is correct
    float epsilon = 1e-5f;
    if (std::abs(result[0] - 7.5f) < epsilon && std::abs(result[1] - 15.0f) < epsilon) {
        std::cout << "TEST PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED!" << std::endl;
        return 1;
    }
}
