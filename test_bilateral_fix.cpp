#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

int main() {
    std::cout << "Testing bilateralFilter fix for issue #28254..." << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl << std::endl;
    
    // Create a small 2x2 CV_32FC1 image with values in range [100, 200]
    // This ensures BORDER_CONSTANT padding (default value 0) is outside the range,
    // which triggers the out-of-bounds condition in the LUT access
    cv::Mat src(2, 2, CV_32FC1);
    src.at<float>(0, 0) = 100.0f;
    src.at<float>(0, 1) = 150.0f;
    src.at<float>(1, 0) = 175.0f;
    src.at<float>(1, 1) = 200.0f;
    cv::Mat dst;
    
    std::cout << "Source image:" << std::endl << src << std::endl << std::endl;
    
    // Parameters that trigger the bug
    int d = -1;
    double sigmaColor = 2.7;
    double sigmaSpace = 44.5;
    int borderType = cv::BORDER_CONSTANT;
    
    std::cout << "Applying bilateralFilter with:" << std::endl;
    std::cout << "  d = " << d << std::endl;
    std::cout << "  sigmaColor = " << sigmaColor << std::endl;
    std::cout << "  sigmaSpace = " << sigmaSpace << std::endl;
    std::cout << "  borderType = BORDER_CONSTANT" << std::endl << std::endl;
    
    try {
        cv::bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType);
        std::cout << "SUCCESS: bilateralFilter completed without crash" << std::endl;
        std::cout << "Output image:" << std::endl << dst << std::endl << std::endl;
        
        // Verify output is valid
        if (dst.empty()) {
            std::cerr << "ERROR: Output image is empty" << std::endl;
            return 1;
        }
        if (dst.size() != src.size()) {
            std::cerr << "ERROR: Output size mismatch" << std::endl;
            return 1;
        }
        if (dst.type() != src.type()) {
            std::cerr << "ERROR: Output type mismatch" << std::endl;
            return 1;
        }
        
        std::cout << "All checks passed!" << std::endl;
        std::cout << std::endl << "NOTE: If you're running this WITHOUT the fix, it should crash" << std::endl;
        std::cout << "      or trigger AddressSanitizer with heap-buffer-overflow." << std::endl;
        return 0;
    } catch (const cv::Exception& e) {
        std::cerr << "ERROR: OpenCV exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception: " << e.what() << std::endl;
        return 1;
    }
}
