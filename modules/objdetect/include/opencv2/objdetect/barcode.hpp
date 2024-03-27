#ifndef OPENCV_OBJDETECT_BARCODE_HPP
#define OPENCV_OBJDETECT_BARCODE_HPP

#include <opencv2/core.hpp>
#include <opencv2/objdetect/graphical_code_detector.hpp>

namespace cv {
namespace barcode {

class CV_EXPORTS_W_SIMPLE BarcodeDetector : public cv::GraphicalCodeDetector {
public:
    CV_WRAP BarcodeDetector();
    CV_WRAP BarcodeDetector(const std::string &prototxt_path, const std::string &model_path);
    ~BarcodeDetector();

    CV_WRAP bool decodeWithType(InputArray img,
                             InputArray points,
                             CV_OUT std::vector<std::string> &decoded_info,
                             CV_OUT std::vector<std::string> &decoded_type) const;

    CV_WRAP bool detectAndDecodeWithType(InputArray img,
                                      CV_OUT std::vector<std::string> &decoded_info,
                                      CV_OUT std::vector<std::string> &decoded_type,
                                      OutputArray points = noArray()) const;

private:
    // Private member variables to store dynamically adjusted parameters
    float gradient_threshold; // Gradient magnitude threshold
    std::vector<float> filter_window_sizes; // Filter window sizes

    // Method to dynamically adjust parameters based on input image resolution
    void adjustParameters(const Mat& img) const {
        // Calculate downsampling factor based on input image resolution
        float downsampling_factor = std::min(512.0f / img.cols, 512.0f / img.rows);

        // Adjust gradient threshold
        gradient_threshold = 64.0f * downsampling_factor;

        // Adjust filter window sizes
        filter_window_sizes = {0.01f * downsampling_factor, 0.03f * downsampling_factor, 
                               0.06f * downsampling_factor, 0.08f * downsampling_factor};
    }
};

BarcodeDetector::BarcodeDetector() {
    gradient_threshold = 64.0f;
    filter_window_sizes = {0.01f, 0.03f, 0.06f, 0.08f};
}

BarcodeDetector::BarcodeDetector(const std::string &prototxt_path, const std::string &model_path) {
    // Initialize with default parameters
    gradient_threshold = 64.0f;
    filter_window_sizes = {0.01f, 0.03f, 0.06f, 0.08f};
}

BarcodeDetector::~BarcodeDetector() {}

bool BarcodeDetector::decodeWithType(InputArray img,
                                     InputArray points,
                                     std::vector<std::string> &decoded_info,
                                     std::vector<std::string> &decoded_type) const {
    // Adjust parameters based on input image resolution
    adjustParameters(img.getMat());
    
    // Your decoding logic here
}

bool BarcodeDetector::detectAndDecodeWithType(InputArray img,
                                              std::vector<std::string> &decoded_info,
                                              std::vector<std::string> &decoded_type,
                                              OutputArray points) const {
    // Adjust parameters based on input image resolution
    adjustParameters(img.getMat());
    
    // Your detection and decoding logic here
}

} // namespace barcode
} // namespace cv

#endif // OPENCV_OBJDETECT_BARCODE_HPP
