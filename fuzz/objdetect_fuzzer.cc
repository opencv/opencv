#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/barcode.hpp>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 5) {
        return 0;
    }

    uint8_t mode = data[0] % 3;
    
    // Simple dimension extraction to create a valid image
    // Use 16-bit to allow up to ~65k, but cap typically at 1024 or 2048 for performance
    uint16_t width = *reinterpret_cast<const uint16_t*>(data + 1) % 1024 + 1;
    uint16_t height = *reinterpret_cast<const uint16_t*>(data + 3) % 1024 + 1;

    size_t header_size = 5;
    size_t pixel_data_size = size - header_size;
    
    // Ensure we have enough data for the requested size, or adjust size to fit data
    // We prefer "adjust size to fit data" or "use data as buffer".
    // Let's create a Mat that wraps the remaining data.
    
    // Calculate max possible area with remaining data
    size_t area = pixel_data_size;
    if (area == 0) return 0;

    // Adjust width/height to fit area
    if ((size_t)width * height > area) {
        // Try to keep width, adjust height
        height = area / width;
        if (height == 0) {
            height = 1;
            width = area;
        }
    }
    
    // Construct Mat (GRAYSCALE for simplicity, most detectors work on gray)
    cv::Mat img(height, width, CV_8UC1, (void*)(data + header_size));

    try {
        if (mode == 0) {
            // QRCode Standard
            cv::QRCodeDetector detector;
            std::vector<cv::Point> points;
            std::string output;
            output = detector.detectAndDecode(img, points);
        } else if (mode == 1) {
            // QRCode Curved
            cv::QRCodeDetector detector;
            // detectAndDecodeCurved returns string directly
            detector.detectAndDecodeCurved(img);
        } else if (mode == 2) {
            // Barcode
            cv::barcode::BarcodeDetector detector;
            std::vector<std::string> decoded_info;
            std::vector<std::string> decoded_type;
            detector.detectAndDecodeWithType(img, decoded_info, decoded_type);
        }
    } catch (...) {
        // Ignore processing errors
    }

    return 0;
}
