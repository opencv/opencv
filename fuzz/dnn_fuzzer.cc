#include <cstdint>
#include <cstddef>
#include <vector>
#include <opencv2/dnn.hpp>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) {
        return 0;
    }

    // Use the first byte to select the parser
    uint8_t parser_selector = data[0] % 4;
    const char* buffer = reinterpret_cast<const char*>(data + 1);
    size_t buffer_len = size - 1;

    try {
        if (parser_selector == 0) {
            // ONNX
            cv::dnn::readNetFromONNX(buffer, buffer_len);
        } else if (parser_selector == 1) {
            // TensorFlow (Buffer, Config Buffer (empty))
            cv::dnn::readNetFromTensorflow(buffer, buffer_len, NULL, 0);
        } else if (parser_selector == 2) {
            // Caffe (Buffer, Config Buffer (empty))
            cv::dnn::readNetFromCaffe(buffer, buffer_len, NULL, 0);
        } else if (parser_selector == 3) {
            // Darknet (Buffer, Config Buffer (empty))
            cv::dnn::readNetFromDarknet(buffer, buffer_len, NULL, 0);
        }
    } catch (...) {
        // Ignore parsing errors
    }

    return 0;
}