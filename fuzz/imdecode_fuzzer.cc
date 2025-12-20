#include <cstdint>
#include <cstddef>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Input size limit (OSS-Fuzz recommendation for image parsers is ~1MB, 
    // but 256KB is often enough for headers/structure)
    if (size == 0 || size > 1024 * 1024) {
        return 0;
    }

    // Construct a 1xSize byte array header pointing to raw data
    cv::Mat raw_data(1, size, CV_8UC1, (void*)data);

    try {
        cv::Mat img = cv::imdecode(raw_data, cv::IMREAD_UNCHANGED);
        
        // Optional: access some pixels to force processing? 
        // usually decode is enough to trigger parser bugs.
        if (!img.empty()) {
            // Just ensure we actually touched the memory
            volatile int w = img.cols;
            volatile int h = img.rows;
            volatile int type = img.type();
            (void)w;
            (void)h;
            (void)type;
        }
    } catch (...) {
        // Catch all exceptions to prevent fuzzer crash (unless it's a segfault/ASan error)
    }

    return 0;
}
