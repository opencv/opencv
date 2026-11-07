#include <cstdint>
#include <cstddef>
#include <string>
#include <opencv2/core.hpp>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0) {
        return 0;
    }

    // Convert data to a string
    std::string input_data(reinterpret_cast<const char*>(data), size);

    try {
        // Try to open as memory buffer
        cv::FileStorage fs(input_data, cv::FileStorage::READ | cv::FileStorage::MEMORY);
        
        // Iterate through nodes to trigger parsing of values
        if (fs.isOpened()) {
            cv::FileNode root = fs.root();
            for (cv::FileNodeIterator it = root.begin(); it != root.end(); ++it) {
                // Just access the item
                cv::FileNode item = *it;
                (void)item.name();
            }
        }
    } catch (...) {
        // Ignore parsing errors
    }

    return 0;
}
