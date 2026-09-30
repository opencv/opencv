#include <cstdint>
#include <cstdio>
#include <string>
#include <unistd.h>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0) return 0;

    // Create a unique temporary file
    char filename[] = "/tmp/fuzz_videoio_XXXXXX";
    int fd = mkstemp(filename);
    if (fd == -1) return 0;

    // Write fuzzer data to file
    if (write(fd, data, size) != (ssize_t)size) {
        close(fd);
        unlink(filename);
        return 0;
    }
    close(fd);

    try {
        // Force use of the internal OpenCV MJPEG/AVI parser
        // This targets modules/videoio/src/cap_mjpeg_decoder.cpp and container_avi.cpp
        cv::VideoCapture cap(filename, cv::CAP_OPENCV_MJPEG);
        
        if (cap.isOpened()) {
            cv::Mat frame;
            // Retrieve frames until end or error
            // Limit to avoid infinite loops on looped videos or DoS
            int max_frames = 100; 
            while (max_frames-- > 0 && cap.grab()) {
                cap.retrieve(frame);
                // Optional: access frame properties to ensure decoding actually happened
                if (!frame.empty()) {
                   volatile int w = frame.cols;
                   (void)w;
                }
            }
        }
    } catch (...) {
        // Handle exceptions
    }

    // Cleanup
    unlink(filename);
    return 0;
}
