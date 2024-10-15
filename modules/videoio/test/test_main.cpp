// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include <opencv2/core/utils/logger.hpp>

#if defined(HAVE_HPX)
    #include <hpx/hpx_main.hpp>
#endif

static
void initTests()
{
    const std::vector<cv::VideoCaptureAPIs> backends = cv::videoio_registry::getStreamBackends();
    const char* requireFFmpeg = getenv("OPENCV_TEST_VIDEOIO_BACKEND_REQUIRE_FFMPEG");
    if (requireFFmpeg && !isBackendAvailable(cv::CAP_FFMPEG, backends))
    {
        CV_LOG_FATAL(NULL, "OpenCV-Test: required FFmpeg backend is not available (broken plugin?). STOP.");
        exit(1);
    }
}

CV_TEST_MAIN("highgui", initTests())
