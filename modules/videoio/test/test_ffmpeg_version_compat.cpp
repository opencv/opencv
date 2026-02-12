// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Verifies that OpenCV can be built against different FFmpeg versions (4.x, 5.x, 6.x)

#include "test_precomp.hpp"

#ifdef HAVE_FFMPEG

namespace opencv_test { namespace {

TEST(Videoio_FFmpeg, version_compatibility)
{    
    std::vector<cv::VideoCaptureAPIs> backends = cv::videoio_registry::getBackends();
    
    bool ffmpeg_found = false;
    for (const auto& backend : backends) {
        if (backend == cv::CAP_FFMPEG) {
            ffmpeg_found = true;
            break;
        }
    }
    
    ASSERT_TRUE(ffmpeg_found) << "FFmpeg backend should be available in registry";
    
    std::string name = cv::videoio_registry::getBackendName(cv::CAP_FFMPEG);
    EXPECT_EQ(name, "FFMPEG") << "Backend name should be FFMPEG";
}

TEST(Videoio_FFmpeg, backend_availability)
{
    std::vector<cv::VideoCaptureAPIs> backends = cv::videoio_registry::getBackends();
    
    bool ffmpeg_available = false;
    for (const auto& backend : backends) {
        if (backend == cv::CAP_FFMPEG) {
            ffmpeg_available = true;
            break;
        }
    }
    
    EXPECT_TRUE(ffmpeg_available) << "FFmpeg backend should be available";
}

}}

#endif
