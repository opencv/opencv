// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_VIDEOIO_VIDEOIO_REGISTRY_HPP__
#define __OPENCV_VIDEOIO_VIDEOIO_REGISTRY_HPP__

namespace cv
{

/** Capabilities bitmask */
enum BackendMode {
    MODE_CAPTURE_BY_INDEX    = 1 << 0,           //!< device index
    MODE_CAPTURE_BY_FILENAME = 1 << 1,           //!< filename or device path (v4l2)
    MODE_WRITER              = 1 << 4,            //!< writer

    MODE_CAPTURE_ALL = MODE_CAPTURE_BY_INDEX + MODE_CAPTURE_BY_FILENAME,
};

struct VideoBackendInfo {
    VideoCaptureAPIs id;
    BackendMode mode;
    int priority;     // 1000-<index*10> - default builtin priority
                      // 0 - disabled (OPENCV_VIDEOIO_PRIORITY_<name> = 0)
                      // >10000 - prioritized list (OPENCV_VIDEOIO_PRIORITY_LIST)
    const char* name;
};

namespace videoio_registry {

std::vector<VideoBackendInfo> getAvailableBackends_CaptureByIndex();
std::vector<VideoBackendInfo> getAvailableBackends_CaptureByFilename();
std::vector<VideoBackendInfo> getAvailableBackends_Writer();

} // namespace

void VideoCapture_create(CvCapture*& capture, Ptr<IVideoCapture>& icap, VideoCaptureAPIs api, int index);
void VideoCapture_create(CvCapture*& capture, Ptr<IVideoCapture>& icap, VideoCaptureAPIs api, const cv::String& filename);
void VideoWriter_create(CvVideoWriter*& writer, Ptr<IVideoWriter>& iwriter, VideoCaptureAPIs api,
        const String& filename, int fourcc, double fps, const Size& frameSize, bool isColor);

} // namespace
#endif // __OPENCV_VIDEOIO_VIDEOIO_REGISTRY_HPP__
