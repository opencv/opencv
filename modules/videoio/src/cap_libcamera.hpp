// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, Advait Dhamorikar all rights reserved.

#ifdef HAVE_LIBCAMERA

#include <iostream>
#include <sys/mman.h>
#include <errno.h>
#include <memory>
#include <queue>
#include <map>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <libcamera/libcamera.h>
#include <libcamera/framebuffer.h>
#include <libcamera/base/span.h>
#include <condition_variable>

using namespace cv;
using namespace libcamera;

namespace cv {

class CvCapture_libcamera_proxy CV_FINAL : public cv::IVideoCapture
{
public:
    bool isOpened() const CV_OVERRIDE { return opened_; }
    bool open();
    bool grabFrame() CV_OVERRIDE;
    bool retrieveFrame(int, OutputArray) CV_OVERRIDE;
    virtual double getProperty(int) const CV_OVERRIDE;
    virtual bool setProperty(int, double) CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE { return cv::CAP_LIBCAMERA; }

    static std::shared_ptr<CameraManager> cm_;

    CvCapture_libcamera_proxy(size_t index = 0)
    {
        if (!cm_)
        {
            cm_ = std::make_shared<CameraManager>();
            cm_->start();
        }
        std::cout << "libcamera cameras(): " << cm_->cameras().size() << std::endl;
        if (index >= cm_->cameras().size())
        {
            CV_LOG_ERROR(NULL, "Invalid camera index");
            return;
        }

        cameraId_ = cm_->cameras()[index]->id();
        camera_ = cm_->get(cameraId_);

        if (!camera_)
        {
            CV_LOG_ERROR(NULL, cv::format("Camera %s not found", cameraId_.c_str()));
            return;
        }

        if (camera_->acquire())
        {
            CV_LOG_ERROR(NULL, cv::format("Failed to acquire camera %s", cameraId_.c_str()));
            return;
        }

        cam_init();
   }


    ~CvCapture_libcamera_proxy()
    {
        if (!opened_)
            return;

        if (camera_)
        {
            camera_->stop();
        }

        for (auto &req : requests_)
        {
            if (req && req->status() == libcamera::Request::RequestPending)
            {
                camera_->release();
            }
        }


        allocator_.reset();
        requests_.clear();
        planes_.clear();
        maps_.clear();

        config_.reset();
        streamConfig_ = {};

        if (camera_)
        {
            camera_->release();
            camera_.reset();
        }

        if (cm_)
        {
            cm_->stop();
            cm_.reset();
        }

        cameraId_.clear();
        opened_ = false;
        open_ = false;
        std::cout << "Closing the device" << std::endl;
    }

    bool getLibcameraPixelFormat(int value = 0)
    {
        if (!config_)
        {
            CV_LOG_ERROR(NULL, "Camera configuration missing.");
            return false;
        }

        // Retrieve supported formats
        const StreamConfiguration &cfg = config_->at(0);
        const StreamFormats &formats = cfg.formats();
        std::map<int, std::pair<libcamera::PixelFormat, int>> formatMap =
        {
            {FMT_MJPEG, {libcamera::formats::MJPEG, FMT_MJPEG}},
            {FMT_YUYV, {libcamera::formats::YUYV, FMT_YUYV}},
            {FMT_NV12, {libcamera::formats::NV12, FMT_NV12}},
            {FMT_NV21, {libcamera::formats::NV21, FMT_NV21}},
            {FMT_RGB888, {libcamera::formats::RGB888, FMT_RGB888}},
            {FMT_BGR888, {libcamera::formats::BGR888, FMT_BGR888}},
            {FMT_UYVY, {libcamera::formats::UYVY, FMT_UYVY}},
            {FMT_YUV420, {libcamera::formats::YUV420, FMT_YUV420}}
        };

        // Log all supported formats
        CV_LOG_DEBUG(NULL, "Supported pixel formats:");
        std::vector<PixelFormat> availableFormats = formats.pixelformats();
        for (PixelFormat pixelformat : availableFormats)
        {
            std::ostringstream formatStream;
            formatStream << pixelformat;
            CV_LOG_INFO(NULL, " * " << formatStream.str() << " -> "
            << formats.range(pixelformat).toString());
        }

        // Ensure requested format is available
        auto it = formatMap.find(value);
        if (it != formatMap.end() && std::find(availableFormats.begin(),
            availableFormats.end(), it->second.first) != availableFormats.end())
        {
            pixelFormat_ = it->second.first;
            pixFmt_ = it->second.second;
            CV_LOG_INFO(NULL, "Pixel format set to: " << pixelFormat_ << ", pixFmt_: " << pixFmt_);
            return true;
        }

        // If requested format isn't supported, fallback to supported format
        for (const auto &entry : formatMap)
        {
            if (std::find(availableFormats.begin(), availableFormats.end(),
                entry.second.first) != availableFormats.end())
            {
                pixelFormat_ = entry.second.first;
                pixFmt_ = entry.second.second;
                CV_LOG_WARNING(NULL, "Requested format not supported. Defaulting to: "
                    << pixelFormat_ << ", pixFmt_: " << pixFmt_);
                return true;
            }
        }

        CV_LOG_ERROR(NULL, "No OpenCV-supported pixel formats available.");
        return false;
    }

    bool getCameraConfiguration(int value)
    {
     switch (value)
       {
        case ROLE_RAW:
            strcfg_ = StreamRole::Raw;
            return true;
        case ROLE_STILL:
            strcfg_ = StreamRole::StillCapture;
            return true;
        case ROLE_VIDEO:
            strcfg_ = StreamRole::VideoRecording;
            return true;
        case ROLE_VIEWFINDER:
            strcfg_ = StreamRole::Viewfinder;
            return true;
        default:
            strcfg_ = StreamRole::VideoRecording;
            return true;
        }
    }

    void requestComplete(Request *request);
    void cam_init();
    void cam_init(int index);
    int mapFrameBuffer(const FrameBuffer *buffer);
    int convertToRgb(libcamera::Request *req, OutputArray &outImage);
    bool icvSetFrameSize(int, int);

    std::queue<Request*> completedRequests_;
    std::unique_ptr<CameraConfiguration> config_;
    std::shared_ptr<Camera> camera_;
    std::string cameraId_;
    std::vector<libcamera::Span<uint8_t>> planes_;
    std::vector<libcamera::Span<uint8_t>> maps_;
    std::vector<std::unique_ptr<Request>> requests_;
    std::unique_ptr<FrameBufferAllocator> allocator_;
    std::condition_variable requestAvailable_;
    std::mutex mutex_;

    int width_ = 480, height_ = 640;
    int pixFmt_;
    int propFmt_;
    int gc = 0;
    unsigned int allocated_;
    bool opened_ = false;
    bool open_ = false;

    struct MappedBufferInfo
    {
        uint8_t *address = nullptr;
        size_t mapLength = 0;
        size_t dmabufLength = 0;
    };
    StreamConfiguration streamConfig_;
    StreamRole strcfg_ = StreamRole::VideoRecording;
    PixelFormat pixelFormat_;
};

std::shared_ptr<CameraManager> CvCapture_libcamera_proxy::cm_ = nullptr;

}

#endif
