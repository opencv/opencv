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

    CvCapture_libcamera_proxy(size_t index = 0)
    {
        cm_ = std::make_unique<CameraManager>();
        cm_->start();
        std::cout << "libcamera cameras(): " << cm_->cameras().size() << std::endl;
        if (index >= cm_->cameras().size())
        {
            std::cerr << "Invalid camera index " << index << std::endl;
            return;
        }

        cameraId_ = cm_->cameras()[index]->id();
        camera_ = cm_->get(cameraId_);

        if (!camera_)
        {
            std::cerr << "Camera " << cameraId_ << " not found" << std::endl;
            return;
        }

        if (camera_->acquire())
        {
            std::cerr << "Failed to acquire camera " << cameraId_ << std::endl;
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

    private:
    bool getLibcameraPixelFormat(int value)
    {
        switch (value)
        {
            case FMT_MJPEG:
                pixelFormat_ = libcamera::formats::MJPEG;
                return true;

            case FMT_YUYV:
                pixelFormat_ = libcamera::formats::YUYV;
                return true;

            case FMT_RGB888:
                pixelFormat_ = libcamera::formats::RGB888;
                return true;

            case FMT_BGR888:
                pixelFormat_ = libcamera::formats::BGR888;
                return true;

            case FMT_NV12:
                pixelFormat_ = libcamera::formats::NV12;
                return true;

            case FMT_YUV420:
                pixelFormat_ = libcamera::formats::YUV420;
                return true;

            default:
                pixelFormat_ = libcamera::formats::YUYV;
                return false;
        }
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
    std::unique_ptr<CameraManager> cm_;
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
    PixelFormat pixelFormat_ = libcamera::formats::MJPEG;
};
}

#endif
