/*
 *  cap_libcamera.cpp
 *  For Video I/O
 *  by Advait Dhamorikar advaitdhamorikar[at]gmail[dot]com
 *  and Umang Jain email[at]uajain[dot]com
 *  Copyright 2025. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "precomp.hpp"
#include "cap_libcamera.hpp"

using namespace cv;
using namespace libcamera;

namespace cv {

void CvCapture_libcamera_proxy::cam_init()
{
   std::cout << "Initializing camera: " << cameraId_ << std::endl;
   opened_ = true;
}

void CvCapture_libcamera_proxy::requestComplete(Request *request)
{
   if (!request || !camera_)
       return;

   if (request->status() == Request::RequestCancelled)
       return;

   std::lock_guard<std::mutex> lock(mutex_);
   completedRequests_.push(request);
   requestAvailable_.notify_one();
}

int CvCapture_libcamera_proxy::mapFrameBuffer(const FrameBuffer *buffer)
{
    int error;
    if (buffer->planes().empty())
    {
        std::cerr << "Buffer has no planes " << std::endl;
        return -EINVAL;
    }
    maps_.clear();
    planes_.clear();
    planes_.reserve(buffer->planes().size());
    std::map<int, MappedBufferInfo> mappedBuffers;
    for (const FrameBuffer::Plane &plane : buffer->planes())
    {
        const int fd = plane.fd.get();
        if (mappedBuffers.find(fd) == mappedBuffers.end())
        {
            const size_t length = lseek(fd, 0, SEEK_END);
            mappedBuffers[fd] = MappedBufferInfo{ nullptr, 0, length };
        }
        const size_t length = mappedBuffers[fd].dmabufLength;
        if (plane.offset > length || plane.offset + plane.length > length)
        {
                    std::cerr << "plane is out of buffer: "
                     << "buffer length=" << length
                     << ", plane offset=" << plane.offset
                     << ", plane length=" << plane.length << std::endl;
            return -ERANGE;
        }
        size_t &mapLength = mappedBuffers[fd].mapLength;
        mapLength = std::max(mapLength,
                           static_cast<size_t>(plane.offset + plane.length));
    }
    for (const FrameBuffer::Plane &plane : buffer->planes())
    {
        const int fd = plane.fd.get();
        auto &info = mappedBuffers[fd];
        if (!info.address)
        {
            void *address = mmap(nullptr, info.mapLength, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, fd, 0);
            if (address == MAP_FAILED)
            {
                error = -errno;
                std::cerr <<  "Failed to mmap plane: "
                         << strerror(-error) << std::endl;
                return -error;
            }
            info.address = static_cast<uint8_t *>(address);
            maps_.emplace_back(info.address, info.mapLength);
        }

        planes_.emplace_back(info.address + plane.offset, plane.length);
    }
    return 0;
}

bool CvCapture_libcamera_proxy::icvSetFrameSize(int width, int height)
{
    if (width > 0)
        width_ = width;
    if (height > 0)
        height_ = height;

    streamConfig_.size.width = width_;
    streamConfig_.size.height = height_;

    return true;
}

int CvCapture_libcamera_proxy::convertToRgb(Request *request, OutputArray &outImage)
{
    FrameBuffer *fb = nullptr;
    const Request::BufferMap &buffers = request->buffers();
    for (const auto &[stream, buffer] : buffers)
    {
        if (stream->configuration().pixelFormat == pixelFormat_)
        {
            fb = buffer;
        }
    }

    int ret = mapFrameBuffer(fb);
    const FrameMetadata &metadata = fb->metadata();
    if (ret < 0 || !fb)
    {
        std::cerr << "Failed to mmap buffer." << std::endl;
        return ret;
    }

    unsigned char* data = static_cast<unsigned char*>(planes_[0].data());
    cv::Mat& destination = outImage.getMatRef();
    switch (pixFmt_)
    {
        case FMT_MJPEG:
        {
            cv::imdecode(
                cv::Mat(1, metadata.planes()[0].bytesused, CV_8U, data),
                IMREAD_COLOR, &destination);
            break;
        }
        case FMT_YUYV:
        {
            if (metadata.planes()[0].bytesused <
                config_->at(0).size.width * config_->at(0).size.height * 2)
            {
                std::cerr << "YUYV: Frame too small." << std::endl;
                return -1;
            }

            cv::cvtColor(
                cv::Mat(config_->at(0).size.height,
                        config_->at(0).size.width,
                        CV_8UC2, data),
                destination, cv::COLOR_YUV2BGR_YUYV);
            break;
        }
        case FMT_NV12:
        {
            cv::cvtColor(
                cv::Mat(config_->at(0).size.height * 3 / 2,
                        config_->at(0).size.width,
                        CV_8UC1, data),
                destination, cv::COLOR_YUV2BGR_NV12);
            break;
        }
        case FMT_RGB888:
        {
            destination = cv::Mat(config_->at(0).size.height,
                                config_->at(0).size.width,
                                CV_8UC3, data).clone();
            break;
        }
        case FMT_BGR888:
        {
            cv::cvtColor(
                cv::Mat(config_->at(0).size.height,
                        config_->at(0).size.width,
                        CV_8UC3, data),
                destination, cv::COLOR_BGR2RGB);
            break;
        }
        case FMT_UYVY:
        {
            cv::Mat yuyvFrame(config_->at(0).size.height,
                            config_->at(0).size.width,
                            CV_8UC2, data);
            cv::cvtColor(yuyvFrame, destination, cv::COLOR_YUV2BGR_YUY2);
            break;
        }
        case FMT_YUV420:
        {
            cv::cvtColor(
                cv::Mat(config_->at(0).size.height * 3 / 2,
                        config_->at(0).size.width,
                        CV_8UC1, data),
                destination, cv::COLOR_YUV2BGR_I420);
            break;
        }
        default:
        {
            if (metadata.planes()[0].bytesused <
                config_->at(0).size.width * config_->at(0).size.height * 2)
            {
                std::cerr << "YUYV: Frame too small." << std::endl;
                return -1;
            }
            cv::cvtColor(
                cv::Mat(config_->at(0).size.height,
                        config_->at(0).size.width,
                        CV_8UC2, data),
                destination, cv::COLOR_YUV2BGR_YUYV);
            break;
        }
    }

    return 0;
}

bool CvCapture_libcamera_proxy::open()
{
    std::unique_ptr<Request> request;
    unsigned int nbuffers = UINT_MAX;
    int ret = 0;
    try
    {
        allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
        for (StreamConfiguration &cfg : *config_)
        {
            ret = allocator_->allocate(cfg.stream());
            if (ret < 0)
            {
                std::cerr << "Can't allocate buffers" << std::endl;
                return false;
            }
            allocated_ = allocator_->buffers(cfg.stream()).size();
            nbuffers = std::min(nbuffers, allocated_);
        }

        for (unsigned int i = 0; i < nbuffers; i++)
        {
            request = camera_->createRequest();
            if (!request)
            {
                std::cerr << "Can't create request" << std::endl;
                return EXIT_FAILURE;
            }
            for (StreamConfiguration &cfg : *config_)
            {
                Stream *stream = cfg.stream();
                const std::vector<std::unique_ptr<FrameBuffer>> &buffers =
                allocator_->buffers(stream);
                const std::unique_ptr<FrameBuffer> &buffer = buffers[i];
                ret = request->addBuffer(stream, buffer.get());
                if (ret < 0)
                {
                    std::cerr << "Can't set buffer for request"<< std::endl;
                    return ret;
                }
            }
            requests_.push_back(std::move(request));
        }
        camera_->requestCompleted.connect(this, &CvCapture_libcamera_proxy::requestComplete);
        camera_->start();
        for (std::unique_ptr<Request> &req : requests_)
             camera_->queueRequest(req.get());

        return 1;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::cerr<<"CvCapture_libcamera_proxy::open failed"<<std::endl;
        opened_ = false;
    }

    return opened_;
}

bool CvCapture_libcamera_proxy::grabFrame()
{
    if (!opened_ && gc > 0)
    {
        ;
    }
    else if (opened_ && gc == 0)
    {
        // Generate configuration
        config_ = camera_->generateConfiguration({ strcfg_ });
        if (!config_ || config_->empty())
        {
            std::cerr << "Failed to generate stream configuration." << std::endl;
            return -1;
        }

        // Update configuration
        StreamConfiguration &cfg = config_->at(0);
        cfg.pixelFormat = pixelFormat_;
        cfg.size.width = width_;
        cfg.size.height = height_;

        // Validate config
        CameraConfiguration::Status status = config_->validate();

        if (status == CameraConfiguration::Invalid)
        {
            std::cerr << "Camera configuration is invalid!" << std::endl;
            return -1;
        }
        if (status == CameraConfiguration::Adjusted)
        {
            std::cout << "Camera configuration was adjusted by libcamera!" << std::endl;
        }

        camera_->configure(config_.get());
        streamConfig_ = cfg;

        gc++;
        open();
    }

    return true;
}

bool CvCapture_libcamera_proxy::retrieveFrame(int, OutputArray &outputFrame)
{
    std::unique_lock<std::mutex> lock(mutex_);
    requestAvailable_.wait(lock, [this] { return !completedRequests_.empty(); });

    Request *request = completedRequests_.front();
    completedRequests_.pop();
    lock.unlock();

    int ret = convertToRgb(request, outputFrame);
    if (ret < 0)
    {
        std::cerr << "convertToRGB failed\n";
        return false;
    }

    request->reuse(Request::ReuseBuffers);
    camera_->queueRequest(request);

    return !outputFrame.empty();
}

double CvCapture_libcamera_proxy::getProperty(int property_id) const
{
    switch (property_id)
    {
        case CAP_PROP_POS_FRAMES: return completedRequests_.front()->sequence();
        case CAP_PROP_FRAME_WIDTH: return streamConfig_.size.width;
        case CAP_PROP_FRAME_HEIGHT: return streamConfig_.size.height;
    }
    return 0;
}

bool CvCapture_libcamera_proxy::setProperty(int property_id, double value)
{
    switch (property_id)
    {
        case CAP_PROP_FRAME_WIDTH:
            return icvSetFrameSize(cvRound(value), height_);
        case CAP_PROP_FRAME_HEIGHT:
            return icvSetFrameSize(width_, cvRound(value));
        case CAP_PROP_MODE:
            pixFmt_ = cvRound(value);
            return getLibcameraPixelFormat(pixFmt_);
        case CAP_PROP_FORMAT:
            propFmt_ = cvRound(value);
            return getCameraConfiguration(propFmt_);
    }
    return false;
}

cv::Ptr<cv::IVideoCapture> create_libcamera_capture_cam(int index)
{
    cv::Ptr<CvCapture_libcamera_proxy> capture = cv::makePtr<CvCapture_libcamera_proxy>(index);
    if (capture)
    {
        return capture;
    }
    return cv::Ptr<cv::IVideoCapture>();
}
}//namespace cv