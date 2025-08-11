// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * Portions of this code are derived from https://github.com/kbarni/LCCV, which is licensed under the BSD 2-Clause "Simplified" License.
 * The original code is Copyright (C) 2021, Raspberry Pi (Trading) Ltd..
 * For the original BSD-licensed code, the following applies:
 *
 * Copyright (C) 2021, Raspberry Pi (Trading) Ltd.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*!
 * \file cap_libcamera.cpp
 * \file cap_libcamera.hpp
 * 
 * \author Xuanrui Zhu <sulingdie@gmail.com>
 * 
 * \author Jianru Xu <vegetableplanes@gmail.com>
 *
 * \author Zhian Chen <czabewin@gmail.com>
 *
 * \brief Use Libcamera to read/write video
 */
#include "precomp.hpp"
#include "cap_libcamera.hpp"
#include <mutex>

/**
 * @brief implementation of the LibcameraApp class and LibcameraCapture
 * The LibcameraApp implements is from LCCV
 * Source: https://github.com/kbarni/LCCV
*/

namespace cv
{

/* ***************Start of methods from original LibcameraApp class*************** */
std::string const &LibcameraCapture::CameraId() const
{
    return camera_->id();
}

void LibcameraCapture::OpenCamera()
{

    if (options_->verbose)
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Opening camera...");
    CV_Assert(getCameraManager()->cameras().size() != 0 && "no cameras available");
    CV_Assert(options_->camera < getCameraManager()->cameras().size() && "camera index out of range");
    std::string const &cam_id = getCameraManager()->cameras()[options_->camera]->id();
    camera_ = getCameraManager()->get(cam_id);
    if (!camera_)
        CV_Error(cv::Error::StsAssert, "failed to find camera " + cam_id);
    if (!camera_acquired_ && camera_->acquire())
        CV_Error(cv::Error::StsAssert, "failed to acquire camera " + cam_id);
    camera_acquired_ = true;
    if (options_->verbose)
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Acquired camera " << cam_id);
}


void LibcameraCapture::CloseCamera()
{
    if (camera_acquired_)
        camera_->release();
    camera_acquired_ = false;

    camera_.reset();

    if (options_->verbose && !options_->help)
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Camera closed");
}

void LibcameraCapture::ConfigureViewfinder()
{
    if (options_->verbose)
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Configuring viewfinder...");

    StreamRoles stream_roles = {StreamRole::Viewfinder};
    configuration_ = camera_->generateConfiguration(stream_roles);
    CV_Assert(configuration_ && "failed to generate viewfinder configuration");

    // Now we get to override any of the default settings from the options_->
    configuration_->at(0).pixelFormat = libcamera::formats::RGB888;
    configuration_->at(0).size.width = options_->video_width;
    configuration_->at(0).size.height = options_->video_height;
    configuration_->at(0).bufferCount = 4;

    //    configuration_->transform = options_->transform;

    configureDenoise(options_->denoise == "auto" ? "cdn_off" : options_->denoise);
    setupCapture();

    streams_["viewfinder"] = configuration_->at(0).stream();

    if (options_->verbose)
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Viewfinder setup complete");
}

void LibcameraCapture::Teardown()
{
    if (options_->verbose && !options_->help)
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Tearing down requests, buffers and configuration");

    for (auto &iter : mapped_buffers_)
    {
        // assert(iter.first->planes().size() == iter.second.size());
        // for (unsigned i = 0; i < iter.first->planes().size(); i++)
        for (auto &span : iter.second)
            munmap(span.data(), span.size());
    }
    mapped_buffers_.clear();

    delete allocator_;
    allocator_ = nullptr;

    configuration_.reset();

    frame_buffers_.clear();

    streams_.clear();
}

void LibcameraCapture::StartCamera()
{
    // This makes all the Request objects that we shall need.
    makeRequests();

    // Build a list of initial controls that we must set in the camera before starting it.
    // We don't overwrite anything the application may have set before calling us.
    if (!controls_.get(controls::ScalerCrop) && options_->roi_width != 0 && options_->roi_height != 0)
    {
        Rectangle sensor_area = *camera_->properties().get(properties::ScalerCropMaximum);
        int x = options_->roi_x * sensor_area.width;
        int y = options_->roi_y * sensor_area.height;
        int w = options_->roi_width * sensor_area.width;
        int h = options_->roi_height * sensor_area.height;
        Rectangle crop(x, y, w, h);
        crop.translateBy(sensor_area.topLeft());
        if (options_->verbose)
            CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Using crop " << crop.toString());
        controls_.set(controls::ScalerCrop, crop);
    }

    // Framerate is a bit weird. If it was set programmatically, we go with that, but
    // otherwise it applies only to preview/video modes. For stills capture we set it
    // as long as possible so that we get whatever the exposure profile wants.
    if (!controls_.get(controls::FrameDurationLimits))
    {
        if (StillStream())
            controls_.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({INT64_C(100), INT64_C(1000000000)}));
        else if (options_->framerate > 0)
        {
            int64_t frame_time = 1000000 / options_->framerate; // in us
            controls_.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({frame_time, frame_time}));
        }
    }

    if (!controls_.get(controls::ExposureTime) && options_->shutter)
        controls_.set(controls::ExposureTime, options_->shutter);
    if (!controls_.get(controls::AnalogueGain) && options_->gain)
        controls_.set(controls::AnalogueGain, options_->gain);
    if (!controls_.get(controls::AeMeteringMode))
        controls_.set(controls::AeMeteringMode, options_->getMeteringMode());
    if (!controls_.get(controls::AeExposureMode))
        controls_.set(controls::AeExposureMode, options_->getExposureMode());
    if (!controls_.get(controls::ExposureValue))
        controls_.set(controls::ExposureValue, options_->ev);
    if (!controls_.get(controls::AwbMode))
        controls_.set(controls::AwbMode, options_->getWhiteBalance());
    if (!controls_.get(controls::ColourGains) && options_->awb_gain_r && options_->awb_gain_b)
        controls_.set(controls::ColourGains, libcamera::Span<const float, 2>({options_->awb_gain_r, options_->awb_gain_b}));
    if (!controls_.get(controls::Brightness))
        controls_.set(controls::Brightness, options_->brightness);
    if (!controls_.get(controls::Contrast))
        controls_.set(controls::Contrast, options_->contrast);
    if (!controls_.get(controls::Saturation))
        controls_.set(controls::Saturation, options_->saturation);
    if (!controls_.get(controls::Sharpness))
        controls_.set(controls::Sharpness, options_->sharpness);

    if (camera_->start(&controls_))
        CV_Error(cv::Error::StsAssert, "failed to start camera");
    controls_.clear();
    camera_started_ = true;
    last_timestamp_ = 0;

    camera_->requestCompleted.connect(this, &LibcameraCapture::requestComplete);

    for (std::unique_ptr<Request> &request : requests_)
    {
        if (camera_->queueRequest(request.get()) < 0)
            CV_Error(cv::Error::StsAssert, "Failed to queue request");
    }

    if (options_->verbose)
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Camera started!");
}

void LibcameraCapture::StopCamera()
{
    {
        // We don't want QueueRequest to run asynchronously while we stop the camera.
        std::lock_guard<std::mutex> lock(camera_stop_mutex_);
        if (camera_started_)
        {
            CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera): Camera tries to stop!");
            if (camera_->stop())
                CV_Error(cv::Error::StsAssert, "failed to stop camera");
            camera_started_ = false;
        }
    }

    if (camera_)
        camera_->requestCompleted.disconnect(this, &LibcameraCapture::requestComplete);

    // An application might be holding a CompletedRequest, so queueRequest will get
    // called to delete it later, but we need to know not to try and re-queue it.
    completed_requests_.clear();

    msg_queue_.Clear();

    while (!free_requests_.empty())
        free_requests_.pop();

    requests_.clear();

    controls_.clear(); // no need for mutex here

    if (options_->verbose && !options_->help)
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Camera stopped!");
}

void LibcameraCapture::ApplyRoiSettings()
{
    if (!controls_.get(controls::ScalerCrop) && options_->roi_width != 0 && options_->roi_height != 0)
    {
        Rectangle sensor_area = *camera_->properties().get(properties::ScalerCropMaximum);
        int x = options_->roi_x * sensor_area.width;
        int y = options_->roi_y * sensor_area.height;
        int w = options_->roi_width * sensor_area.width;
        int h = options_->roi_height * sensor_area.height;
        Rectangle crop(x, y, w, h);
        crop.translateBy(sensor_area.topLeft());
        if (options_->verbose)
            CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Using crop " << crop.toString());
        controls_.set(controls::ScalerCrop, crop);
    }
}

LibcameraCapture::Msg LibcameraCapture::Wait()
{
    return msg_queue_.Wait();
}

void LibcameraCapture::queueRequest(CompletedRequest *completed_request)
{
    BufferMap buffers(std::move(completed_request->buffers));

    Request *request = completed_request->request;
    assert(request);

    // This function may run asynchronously so needs protection from the
    // camera stopping at the same time.
    std::lock_guard<std::mutex> stop_lock(camera_stop_mutex_);
    if (!camera_started_)
        return;

    // An application could be holding a CompletedRequest while it stops and re-starts
    // the camera, after which we don't want to queue another request now.
    {
        std::lock_guard<std::mutex> lock(completed_requests_mutex_);
        auto it = completed_requests_.find(completed_request);
        delete completed_request;
        if (it == completed_requests_.end())
            return;
        completed_requests_.erase(it);
    }

    for (auto const &p : buffers)
    {
        if (request->addBuffer(p.first, p.second) < 0)
            CV_Error(cv::Error::StsAssert, "failed to add buffer to request in QueueRequest");
    }

    {
        std::lock_guard<std::mutex> lock(control_mutex_);
        request->controls() = std::move(controls_);
    }

    if (camera_->queueRequest(request) < 0)
        CV_Error(cv::Error::StsAssert, "failed to queue request");
}

void LibcameraCapture::PostMessage(MsgType &t, MsgPayload &p)
{
    Msg msg(t);
    msg.payload = p;
    msg_queue_.Post(std::move(msg));
}

libcamera::Stream *LibcameraCapture::GetStream(std::string const &name, unsigned int *w, unsigned int *h,
                                            unsigned int *stride) const
{
    auto it = streams_.find(name);
    if (it == streams_.end())
        return nullptr;
    StreamDimensions(it->second, w, h, stride);
    return it->second;
}

libcamera::Stream *LibcameraCapture::ViewfinderStream(unsigned int *w, unsigned int *h, unsigned int *stride) const
{
    return GetStream("viewfinder", w, h, stride);
}

libcamera::Stream *LibcameraCapture::StillStream(unsigned int *w, unsigned int *h, unsigned int *stride) const
{
    return GetStream("still", w, h, stride);
}

libcamera::Stream *LibcameraCapture::RawStream(unsigned int *w, unsigned int *h, unsigned int *stride) const
{
    return GetStream("raw", w, h, stride);
}

libcamera::Stream *LibcameraCapture::VideoStream(unsigned int *w, unsigned int *h, unsigned int *stride) const
{
    return GetStream("video", w, h, stride);
}

libcamera::Stream *LibcameraCapture::LoresStream(unsigned int *w, unsigned int *h, unsigned int *stride) const
{
    return GetStream("lores", w, h, stride);
}

libcamera::Stream *LibcameraCapture::GetMainStream() const
{
    for (auto &p : streams_)
    {
        if (p.first == "viewfinder" || p.first == "still" || p.first == "video")
            return p.second;
    }

    return nullptr;
}

std::vector<libcamera::Span<uint8_t>> LibcameraCapture::Mmap(FrameBuffer *buffer) const
{
    auto item = mapped_buffers_.find(buffer);
    if (item == mapped_buffers_.end())
        return {};
    return item->second;
}

void LibcameraCapture::SetControls(ControlList &controls)
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    controls_ = std::move(controls);
}


void LibcameraCapture::StreamDimensions(Stream const *stream, unsigned int *w, unsigned int *h, unsigned int *stride) const
{
    StreamConfiguration const &cfg = stream->configuration();
    if (w)
        *w = cfg.size.width;
    if (h)
        *h = cfg.size.height;
    if (stride)
        *stride = cfg.stride;
}

void LibcameraCapture::setupCapture()
{
    // First finish setting up the configuration.

    CameraConfiguration::Status validation = configuration_->validate();
    if (validation == CameraConfiguration::Invalid)
        CV_Error(cv::Error::StsAssert, "failed to valid stream configurations");
    else if (validation == CameraConfiguration::Adjusted)
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Stream configuration adjusted");

    if (camera_->configure(configuration_.get()) < 0)
        CV_Error(cv::Error::StsAssert, "failed to configure streams");

    if (options_->verbose)
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Camera streams configured");

    // Next allocate all the buffers we need, mmap them and store them on a free list.

    allocator_ = new FrameBufferAllocator(camera_);
    for (StreamConfiguration &config : *configuration_)
    {
        Stream *stream = config.stream();

        if (allocator_->allocate(stream) < 0)
            CV_Error(cv::Error::StsAssert, "failed to allocate capture buffers");

        for (const std::unique_ptr<FrameBuffer> &buffer : allocator_->buffers(stream))
        {
            // "Single plane" buffers appear as multi-plane here, but we can spot them because then
            // planes all share the same fd. We accumulate them so as to mmap the buffer only once.
            size_t buffer_size = 0;
            for (unsigned i = 0; i < buffer->planes().size(); i++)
            {
                const FrameBuffer::Plane &plane = buffer->planes()[i];
                buffer_size += plane.length;
                if (i == buffer->planes().size() - 1 || plane.fd.get() != buffer->planes()[i + 1].fd.get())
                {
                    void *memory = mmap(NULL, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
                    mapped_buffers_[buffer.get()].push_back(
                        libcamera::Span<uint8_t>(static_cast<uint8_t *>(memory), buffer_size));
                    buffer_size = 0;
                }
            }
            frame_buffers_[stream].push(buffer.get());
        }
    }
    if (options_->verbose)
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Buffers allocated and mapped");

    // The requests will be made when StartCamera() is called.
}

void LibcameraCapture::makeRequests()
{
    auto free_buffers(frame_buffers_);
    while (true)
    {
        for (StreamConfiguration &config : *configuration_)
        {
            Stream *stream = config.stream();
            if (stream == configuration_->at(0).stream())
            {
                if (free_buffers[stream].empty())
                {
                    if (options_->verbose)
                        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Requests created");
                    return;
                }
                std::unique_ptr<Request> request = camera_->createRequest();
                if (!request)
                    CV_Error(cv::Error::StsAssert, "failed to make request");
                requests_.push_back(std::move(request));
            }
            else if (free_buffers[stream].empty())
                CV_Error(cv::Error::StsAssert, "concurrent streams need matching numbers of buffers");

            FrameBuffer *buffer = free_buffers[stream].front();
            free_buffers[stream].pop();
            if (requests_.back()->addBuffer(stream, buffer) < 0)
                CV_Error(cv::Error::StsAssert, "failed to add buffer to request");
        }
    }
}

void LibcameraCapture::requestComplete(Request *request)
{
    if (request->status() == Request::RequestCancelled)
        return;

    CompletedRequest *r = new CompletedRequest(sequence_++, request);
    CompletedRequestPtr payload(r, [this](CompletedRequest *cr)
                                { this->queueRequest(cr); });
    {
        std::lock_guard<std::mutex> lock(completed_requests_mutex_);
        completed_requests_.insert(r);
    }

    // We calculate the instantaneous framerate in case anyone wants it.
    uint64_t timestamp = payload->buffers.begin()->second->metadata().timestamp;
    if (last_timestamp_ == 0 || last_timestamp_ == timestamp)
        payload->framerate = 0;
    else
        payload->framerate = 1e9 / (timestamp - last_timestamp_);
    last_timestamp_ = timestamp;

    msg_queue_.Post(Msg(MsgType::RequestComplete, std::move(payload)));
}

void LibcameraCapture::configureDenoise(const std::string &denoise_mode)
{
    using namespace libcamera::controls::draft;

    static const std::map<std::string, NoiseReductionModeEnum> denoise_table = {
        {"off", NoiseReductionModeOff},
        {"cdn_off", NoiseReductionModeMinimal},
        {"cdn_fast", NoiseReductionModeFast},
        {"cdn_hq", NoiseReductionModeHighQuality}};
    NoiseReductionModeEnum denoise;

    auto const mode = denoise_table.find(denoise_mode);
    if (mode == denoise_table.end())
        CV_Error(cv::Error::StsAssert, "Invalid denoise mode " + denoise_mode);
    denoise = mode->second;

    controls_.set(NoiseReductionMode, denoise);
}

/* ***************End of methods from original LibcameraApp class*************** */

LibcameraCapture::LibcameraCapture() : LibcameraCapture(-1) {}

LibcameraCapture::LibcameraCapture(int camera_index)
{
    options_ = std::unique_ptr<Options>(new Options());
    controls_ = controls::controls;
    if (!options_)
        options_ = std::make_unique<Options>();
    controls_.clear();
    options = static_cast<Options *>(GetOptions());
    still_flags = FLAG_STILL_NONE;
    options->photo_width = 2560;
    options->photo_height = 1440;
    options->video_width = 640;
    options->video_height = 480;
    options->framerate = 30;
    options->denoise = "auto";
    options->timeout = 1000;
    options->setMetering(Metering_Modes::METERING_MATRIX);
    options->setExposureMode(Exposure_Modes::EXPOSURE_NORMAL);
    options->setWhiteBalance(WhiteBalance_Modes::WB_AUTO);
    options->contrast = 1.0f;
    options->saturation = 1.0f;
    options->camera = camera_index;
    // still_flags |= FLAG_STILL_RGB;
    still_flags |= FLAG_STILL_BGR;
    needsReconfigure.store(false, std::memory_order_release);
    camerastarted = false;
    current_request_ = nullptr;
    // Automatically open the camera
    if (camera_index >= 0) {
        open(camera_index);
    }
}

LibcameraCapture::~LibcameraCapture()
{
    stopVideo();
    // delete app;
    CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera): End of ~LibcameraCapture() call");
}

bool LibcameraCapture::startVideo() // not resolved
{
    if (camerastarted)
    {
        CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera): Camera already started");
        return false;
    }
    
    LibcameraCapture::OpenCamera();
    LibcameraCapture::ConfigureViewfinder();
    LibcameraCapture::StartCamera();
    
    // Get stream dimensions for later use
    libcamera::Stream *stream = ViewfinderStream(&vw, &vh, &vstr);
    if (!stream)
    {
        CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Error getting viewfinder stream");
        return false;
    }
    
    camerastarted = true;
    return true;
}

void LibcameraCapture::stopVideo() // not resolved
{
    if (!camerastarted)
        return;

    LibcameraCapture::StopCamera();
    LibcameraCapture::Teardown();
    LibcameraCapture::CloseCamera();
    
    // Clear any pending request
    {
        std::lock_guard<std::mutex> lock(request_mutex_);
        current_request_ = nullptr;
    }
    
    camerastarted = false;
}

/**
 * @brief Check if a frame is available and ready for retrieval.
 *
 * This function checks if libcamera has a completed request ready.
 * If the camera is not started, it will start the camera first.
 * It uses libcamera's asynchronous message system to check for available frames.
 *
 * @return `true` if a frame is ready for retrieval.
 *         `false` if no frame is available or camera failed to start.
 */
bool LibcameraCapture::grabFrame()
{
    if (!camerastarted)
    {
        if (!startVideo())
        {
            CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Failed to start camera");
            return false;
        }
    }
    
    // Try to get a message from libcamera (non-blocking check)
    try 
    {
        // Use Wait() with short timeout to check for available messages
        LibcameraCapture::Msg msg = Wait();

        if (msg.type == LibcameraCapture::MsgType::RequestComplete)
        {
            CompletedRequestPtr payload = msg.getCompletedRequest();
            {
                std::lock_guard<std::mutex> lock(request_mutex_);
                current_request_ = payload;
            }
            return true;
        }
        else if (msg.type == LibcameraCapture::MsgType::Quit)
        {
            CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Quit message received");
            return false;
        }
    }
    catch (const std::exception& e)
    {
        // No message available or error occurred
        return false;
    }
    
    return false;
}

/**
 * @brief Retrieve the frame data from the current completed request.
 *
 * This function extracts frame data directly from libcamera's memory-mapped buffers
 * without additional copying. It uses the completed request obtained in grabFrame().
 *
 * @param int Unused parameter.
 * @param dst An OpenCV `OutputArray` where the retrieved frame will be stored.
 *            The frame is stored in RGB format (8-bit, 3 channels, CV_8UC3).
 *
 * @return `true` if a frame is successfully retrieved and copied to `dst`.
 *         `false` if no frame is ready or an error occurred.
 */
bool LibcameraCapture::retrieveFrame(int, OutputArray dst)
{
    CompletedRequestPtr request;
    {
        std::lock_guard<std::mutex> lock(request_mutex_);
        if (!current_request_)
        {
            return false;
        }
        request = current_request_;
        current_request_ = nullptr; // Clear after use
    }
    
    // Get the viewfinder stream
    libcamera::Stream *stream = ViewfinderStream(&vw, &vh, &vstr);
    if (!stream)
    {
        CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Error getting viewfinder stream");
        return false;
    }
    
    // Get memory mapped buffer directly from libcamera
    std::vector<libcamera::Span<uint8_t>> mem = Mmap(request->buffers[stream]);
    if (mem.empty())
    {
        CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Error getting memory mapped buffer");
        return false;
    }
    
    // Create OpenCV Mat directly from libcamera buffer
    Mat frame(vh, vw, CV_8UC3);
    uint8_t *src_ptr = mem[0].data();
    uint line_size = vw * 3;
    
    // Copy line by line to handle stride correctly
    for (unsigned int i = 0; i < vh; i++, src_ptr += vstr)
    {
        memcpy(frame.ptr(i), src_ptr, line_size);
    }
    
    frame.copyTo(dst);
    return true;
}

bool LibcameraCapture::retrieve(cv::Mat& frame, int stream_idx)
{
    cv::OutputArray dst(frame);
    return retrieveFrame(stream_idx, dst);
}

double LibcameraCapture::getProperty(int propId) const
{
    switch (propId)
    {
    case cv::CAP_PROP_BRIGHTNESS:
        return options->brightness;

    case cv::CAP_PROP_CONTRAST:
        return options->contrast;

    case cv::CAP_PROP_SATURATION:
        return options->saturation;

    case cv::CAP_PROP_SHARPNESS:
        return options->sharpness;

    case cv::CAP_PROP_AUTO_EXPOSURE:
        return options->getExposureMode() == Exposure_Modes::EXPOSURE_NORMAL;

    case cv::CAP_PROP_EXPOSURE:
        return options->shutter;

    case cv::CAP_PROP_AUTO_WB:
        return options->getWhiteBalance() == WhiteBalance_Modes::WB_AUTO;

    case cv::CAP_PROP_WB_TEMPERATURE:
        // Since we don't have a direct WB temperature, return an approximation based on the current setting
        switch (options->getWhiteBalance())
        {
        case WhiteBalance_Modes::WB_TUNGSTEN:
            return 3000.0; // Approximate value for tungsten
        case WhiteBalance_Modes::WB_INDOOR:
            return 4500.0; // Approximate value for indoor
        case WhiteBalance_Modes::WB_DAYLIGHT:
            return 5500.0; // Approximate value for daylight
        case WhiteBalance_Modes::WB_CLOUDY:
            return 7000.0; // Approximate value for cloudy
        default:
            return 5000.0; // Default approximation if none of the above
        }

    case cv::CAP_PROP_XI_AEAG_ROI_OFFSET_X:
        return options->roi_x;

    case cv::CAP_PROP_XI_AEAG_ROI_OFFSET_Y:
        return options->roi_y;

    case cv::CAP_PROP_XI_AEAG_ROI_WIDTH:
        return options->roi_width;

    case cv::CAP_PROP_XI_AEAG_ROI_HEIGHT:
        return options->roi_height;

    case cv::CAP_PROP_FOURCC:
    {
        // Return the FOURCC code of the current video format.
        // This is a placeholder. You should replace it with the actual FOURCC code.
        // return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        // return options->getFourCC();
        CV_LOG_WARNING(NULL, "VIDEOIO(Libcamera): Property FOURCC is not implemented yet");
        return 0;
    }

    case cv::CAP_PROP_FRAME_WIDTH:
        if (options->video_width != 0)
        {
            return options->video_width;
        }
        else
        {
            return options->photo_width;
        }

    case cv::CAP_PROP_FRAME_HEIGHT:
        if (options->video_height != 0)
        {
            return options->video_height;
        }
        else
        {
            return options->photo_height;
        }

    case cv::CAP_PROP_FPS:
        return options->framerate;

    case cv::CAP_PROP_AUTOFOCUS:
    case cv::CAP_PROP_BUFFERSIZE:
    case cv::CAP_PROP_PAN:
    case cv::CAP_PROP_TILT:
    case cv::CAP_PROP_ROLL:
    case cv::CAP_PROP_IRIS:
        // Not implemented, return a default value or an error code
        CV_LOG_WARNING(NULL, "VIDEOIO(Libcamera): Property " << propId << " is not supported.");
        return 0; // Or some other value indicating an error or not supported

    default:
        CV_LOG_WARNING(NULL, "VIDEOIO(Libcamera): Unsupported property: " << propId);
        return 0;
    }
}

bool LibcameraCapture::setProperty(int propId, double value)
{
    switch (propId)
    {
    case cv::CAP_PROP_BRIGHTNESS:
        options->brightness = value;
        needsReconfigure.store(true, std::memory_order_release);
        break;

    case cv::CAP_PROP_CONTRAST:
        options->contrast = value;
        needsReconfigure.store(true, std::memory_order_release);
        break;

    case cv::CAP_PROP_SATURATION:
        options->saturation = value;
        needsReconfigure.store(true, std::memory_order_release);
        break;

    case cv::CAP_PROP_SHARPNESS:
        options->sharpness = value;
        needsReconfigure.store(true, std::memory_order_release);
        break;

    case cv::CAP_PROP_AUTO_EXPOSURE:
        if (value)
        {
            options->setExposureMode(Exposure_Modes::EXPOSURE_NORMAL);
        }
        else
        {
            options->setExposureMode(Exposure_Modes::EXPOSURE_SHORT);
        }
        needsReconfigure.store(true, std::memory_order_release);
        break;

    case cv::CAP_PROP_EXPOSURE:
        options->shutter = value; // Assumes value is in milliseconds, libcamera uses seconds
        needsReconfigure.store(true, std::memory_order_release);
        break;

    case cv::CAP_PROP_AUTO_WB:
        options->setWhiteBalance(value ? WhiteBalance_Modes::WB_AUTO : WhiteBalance_Modes::WB_INDOOR);
        needsReconfigure.store(true, std::memory_order_release);
        break;

    case cv::CAP_PROP_WB_TEMPERATURE:
        // Libcamera does not have a direct WB temperature setting,
        // you might need to convert this to r/b gains for manual control.
        // For now, let's assume a simplified approach.
        if (value < 4000)
        {
            options->setWhiteBalance(WhiteBalance_Modes::WB_TUNGSTEN);
        }
        else if (value < 5000)
        {
            options->setWhiteBalance(WhiteBalance_Modes::WB_INDOOR);
        }
        else if (value < 6500)
        {
            options->setWhiteBalance(WhiteBalance_Modes::WB_DAYLIGHT);
        }
        else
        {
            options->setWhiteBalance(WhiteBalance_Modes::WB_CLOUDY);
        }
        needsReconfigure.store(true, std::memory_order_release);
        break;

    case cv::CAP_PROP_XI_AEAG_ROI_OFFSET_X:
        options->roi_x = value;
        ApplyRoiSettings();
        break;

    case cv::CAP_PROP_XI_AEAG_ROI_OFFSET_Y:
        options->roi_y = value;
        ApplyRoiSettings();
        break;

    case cv::CAP_PROP_XI_AEAG_ROI_WIDTH:
        options->roi_width = value;
        ApplyRoiSettings();
        break;

    case cv::CAP_PROP_XI_AEAG_ROI_HEIGHT:
        options->roi_height = value;
        ApplyRoiSettings();
        break;

    case cv::CAP_PROP_FOURCC:
    {
        // Not implemented yet
        break;
    }

    case cv::CAP_PROP_FRAME_WIDTH:
        options->video_width = options->photo_width = (int)value;
        needsReconfigure.store(true, std::memory_order_release);
        break;

    case cv::CAP_PROP_FRAME_HEIGHT:
        options->video_height = options->photo_height = (int)value;
        needsReconfigure.store(true, std::memory_order_release);
        break;

    case cv::CAP_PROP_FPS:
        options->framerate = (float)value;
        needsReconfigure.store(true, std::memory_order_release);
        break;
    case cv::CAP_PROP_AUTOFOCUS:  // Not implemented
    case cv::CAP_PROP_BUFFERSIZE: // Not implemented
    case cv::CAP_PROP_PAN:        // Not implemented
    case cv::CAP_PROP_TILT:       // Not implemented
    case cv::CAP_PROP_ROLL:       // Not implemen ted
    case cv::CAP_PROP_IRIS:       // Not implemented
        // These properties might need to trigger a re-configuration of the camera.
        // You can handle them here if you want to support changing resolution or framerate on-the-fly.
        // For now, we'll return false to indicate that these properties are not supported for dynamic changes.
        CV_LOG_WARNING(NULL, "VIDEOIO(Libcamera): Property " << propId << " is not supported for dynamic changes.");
        return false;

    default:
        CV_LOG_WARNING(NULL, "VIDEOIO(Libcamera): Unsupported property: " << propId);
        return false;
    }
    return true;
}

bool LibcameraCapture::open(int _index)
{
    cv::String name;
    /* Select camera, or rather, V4L video source */
    if (_index < 0) // Asking for the first device available
    {
        for (int autoindex = 0; autoindex < 8; ++autoindex) // 8=MAX_CAMERAS
        {
            name = cv::format("/dev/video%d", autoindex);
            /* Test using an open to see if this new device name really does exists. */
            int h = ::open(name.c_str(), O_RDONLY);
            if (h != -1)
            {
                ::close(h);
                _index = autoindex;
                break;
            }
        }
        if (_index < 0)
        {
            CV_LOG_WARNING(NULL, "VIDEOIO(Libcamera): can't find camera device");
            name.clear();
            return false;
        }
    }
    else
    {
        name = cv::format("/dev/video%d", _index);
    }

    bool res = open(name);
    if (!res)
    {
        CV_LOG_WARNING(NULL, "VIDEOIO(Libcamera:" << name << "): can't open camera by index");
    }
    return res;
}

bool LibcameraCapture::open(const std::string &_deviceName)
{
    CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera:" << _deviceName << "): opening...");
    // Some parameters initialization here, maybe more needed.
    options->video_width = 640;
    options->video_height = 480;
    options->framerate = 30;
    options->verbose = true;
    return startVideo();
}

bool LibcameraCapture::isOpened() const
{
    return camerastarted;
}

Ptr<IVideoCapture> createLibcameraCapture_file(const std::string &filename)
{
    auto ret = makePtr<LibcameraCapture>();
    if (ret->open(filename))
        return ret;
    return NULL;
}

Ptr<IVideoCapture> createLibcameraCapture_cam(int index)
{
    Ptr<LibcameraCapture> cap = makePtr<LibcameraCapture>();
    if (cap && cap->open(index))
        return cap;
    return Ptr<IVideoCapture>();
}

} // namespace