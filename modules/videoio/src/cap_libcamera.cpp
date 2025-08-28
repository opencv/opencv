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
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/videoio.hpp>

#include <sys/mman.h>

#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <variant>
#include <any>
#include <map>
#include <iomanip>
#include <atomic>
#include <fcntl.h>
#include <fstream>
#include <chrono>

#include <libcamera/base/span.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/property_ids.h>
#include <libcamera/stream.h>
#include "cap_libcamera.hpp"


namespace cv
{
// Forward declaration
class LibcameraCapture;

/* ******************************************************************* */
// LibcameraFrameAllocator implementation

LibcameraFrameAllocator::LibcameraFrameAllocator(LibcameraApp* app, libcamera::Request* request)
    : app_(app), request_to_recycle_(request)
{
    if (!app_) {
        CV_LOG_ERROR(NULL, "LibcameraFrameAllocator constructed with null app");
        throw std::invalid_argument("Cannot create LibcameraFrameAllocator with null app");
    }
    if (!request_to_recycle_) {
        CV_LOG_ERROR(NULL, "LibcameraFrameAllocator constructed with null request");
        throw std::invalid_argument("Cannot create LibcameraFrameAllocator with null request");
    }
}

LibcameraFrameAllocator::~LibcameraFrameAllocator()
{
    // Recycle request if deallocate() hasn't been called (fallback mechanism)
    if (app_ && request_to_recycle_) {
        app_->recycleRequest(request_to_recycle_);
    }
}

void LibcameraFrameAllocator::resetRequest(libcamera::Request* request)
{
    if (app_ && request_to_recycle_) {
        app_->recycleRequest(request_to_recycle_);
    }
    request_to_recycle_ = request;
}

cv::UMatData* LibcameraFrameAllocator::allocate(int dims, const int* sizes, int type, void* data, 
                                                 size_t* step, cv::AccessFlag flags, cv::UMatUsageFlags usageFlags) const
{
    (void)step;
    (void)flags;
    (void)usageFlags;
    
    if (!data) {
        return nullptr;
    }

    cv::UMatData* u = new cv::UMatData(this);
    u->data = u->origdata = static_cast<uchar*>(data);
    u->flags = cv::UMatData::USER_ALLOCATED;
    u->handle = 0;
    u->userdata = 0;
    
    size_t total_size = CV_ELEM_SIZE(type);
    for (int i = 0; i < dims; i++) {
        total_size *= sizes[i];
    }
    u->size = total_size;
    
    return u;
}

bool LibcameraFrameAllocator::allocate(cv::UMatData* data, cv::AccessFlag accessFlags, cv::UMatUsageFlags usageFlags) const
{
    (void)data;         
    (void)accessFlags;  
    (void)usageFlags;   
    
    return false;
}

void LibcameraFrameAllocator::deallocate(cv::UMatData* data) const
{
    if (data) {
        delete data;
    }
    
    if (app_ && request_to_recycle_) {
        app_->recycleRequest(request_to_recycle_);
        request_to_recycle_ = nullptr;
    }
    
    delete this;
}


LibcameraApp::LibcameraApp(std::unique_ptr<Options> opts)
    : options_(std::move(opts)), capture_instance_(nullptr), controls_(controls::controls)
{
    if (!options_)
        options_ = std::make_unique<Options>();
    controls_.clear();
}

LibcameraApp::~LibcameraApp()
{
    try {
        StopCamera();
        Teardown();
        CloseCamera();
    } catch (const std::exception& e) {
        CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Error in destructor: " << e.what());
    }
    
    CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera): End of ~LibcameraApp() call");
}

// 获取当前摄像头的设备ID字符串
// Get the current camera's device ID string
std::string const &LibcameraApp::CameraId() const
{
    return camera_->id();
}

void LibcameraApp::OpenCamera()
{

    CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Opening camera...");

    if (camera_started_ || camera_acquired_ || camera_) {
        CV_LOG_WARNING(NULL, "VIDEOIO(Libcamera): Camera not properly closed, forcing cleanup");
        try {
            StopCamera();
            Teardown();
            CloseCamera();
        } catch (...) {
            CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Error during cleanup before opening camera");
        }
    }

    CV_Assert(getCameraManager()->cameras().size() != 0 && "no cameras available");
    CV_Assert(options_->camera < getCameraManager()->cameras().size() && "camera index out of range");

    std::string const &cam_id = getCameraManager()->cameras()[options_->camera]->id();
    camera_ = getCameraManager()->get(cam_id);
    if (!camera_)
        CV_Error(cv::Error::StsAssert, "failed to find camera " + cam_id);

    if (!camera_acquired_ && camera_->acquire())
        CV_Error(cv::Error::StsAssert, "failed to acquire camera " + cam_id);
    camera_acquired_ = true;

    CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Acquired camera " << cam_id);
}

void LibcameraApp::CloseCamera()
{
    if (camera_started_) {
        StopCamera();
    }
    
    if (camera_acquired_ && camera_) {
        camera_->release();
        camera_acquired_ = false;
    }

    camera_.reset();

    CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Camera closed");
}

void LibcameraApp::ConfigureViewfinder()
{
    CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Configuring viewfinder...");

    StreamRoles stream_roles = {StreamRole::Viewfinder};
    configuration_ = camera_->generateConfiguration(stream_roles);
    if (!configuration_)
        CV_Error(cv::Error::StsAssert, "failed to generate viewfinder configuration");

    // Now we get to override any of the default settings from the options_->
    configuration_->at(0).pixelFormat = libcamera::formats::RGB888;
    configuration_->at(0).size.width = options_->video_width;
    configuration_->at(0).size.height = options_->video_height;
    
    if (capture_instance_) {
        unsigned int min_safe_buffers = 4;
        
        if (options_->buffer_count < min_safe_buffers) {
            options_->buffer_count = min_safe_buffers;
            CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Auto-adjusted buffer count to " << options_->buffer_count 
                      << " for safe operation");
        }
    }
    
    configuration_->at(0).bufferCount = options_->buffer_count;


    configureDenoise(options_->denoise == "auto" ? "cdn_off" : options_->denoise);
    setupCapture();

    streams_["viewfinder"] = configuration_->at(0).stream();

    CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Viewfinder setup complete");
}

// 清理所有已分配的资源：内存映射、缓冲区分配器、配置等
// Teardown all allocated resources: memory mappings, buffer allocator, configuration, etc.
void LibcameraApp::Teardown()
{
    CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Tearing down requests, buffers and configuration");

    if (camera_started_) {
        StopCamera();
    }

    for (auto &iter : mapped_buffers_)
    {
        // assert(iter.first->planes().size() == iter.second.size());
        // for (unsigned i = 0; i < iter.first->planes().size(); i++)
        for (auto &span : iter.second)
            munmap(span.data(), span.size());
    }
    mapped_buffers_.clear();

    if (allocator_) {
        delete allocator_;
        allocator_ = nullptr;
    }

    configuration_.reset();

    frame_buffers_.clear();

    streams_.clear();
    
    camera_started_ = false;
}

// 启动摄像头：创建请求、设置控制参数、连接回调、开始捕获
// Start the camera: create requests, set control parameters, connect callbacks, and start capturing
void LibcameraApp::StartCamera()
{
    makeRequests();

    if (!controls_.get(controls::ScalerCrop) && options_->roi_width != 0 && options_->roi_height != 0)
    {
        Rectangle sensor_area = *camera_->properties().get(properties::ScalerCropMaximum);
        int x = options_->roi_x * sensor_area.width;
        int y = options_->roi_y * sensor_area.height;
        int w = options_->roi_width * sensor_area.width;
        int h = options_->roi_height * sensor_area.height;
        Rectangle crop(x, y, w, h);
        crop.translateBy(sensor_area.topLeft());
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
            int64_t frame_time = 1000000 / options_->framerate;
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
        CV_Error(cv::Error::StsError, "failed to start camera");
    controls_.clear();
    camera_started_ = true;
    last_timestamp_ = 0;

    camera_->requestCompleted.connect(this, &LibcameraApp::requestComplete);

    for (std::unique_ptr<Request> &request : requests_) {
        free_requests_.push(request.get());
    }
    CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera): On-demand mode: " << free_requests_.size() << " requests available");

    CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Camera started!");
}

void LibcameraApp::StopCamera()
{
    {
        std::lock_guard<std::mutex> lock(camera_stop_mutex_);
        if (camera_started_)
        {
            CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera): Camera tries to stop!");
            
            if (camera_)
                camera_->requestCompleted.disconnect(this, &LibcameraApp::requestComplete);
            
            if (camera_->stop())
                CV_Error(cv::Error::StsError, "failed to stop camera");

            camera_started_ = false;
        }
    }

    while (!free_requests_.empty())
        free_requests_.pop();

    requests_.clear();

    controls_.clear(); 

    CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Camera stopped!");
}

void LibcameraApp::ApplyRoiSettings()
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
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Using crop " << crop.toString());
        controls_.set(controls::ScalerCrop, crop);
    }
}

bool LibcameraApp::submitSingleRequest()
{
    std::lock_guard<std::mutex> stop_lock(camera_stop_mutex_);
    if (!camera_started_)
        return false;

    if (free_requests_.empty()) {
        CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): No free requests available for single request");
        return false;
    }

    Request* request = free_requests_.front();
    free_requests_.pop();

    if (!request->buffers().empty()) {
        request->reuse();
    }

    for (StreamConfiguration &config : *configuration_) {
        Stream *stream = config.stream();
        if (frame_buffers_[stream].empty()) {
            CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): No free buffers available for stream");
            free_requests_.push(request);
            return false;
        }
        
        FrameBuffer *buffer = frame_buffers_[stream].front();
        frame_buffers_[stream].pop();
        if (request->addBuffer(stream, buffer) < 0) {
            CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Failed to add buffer to request");
            frame_buffers_[stream].push(buffer);
            free_requests_.push(request);
            return false;
        }
        frame_buffers_[stream].push(buffer);
    }

    {
        std::lock_guard<std::mutex> lock(control_mutex_);
        if (!controls_.empty()) {
            request->controls() = std::move(controls_);
        }
    }

    if (camera_->queueRequest(request) < 0) {
        CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Failed to queue single request");
        free_requests_.push(request);
        return false;
    }
    CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera): Single request submitted, remaining free requests: " << free_requests_.size());

    return true;
}


size_t LibcameraApp::getFreeRequestsCount() const
{
    std::lock_guard<std::mutex> stop_lock(camera_stop_mutex_);
    return free_requests_.size();
}

void LibcameraApp::recycleRequest(libcamera::Request* request)
{
    if (!request) {
        CV_LOG_WARNING(NULL, "VIDEOIO(Libcamera): Attempted to recycle null request");
        return;
    }

    std::lock_guard<std::mutex> stop_lock(camera_stop_mutex_);
    if (!camera_started_) {
        return;
    }

    request->reuse();  
    free_requests_.push(request);
}


libcamera::Stream *LibcameraApp::GetStream(std::string const &name, unsigned int *w, unsigned int *h,
                                           unsigned int *stride) const
{
    auto it = streams_.find(name);
    if (it == streams_.end())
        return nullptr;
    StreamDimensions(it->second, w, h, stride);
    return it->second;
}

libcamera::Stream *LibcameraApp::ViewfinderStream(unsigned int *w, unsigned int *h, unsigned int *stride) const
{
    return GetStream("viewfinder", w, h, stride);
}

libcamera::Stream *LibcameraApp::StillStream(unsigned int *w, unsigned int *h, unsigned int *stride) const
{
    return GetStream("still", w, h, stride);
}

libcamera::Stream *LibcameraApp::RawStream(unsigned int *w, unsigned int *h, unsigned int *stride) const
{
    return GetStream("raw", w, h, stride);
}


libcamera::Stream *LibcameraApp::VideoStream(unsigned int *w, unsigned int *h, unsigned int *stride) const
{
    return GetStream("video", w, h, stride);
}

libcamera::Stream *LibcameraApp::LoresStream(unsigned int *w, unsigned int *h, unsigned int *stride) const
{
    return GetStream("lores", w, h, stride);
}

libcamera::Stream *LibcameraApp::GetMainStream() const
{
    for (auto &p : streams_)
    {
        if (p.first == "viewfinder" || p.first == "still" || p.first == "video")
            return p.second;
    }

    return nullptr;
}

// 返回可访问的内存区域列表
// Return the list of accessible memory regions
std::vector<libcamera::Span<uint8_t>> LibcameraApp::Mmap(FrameBuffer *buffer) const
{
    auto item = mapped_buffers_.find(buffer);
    if (item == mapped_buffers_.end())
        return {};
    return item->second;
}

void LibcameraApp::SetControls(ControlList &controls)
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    controls_ = std::move(controls);
}

void LibcameraApp::StreamDimensions(Stream const *stream, unsigned int *w, unsigned int *h, unsigned int *stride) const
{
    StreamConfiguration const &cfg = stream->configuration();
    if (w)
        *w = cfg.size.width;
    if (h)
        *h = cfg.size.height;
    if (stride)
        *stride = cfg.stride;
}

// 设置捕获参数：验证配置、配置摄像头、分配缓冲区、建立内存映射
// Setup capture parameters: validate configuration, configure camera, allocate buffers, establish memory mappings
void LibcameraApp::setupCapture()
{
    // First finish setting up the configuration.

    CameraConfiguration::Status validation = configuration_->validate();
    if (validation == CameraConfiguration::Invalid)
        CV_Error(cv::Error::StsError, "failed to validate stream configurations");
    else if (validation == CameraConfiguration::Adjusted)
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Stream configuration adjusted");

    if (camera_->configure(configuration_.get()) < 0)
        CV_Error(cv::Error::StsError, "failed to configure streams");

    CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Camera streams configured");


    allocator_ = new FrameBufferAllocator(camera_);
    for (StreamConfiguration &config : *configuration_)
    {
        Stream *stream = config.stream();

        // Allocate buffers for this stream (count was set in ConfigureViewfinder)
        if (allocator_->allocate(stream) < 0)
            CV_Error(cv::Error::StsError, "failed to allocate capture buffers");

        CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera): Allocated " << allocator_->buffers(stream).size() 
                  << " buffers for stream (requested: " << options_->buffer_count << ")");

        for (const std::unique_ptr<FrameBuffer> &buffer : allocator_->buffers(stream))
        {
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
    CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera): Buffers allocated and mapped");

    // The requests will be made when StartCamera() is called.
}

// 创建所有需要的请求对象，为每个流分配缓冲区
// Create all the required request objects and allocate buffers for each stream
void LibcameraApp::makeRequests()
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
                    CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Buffers allocated and mapped");
                    return;
                }
                std::unique_ptr<Request> request = camera_->createRequest();
                if (!request)
                    CV_Error(cv::Error::StsError, "failed to make request");
                requests_.push_back(std::move(request));
            }
            else if (free_buffers[stream].empty())
                CV_Error(cv::Error::StsAssert, "concurrent streams need matching numbers of buffers");

            FrameBuffer *buffer = free_buffers[stream].front();
            free_buffers[stream].pop();
            if (requests_.back()->addBuffer(stream, buffer) < 0)
                CV_Error(cv::Error::StsError, "failed to add buffer to request");
        }
    }
}

// 回调函数：处理完成的请求，计算帧率，分发给消费者
// Callback function to handle completed requests, calculate framerate, and dispatch to consumers
void LibcameraApp::requestComplete(Request *request)
{
    if (request->status() == Request::RequestCancelled)
        return;

    CompletedRequest completed_req(sequence_++, request);
    CompletedRequestPtr payload = std::make_shared<CompletedRequest>(std::move(completed_req));

    uint64_t timestamp = payload->buffers.begin()->second->metadata().timestamp;
    if (last_timestamp_ == 0 || last_timestamp_ == timestamp)
        payload->framerate = 0;
    else
        payload->framerate = 1e9 / (timestamp - last_timestamp_);
    last_timestamp_ = timestamp;

    if (capture_instance_) {
        capture_instance_->onRequestComplete(std::move(payload));
    }
}

// 配置降噪模式
// Configure the denoise mode
void LibcameraApp::configureDenoise(const std::string &denoise_mode)
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
        CV_Error(cv::Error::StsBadArg, "Invalid denoise mode " + denoise_mode);
    denoise = mode->second;

    controls_.set(NoiseReductionMode, denoise);
}

LibcameraCapture::LibcameraCapture()
{
    auto opts = std::make_unique<Options>();
    app = new LibcameraApp(std::move(opts));
    options = static_cast<Options *>(app->GetOptions());
    
    app->SetCaptureInstance(this);
    
    still_flags = LibcameraApp::FLAG_STILL_NONE;
    options->photo_width = 640;
    options->photo_height = 480;
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
    still_flags |= LibcameraApp::FLAG_STILL_BGR;
    camera_started_.store(false, std::memory_order_release);
    needsReconfigure.store(false, std::memory_order_release);
    vw = vh = vstr = 0;
}

LibcameraCapture::LibcameraCapture(int camera_index) : LibcameraCapture()
{
    options->camera = camera_index;
    open(camera_index);
}

LibcameraCapture::~LibcameraCapture()
{
    stopVideo();
    
    if (app) {
        app->SetCaptureInstance(nullptr);
        delete app;
        app = nullptr;
    }
    
    CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera): End of ~LibcameraCapture() call");
}


// 打开摄像头、配置流、获取流信息、启动捕获
// Open the camera, configure the stream, get stream information, and start capturing
bool LibcameraCapture::startVideo()
{
    if (camera_started_.load(std::memory_order_relaxed))
    {
        CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera): Camera already started");
        return true;
    }

    try {
        app->OpenCamera();
        app->ConfigureViewfinder();
        
        libcamera::Stream *stream = app->ViewfinderStream(&vw, &vh, &vstr);
        if (!stream) {
            CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Error getting viewfinder stream");
            return false;
        }
        
        app->StartCamera();
        camera_started_.store(true, std::memory_order_release);
        
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Camera started successfully");
        return true;
    } catch (const std::exception& e) {
        CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Error starting camera: " << e.what());
        return false;
    }
}

// 停止视频捕获
// Stop video capture
void LibcameraCapture::stopVideo()
{
    if (!camera_started_.load(std::memory_order_relaxed))
        return;

    camera_started_.store(false, std::memory_order_release);

    {
        std::lock_guard<std::mutex> lock(completed_requests_mutex_);
        while (!completed_requests_.empty()) {
            completed_requests_.pop();
        }
    }
    completed_requests_cv_.notify_all();

    try {
        if (app) {
            app->StopCamera();    
            app->Teardown();      
            app->CloseCamera();   
        }
    } catch (const std::exception& e) {
        CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Error stopping camera: " << e.what());
    }
    
    CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera): Camera stopped");
}

// 接收完成的请求
// Receive completed requests as a producer
void LibcameraCapture::onRequestComplete(CompletedRequestPtr completed_request)
{
    std::lock_guard<std::mutex> lock(completed_requests_mutex_);
    
    completed_requests_.push(std::move(completed_request));
    
    completed_requests_cv_.notify_one();
}

// 等待帧数据可用，支持超时等待
// Wait for frame data to be available, supporting timeout waiting
bool LibcameraCapture::waitForFrame(unsigned int timeout_ms)
{
    std::unique_lock<std::mutex> lock(completed_requests_mutex_);
    
    if (timeout_ms == 0) {
        // Infinite wait
        completed_requests_cv_.wait(lock, [this] { 
            return !completed_requests_.empty() || !camera_started_.load(std::memory_order_acquire); 
        });
    } else {
        // Timeout wait
        auto timeout = std::chrono::milliseconds(timeout_ms);
        if (!completed_requests_cv_.wait_for(lock, timeout, [this] { 
            return !completed_requests_.empty() || !camera_started_.load(std::memory_order_acquire); 
        })) {
            return false; // Timeout
        }
    }
    
    return !completed_requests_.empty();
}

// 从队列中获取一个完成的请求
// Get a completed request from the queue
CompletedRequestPtr LibcameraCapture::getCompletedRequest()
{
    CompletedRequestPtr request;
    
    {
        std::lock_guard<std::mutex> lock(completed_requests_mutex_);
        if (completed_requests_.empty()) {
            return nullptr;
        }
        request = std::move(completed_requests_.front());
        completed_requests_.pop();
    } 
    return request;
}

/**
 * @brief Ensure camera is started and ready to capture frames
 * @return true if camera is ready to capture frames
 * @return false if failed to start camera
 */
bool LibcameraCapture::grabFrame()
{
    if (needsReconfigure.load(std::memory_order_acquire))
    {
        CV_LOG_INFO(NULL, "VIDEOIO(Libcamera): Reconfiguring camera...");
        stopVideo();
        if (!startVideo()) {
            CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Failed to restart camera after reconfiguration");
            return false;
        }
        needsReconfigure.store(false, std::memory_order_release);
    }
    
    if (!camera_started_.load(std::memory_order_acquire))
    {
        if (!startVideo()) {
            CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Failed to start camera");
            return false;
        }
    }
    
    // Submit a single request and record timing
    auto now = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    last_request_submit_time_ns_.store(ns, std::memory_order_relaxed);
    
    if (app && !app->submitSingleRequest()) {
        CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Failed to submit single request in on-demand mode");
        return false;
    }
    
    return camera_started_.load(std::memory_order_acquire);
}

bool LibcameraCapture::retrieveFrame(int, OutputArray dst)
{
    if (!camera_started_.load(std::memory_order_acquire)) {
        CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Camera not started in retrieveFrame");
        return false;
    }
        
    auto wait_start = std::chrono::steady_clock::now();
    auto wait_start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(wait_start.time_since_epoch()).count();
    last_wait_start_time_ns_.store(wait_start_ns, std::memory_order_relaxed);
    
    if (!waitForFrame(options->timeout)) {
        return false;  // 超时
    }
    
    CompletedRequestPtr completed_request = getCompletedRequest();
    if (!completed_request) {
        return false;  // 无可用请求
    }
    
    auto frame_complete = std::chrono::steady_clock::now();
    auto frame_complete_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(frame_complete.time_since_epoch()).count();
    last_frame_complete_time_ns_.store(frame_complete_ns, std::memory_order_relaxed);
    
    last_capture_timestamp_ns_.store(completed_request->capture_timestamp_ns, std::memory_order_relaxed);
    
    try {
        auto stream = app->ViewfinderStream(&vw, &vh, &vstr);
        if (!stream) {
            CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Failed to get viewfinder stream");
            return false;
        }
        
        auto mem = app->Mmap(completed_request->buffers[stream]);
        if (mem.empty()) {
            CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Failed to get memory mapping");
            return false;
        }
        
        const uint32_t pixel_bytes = 3;  // RGB888
        const uint32_t expected_row_bytes = vw * pixel_bytes;
        uint8_t *libcamera_buffer_ptr = mem[0].data();
        
        if (vstr == expected_row_bytes) {
            LibcameraFrameAllocator* allocator = new LibcameraFrameAllocator(app, completed_request->request);
            
            reuse_dims_[0] = (int)vh;
            reuse_dims_[1] = (int)vw;
            
            cv::UMatData* u = allocator->allocate(2, reuse_dims_, CV_8UC3, 
                                                  libcamera_buffer_ptr, nullptr, cv::ACCESS_READ, cv::USAGE_DEFAULT);
            if (!u) {
                CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Failed to allocate UMatData via LibcameraFrameAllocator");
                return false;
            }
            
            int sizes[] = {(int)vh, (int)vw};
            size_t steps[] = {vstr, pixel_bytes};
            
            Mat zero_copy_mat(2, sizes, CV_8UC3, libcamera_buffer_ptr, steps);
            zero_copy_mat.allocator = allocator; 
            
            if (!zero_copy_mat.u) {
                zero_copy_mat.u = u;
                u->refcount = 1;
            }
            
            dst.assign(zero_copy_mat);
            return true;
        } else {
            // (回退)步长不兼容，使用拷贝
            CV_LOG_WARNING(NULL, cv::format("Stride mismatch: libcamera=%d, expected=%d", 
                                           vstr, expected_row_bytes));
            
            dst.create(vh, vw, CV_8UC3);
            Mat target_frame = dst.getMat();
            
            const uint8_t *src_ptr = libcamera_buffer_ptr;
            for (unsigned int row = 0; row < vh; row++) {
                memcpy(target_frame.ptr(row), src_ptr, expected_row_bytes);
                src_ptr += vstr;  
            }
            return true;
        }
        
    } catch (const std::exception& e) {
        CV_LOG_ERROR(NULL, "VIDEOIO(Libcamera): Error in retrieveFrame: " << e.what());
        return false;
    }
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
        CV_LOG_WARNING(NULL, "VIDEOIO(Libcamera): Property FOURCC not implemented yet");
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

    case cv::CAP_PROP_POS_MSEC:
        {
            uint64_t timestamp_ns = last_capture_timestamp_ns_.load(std::memory_order_relaxed);
            return timestamp_ns / 1000000.0; 
        }

    case cv::CAP_PROP_AUTOFOCUS:
    case cv::CAP_PROP_BUFFERSIZE:
    case cv::CAP_PROP_PAN:
    case cv::CAP_PROP_TILT:
    case cv::CAP_PROP_ROLL:
    case cv::CAP_PROP_IRIS:
        // Not implemented, return a default value or an error code
        CV_LOG_WARNING(NULL, "VIDEOIO(Libcamera): Property " << propId << " is not supported");
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

        // case cv::CAP_PROP_ZOOM: // This is a custom property for ROI
        //     options->roi_x = options->roi_y = (1.0 - value) / 2.0; // Assuming value is normalized zoom level (0.0 - 1.0)
        //     options->roi_width = options->roi_height = value;
        //     break;

    case cv::CAP_PROP_XI_AEAG_ROI_OFFSET_X:
        options->roi_x = value;
        app->ApplyRoiSettings();
        break;

    case cv::CAP_PROP_XI_AEAG_ROI_OFFSET_Y:
        options->roi_y = value;
        app->ApplyRoiSettings();
        break;

    case cv::CAP_PROP_XI_AEAG_ROI_WIDTH:
        options->roi_width = value;
        app->ApplyRoiSettings();
        break;

    case cv::CAP_PROP_XI_AEAG_ROI_HEIGHT:
        options->roi_height = value;
        app->ApplyRoiSettings();
        break;

    case cv::CAP_PROP_FOURCC:
    {
        // Not implemented yet

        // char fourcc[4];
        // fourcc[0] = (char)((int)value & 0XFF);
        // fourcc[1] = (char)(((int)value >> 8) & 0XFF);
        // fourcc[2] = (char)(((int)value >> 16) & 0XFF);
        // fourcc[3] = (char)(((int)value >> 24) & 0XFF);
        // if(fourcc[0]=='M'&&fourcc[1]=='J'&&fourcc[2]=='P'&&fourcc[3]=='G'){

        // }
        // else if(fourcc[0]=='Y'&&fourcc[1]=='U'&&fourcc[2]=='Y'&&fourcc[3]=='V'){

        // }
        // else if(fourcc[0]=='R'&&fourcc[1]=='G'&&fourcc[2]=='B'&&fourcc[3]=='3'){
        //     still_flags = LibcameraApp::FLAG_STILL_RGB;
        // }
        // else{
        //     std::cerr << "Warning: FourCC code " << fourcc << " not supported." << std::endl;
        //     return false;
        // }
        // // needsReconfigure.store(true, std::memory_order_release);
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
    case cv::CAP_PROP_ROLL:       // Not implemented
    case cv::CAP_PROP_IRIS:       // Not implemented
        // These properties might need to trigger a re-configuration of the camera.
        // You can handle them here if you want to support changing resolution or framerate on-the-fly.
        // For now, we'll return false to indicate that these properties are not supported for dynamic changes.
        CV_LOG_WARNING(NULL, "VIDEOIO(Libcamera): Property " << propId << " is not supported for dynamic changes");
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
    (void)_deviceName;
    
    options->video_width = 1280;
    options->video_height = 720;
    options->framerate = 30;
    
    return startVideo();
}

Ptr<IVideoCapture> createLibcameraCapture_file(const std::string &filename)
{
    Ptr<LibcameraCapture> cap = makePtr<LibcameraCapture>();
    if (cap && cap->open(filename))
        return cap.staticCast<IVideoCapture>();
    return Ptr<IVideoCapture>();
}

Ptr<IVideoCapture> createLibcameraCapture_cam(int index)
{
    Ptr<LibcameraCapture> cap = makePtr<LibcameraCapture>();
    if (cap && cap->open(index))
        return cap.staticCast<IVideoCapture>();
    return Ptr<IVideoCapture>();
}

}