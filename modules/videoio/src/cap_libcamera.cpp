#include "precomp.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/filesystem.hpp>

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

#include <libcamera/base/span.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/property_ids.h>
#inclued <libcamera/stream.h>
#include <cap_libcamera.hpp>


namespace cv 
{

LibcameraApp::LibcameraApp(std::unique_ptr<Options> opts)
	: options_(std::move(opts)), controls_(controls::controls)

{
	if (!options_)
		options_ = std::make_unique<Options>();
	controls_.clear();
}

LibcameraApp::~LibcameraApp()
{
	StopCamera();
	Teardown();
	CloseCamera();
}

std::string const &LibcameraApp::CameraId() const
{
	return camera_->id();
}

void LibcameraApp::OpenCamera()
{

	if (options_->verbose)
		std::cerr << "Opening camera..." << std::endl;

	camera_manager_ = std::make_unique<CameraManager>();
	int ret = camera_manager_->start();
	if (ret)
		throw std::runtime_error("camera manager failed to start, code " + std::to_string(-ret));

	if (camera_manager_->cameras().size() == 0)
		throw std::runtime_error("no cameras available");
	if (options_->camera >= camera_manager_->cameras().size())
		throw std::runtime_error("selected camera is not available");

	std::string const &cam_id = camera_manager_->cameras()[options_->camera]->id();
	camera_ = camera_manager_->get(cam_id);
	if (!camera_)
		throw std::runtime_error("failed to find camera " + cam_id);

	if (camera_->acquire())
		throw std::runtime_error("failed to acquire camera " + cam_id);
	camera_acquired_ = true;

	if (options_->verbose)
		std::cerr << "Acquired camera " << cam_id << std::endl;

}

void LibcameraApp::CloseCamera()
{
	if (camera_acquired_)
		camera_->release();
	camera_acquired_ = false;

	camera_.reset();

	camera_manager_.reset();

	if (options_->verbose && !options_->help)
		std::cerr << "Camera closed" << std::endl;
}

void LibcameraApp::ConfigureStill(unsigned int flags)
{
	if (options_->verbose)
		std::cerr << "Configuring still capture..." << std::endl;

	// Always request a raw stream as this forces the full resolution capture mode.
	// (options_->mode can override the choice of camera mode, however.)
	StreamRoles stream_roles = { StreamRole::StillCapture, StreamRole::Raw };
	configuration_ = camera_->generateConfiguration(stream_roles);
	if (!configuration_)
		throw std::runtime_error("failed to generate still capture configuration");

	// Now we get to override any of the default settings from the options_->
	if (flags & FLAG_STILL_BGR)
		configuration_->at(0).pixelFormat = libcamera::formats::BGR888;
	else if (flags & FLAG_STILL_RGB)
		configuration_->at(0).pixelFormat = libcamera::formats::RGB888;
	else
		configuration_->at(0).pixelFormat = libcamera::formats::YUV420;
	if ((flags & FLAG_STILL_BUFFER_MASK) == FLAG_STILL_DOUBLE_BUFFER)
		configuration_->at(0).bufferCount = 2;
	else if ((flags & FLAG_STILL_BUFFER_MASK) == FLAG_STILL_TRIPLE_BUFFER)
		configuration_->at(0).bufferCount = 3;
    if (options_->photo_width)
        configuration_->at(0).size.width = options_->photo_width;
    if (options_->photo_height)
        configuration_->at(0).size.height = options_->photo_height;

//    configuration_->transform = options_->transform;

	//if (have_raw_stream && !options_->rawfull)
	{
		configuration_->at(1).size.width = configuration_->at(0).size.width;
		configuration_->at(1).size.height = configuration_->at(0).size.height;
	}
	configuration_->at(1).bufferCount = configuration_->at(0).bufferCount;

	configureDenoise(options_->denoise == "auto" ? "cdn_hq" : options_->denoise);
	setupCapture();

	streams_["still"] = configuration_->at(0).stream();
	streams_["raw"] = configuration_->at(1).stream();

	if (options_->verbose)
		std::cerr << "Still capture setup complete" << std::endl;
}

void LibcameraApp::ConfigureViewfinder()
{
    if (options_->verbose)
        std::cerr << "Configuring viewfinder..." << std::endl;

    StreamRoles stream_roles = { StreamRole::Viewfinder };
    configuration_ = camera_->generateConfiguration(stream_roles);
    if (!configuration_)
        throw std::runtime_error("failed to generate viewfinder configuration");

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
        std::cerr << "Viewfinder setup complete" << std::endl;
}

void LibcameraApp::Teardown()
{
	if (options_->verbose && !options_->help)
		std::cerr << "Tearing down requests, buffers and configuration" << std::endl;

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

void LibcameraApp::StartCamera()
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
			std::cerr << "Using crop " << crop.toString() << std::endl;
		controls_.set(controls::ScalerCrop, crop);
	}

	// Framerate is a bit weird. If it was set programmatically, we go with that, but
	// otherwise it applies only to preview/video modes. For stills capture we set it
	// as long as possible so that we get whatever the exposure profile wants.
	if (!controls_.get(controls::FrameDurationLimits))
	{
		if (StillStream())
			controls_.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({ INT64_C(100), INT64_C(1000000000) }));
		else if (options_->framerate > 0)
		{
			int64_t frame_time = 1000000 / options_->framerate; // in us
			controls_.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({ frame_time, frame_time }));
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
		controls_.set(controls::ColourGains, libcamera::Span<const float, 2>({ options_->awb_gain_r, options_->awb_gain_b }));
	if (!controls_.get(controls::Brightness))
		controls_.set(controls::Brightness, options_->brightness);
	if (!controls_.get(controls::Contrast))
		controls_.set(controls::Contrast, options_->contrast);
	if (!controls_.get(controls::Saturation))
		controls_.set(controls::Saturation, options_->saturation);
	if (!controls_.get(controls::Sharpness))
		controls_.set(controls::Sharpness, options_->sharpness);

	if (camera_->start(&controls_))
		throw std::runtime_error("failed to start camera");
	controls_.clear();
	camera_started_ = true;
	last_timestamp_ = 0;

	camera_->requestCompleted.connect(this, &LibcameraApp::requestComplete);

	for (std::unique_ptr<Request> &request : requests_)
	{
		if (camera_->queueRequest(request.get()) < 0)
			throw std::runtime_error("Failed to queue request");
	}

	if (options_->verbose)
		std::cerr << "Camera started!" << std::endl;
}

void LibcameraApp::StopCamera()
{
	{
		// We don't want QueueRequest to run asynchronously while we stop the camera.
		std::lock_guard<std::mutex> lock(camera_stop_mutex_);
		if (camera_started_)
		{
			if (camera_->stop())
				throw std::runtime_error("failed to stop camera");

			camera_started_ = false;
		}
	}

	if (camera_)
		camera_->requestCompleted.disconnect(this, &LibcameraApp::requestComplete);

	// An application might be holding a CompletedRequest, so queueRequest will get
	// called to delete it later, but we need to know not to try and re-queue it.
	completed_requests_.clear();

	msg_queue_.Clear();

	while (!free_requests_.empty())
		free_requests_.pop();

	requests_.clear();

	controls_.clear(); // no need for mutex here

	if (options_->verbose && !options_->help)
		std::cerr << "Camera stopped!" << std::endl;
}

void LibcameraApp::ApplyRoiSettings(){
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
            std::cerr << "Using crop " << crop.toString() << std::endl;
        controls_.set(controls::ScalerCrop, crop);
    }
}

LibcameraApp::Msg LibcameraApp::Wait()
{
	return msg_queue_.Wait();
}

void LibcameraApp::queueRequest(CompletedRequest *completed_request)
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
			throw std::runtime_error("failed to add buffer to request in QueueRequest");
	}

	{
		std::lock_guard<std::mutex> lock(control_mutex_);
		request->controls() = std::move(controls_);
	}

	if (camera_->queueRequest(request) < 0)
		throw std::runtime_error("failed to queue request");
}

void LibcameraApp::PostMessage(MsgType &t, MsgPayload &p)
{
	msg_queue_.Post(Msg(t, std::move(p)));
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

void LibcameraApp::setupCapture()
{
	// First finish setting up the configuration.

	CameraConfiguration::Status validation = configuration_->validate();
	if (validation == CameraConfiguration::Invalid)
		throw std::runtime_error("failed to valid stream configurations");
	else if (validation == CameraConfiguration::Adjusted)
		std::cerr << "Stream configuration adjusted" << std::endl;

	if (camera_->configure(configuration_.get()) < 0)
		throw std::runtime_error("failed to configure streams");
	if (options_->verbose)
		std::cerr << "Camera streams configured" << std::endl;

	// Next allocate all the buffers we need, mmap them and store them on a free list.

	allocator_ = new FrameBufferAllocator(camera_);
	for (StreamConfiguration &config : *configuration_)
	{
		Stream *stream = config.stream();

		if (allocator_->allocate(stream) < 0)
			throw std::runtime_error("failed to allocate capture buffers");

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
		std::cerr << "Buffers allocated and mapped" << std::endl;

	// The requests will be made when StartCamera() is called.
}

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
					if (options_->verbose)
						std::cerr << "Requests created" << std::endl;
					return;
				}
				std::unique_ptr<Request> request = camera_->createRequest();
				if (!request)
					throw std::runtime_error("failed to make request");
				requests_.push_back(std::move(request));
			}
			else if (free_buffers[stream].empty())
				throw std::runtime_error("concurrent streams need matching numbers of buffers");

			FrameBuffer *buffer = free_buffers[stream].front();
			free_buffers[stream].pop();
			if (requests_.back()->addBuffer(stream, buffer) < 0)
				throw std::runtime_error("failed to add buffer to request");
		}
	}
}

void LibcameraApp::requestComplete(Request *request)
{
	if (request->status() == Request::RequestCancelled)
		return;

	CompletedRequest *r = new CompletedRequest(sequence_++, request);
	CompletedRequestPtr payload(r, [this](CompletedRequest *cr) { this->queueRequest(cr); });
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

void LibcameraApp::configureDenoise(const std::string &denoise_mode)
{
	using namespace libcamera::controls::draft;

	static const std::map<std::string, NoiseReductionModeEnum> denoise_table = {
		{ "off", NoiseReductionModeOff },
		{ "cdn_off", NoiseReductionModeMinimal },
		{ "cdn_fast", NoiseReductionModeFast },
		{ "cdn_hq", NoiseReductionModeHighQuality }
	};
	NoiseReductionModeEnum denoise;

	auto const mode = denoise_table.find(denoise_mode);
	if (mode == denoise_table.end())
		throw std::runtime_error("Invalid denoise mode " + denoise_mode);
	denoise = mode->second;

	controls_.set(NoiseReductionMode, denoise);
}

/* ******************************************************************* */
class LibcameraCapture CV_FINAL : public IVideoCapture
{
private:

public:
    LibcameraCapture();
    virtual ~LibcameraCapture() CV_OVERRIDE;

    Options *options;

    bool startPhoto();
    bool capturePhoto(cv::Mat &frame);
    bool stopPhoto();

    bool startVideo();
    bool getVideoFrame(cv::Mat &frame, unsigned int timeout);
    void stopVideo();

    bool open(int _index);
    bool open(const std::string & filename);

    virtual bool grabFrame() CV_OVERRIDE;
    virtual bool retrieveFrame(int /*unused*/, OutputArray dst) CV_OVERRIDE;
    virtual double getProperty(int propId) const CV_OVERRIDE;
    virtual bool setProperty(int propId, double value) CV_OVERRIDE;
    // virtual bool isOpened() const CV_OVERRIDE { return (bool)pipeline; }
    virtual int getCaptureDomain() CV_OVERRIDE { return cv::CAP_LIBCAMERA; } // Need to modify videoio.hpp/enum VideoCaptureAPIs
    bool configureHW(const cv::VideoCaptureParameters&);
    bool configureStreamsProperty(const cv::VideoCaptureParameters&);
    bool isOpened() const CV_OVERRIDE { return camerastarted; }

protected:

    LibcameraApp *app;
    void getImage(cv::Mat &frame, CompletedRequestPtr &payload);
    static void *videoThreadFunc(void *p);
    pthread_t videothread;
    unsigned int still_flags;
    unsigned int vw,vh,vstr;
    std::atomic<bool> running,frameready;
    uint8_t *framebuffer;
    std::mutex mtx;
    bool camerastarted;
    bool isFramePending;
};

LibcameraCapture::LibcameraCapture()
{
	app = new LibcameraApp(std::make_unique<Options>());
    options = static_cast<Options *>(app->GetOptions());
    still_flags = LibcameraApp::FLAG_STILL_NONE;
    options->photo_width = 4056;
    options->photo_height = 3040;
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
    still_flags |= LibcameraApp::FLAG_STILL_RGB;
    running.store(false, std::memory_order_release);;
    frameready.store(false, std::memory_order_release);;
    framebuffer=nullptr;
    camerastarted=false;
    isFramePending=false;
}

LibcameraCapture::~LibcameraCapture()
{
    delete app;
}

// using namespace LibcameraApp;


void LibcameraCapture::getImage(cv::Mat &frame, CompletedRequestPtr &payload)
{
    unsigned int w, h, stride;
    libcamera::Stream *stream = app->StillStream();
	app->StreamDimensions(stream, &w, &h, &stride);
    const std::vector<libcamera::Span<uint8_t>> mem = app->Mmap(payload->buffers[stream]);
    frame.create(h,w,CV_8UC3);
    uint ls = w*3;
    uint8_t *ptr = (uint8_t *)mem[0].data();
    for (unsigned int i = 0; i < h; i++, ptr += stride)
    {
        memcpy(frame.ptr(i),ptr,ls);
    }
}

bool LibcameraCapture::startPhoto()
{
    LibcameraCapture::app->OpenCamera();
    LibcameraCapture::app->ConfigureStill(still_flags);
    camerastarted=true;
    return true;
}

bool LibcameraCapture::stopPhoto()
{
    if(camerastarted){
        camerastarted=false;
        LibcameraCapture::app->Teardown();
        LibcameraCapture::app->CloseCamera();
    }
    return true;
}

bool LibcameraCapture::capturePhoto(cv::Mat &frame)
{   
    if(!camerastarted){
        LibcameraCapture::app->OpenCamera();
        LibcameraCapture::app->ConfigureStill(still_flags);
    }
    LibcameraCapture::app->StartCamera();
    LibcameraApp::Msg msg = LibcameraCapture::app->Wait();
    if (msg.type == LibcameraApp::MsgType::Quit)
        return false;
    else if (msg.type != LibcameraApp::MsgType::RequestComplete)
        return false;
    if (LibcameraCapture::app->StillStream())
    {
        LibcameraCapture::app->StopCamera();
        getImage(frame, std::get<CompletedRequestPtr>(msg.payload));
        LibcameraCapture::app->Teardown();
        LibcameraCapture::app->CloseCamera();
    } else {
        std::cerr<<"Incorrect stream received"<<std::endl;
        return false;
        LibcameraCapture::app->StopCamera();
        if(!camerastarted){
            LibcameraCapture::app->Teardown();
            LibcameraCapture::app->CloseCamera();
        }
    }
    return true;
}

void *LibcameraCapture::videoThreadFunc(void *p) //not resolved
{   
    LibcameraCapture *t = (LibcameraCapture *)p;
    t->running.store(true, std::memory_order_release);
    //allocate framebuffer
    //unsigned int vw,vh,vstr;
    libcamera::Stream *stream = t->app->ViewfinderStream(&t->vw,&t->vh,&t->vstr);
    int buffersize=t->vh*t->vstr;
    if(t->framebuffer)delete[] t->framebuffer;
    t->framebuffer=new uint8_t[buffersize];
    std::vector<libcamera::Span<uint8_t>> mem;

    //main loop
    while(t->running.load(std::memory_order_acquire)){
        LibcameraApp::Msg msg = t->app->Wait();
        if (msg.type == LibcameraApp::MsgType::Quit){
            std::cerr<<"Quit message received"<<std::endl;
            t->running.store(false,std::memory_order_release);
        }
        else if (msg.type != LibcameraApp::MsgType::RequestComplete)
            throw std::runtime_error("unrecognised message!");


        CompletedRequestPtr payload = std::get<CompletedRequestPtr>(msg.payload);
        mem = t->app->Mmap(payload->buffers[stream]);
        t->mtx.lock();
            memcpy(t->framebuffer,mem[0].data(),buffersize);
        t->mtx.unlock();
        t->frameready.store(true, std::memory_order_release);
    }
    if(t->framebuffer){
        delete[] t->framebuffer;
        t->framebuffer=nullptr;
    }
    return NULL;
}

bool LibcameraCapture::startVideo() //not resolved
{   
    if(camerastarted) stopPhoto();
    if(running.load(std::memory_order_relaxed)){
        std::cerr<<"Video thread already running";
        return false;
    }
    frameready.store(false, std::memory_order_release);
    LibcameraCapture::app->OpenCamera();
    LibcameraCapture::app->ConfigureViewfinder();
    LibcameraCapture::app->StartCamera();

    int ret = pthread_create(&videothread, NULL, &videoThreadFunc, this);
    if (ret != 0) {
        std::cerr<<"Error starting video thread";
        return false;
    }
    return true;
}

void LibcameraCapture::stopVideo() //not resolved
{
    if(!running)return;

    running.store(false, std::memory_order_release);;

    //join thread
    void *status;
    int ret = pthread_join(videothread, &status);
    if(ret<0)
        std::cerr<<"Error joining thread"<<std::endl;

    LibcameraCapture::app->StopCamera();
    LibcameraCapture::app->Teardown();
    LibcameraCapture::app->CloseCamera();
    frameready.store(false, std::memory_order_release);;
}


/**
 * @brief Attempt to start the camera and ensure a frame is pending for capture.
 *
 * This function checks whether a frame is already pending. If a frame is pending, 
 * it returns immediately with `true`. If no frame is pending, the function attempts 
 * to configure the camera, start the camera stream, and create a video thread 
 * to handle video capturing. The `isFramePending` flag is updated accordingly.
 *
 * @return `true` if a frame is pending for capture.
 *         `false` if an error occurs while starting the video thread or configuring the camera.
 */
bool LibcameraCapture::grabFrame()
{   
    if(isFramePending)
    {
        return true;
    }
    else 
	{
		LibcameraCapture::app->ConfigureViewfinder();
        LibcameraCapture::app->StartCamera();

        int ret = pthread_create(&videothread, NULL, &videoThreadFunc, this);
        if (ret != 0) {
            std::cerr<<"Error starting video thread";
            return false;
        }
        isFramePending = true;
	}
    return isFramePending;
}



/**
* @brief Retrieve a single frame from the video stream and copy it to the destination.
*
* This function waits for a frame to be ready in the framebuffer, extracts the frame data,
* and copies it to the provided OpenCV `OutputArray`. It uses a timeout mechanism to avoid
* indefinite blocking if no frame becomes available.
*
* @param int Unused parameter. 
* @param dst An OpenCV `OutputArray` where the retrieved frame will be stored.
*            The frame is stored in RGB format (8-bit, 3 channels, CV_8UC3).
*
* @return `true` if a frame is successfully retrieved and copied to `dst`.
*         `false` if no frame is ready (e.g., due to timeout or video not running).
*/
bool LibcameraCapture::retrieveFrame(int, OutputArray dst)
{   
    
	if(!running.load(std::memory_order_acquire))return false;
    auto start_time = std::chrono::high_resolution_clock::now();
    bool timeout_reached = false;
    timespec req;
    req.tv_sec=0;
    req.tv_nsec=1000000;//1ms
    while((!frameready.load(std::memory_order_acquire))&&(!timeout_reached)){
        nanosleep(&req,NULL);
        timeout_reached = (std::chrono::high_resolution_clock::now() - start_time > std::chrono::milliseconds(1000));
        //timeout=1000. Need to be modified in this class.
    }
    if(frameready.load(std::memory_order_acquire)){
        Mat frame(vh,vw,CV_8UC3);
        uint ls = vw*3;
        mtx.lock();
            uint8_t *ptr = framebuffer;
            for (unsigned int i = 0; i < vh; i++, ptr += vstr)
                memcpy(frame.ptr(i),ptr,ls);
        mtx.unlock();
        frameready.store(false, std::memory_order_release);;
        frame.copyTo(dst);
        return true;
    }
    else
        return false;
}

bool LibcameraCapture::open(int _index)
{
    cv::String name;
    /* Select camera, or rather, V4L video source */
    if (_index < 0) // Asking for the first device available
    {
        for (int autoindex = 0; autoindex < 8; ++autoindex)//8=MAX_CAMERAS
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

bool LibcameraCapture::open(const std::string & _deviceName)
{
    CV_LOG_DEBUG(NULL, "VIDEOIO(Libcamera:" << _deviceName << "): opening...");
    //Some parameters initialization here, maybe more needed.
    options->video_width=1024;
    options->video_height=768;
    options->framerate=5;
    options->verbose=true;
    //same procedure as startVideo() below, try to replace startVideo() later.
    if(camerastarted) stopPhoto();
    if(running.load(std::memory_order_relaxed)){
        std::cerr<<"Video thread already running";
        return false;
    }
    frameready.store(false, std::memory_order_release);
    LibcameraCapture::app->OpenCamera();
    return true;
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

} //namespace
