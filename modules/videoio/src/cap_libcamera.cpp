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

#include <libcamera/base/span.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/property_ids.h>
#include <libcamera/transform.h>


namespace cv 
{

enum Exposure_Modes {
    EXPOSURE_NORMAL = libcamera::controls::ExposureNormal,
    EXPOSURE_SHORT = libcamera::controls::ExposureShort,
    EXPOSURE_CUSTOM = libcamera::controls::ExposureCustom
};

enum Metering_Modes {
    METERING_CENTRE = libcamera::controls::MeteringCentreWeighted,
    METERING_SPOT = libcamera::controls::MeteringSpot,
    METERING_MATRIX = libcamera::controls::MeteringMatrix,
    METERING_CUSTOM = libcamera::controls::MeteringCustom
};

enum WhiteBalance_Modes {
    WB_AUTO = libcamera::controls::AwbAuto,
    WB_NORMAL = libcamera::controls::AwbAuto,
    WB_INCANDESCENT = libcamera::controls::AwbIncandescent,
    WB_TUNGSTEN = libcamera::controls::AwbTungsten,
    WB_FLUORESCENT = libcamera::controls::AwbFluorescent,
    WB_INDOOR = libcamera::controls::AwbIndoor,
    WB_DAYLIGHT = libcamera::controls::AwbDaylight,
    WB_CLOUDY = libcamera::controls::AwbCloudy,
    WB_CUSTOM = libcamera::controls::AwbAuto
};

class Options
{
public:
	Options()
	{
        timeout=1000;
        metering_index = Metering_Modes::METERING_CENTRE;
        exposure_index=Exposure_Modes::EXPOSURE_NORMAL;
        awb_index=WhiteBalance_Modes::WB_AUTO;
        saturation=1.0f;
        contrast=1.0f;
        sharpness=1.0f;
	brightness=0.0f;
	shutter=0.0f;
	gain=0.0f;
	ev=0.0f;
	roi_x=roi_y=roi_width=roi_height=0;
	awb_gain_r=awb_gain_b=0;
        denoise="auto";
        verbose=false;
	transform=libcamera::Transform::Identity;
	camera=0;
	}

	virtual ~Options() {}

	virtual void Print() const;

    void setMetering(Metering_Modes meteringmode){metering_index=meteringmode;}
    void setWhiteBalance(WhiteBalance_Modes wb){awb_index = wb;}
    void setExposureMode(Exposure_Modes exp){exposure_index = exp;}

    int getExposureMode(){return exposure_index;}
    int getMeteringMode(){return metering_index;}
    int getWhiteBalance(){return awb_index;}

	bool help;
	bool version;
	bool list_cameras;
	bool verbose;
	uint64_t timeout; // in ms
    unsigned int photo_width, photo_height;
    unsigned int video_width, video_height;
	bool rawfull;
	libcamera::Transform transform;
	float roi_x, roi_y, roi_width, roi_height;
	float shutter;
	float gain;
	float ev;
	float awb_gain_r;
	float awb_gain_b;
	float brightness;
	float contrast;
	float saturation;
	float sharpness;
	float framerate;
	std::string denoise;
	std::string info_text;
	unsigned int camera;

protected:
	int metering_index;
	int exposure_index;
    int awb_index;

private:
};
struct CompletedRequest;
using CompletedRequestPtr = std::shared_ptr<CompletedRequest>;

namespace controls = libcamera::controls;
namespace properties = libcamera::properties;

class LibcameraApp
{
public:
	using Stream = libcamera::Stream;
	using FrameBuffer = libcamera::FrameBuffer;
	using ControlList = libcamera::ControlList;
	using Request = libcamera::Request;
	using CameraManager = libcamera::CameraManager;
	using Camera = libcamera::Camera;
	using CameraConfiguration = libcamera::CameraConfiguration;
	using FrameBufferAllocator = libcamera::FrameBufferAllocator;
	using StreamRole = libcamera::StreamRole;
	using StreamRoles = std::vector<libcamera::StreamRole>;
	using PixelFormat = libcamera::PixelFormat;
	using StreamConfiguration = libcamera::StreamConfiguration;
	using BufferMap = Request::BufferMap;
	using Size = libcamera::Size;
	using Rectangle = libcamera::Rectangle;
	enum class MsgType
	{
		RequestComplete,
		Quit
	};
	typedef std::variant<CompletedRequestPtr> MsgPayload;
	struct Msg
	{
		Msg(MsgType const &t) : type(t) {}
		template <typename T>
		Msg(MsgType const &t, T p) : type(t), payload(std::forward<T>(p))
		{
		}
		MsgType type;
		MsgPayload payload;
	};

	// Some flags that can be used to give hints to the camera configuration.
	static constexpr unsigned int FLAG_STILL_NONE = 0;
	static constexpr unsigned int FLAG_STILL_BGR = 1; // supply BGR images, not YUV
	static constexpr unsigned int FLAG_STILL_RGB = 2; // supply RGB images, not YUV
	static constexpr unsigned int FLAG_STILL_RAW = 4; // request raw image stream
	static constexpr unsigned int FLAG_STILL_DOUBLE_BUFFER = 8; // double-buffer stream
	static constexpr unsigned int FLAG_STILL_TRIPLE_BUFFER = 16; // triple-buffer stream
	static constexpr unsigned int FLAG_STILL_BUFFER_MASK = 24; // mask for buffer flags

	static constexpr unsigned int FLAG_VIDEO_NONE = 0;
	static constexpr unsigned int FLAG_VIDEO_RAW = 1; // request raw image stream
	static constexpr unsigned int FLAG_VIDEO_JPEG_COLOURSPACE = 2; // force JPEG colour space

	LibcameraApp(std::unique_ptr<Options> const opts = nullptr);
	virtual ~LibcameraApp();

	Options *GetOptions() const { return options_.get(); }

	std::string const &CameraId() const;
	void OpenCamera();
	void CloseCamera();

	void ConfigureStill(unsigned int flags = FLAG_STILL_NONE);
    void ConfigureViewfinder();

	void Teardown();
	void StartCamera();
	void StopCamera();

    void ApplyRoiSettings();

	Msg Wait();
	void PostMessage(MsgType &t, MsgPayload &p);

	Stream *GetStream(std::string const &name, unsigned int *w = nullptr, unsigned int *h = nullptr,
					  unsigned int *stride = nullptr) const;
	Stream *ViewfinderStream(unsigned int *w = nullptr, unsigned int *h = nullptr,
							 unsigned int *stride = nullptr) const;
	Stream *StillStream(unsigned int *w = nullptr, unsigned int *h = nullptr, unsigned int *stride = nullptr) const;
	Stream *RawStream(unsigned int *w = nullptr, unsigned int *h = nullptr, unsigned int *stride = nullptr) const;
	Stream *VideoStream(unsigned int *w = nullptr, unsigned int *h = nullptr, unsigned int *stride = nullptr) const;
	Stream *LoresStream(unsigned int *w = nullptr, unsigned int *h = nullptr, unsigned int *stride = nullptr) const;
	Stream *GetMainStream() const;

	std::vector<libcamera::Span<uint8_t>> Mmap(FrameBuffer *buffer) const;

	void SetControls(ControlList &controls);
	void StreamDimensions(Stream const *stream, unsigned int *w, unsigned int *h, unsigned int *stride) const;

protected:
	std::unique_ptr<Options> options_;

private:
	template <typename T>
	class MessageQueue
	{
	public:
		template <typename U>
		void Post(U &&msg)
		{
			std::unique_lock<std::mutex> lock(mutex_);
			queue_.push(std::forward<U>(msg));
			cond_.notify_one();
		}
		T Wait()
		{
			std::unique_lock<std::mutex> lock(mutex_);
			cond_.wait(lock, [this] { return !queue_.empty(); });
			T msg = std::move(queue_.front());
			queue_.pop();
			return msg;
		}
		void Clear()
		{
			std::unique_lock<std::mutex> lock(mutex_);
			queue_ = {};
		}

	private:
		std::queue<T> queue_;
		std::mutex mutex_;
		std::condition_variable cond_;
	};

	void setupCapture();
	void makeRequests();
	void queueRequest(CompletedRequest *completed_request);
	void requestComplete(Request *request);
	void configureDenoise(const std::string &denoise_mode);

	std::unique_ptr<CameraManager> camera_manager_;
	std::shared_ptr<Camera> camera_;
	bool camera_acquired_ = false;
	std::unique_ptr<CameraConfiguration> configuration_;
	std::map<FrameBuffer *, std::vector<libcamera::Span<uint8_t>>> mapped_buffers_;
	std::map<std::string, Stream *> streams_;
	FrameBufferAllocator *allocator_ = nullptr;
	std::map<Stream *, std::queue<FrameBuffer *>> frame_buffers_;
	std::queue<Request *> free_requests_;
	std::vector<std::unique_ptr<Request>> requests_;
	std::mutex completed_requests_mutex_;
	std::set<CompletedRequest *> completed_requests_;
	bool camera_started_ = false;
	std::mutex camera_stop_mutex_;
	MessageQueue<Msg> msg_queue_;
	// For setting camera controls.
	std::mutex control_mutex_;
	ControlList controls_;
	// Other:
	uint64_t last_timestamp_;
	uint64_t sequence_ = 0;
};

struct FrameInfo
{
	FrameInfo(libcamera::ControlList &ctrls)
		: exposure_time(0.0), digital_gain(0.0), colour_gains({ { 0.0f, 0.0f } }), focus(0.0), aelock(false)
	{
        auto exp = ctrls.get(libcamera::controls::ExposureTime);
		if (exp)
			exposure_time = *exp;

		auto ag = ctrls.get(libcamera::controls::AnalogueGain);
		if (ag)
			analogue_gain = *ag;

		auto dg = ctrls.get(libcamera::controls::DigitalGain);
		if (dg)
			digital_gain = *dg;

		auto cg = ctrls.get(libcamera::controls::ColourGains);
		if (cg)
		{
			colour_gains[0] = (*cg)[0], colour_gains[1] = (*cg)[1];
		}

		auto fom = ctrls.get(libcamera::controls::FocusFoM);
		if (fom)
			focus = *fom;

		auto ae = ctrls.get(libcamera::controls::AeLocked);
		if (ae)
			aelock = *ae;
	}

	std::string ToString(std::string &info_string) const
	{
		std::string parsed(info_string);

		for (auto const &t : tokens)
		{
			std::size_t pos = parsed.find(t);
			if (pos != std::string::npos)
			{
				std::stringstream value;
				value << std::fixed << std::setprecision(2);

				if (t == "%frame")
					value << sequence;
				else if (t == "%fps")
					value << fps;
				else if (t == "%exp")
					value << exposure_time;
				else if (t == "%ag")
					value << analogue_gain;
				else if (t == "%dg")
					value << digital_gain;
				else if (t == "%rg")
					value << colour_gains[0];
				else if (t == "%bg")
					value << colour_gains[1];
				else if (t == "%focus")
					value << focus;
				else if (t == "%aelock")
					value << aelock;

				parsed.replace(pos, t.length(), value.str());
			}
		}

		return parsed;
	}

	unsigned int sequence;
	float exposure_time;
	float analogue_gain;
	float digital_gain;
	std::array<float, 2> colour_gains;
	float focus;
	float fps;
	bool aelock;

private:
	// Info text tokens.
	inline static const std::string tokens[] = { "%frame", "%fps", "%exp",	 "%ag",	   "%dg",
												 "%rg",	   "%bg",  "%focus", "%aelock" };
};

class Metadata
{
public:
	Metadata() = default;

	Metadata(Metadata const &other)
	{
		std::scoped_lock other_lock(other.mutex_);
		data_ = other.data_;
	}

	Metadata(Metadata &&other)
	{
		std::scoped_lock other_lock(other.mutex_);
		data_ = std::move(other.data_);
		other.data_.clear();
	}

	template <typename T>
	void Set(std::string const &tag, T &&value)
	{
		std::scoped_lock lock(mutex_);
		data_.insert_or_assign(tag, std::forward<T>(value));
	}

	template <typename T>
	int Get(std::string const &tag, T &value) const
	{
		std::scoped_lock lock(mutex_);
		auto it = data_.find(tag);
		if (it == data_.end())
			return -1;
		value = std::any_cast<T>(it->second);
		return 0;
	}

	void Clear()
	{
		std::scoped_lock lock(mutex_);
		data_.clear();
	}

	Metadata &operator=(Metadata const &other)
	{
		std::scoped_lock lock(mutex_, other.mutex_);
		data_ = other.data_;
		return *this;
	}

	Metadata &operator=(Metadata &&other)
	{
		std::scoped_lock lock(mutex_, other.mutex_);
		data_ = std::move(other.data_);
		other.data_.clear();
		return *this;
	}

	void Merge(Metadata &other)
	{
		std::scoped_lock lock(mutex_, other.mutex_);
		data_.merge(other.data_);
	}

	template <typename T>
	T *GetLocked(std::string const &tag)
	{
		// This allows in-place access to the Metadata contents,
		// for which you should be holding the lock.
		auto it = data_.find(tag);
		if (it == data_.end())
			return nullptr;
		return std::any_cast<T>(&it->second);
	}

	template <typename T>
	void SetLocked(std::string const &tag, T &&value)
	{
		// Use this only if you're holding the lock yourself.
		data_.insert_or_assign(tag, std::forward<T>(value));
	}

	// Note: use of (lowercase) lock and unlock means you can create scoped
	// locks with the standard lock classes.
	// e.g. std::lock_guard<RPiController::Metadata> lock(metadata)
	void lock() { mutex_.lock(); }
	void unlock() { mutex_.unlock(); }

private:
	mutable std::mutex mutex_;
	std::map<std::string, std::any> data_;
};

struct CompletedRequest
{
	using BufferMap = libcamera::Request::BufferMap;
	using ControlList = libcamera::ControlList;
	using Request = libcamera::Request;

	CompletedRequest(unsigned int seq, Request *r)
		: sequence(seq), buffers(r->buffers()), metadata(r->metadata()), request(r)
	{
		r->reuse();
	}
	unsigned int sequence;
	BufferMap buffers;
	ControlList metadata;
	Request *request;
	float framerate;
	Metadata post_process_metadata;
};


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
        timeout_reached = (std::chrono::high_resolution_clock::now() - start_time > std::chrono::milliseconds(1000));//timeout=1000. Need to be modified in this class.
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
