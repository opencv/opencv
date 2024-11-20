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

#include <libcamera/base/span.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/property_ids.h>


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


/* ******************************************************************* */
class LibcameraCapture CV_FINAL : public IVideoCapture
{
private:

public:
    LibcameraCapture();
    virtual ~LibcameraCapture() CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual bool retrieveFrame(int /*unused*/, OutputArray dst) CV_OVERRIDE;
    bool configureAudioFrame();
    bool grabVideoFrame();
    bool grabAudioFrame();
    bool retrieveVideoFrame(int /*unused*/, OutputArray dst);
    bool retrieveAudioFrame(int /*unused*/, OutputArray dst);
    virtual double getProperty(int propId) const CV_OVERRIDE;
    virtual bool setProperty(int propId, double value) CV_OVERRIDE;
    // virtual bool isOpened() const CV_OVERRIDE { return (bool)pipeline; }
    virtual int getCaptureDomain() CV_OVERRIDE { return cv::CAP_LIBCAMERA; } // Need to modify videoio.hpp/enum VideoCaptureAPIs
    bool open(int id, const cv::VideoCaptureParameters& params);
    bool open(const String &filename_, const cv::VideoCaptureParameters& params);
    bool configureHW(const cv::VideoCaptureParameters&);
    bool configureStreamsProperty(const cv::VideoCaptureParameters&);
    bool setAudioProperties(const cv::VideoCaptureParameters&);

protected:
    // bool isPipelinePlaying();
    // void startPipeline();
    // void stopPipeline();
    // void restartPipeline();
    // void setFilter(const char *prop, int type, int v1, int v2);
    // void removeFilter(const char *filter);

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
    if(startVideo()) // Check if video framebuffer works
    { // Implemented from lccv::Picamera::capturePhoto
        stopVideo();
		options->photo_width=2028;
		options->photo_height=1520;
		options->verbose=true;
		cv::Mat frame;
		if(!capturePhoto(frame)) // Check if real cv::Mat frame can be grabbed
		{
			std::cerr<<"grabframe() -> capturePhoto() Failed.";
			return false;
		}

    }
    else 
	{
		std::cerr<<"grabFrame() -> startVideo() Failed.";
		return false;
	}
}

bool LibcameraCapture::retrieveFrame()
{
	startVideo();
	if(!running.load(std::memory_order_acquire))return false;//判断相机是否运行
    auto start_time = std::chrono::high_resolution_clock::now();//开始计时
    bool timeout_reached = false;
    timespec req;
    req.tv_sec=0;
    req.tv_nsec=1000000;//1ms
    //等待帧就绪或超时
    while((!frameready.load(std::memory_order_acquire))&&(!timeout_reached)){
        nanosleep(&req,NULL);
        timeout_reached = (std::chrono::high_resolution_clock::now() - start_time > std::chrono::milliseconds(timeout));
    }
    if(frameready.load(std::memory_order_acquire)){
        frame.create(vh,vw,CV_8UC3);
        uint ls = vw*3;
        mtx.lock();
            uint8_t *ptr = framebuffer;
            for (unsigned int i = 0; i < vh; i++, ptr += vstr)
                memcpy(frame.ptr(i),ptr,ls);
        mtx.unlock();
        frameready.store(false, std::memory_order_release);;
        return true;
    }
    else
        return false;
}

} //namespace
