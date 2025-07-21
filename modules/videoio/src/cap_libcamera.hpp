// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#pragma once

#include <fstream>
#include <iostream>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <functional>
#include <variant>
#include <any>
#include <map>

#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/property_ids.h>
#include <libcamera/transform.h>
#include <mutex>
#include <opencv2/core/mat.hpp>

#include "cap_interface.hpp"

namespace cv
{
// Forward declarations
class LibcameraApp;
struct CompletedRequest;
using CompletedRequestPtr = std::shared_ptr<CompletedRequest>;

// Enumerations for camera settings
enum Exposure_Modes
{
    EXPOSURE_NORMAL = libcamera::controls::ExposureNormal,
    EXPOSURE_SHORT = libcamera::controls::ExposureShort,
    EXPOSURE_CUSTOM = libcamera::controls::ExposureCustom
};

enum Metering_Modes
{
    METERING_CENTRE = libcamera::controls::MeteringCentreWeighted,
    METERING_SPOT = libcamera::controls::MeteringSpot,
    METERING_MATRIX = libcamera::controls::MeteringMatrix,
    METERING_CUSTOM = libcamera::controls::MeteringCustom
};

enum WhiteBalance_Modes
{
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
        timeout = 1000;
        metering_index = Metering_Modes::METERING_CENTRE;
        exposure_index = Exposure_Modes::EXPOSURE_NORMAL;
        awb_index = WhiteBalance_Modes::WB_AUTO;
        saturation = 1.0f;
        contrast = 1.0f;
        sharpness = 1.0f;
        brightness = 0.0f;
        shutter = 0.0f;
        gain = 0.0f;
        ev = 0.0f;
        roi_x = roi_y = roi_width = roi_height = 0;
        awb_gain_r = awb_gain_b = 0;
        denoise = "auto";
        verbose = false;
        transform = libcamera::Transform::Identity;
        camera = 0;
        buffer_count = 4;
    }

    ~Options() {}

    void setMetering(Metering_Modes meteringmode) { metering_index = meteringmode; }
    void setWhiteBalance(WhiteBalance_Modes wb) { awb_index = wb; }
    void setExposureMode(Exposure_Modes exp) { exposure_index = exp; }

    int getExposureMode() { return exposure_index; }
    int getMeteringMode() { return metering_index; }
    int getWhiteBalance() { return awb_index; }

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
    unsigned int buffer_count;  // Number of buffers to allocate per stream

protected:
    int metering_index;
    int exposure_index;
    int awb_index;

private:
};

class LibcameraCapture;

class LibcameraFrameAllocator : public cv::MatAllocator
{
private:
    LibcameraApp* app_;                          // 指向App实例，用于归还请求
    mutable libcamera::Request* request_to_recycle_; // 需要回收的请求
    
public:
    LibcameraFrameAllocator(LibcameraApp* app, libcamera::Request* request);
    virtual ~LibcameraFrameAllocator();
    
    // MatAllocator interface
    cv::UMatData* allocate(int dims, const int* sizes, int type, void* data, size_t* step, 
                           cv::AccessFlag flags, cv::UMatUsageFlags usageFlags) const override;
    bool allocate(cv::UMatData* data, cv::AccessFlag accessFlags, cv::UMatUsageFlags usageFlags) const override;
    void deallocate(cv::UMatData* data) const override;
};

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
    static constexpr unsigned int FLAG_STILL_BGR = 1;           
    static constexpr unsigned int FLAG_STILL_RGB = 2;           
    static constexpr unsigned int FLAG_STILL_RAW = 4;            // request raw image stream
    static constexpr unsigned int FLAG_STILL_DOUBLE_BUFFER = 8;  // double-buffer stream
    static constexpr unsigned int FLAG_STILL_TRIPLE_BUFFER = 16; // triple-buffer stream
    static constexpr unsigned int FLAG_STILL_BUFFER_MASK = 24;   // mask for buffer flags

    static constexpr unsigned int FLAG_VIDEO_NONE = 0;
    static constexpr unsigned int FLAG_VIDEO_RAW = 1;              // request raw image stream
    static constexpr unsigned int FLAG_VIDEO_JPEG_COLOURSPACE = 2; // force JPEG colour space

    LibcameraApp(std::unique_ptr<Options> opts = nullptr);
    virtual ~LibcameraApp();

    Options *GetOptions() const { return options_.get(); }
    
    void SetCaptureInstance(LibcameraCapture* capture) { capture_instance_ = capture; }

    std::string const &CameraId() const;
    void OpenCamera();
    void CloseCamera();

    void ConfigureStill(unsigned int flags = FLAG_STILL_NONE);
    void ConfigureViewfinder();

    void Teardown();
    void StartCamera();
    void StopCamera();
    
    // (按需模式)手动提交一个请求
    bool submitSingleRequest();
    
    // (按需模式)手动归还请求到队列
    void returnRequestToQueue(libcamera::Request* request);
    
    // 获取当前空闲请求数量
    size_t getFreeRequestsCount() const;
    
    // 请求回收
    void recycleRequest(libcamera::Request* request);

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
    
    // LibcameraCapture 实例的引用，用于直接回调
    LibcameraCapture* capture_instance_;

private:
    static std::shared_ptr<CameraManager> getCameraManager()
    {
        static std::shared_ptr<CameraManager> camera_manager_;
        if (!camera_manager_)
        {
            std::cerr << "creating manager" << std::endl;
            camera_manager_ = std::make_shared<CameraManager>();
            int ret = camera_manager_->start();
            if (ret)
                throw std::runtime_error("camera manager failed to start,"
                                         "code " +
                                         std::to_string(-ret));
        }

        return camera_manager_;
    }

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
            cond_.wait(lock, [this]
                       { return !queue_.empty(); });
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

    // std::unique_ptr<CameraManager> camera_manager_;
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
    std::set<CompletedRequest *> active_requests_;
    bool camera_started_ = false;
    mutable std::mutex camera_stop_mutex_;
    MessageQueue<Msg> msg_queue_;
    // For setting camera controls.
    std::mutex control_mutex_;
    ControlList controls_;
    // Other:
    uint64_t last_timestamp_;
    uint64_t sequence_ = 0;
};

class Metadata
{
public:
    Metadata() = default;

    Metadata(Metadata const &other)
    {
        std::lock_guard<std::mutex> other_lock(other.mutex_);
        data_ = other.data_;
    }

    Metadata(Metadata &&other)
    {
        std::lock_guard<std::mutex> other_lock(other.mutex_);
        data_ = std::move(other.data_);
        other.data_.clear();
    }

    template <typename T>
    void Set(std::string const &tag, T &&value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
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
        std::lock_guard<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> other_lock(other.mutex_);
        data_ = other.data_;
        return *this;
    }

    Metadata &operator=(Metadata &&other)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> other_lock(other.mutex_);
        data_ = std::move(other.data_);
        other.data_.clear();
        return *this;
    }

    void Merge(Metadata &other)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> other_lock(other.mutex_);
        
        for (auto& pair : other.data_) {
            data_[pair.first] = std::move(pair.second);
        }
        other.data_.clear();
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

class LibcameraCapture CV_FINAL : public IVideoCapture
{
public:
    LibcameraCapture();
    LibcameraCapture(int camera_index);
    virtual ~LibcameraCapture() CV_OVERRIDE;

    Options *options;

    bool startVideo();
    void stopVideo();

    bool open(int _index);
    bool open(const std::string &filename);

    virtual bool grabFrame() CV_OVERRIDE;
    virtual bool retrieveFrame(int /*unused*/, OutputArray dst) CV_OVERRIDE;
    
    // 供插件使用
    bool grab() { return grabFrame(); }
    bool retrieve(cv::Mat& frame, int stream_idx = 0) 
    { 
        return retrieveFrame(stream_idx, frame); 
    }
    
    virtual double getProperty(int propId) const CV_OVERRIDE;
    virtual bool setProperty(int propId, double value) CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE { return cv::CAP_LIBCAMERA; }
    
    bool isOpened() const CV_OVERRIDE
    {
        return camera_started_.load(std::memory_order_acquire);
    }

    // 新的回调接口，由 LibcameraApp 调用
    void onRequestComplete(CompletedRequestPtr completed_request);
    size_t getMaxQueueSize() const { return MAX_QUEUE_SIZE; }

protected:
    LibcameraApp *app;
    unsigned int still_flags;
    unsigned int vw, vh, vstr;
    std::atomic<bool> camera_started_;
    std::atomic<bool> needsReconfigure;

    std::queue<CompletedRequestPtr> completed_requests_;
    mutable std::mutex completed_requests_mutex_;
    std::condition_variable completed_requests_cv_;
    
    // MAX_QUEUE_SIZE == 0: On demand mode, sending a request when user needs a img
    // MAX_QUEUE_SIZE > 0: Producer-consumer queue. 
    static constexpr size_t MAX_QUEUE_SIZE = 0;
            
    bool waitForFrame(unsigned int timeout_ms);
    CompletedRequestPtr getCompletedRequest(); 
};

};