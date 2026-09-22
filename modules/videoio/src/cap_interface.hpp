// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef CAP_INTERFACE_HPP
#define CAP_INTERFACE_HPP

#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/videoio/utils.private.hpp"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

//===================================================

// Modern classes

namespace cv
{
namespace
{
template <class T>
inline T castParameterTo(int paramValue)
{
    return static_cast<T>(paramValue);
}

template <>
inline bool castParameterTo(int paramValue)
{
    return paramValue != 0;
}
}

class VideoParameters
{
public:
    struct VideoParameter {
        VideoParameter() = default;

        VideoParameter(int key_, int value_) : key(key_), value(value_) {}

        int key{-1};
        int value{-1};
        mutable bool isConsumed{false};
    };

    VideoParameters() = default;

    explicit VideoParameters(const std::vector<int>& params)
    {
        const auto count = params.size();
        if (count % 2 != 0)
        {
            CV_Error_(Error::StsVecLengthErr,
                      ("Vector of VideoWriter parameters should have even length"));
        }
        params_.reserve(count / 2);
        for (std::size_t i = 0; i < count; i += 2)
        {
            add(params[i], params[i + 1]);
        }
    }

    VideoParameters(int* params, unsigned n_params)
    {
        params_.reserve(n_params);
        for (unsigned i = 0; i < n_params; ++i)
        {
            add(params[2*i], params[2*i + 1]);
        }
    }

    void add(int key, int value)
    {
        params_.emplace_back(key, value);
    }

    bool has(int key) const
    {
        auto it = std::find_if(params_.begin(), params_.end(),
            [key](const VideoParameter &param)
            {
                return param.key == key;
            }
        );
        return it != params_.end();
    }

    template <class ValueType>
    ValueType get(int key) const
    {
        auto it = std::find_if(params_.begin(), params_.end(),
            [key](const VideoParameter &param)
            {
                return param.key == key;
            }
        );
        if (it != params_.end())
        {
            it->isConsumed = true;
            return castParameterTo<ValueType>(it->value);
        }
        else
        {
            CV_Error_(Error::StsBadArg, ("Missing value for parameter: [%d]", key));
        }
    }

    template <class ValueType>
    ValueType get(int key, ValueType defaultValue) const
    {
        auto it = std::find_if(params_.begin(), params_.end(),
            [key](const VideoParameter &param)
            {
                return param.key == key;
            }
        );
        if (it != params_.end())
        {
            it->isConsumed = true;
            return castParameterTo<ValueType>(it->value);
        }
        else
        {
            return defaultValue;
        }
    }

    std::vector<int> getUnused() const
    {
        std::vector<int> unusedParams;
        for (const auto &param : params_)
        {
            if (!param.isConsumed)
            {
                unusedParams.push_back(param.key);
            }
        }
        return unusedParams;
    }

    std::vector<int> getIntVector() const
    {
        std::vector<int> vint_params;
        for (const auto& param : params_)
        {
            vint_params.push_back(param.key);
            vint_params.push_back(param.value);
        }
        return vint_params;
    }

    bool empty() const
    {
        return params_.empty();
    }

    bool warnUnusedParameters() const
    {
        bool found = false;
        for (const auto &param : params_)
        {
            if (!param.isConsumed)
            {
                found = true;
                CV_LOG_INFO(NULL, "VIDEOIO: unused parameter: [" << param.key << "]=" <<
                    cv::format("%lld / 0x%016llx", (long long)param.value, (long long)param.value));
            }
        }
        return found;
    }


private:
    std::vector<VideoParameter> params_;
};

class VideoWriterParameters : public VideoParameters
{
public:
    using VideoParameters::VideoParameters;  // reuse constructors
};

class VideoCaptureParameters : public VideoParameters
{
public:
    using VideoParameters::VideoParameters;  // reuse constructors
};

class IVideoCapture
{
public:
    virtual ~IVideoCapture() {}
    virtual double getProperty(int) const { return CAP_PROP_UNKNOWN; }
    virtual bool setProperty(int, double) { return false; }
    virtual bool grabFrame() = 0;
    virtual bool retrieveFrame(int, OutputArray) = 0;
    virtual bool isOpened() const = 0;
    virtual int getCaptureDomain() { return CAP_ANY; } // Return the type of the capture object: CAP_DSHOW, etc...
};

// Decorator adding read-ahead prefetching to any IVideoCapture backend
class PrefetchCapture CV_FINAL : public IVideoCapture
{
public:
    PrefetchCapture(const Ptr<IVideoCapture>& backend, size_t depth)
        : inner(backend), prefetchDepth(depth), prefetchDrop(false), prefetchStop(false) {}

    ~PrefetchCapture() CV_OVERRIDE { stopWorker(); }

    // Stops the worker but keeps serving queued frames until the queue is empty,
    // then reads straight from the backend. Unwrapping here would lose those frames.
    void disablePrefetch()
    {
        stopWorker();
        prefetchDepth = 0;
    }

    // Prefetching is active while the worker runs or frames are still queued.
    bool isPrefetching() const
    {
        std::unique_lock<std::mutex> lock(prefetchMutex);
        return prefetchDepth != 0 || !prefetchQueue.empty();
    }

    double getProperty(int propId) const CV_OVERRIDE
    {
        if (propId == cv::CAP_PROP_PREFETCH_FRAMES)
            return static_cast<double>(prefetchDepth);
        if (propId == cv::CAP_PROP_PREFETCH_DROP)
            return static_cast<double>(prefetchDrop);

        const int idx = framePropIndex(propId);
        if (idx >= 0 && currentIsQueued && !current.mat.empty())
            return current.meta[idx];

        std::unique_lock<std::mutex> lock(backendMutex);
        return inner->getProperty(propId);
    }

    bool setProperty(int propId, double value) CV_OVERRIDE
    {
        // Prefetch properties do not move the stream, so queued frames stay valid.
        // Anything else goes to the backend and may move it, so the queue is dropped.
        const bool prefetchOnly = (propId == cv::CAP_PROP_PREFETCH_FRAMES ||
                                   propId == cv::CAP_PROP_PREFETCH_DROP);
        Pause pause(*this, !prefetchOnly);
        switch (propId)
        {
            case cv::CAP_PROP_PREFETCH_FRAMES:
            {
                const int depth = cvRound(value);
                if (depth < 0)
                    return false;
                prefetchDepth = static_cast<size_t>(depth);
                return true;
            }

            case cv::CAP_PROP_PREFETCH_DROP:
                prefetchDrop = (value != 0);
                return true;

            default:
                return inner->setProperty(propId, value);
        }
    }

    bool grabFrame() CV_OVERRIDE
    {
        if (ensureWorker())
        {
            std::unique_lock<std::mutex> lock(prefetchMutex);
            prefetchCond.wait(lock, [this]{ return prefetchStop || prefetchEof || !prefetchQueue.empty(); });
            return takeQueuedLocked();
        }

        // Prefetching is off, but a stopped worker may have left frames behind.
        {
            std::unique_lock<std::mutex> lock(prefetchMutex);
            if (!prefetchQueue.empty())
                return takeQueuedLocked();
        }

        currentIsQueued = false;
        std::unique_lock<std::mutex> lock(backendMutex);
        return inner->grabFrame();
    }

    bool retrieveFrame(int channel, OutputArray image) CV_OVERRIDE
    {
        if (!currentIsQueued)
        {
            std::unique_lock<std::mutex> lock(backendMutex);
            return inner->retrieveFrame(channel, image);
        }

        if (channel != 0 || current.mat.empty())
        {
            image.release();
            return false;
        }

        image.assign(current.mat);
        return true;
    }

    bool isOpened() const CV_OVERRIDE { return inner->isOpened(); }
    int getCaptureDomain() CV_OVERRIDE { return inner->getCaptureDomain(); }

private:
    // per-frame properties: travel with the queued frame, since the worker is already ahead of the consumer
    static constexpr int framePropIds[] = {
        cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, cv::CAP_PROP_POS_AVI_RATIO,
        cv::CAP_PROP_PTS, cv::CAP_PROP_FRAME_TYPE, cv::CAP_PROP_LRF_HAS_KEY_FRAME
    };
    static constexpr size_t numFrameProps = sizeof(framePropIds) / sizeof(framePropIds[0]);

    struct Frame
    {
        Mat mat;
        double meta[numFrameProps] = {};
    };

    static int framePropIndex(int propId)
    {
        for (size_t i = 0; i < numFrameProps; i++)
            if (framePropIds[i] == propId)
                return static_cast<int>(i);
        return -1;
    }

    struct Pause
    {
        PrefetchCapture& cap;
        Pause(PrefetchCapture& c, bool discardQueued) : cap(c)
        {
            cap.stopWorker();
            if (discardQueued)
                cap.discardQueue();
        }
        ~Pause() { try { cap.startWorker(); } catch (...) {} }
    };

    bool ensureWorker()
    {
        if (prefetchDepth != 0 && !prefetchWorker.joinable())
            startWorker();
        return prefetchDepth != 0 && prefetchWorker.joinable();
    }

    void startWorker()
    {
        if (prefetchDepth == 0 || prefetchEof || !inner->isOpened())
            return;
        prefetchWorker = std::thread(&PrefetchCapture::loop, this);
    }

    void stopWorker()
    {
        if (!prefetchWorker.joinable())
            return;
        {
            std::unique_lock<std::mutex> lock(prefetchMutex);
            prefetchStop = true;
            prefetchCond.notify_all();
        }
        prefetchWorker.join();
        prefetchStop = false;
    }

    void discardQueue()
    {
        prefetchQueue.clear();
        current = Frame();
        currentIsQueued = false;
        prefetchEof = false;
    }

    // Take the oldest queued frame. Caller holds prefetchMutex.
    bool takeQueuedLocked()
    {
        while (!prefetchQueue.empty() && prefetchQueue.front().mat.empty())
        {
            prefetchQueue.pop_front();
            prefetchEof = true;
        }
        if (prefetchQueue.empty())
        {
            current = Frame();
            currentIsQueued = false;
            return false;
        }
        current = std::move(prefetchQueue.front());
        prefetchQueue.pop_front();
        currentIsQueued = true;
        prefetchCond.notify_all();
        return true;
    }

    void loop()
    {
        for (;;)
        {
            Frame frame;
            bool ok = false;
            try
            {
                std::unique_lock<std::mutex> lock(backendMutex);
                if (inner->grabFrame())
                    ok = inner->retrieveFrame(0, frame.mat);
                if (ok)
                {
                    for (size_t i = 0; i < numFrameProps; i++)
                        frame.meta[i] = inner->getProperty(framePropIds[i]);
                }
            }
            catch (...)
            {
                ok = false;
            }

            std::unique_lock<std::mutex> lock(prefetchMutex);
            if (prefetchDrop)
            {
                while (!prefetchQueue.empty() && prefetchQueue.size() >= prefetchDepth)
                    prefetchQueue.pop_front();
            }
            else
            {
                prefetchCond.wait(lock, [this]{ return prefetchStop || prefetchQueue.size() < prefetchDepth; });
            }
            if (prefetchStop)
            {
                // A frame was decoded before the stop was seen. Keep it even if that
                // goes one over the depth, or every pause would lose a frame.
                if (ok)
                    prefetchQueue.push_back(std::move(frame));
                return;
            }

            prefetchQueue.push_back(ok ? std::move(frame) : Frame());
            prefetchCond.notify_all();
            if (!ok)
                return;
        }
    }

    Ptr<IVideoCapture> inner;
    mutable std::mutex backendMutex;
    mutable std::mutex prefetchMutex;
    std::condition_variable prefetchCond;
    std::deque<Frame> prefetchQueue;
    Frame current; // frame taken by the last grabFrame(); only ever touched by the consumer thread
    bool currentIsQueued = false; // did the last grabFrame() come from the queue?
    bool prefetchEof = false; // the worker hit the end of the stream; cleared by anything that moves the stream
    std::thread prefetchWorker;
    size_t prefetchDepth;
    bool prefetchDrop;
    bool prefetchStop;
};

class IVideoWriter
{
public:
    virtual ~IVideoWriter() {}
    virtual double getProperty(int) const { return CAP_PROP_UNKNOWN; }
    virtual bool setProperty(int, double) { return false; }
    virtual bool isOpened() const = 0;
    virtual bool write(InputArray) = 0;
    virtual int getCaptureDomain() const { return cv::CAP_ANY; } // Return the type of the capture object: CAP_FFMPEG, etc...
};

namespace internal {
class VideoCapturePrivateAccessor
{
public:
    static
    IVideoCapture* getIVideoCapture(const VideoCapture& cap) { return cap.icap.get(); }
};
} // namespace


// Advanced base class for VideoCapture backends providing some extra functionality
class VideoCaptureBase : public IVideoCapture
{
public:
    VideoCaptureBase() : autorotate(true) {}
    double getProperty(int propId) const CV_OVERRIDE
    {
        switch(propId)
        {
            case cv::CAP_PROP_ORIENTATION_AUTO:
                return static_cast<double>(autorotate);

            case cv::CAP_PROP_FRAME_WIDTH:
                return shouldSwapWidthHeight() ? getProperty_(cv::CAP_PROP_FRAME_HEIGHT) : getProperty_(cv::CAP_PROP_FRAME_WIDTH);

            case cv::CAP_PROP_FRAME_HEIGHT:
                return shouldSwapWidthHeight() ? getProperty_(cv::CAP_PROP_FRAME_WIDTH) : getProperty_(cv::CAP_PROP_FRAME_HEIGHT);

            default:
                return getProperty_(propId);
        }
    }
    bool setProperty(int propId, double value) CV_OVERRIDE
    {
        switch(propId)
        {
            case cv::CAP_PROP_ORIENTATION_AUTO:
                autorotate = (value != 0);
                return true;

            default:
                return setProperty_(propId, value);
        }
    }
    bool retrieveFrame(int channel, OutputArray image) CV_OVERRIDE
    {
        const bool res = retrieveFrame_(channel, image);
        if (res)
            applyMetadataRotation(image);
        return res;
    }

protected:
    virtual double getProperty_(int) const = 0;
    virtual bool setProperty_(int, double) = 0;
    virtual bool retrieveFrame_(int, OutputArray) = 0;

protected:
    bool shouldSwapWidthHeight() const
    {
        if (!autorotate)
            return false;
        int rotation = static_cast<int>(getProperty(cv::CAP_PROP_ORIENTATION_META));
        return std::abs(rotation % 180) == 90;
    }
    void applyMetadataRotation(OutputArray mat) const
    {
        bool rotation_auto = 0 != getProperty(CAP_PROP_ORIENTATION_AUTO);
        int rotation_angle = static_cast<int>(getProperty(CAP_PROP_ORIENTATION_META));
        if(!rotation_auto || rotation_angle%360 == 0)
        {
            return;
        }
        cv::RotateFlags flag;
        if(rotation_angle == 90 || rotation_angle == -270) { // Rotate clockwise 90 degrees
            flag = cv::ROTATE_90_CLOCKWISE;
        } else if(rotation_angle == 270 || rotation_angle == -90) { // Rotate clockwise 270 degrees
            flag = cv::ROTATE_90_COUNTERCLOCKWISE;
        } else if(rotation_angle == 180 || rotation_angle == -180) { // Rotate clockwise 180 degrees
            flag = cv::ROTATE_180;
        } else { // Unsupported rotation
            return;
        }
        cv::rotate(mat, mat, flag);
    }

protected:
    bool autorotate;
};


//==================================================================================================

Ptr<IVideoCapture> cvCreateFileCapture_FFMPEG_proxy(const std::string &filename, const VideoCaptureParameters& params);
Ptr<IVideoCapture> cvCreateCameraCapture_FFMPEG_proxy(int index, const VideoCaptureParameters& params);
Ptr<IVideoCapture> cvCreateStreamCapture_FFMPEG_proxy(const Ptr<IStreamReader>& stream, const VideoCaptureParameters& params);
Ptr<IVideoWriter> cvCreateVideoWriter_FFMPEG_proxy(const std::string& filename, int fourcc,
                                                   double fps, const Size& frameSize,
                                                   const VideoWriterParameters& params);

Ptr<IVideoCapture> createGStreamerCapture_file(const std::string& filename, const cv::VideoCaptureParameters& params);
Ptr<IVideoCapture> createGStreamerCapture_cam(int index, const cv::VideoCaptureParameters& params);
Ptr<IVideoWriter> create_GStreamer_writer(const std::string& filename, int fourcc,
                                          double fps, const Size& frameSize,
                                          const VideoWriterParameters& params);

Ptr<IVideoCapture> create_MFX_capture(const std::string &filename);
Ptr<IVideoWriter> create_MFX_writer(const std::string& filename, int _fourcc,
                                    double fps, const Size& frameSize,
                                    const VideoWriterParameters& params);

Ptr<IVideoCapture> create_AVFoundation_capture_file(const std::string &filename);
Ptr<IVideoCapture> create_AVFoundation_capture_cam(int index);
Ptr<IVideoWriter> create_AVFoundation_writer(const std::string& filename, int fourcc,
                                             double fps, const Size& frameSize,
                                             const VideoWriterParameters& params);

Ptr<IVideoCapture> create_WRT_capture(int device);

Ptr<IVideoCapture> cvCreateCapture_MSMF(int index, const VideoCaptureParameters& params);
Ptr<IVideoCapture> cvCreateCapture_MSMF(const std::string& filename, const VideoCaptureParameters& params);
Ptr<IVideoCapture> cvCreateCapture_MSMF(const Ptr<IStreamReader>& stream, const VideoCaptureParameters& params);
Ptr<IVideoWriter> cvCreateVideoWriter_MSMF(const std::string& filename, int fourcc,
                                           double fps, const Size& frameSize,
                                           const VideoWriterParameters& params);

Ptr<IVideoCapture> create_DShow_capture(int index, const VideoCaptureParameters& params);

Ptr<IVideoCapture> create_V4L_capture_cam(int index);
Ptr<IVideoCapture> create_V4L_capture_file(const std::string &filename);

Ptr<IVideoCapture> create_OpenNI2_capture_cam( int index );
Ptr<IVideoCapture> create_OpenNI2_capture_file( const std::string &filename );

Ptr<IVideoCapture> create_Images_capture(const std::string& filename, const VideoCaptureParameters& params);
Ptr<IVideoWriter> create_Images_writer(const std::string& filename, int fourcc,
                                       double fps, const Size& frameSize,
                                       const VideoWriterParameters& params);

Ptr<IVideoCapture> create_DC1394_capture(int index);

Ptr<IVideoCapture> create_RealSense_capture(int index);

Ptr<IVideoCapture> create_PvAPI_capture( int index );

Ptr<IVideoCapture> create_XIMEA_capture_cam( int index );
Ptr<IVideoCapture> create_XIMEA_capture_file( const std::string &serialNumber );

Ptr<IVideoCapture> create_ueye_camera(int camera);

Ptr<IVideoCapture> create_Aravis_capture( int index );

Ptr<IVideoCapture> createMotionJpegCapture(const std::string& filename);
Ptr<IVideoWriter> createMotionJpegWriter(const std::string& filename, int fourcc,
                                         double fps, const Size& frameSize,
                                         const VideoWriterParameters& params);

Ptr<IVideoCapture> createGPhoto2Capture(int index);
Ptr<IVideoCapture> createGPhoto2Capture(const std::string& deviceName);

Ptr<IVideoCapture> createXINECapture(const std::string &filename);

Ptr<IVideoCapture> createAndroidCapture_cam(int index, const VideoCaptureParameters& params);
Ptr<IVideoCapture> createAndroidCapture_file(const std::string &filename, const VideoCaptureParameters& params);
Ptr<IVideoWriter> createAndroidVideoWriter(const std::string& filename, int fourcc,
                                           double fps, const Size& frameSize,
                                           const VideoWriterParameters& params);

Ptr<IVideoCapture> create_obsensor_capture(int index, const cv::VideoCaptureParameters& params);

bool VideoCapture_V4L_waitAny(
        const std::vector<VideoCapture>& streams,
        CV_OUT std::vector<int>& ready,
        int64 timeoutNs);

static inline
std::ostream& operator<<(std::ostream& out, const VideoAccelerationType& va_type)
{
    switch (va_type)
    {
    case VIDEO_ACCELERATION_NONE: out << "NONE"; return out;
    case VIDEO_ACCELERATION_ANY: out << "ANY"; return out;
    case VIDEO_ACCELERATION_D3D11: out << "D3D11"; return out;
    case VIDEO_ACCELERATION_VAAPI: out << "VAAPI"; return out;
    case VIDEO_ACCELERATION_MFX: out << "MFX"; return out;
    case VIDEO_ACCELERATION_DRM: out << "DRM"; return out;
    }
    out << cv::format("UNKNOWN(0x%ux)", static_cast<unsigned int>(va_type));
    return out;
}

} // cv::

#endif // CAP_INTERFACE_HPP
