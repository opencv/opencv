// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "videoio_registry.hpp"

#include "opencv2/videoio/registry.hpp"

#ifdef HAVE_DSHOW
    #include "cap_dshow.hpp"
#endif

#ifdef HAVE_LIBREALSENSE
    #include "cap_librealsense.hpp"
#endif

#ifdef HAVE_MFX
    #include "cap_mfx_reader.hpp"
    #include "cap_mfx_writer.hpp"
#endif

#include "plugin_api.hpp"

// All WinRT versions older than 8.0 should provide classes used for video support
#if defined(WINRT) && !defined(WINRT_8_0) && defined(__cplusplus_winrt)
    #include "cap_winrt_capture.hpp"
    #include "cap_winrt_bridge.hpp"
    #define WINRT_VIDEO
#endif

#if defined _M_X64 && defined _MSC_VER && !defined CV_ICC
    #pragma optimize("", off)
    #pragma warning(disable : 4748)
#endif

#include <iterator>

namespace cv {

namespace {

#define DECLARE_DYNAMIC_BACKEND(cap, name, mode)                                    \
    {                                                                               \
        cap, (BackendMode)(mode), 1000, name, createPluginBackendFactory(cap, name) \
    }

#define DECLARE_STATIC_BACKEND(cap, name, mode, createCaptureFile, createCaptureCamera, \
                               createWriter)                                            \
    {                                                                                   \
        cap, (BackendMode)(mode), 1000, name,                                           \
            createBackendFactory(createCaptureFile, createCaptureCamera, createWriter)  \
    }

/** Ordering guidelines:
- modern optimized, multi-platform libraries: ffmpeg, gstreamer, Media SDK
- platform specific universal SDK: WINRT, AVFOUNDATION, MSMF/DSHOW, V4L/V4L2
- RGB-D: OpenNI/OpenNI2, REALSENSE
- special OpenCV (file-based): "images", "mjpeg"
- special camera SDKs, including stereo: other special SDKs: FIREWIRE/1394, XIMEA/ARAVIS/GIGANETIX/PVAPI(GigE)
- other: XINE, gphoto2, etc
*/
const struct VideoBackendInfo builtin_backends[] = {
#ifdef HAVE_FFMPEG
    DECLARE_STATIC_BACKEND(CAP_FFMPEG, "FFMPEG", MODE_CAPTURE_BY_FILENAME | MODE_WRITER,
                           cvCreateFileCapture_FFMPEG_proxy, 0, cvCreateVideoWriter_FFMPEG_proxy),
#elif defined(ENABLE_PLUGINS) || defined(HAVE_FFMPEG_WRAPPER)
    DECLARE_DYNAMIC_BACKEND(CAP_FFMPEG, "FFMPEG", MODE_CAPTURE_BY_FILENAME | MODE_WRITER),
#endif

#ifdef HAVE_GSTREAMER
    DECLARE_STATIC_BACKEND(CAP_GSTREAMER, "GSTREAMER", MODE_CAPTURE_ALL | MODE_WRITER,
                           createGStreamerCapture_file, createGStreamerCapture_cam,
                           create_GStreamer_writer),
#elif defined(ENABLE_PLUGINS)
    DECLARE_DYNAMIC_BACKEND(CAP_GSTREAMER, "GSTREAMER", MODE_CAPTURE_ALL | MODE_WRITER),
#endif

#ifdef HAVE_MFX // Media SDK
    DECLARE_STATIC_BACKEND(CAP_INTEL_MFX, "INTEL_MFX", MODE_CAPTURE_BY_FILENAME | MODE_WRITER,
                           create_MFX_capture, 0, create_MFX_writer),
#elif defined(ENABLE_PLUGINS)
    DECLARE_DYNAMIC_BACKEND(CAP_INTEL_MFX, "INTEL_MFX", MODE_CAPTURE_BY_FILENAME | MODE_WRITER),
#endif

// Apple platform
#ifdef HAVE_AVFOUNDATION
    DECLARE_STATIC_BACKEND(CAP_AVFOUNDATION, "AVFOUNDATION", MODE_CAPTURE_ALL | MODE_WRITER,
                           create_AVFoundation_capture_file, create_AVFoundation_capture_cam,
                           create_AVFoundation_writer),
#endif

// Windows
#ifdef WINRT_VIDEO
    DECLARE_STATIC_BACKEND(CAP_WINRT, "WINRT", MODE_CAPTURE_BY_INDEX, 0, create_WRT_capture, 0),
#endif
#ifdef HAVE_MSMF
    DECLARE_STATIC_BACKEND(CAP_MSMF, "MSMF", MODE_CAPTURE_ALL | MODE_WRITER, cvCreateCapture_MSMF,
                           cvCreateCapture_MSMF, cvCreateVideoWriter_MSMF),
#endif
#ifdef HAVE_DSHOW
    DECLARE_STATIC_BACKEND(CAP_DSHOW, "DSHOW", MODE_CAPTURE_BY_INDEX, 0, create_DShow_capture, 0),
#endif

// Linux, some Unix
#if defined HAVE_CAMV4L2
    DECLARE_STATIC_BACKEND(CAP_V4L2, "V4L2", MODE_CAPTURE_ALL, create_V4L_capture_file,
                           create_V4L_capture_cam, 0),
#elif defined HAVE_VIDEOIO
    DECLARE_STATIC_BACKEND(CAP_V4L, "V4L_BSD", MODE_CAPTURE_ALL, create_V4L_capture_file,
                           create_V4L_capture_cam, 0),
#endif


// RGB-D universal
#ifdef HAVE_OPENNI2
    DECLARE_STATIC_BACKEND(CAP_OPENNI2, "OPENNI2", MODE_CAPTURE_ALL, create_OpenNI2_capture_file,
                           create_OpenNI2_capture_cam, 0),
#endif

#ifdef HAVE_LIBREALSENSE
    DECLARE_STATIC_BACKEND(CAP_REALSENSE, "INTEL_REALSENSE", MODE_CAPTURE_BY_INDEX, 0,
                           create_RealSense_capture, 0),
#endif

    // OpenCV file-based only
    DECLARE_STATIC_BACKEND(CAP_IMAGES, "CV_IMAGES", MODE_CAPTURE_BY_FILENAME | MODE_WRITER,
                           create_Images_capture, 0, create_Images_writer),
    DECLARE_STATIC_BACKEND(CAP_OPENCV_MJPEG, "CV_MJPEG", MODE_CAPTURE_BY_FILENAME | MODE_WRITER,
                           createMotionJpegCapture, 0, createMotionJpegWriter),

// special interfaces / stereo cameras / other SDKs
#if defined(HAVE_DC1394_2)
    DECLARE_STATIC_BACKEND(CAP_FIREWIRE, "FIREWIRE", MODE_CAPTURE_BY_INDEX, 0,
                           create_DC1394_capture, 0),
#endif
// GigE
#ifdef HAVE_PVAPI
    DECLARE_STATIC_BACKEND(CAP_PVAPI, "PVAPI", MODE_CAPTURE_BY_INDEX, 0, create_PvAPI_capture, 0),
#endif
#ifdef HAVE_XIMEA
    DECLARE_STATIC_BACKEND(CAP_XIAPI, "XIMEA", MODE_CAPTURE_ALL, create_XIMEA_capture_file,
                           create_XIMEA_capture_cam, 0),
#endif
#ifdef HAVE_ARAVIS_API
    DECLARE_STATIC_BACKEND(CAP_ARAVIS, "ARAVIS", MODE_CAPTURE_BY_INDEX, 0, create_Aravis_capture, 0),
#endif

#ifdef HAVE_GPHOTO2
    DECLARE_STATIC_BACKEND(CAP_GPHOTO2, "GPHOTO2", MODE_CAPTURE_ALL, createGPhoto2Capture,
                           createGPhoto2Capture, 0),
#endif
#ifdef HAVE_XINE
    DECLARE_STATIC_BACKEND(CAP_XINE, "XINE", MODE_CAPTURE_BY_FILENAME, createXINECapture, 0, 0),
#endif
#ifdef HAVE_ANDROID_MEDIANDK
    DECLARE_STATIC_BACKEND(CAP_ANDROID, "ANDROID_MEDIANDK", MODE_CAPTURE_BY_FILENAME,
                           createAndroidCapture_file, 0, 0),
#endif
    // dropped backends: MIL, TYZX
};

bool sortByPriority(const VideoBackendInfo& lhs, const VideoBackendInfo& rhs)
{
    return lhs.priority > rhs.priority;
}

template<class T, std::size_t N>
constexpr std::size_t getSize(const T (&/*array */)[N]) CV_NOEXCEPT
{
    return N;
}

/** @brief Manages list of enabled backends
 */
class VideoBackendRegistry
{
public:
    std::string dumpBackends() const
    {
        std::ostringstream os;
        for (size_t i = 0; i < enabledBackends.size(); i++)
        {
            if (i > 0)
            {
                os << "; ";
            }
            const VideoBackendInfo& info = enabledBackends[i];
            os << info.name << '(' << info.priority << ')';
        }
        return os.str();
    }

    static VideoBackendRegistry& getInstance()
    {
        static VideoBackendRegistry g_instance;
        return g_instance;
    }

    inline std::vector<VideoBackendInfo> getEnabledBackends() const { return enabledBackends; }

    inline std::vector<VideoBackendInfo> getAvailableBackends_CaptureByIndex() const
    {
        return filterBackends(BackendMode::MODE_CAPTURE_BY_INDEX);
    }

    inline std::vector<VideoBackendInfo> getAvailableBackends_CaptureByFilename() const
    {
        return filterBackends(BackendMode::MODE_CAPTURE_BY_FILENAME);
    }

    inline std::vector<VideoBackendInfo> getAvailableBackends_Writer() const
    {
        return filterBackends(BackendMode::MODE_WRITER);
    }

protected:
    std::vector<VideoBackendInfo> enabledBackends;

    VideoBackendRegistry()
    {
        enabledBackends.assign(std::begin(builtin_backends), std::end(builtin_backends));
        const size_t builtin_backends_size = getSize(builtin_backends);
        for (size_t i = 0; i < builtin_backends_size; ++i)
        {
            VideoBackendInfo& info = enabledBackends[i];
            info.priority = static_cast<int>(1000 - i * 10);
        }
        CV_LOG_DEBUG(nullptr, "VIDEOIO: Builtin backends(" << builtin_backends_size
                                                           << "): " << dumpBackends());
        if (readPrioritySettings())
        {
            CV_LOG_INFO(nullptr, "VIDEOIO: Updated backends priorities: " << dumpBackends());
        }
        size_t enabled = 0;
        for (size_t i = 0; i < builtin_backends_size; ++i)
        {
            VideoBackendInfo& info = enabledBackends[enabled];
            if (enabled != i)
            {
                info = enabledBackends[i];
            }
            size_t param_priority = utils::getConfigurationParameterSizeT(
                cv::format("OPENCV_VIDEOIO_PRIORITY_%s", info.name).c_str(), (size_t)info.priority);
            CV_Assert(param_priority == (size_t)(int)param_priority); // overflow check
            if (param_priority > 0)
            {
                info.priority = (int)param_priority;
                enabled++;
            }
            else
            {
                CV_LOG_INFO(nullptr, "VIDEOIO: Disable backend: " << info.name);
            }
        }
        enabledBackends.resize(enabled);
        CV_LOG_DEBUG(nullptr, "VIDEOIO: Available backends(" << enabled << "): " << dumpBackends());
        std::sort(enabledBackends.begin(), enabledBackends.end(), sortByPriority);
        CV_LOG_INFO(nullptr, "VIDEOIO: Enabled backends("
                                 << enabled << ", sorted by priority): " << dumpBackends());
    }

    static std::vector<std::string> tokenizeString(const std::string& input, char token)
    {
        std::vector<std::string> result;
        std::string::size_type prev_pos = 0, pos = 0;
        while ((pos = input.find(token, pos)) != std::string::npos)
        {
            result.push_back(input.substr(prev_pos, pos - prev_pos));
            prev_pos = ++pos;
        }
        result.push_back(input.substr(prev_pos));
        return result;
    }

    bool readPrioritySettings()
    {
        bool hasChanges = false;
        cv::String prioritized_backends =
            utils::getConfigurationParameterString("OPENCV_VIDEOIO_PRIORITY_LIST", nullptr);
        if (prioritized_backends.empty())
        {
            return hasChanges;
        }
        CV_LOG_INFO(nullptr, "VIDEOIO: Configured priority list (OPENCV_VIDEOIO_PRIORITY_LIST): "
                                 << prioritized_backends);
        const std::vector<std::string> names = tokenizeString(prioritized_backends, ',');
        for (size_t i = 0; i < names.size(); i++)
        {
            const std::string& name = names[i];
            bool found = false;
            for (auto& info : enabledBackends)
            {
                if (name == info.name)
                {
                    info.priority = static_cast<int>(100000 + (names.size() - i) * 1000);
                    CV_LOG_DEBUG(nullptr, "VIDEOIO: New backend priority: '" << name << "' => "
                                                                             << info.priority);
                    found = true;
                    hasChanges = true;
                    break;
                }
            }
            if (!found)
            {
                CV_LOG_WARNING(nullptr, "VIDEOIO: Can't prioritize unknown/unavailable backend: '"
                                            << name << "'");
            }
        }
        return hasChanges;
    }

private:
    inline std::vector<VideoBackendInfo> filterBackends(BackendMode mode) const
    {
        std::vector<VideoBackendInfo> result;
        result.reserve(enabledBackends.size());
        std::copy_if(enabledBackends.cbegin(), enabledBackends.cend(), std::back_inserter(result),
                     [mode](const VideoBackendInfo& info) { return info.mode & mode; });
        return result;
    }
};

std::vector<VideoCaptureAPIs> getAvailableCaptureApi(const std::vector<VideoBackendInfo>& backends)
{
    std::vector<VideoCaptureAPIs> result;
    result.reserve(backends.size());
    for (const auto& backend : backends)
    {
        result.push_back(static_cast<VideoCaptureAPIs>(backend.id));
    }
    return result;
}

std::vector<VideoBackendInfo> getEnabledBackends()
{
    return VideoBackendRegistry::getInstance().getEnabledBackends();
}

} // namespace

namespace videoio_registry {

std::vector<VideoBackendInfo> getAvailableBackends_CaptureByIndex()
{
    return VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByIndex();
}

std::vector<VideoBackendInfo> getAvailableBackends_CaptureByFilename()
{
    return VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByFilename();
}

std::vector<VideoBackendInfo> getAvailableBackends_Writer()
{
    return VideoBackendRegistry::getInstance().getAvailableBackends_Writer();
}

cv::String getBackendName(VideoCaptureAPIs api)
{
    if (api == CAP_ANY)
    {
        return "CAP_ANY"; // special case, not a part of backends list
    }
    for (const auto& backend : builtin_backends)
    {
        if (backend.id == api)
        {
            return backend.name;
        }
    }
    return cv::format("UnknownVideoAPI(%d)", (int)api);
}

std::vector<VideoCaptureAPIs> getBackends()
{
    return getAvailableCaptureApi(getEnabledBackends());
}

std::vector<VideoCaptureAPIs> getCameraBackends()
{
    return getAvailableCaptureApi(getAvailableBackends_CaptureByIndex());
}

std::vector<VideoCaptureAPIs> getStreamBackends()
{
    return getAvailableCaptureApi(getAvailableBackends_CaptureByFilename());
}

std::vector<VideoCaptureAPIs> getWriterBackends()
{
    return getAvailableCaptureApi(getAvailableBackends_Writer());
}

bool hasBackend(VideoCaptureAPIs api)
{
    const auto& enabled_backends = getEnabledBackends();
    for (const auto& backend : enabled_backends)
    {
        if (api == backend.id)
        {
            CV_Assert(!backend.backendFactory.empty());
            return !backend.backendFactory->getBackend().empty();
        }
    }
    return false;
}

} // namespace videoio_registry

} // namespace cv
