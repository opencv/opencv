// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <map>

#include "videoio_registry.hpp"

#include "opencv2/videoio/registry.hpp"

#include <iostream>

#include "cap_librealsense.hpp"
#include "cap_dshow.hpp"

#ifdef HAVE_MFX
#include "cap_mfx_reader.hpp"
#include "cap_mfx_writer.hpp"
#endif

#include "plugin_api.hpp"

// All WinRT versions older than 8.0 should provide classes used for video support
#if defined(WINRT) && !defined(WINRT_8_0) && defined(__cplusplus_winrt)
#   include "cap_winrt_capture.hpp"
#   include "cap_winrt_bridge.hpp"
#   define WINRT_VIDEO
#endif

#if defined _M_X64 && defined _MSC_VER && !defined CV_ICC
#pragma optimize("",off)
#pragma warning(disable: 4748)
#endif

//=================================================================
// Private interface
//=================================================================

namespace cv
{
#define DECLARE_DYNAMIC_BACKEND(cap, name, mode) \
{ \
    cap, (BackendMode)(mode | MODE_DYNAMIC), 1000, name "_DYNAMIC", 0 \
}

#define DECLARE_STATIC_BACKEND(cap, name, mode, f1, f2, f3) \
{ \
    cap, (BackendMode)(mode), 1000, name, Ptr<StaticBackend>(new StaticBackend(f1, f2, f3)) \
}

/** Ordering guidelines:
- modern optimized, multi-platform libraries: ffmpeg, gstreamer, Media SDK
- platform specific universal SDK: WINRT, AVFOUNDATION, MSMF/DSHOW, V4L/V4L2
- RGB-D: OpenNI/OpenNI2, REALSENSE
- special OpenCV (file-based): "images", "mjpeg"
- special camera SDKs, including stereo: other special SDKs: FIREWIRE/1394, XIMEA/ARAVIS/GIGANETIX/PVAPI(GigE)
- other: XINE, gphoto2, etc
*/
static const struct VideoBackendInfo builtin_backends[] =
{
#ifdef HAVE_FFMPEG
    DECLARE_STATIC_BACKEND(CAP_FFMPEG, "FFMPEG", MODE_CAPTURE_BY_FILENAME | MODE_WRITER, cvCreateFileCapture_FFMPEG_proxy, 0, cvCreateVideoWriter_FFMPEG_proxy),
#elif defined(ENABLE_PLUGINS)
    DECLARE_DYNAMIC_BACKEND(CAP_FFMPEG, "FFMPEG", MODE_CAPTURE_BY_FILENAME | MODE_WRITER),
#endif

#ifdef HAVE_GSTREAMER
    DECLARE_STATIC_BACKEND(CAP_GSTREAMER, "GSTREAMER", MODE_CAPTURE_ALL | MODE_WRITER, createGStreamerCapture_file, createGStreamerCapture_cam, create_GStreamer_writer),
#elif defined(ENABLE_PLUGINS)
    DECLARE_DYNAMIC_BACKEND(CAP_GSTREAMER, "GSTREAMER", MODE_CAPTURE_ALL | MODE_WRITER),
#endif

#ifdef HAVE_MFX // Media SDK
    DECLARE_STATIC_BACKEND(CAP_INTEL_MFX, "INTEL_MFX", MODE_CAPTURE_BY_FILENAME | MODE_WRITER, create_MFX_capture, 0, create_MFX_writer),
#endif

    // Apple platform
#ifdef HAVE_AVFOUNDATION
    DECLARE_STATIC_BACKEND(CAP_AVFOUNDATION, "AVFOUNDATION", MODE_CAPTURE_ALL | MODE_WRITER, create_AVFoundation_capture_file, create_AVFoundation_capture_cam, create_AVFoundation_writer),
#endif

    // Windows
#ifdef WINRT_VIDEO
    DECLARE_STATIC_BACKEND(CAP_WINRT, "WINRT", MODE_CAPTURE_BY_INDEX, 0, create_WRT_capture, 0),
#endif
#ifdef HAVE_MSMF
    DECLARE_STATIC_BACKEND(CAP_MSMF, "MSMF", MODE_CAPTURE_ALL | MODE_WRITER, cvCreateCapture_MSMF, cvCreateCapture_MSMF, cvCreateVideoWriter_MSMF),
#endif
#ifdef HAVE_DSHOW
    DECLARE_STATIC_BACKEND(CAP_DSHOW, "DSHOW", MODE_CAPTURE_BY_INDEX, 0, create_DShow_capture, 0),
#endif

    // Linux, some Unix
#if defined HAVE_CAMV4L2
    DECLARE_STATIC_BACKEND(CAP_V4L2, "V4L2", MODE_CAPTURE_ALL, create_V4L_capture_file, create_V4L_capture_cam, 0),
#elif defined HAVE_VIDEOIO
    DECLARE_STATIC_BACKEND(CAP_V4L, "V4L_BSD", MODE_CAPTURE_ALL, create_V4L_capture_file, create_V4L_capture_cam, 0),
#endif


    // RGB-D universal
#ifdef HAVE_OPENNI2
    DECLARE_STATIC_BACKEND(CAP_OPENNI2, "OPENNI2", MODE_CAPTURE_ALL, create_OpenNI2_capture_file, create_OpenNI2_capture_cam, 0),
#endif

#ifdef HAVE_LIBREALSENSE
    DECLARE_STATIC_BACKEND(CAP_REALSENSE, "INTEL_REALSENSE", MODE_CAPTURE_BY_INDEX, 0, create_RealSense_capture, 0),
#endif

    // OpenCV file-based only
    DECLARE_STATIC_BACKEND(CAP_IMAGES, "CV_IMAGES", MODE_CAPTURE_BY_FILENAME | MODE_WRITER, create_Images_capture, 0, create_Images_writer),
    DECLARE_STATIC_BACKEND(CAP_OPENCV_MJPEG, "CV_MJPEG", MODE_CAPTURE_BY_FILENAME | MODE_WRITER, createMotionJpegCapture, 0, createMotionJpegWriter),

    // special interfaces / stereo cameras / other SDKs
#if defined(HAVE_DC1394_2)
    DECLARE_STATIC_BACKEND(CAP_FIREWIRE, "FIREWIRE", MODE_CAPTURE_BY_INDEX, 0, create_DC1394_capture, 0),
#endif
    // GigE
#ifdef HAVE_PVAPI
    DECLARE_STATIC_BACKEND(CAP_PVAPI, "PVAPI", MODE_CAPTURE_BY_INDEX, 0, create_PvAPI_capture, 0),
#endif
#ifdef HAVE_XIMEA
    DECLARE_STATIC_BACKEND(CAP_XIAPI, "XIMEA", MODE_CAPTURE_ALL, create_XIMEA_capture_file, create_XIMEA_capture_cam, 0),
#endif
#ifdef HAVE_ARAVIS_API
    DECLARE_STATIC_BACKEND(CAP_ARAVIS, "ARAVIS", MODE_CAPTURE_BY_INDEX, 0, create_Aravis_capture, 0),
#endif

#ifdef HAVE_GPHOTO2
    DECLARE_STATIC_BACKEND(CAP_GPHOTO2, "GPHOTO2", MODE_CAPTURE_ALL, createGPhoto2Capture, createGPhoto2Capture, 0),
#endif
#ifdef HAVE_XINE
    DECLARE_STATIC_BACKEND(CAP_XINE, "XINE", MODE_CAPTURE_BY_FILENAME, createXINECapture, 0, 0),
#endif
    // dropped backends: MIL, TYZX, Android
};

inline static bool sortByPriority(const VideoBackendInfo &lhs, const VideoBackendInfo &rhs)
{
    return lhs.priority > rhs.priority;
}

inline static std::vector<std::string> tokenize_string(const std::string& input, char token = ',')
{
    std::vector<std::string> result;
    std::string::size_type prev_pos = 0, pos = 0;
    while((pos = input.find(token, pos)) != std::string::npos)
    {
        result.push_back(input.substr(prev_pos, pos-prev_pos));
        prev_pos = ++pos;
    }
    result.push_back(input.substr(prev_pos));
    return result;
}

VideoBackendRegistry::VideoBackendRegistry()
{
    typedef std::vector<std::string> PriorityVec;
    using namespace cv::utils;
    const std::string backendOrder_str = getConfigurationParameterString("OPENCV_VIDEOIO_PRIORITY_LIST", NULL);
    const PriorityVec backendOrder = tokenize_string(backendOrder_str);
    if (!backendOrder.empty())
    {
        CV_LOG_INFO(NULL, "VIDEOIO: Configured priority list (OPENCV_VIDEOIO_PRIORITY_LIST): " << backendOrder_str);
    }

    const int N = sizeof(builtin_backends)/sizeof(builtin_backends[0]);
    for (int i = 0; i < N; ++i)
    {
        VideoBackendInfo be = builtin_backends[i];

        // Check if backend needs plugin
        if (be.mode & MODE_DYNAMIC)
        {
            Ptr<DynamicBackend> plugin = DynamicBackend::load(be.id, (int)be.mode);
            if (!plugin)
            {
                CV_LOG_INFO(NULL, "VIDEOIO: Disable backend: " << be.name << " (no plugin)");
                continue;
            }
            else
            {
                be.backendFactory = plugin;
            }
        }

        // initial priority (e.g. for 4 elems: 1000, 990, 980, 970)
        be.priority = 1000 - i * 10;
        CV_LOG_INFO(NULL, "VIDEOIO: Init backend priority: " << be.name << " -> " << be.priority);

        // priority from environment list (e.g. for 4 elems: 13000, 12000, 11000, 10000)
        PriorityVec::const_iterator backendPos = std::find(backendOrder.begin(), backendOrder.end(), be.name);
        if (backendPos != backendOrder.end())
        {
            const int env_priority2 = static_cast<int>(backendOrder.end() - backendPos - 1);
            be.priority = 10000 + 1000 * env_priority2;
            CV_LOG_INFO(NULL, "VIDEOIO: Update backend priority: " << be.name << " -> " << be.priority);
        }

        // priority from environment variable
        const std::string priority_var = std::string("OPENCV_VIDEOIO_PRIORITY_") + be.name;
        const size_t env_priority2 = getConfigurationParameterSizeT(priority_var.c_str(), (size_t)be.priority);
        CV_Assert(env_priority2 == (size_t)(int)env_priority2); // overflow check
        if (env_priority2 == 0)
        {
            CV_LOG_INFO(NULL, "VIDEOIO: Disable backend: " << be.name << " (user)");
            continue;
        }
        else if (be.priority != (int)env_priority2)
        {
            be.priority = (int)env_priority2;
            CV_LOG_INFO(NULL, "VIDEOIO: Update backend priority: " << be.name << " -> " << be.priority);
        }

        enabledBackends.push_back(be);
    }
    std::sort(enabledBackends.begin(), enabledBackends.end(), sortByPriority);
    CV_LOG_INFO(NULL, "VIDEOIO: Enabled backends: " << dumpBackends());
}

std::string VideoBackendRegistry::dumpBackends() const
{
    std::ostringstream os;
    for (size_t i = 0; i < enabledBackends.size(); i++)
    {
        if (i > 0) os << "; ";
        const VideoBackendInfo& info = enabledBackends[i];
        os << info.name << '(' << info.priority << ')';
    }
    return os.str();
}

VideoBackendRegistry &VideoBackendRegistry::getInstance()
{
    static VideoBackendRegistry g_instance;
    return g_instance;
}

Ptr<IBackend> VideoBackendRegistry::getBackend(VideoCaptureAPIs api) const
{
    BackendsVec result;
    for (BackendsVec::const_iterator i = enabledBackends.begin(); i != enabledBackends.end(); i++)
        if (api == i->id)
            return i->backendFactory;
    return Ptr<IBackend>(0);
}

VideoBackendRegistry::BackendsVec VideoBackendRegistry::getBackends(int capabilityMask, VideoCaptureAPIs filter) const
{
    BackendsVec result;
    for (BackendsVec::const_iterator i = enabledBackends.begin(); i != enabledBackends.end(); i++)
    {
        if (filter != CAP_ANY && filter != i->id)
            continue;
        if (i->mode & capabilityMask)
            result.push_back(*i);
    }
    return result;
}

bool VideoBackendRegistry::hasBackend(int mask, VideoCaptureAPIs api) const
{
    for (BackendsVec::const_iterator i = enabledBackends.begin(); i != enabledBackends.end(); i++)
        if (api == i->id && mask & i->mode)
            return true;
    return false;
}

} // cv::

//=================================================================
// Public interface
//=================================================================

namespace cv {  namespace videoio_registry {

cv::String getBackendName(VideoCaptureAPIs api)
{
    if (api == CAP_ANY)
        return "CAP_ANY";  // special case, not a part of backends list
    const int N = sizeof(builtin_backends)/sizeof(builtin_backends[0]);
    for (size_t i = 0; i < N; i++)
    {
        const VideoBackendInfo& backend = builtin_backends[i];
        if (backend.id == api)
            return backend.name;
    }
    return cv::format("UnknownVideoAPI(%d)", (int)api);
}

bool hasBackend(VideoCaptureAPIs api, Capability cap)
{
    int mask = 0;
    if (cap == Read || cap == ReadWrite)
        mask |= MODE_CAPTURE_ALL;
    if (cap == Write || cap == ReadWrite)
        mask |= MODE_WRITER;
    return VideoBackendRegistry::getInstance().hasBackend(mask, api);
}

inline static std::vector<VideoCaptureAPIs> toIDs(const std::vector<VideoBackendInfo> &backends)
{
    std::vector<VideoCaptureAPIs> result;
    for (size_t i = 0; i < backends.size(); i++)
        result.push_back((VideoCaptureAPIs)backends[i].id);
    return result;
}

std::vector<VideoCaptureAPIs> getBackends()
{
    return toIDs(VideoBackendRegistry::getInstance().getBackends(MODE_CAPTURE_ALL + MODE_WRITER));
}

std::vector<VideoCaptureAPIs> getCameraBackends()
{
    return toIDs(VideoBackendRegistry::getInstance().getBackends(MODE_CAPTURE_BY_INDEX));
}

std::vector<VideoCaptureAPIs> getStreamBackends()
{
    return toIDs(VideoBackendRegistry::getInstance().getBackends(MODE_CAPTURE_BY_FILENAME));
}

std::vector<VideoCaptureAPIs> getWriterBackends()
{
    return toIDs(VideoBackendRegistry::getInstance().getBackends(MODE_WRITER));
}

}} // cv::videoio_registry::
