// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "videoio_registry.hpp"

#include "opencv2/videoio/registry.hpp"

#include "cap_intelperc.hpp"
#include "cap_dshow.hpp"

#ifdef HAVE_MFX
#include "cap_mfx_reader.hpp"
#include "cap_mfx_writer.hpp"
#endif

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

using namespace cv;

namespace cv
{

static bool param_VIDEOIO_DEBUG = utils::getConfigurationParameterBool("OPENCV_VIDEOIO_DEBUG", false);
static bool param_VIDEOCAPTURE_DEBUG = utils::getConfigurationParameterBool("OPENCV_VIDEOCAPTURE_DEBUG", false);
static bool param_VIDEOWRITER_DEBUG = utils::getConfigurationParameterBool("OPENCV_VIDEOWRITER_DEBUG", false);

namespace {

#define DECLARE_BACKEND(cap, name, mode) { cap, (BackendMode)(mode), 1000, name }

/** Ordering guidelines:
- modern optimized, multi-platform libraries: ffmpeg, gstreamer, Media SDK
- platform specific universal SDK: WINRT, QTKIT/AVFOUNDATION, MSMF/VFW/DSHOW, V4L/V4L2
- RGB-D: OpenNI/OpenNI2, INTELPERC/REALSENSE
- special OpenCV (file-based): "images", "mjpeg"
- special camera SDKs, including stereo: other special SDKs: FIREWIRE/1394, XIMEA/ARAVIS/GIGANETIX/PVAPI(GigE), UNICAP
- other: XINE, gphoto2, etc
*/
static const struct VideoBackendInfo builtin_backends[] =
{
#ifdef HAVE_FFMPEG
    DECLARE_BACKEND(CAP_FFMPEG, "FFMPEG", MODE_CAPTURE_BY_FILENAME | MODE_WRITER),
#endif
#ifdef HAVE_GSTREAMER
    DECLARE_BACKEND(CAP_GSTREAMER, "GSTREAMER", MODE_CAPTURE_ALL | MODE_WRITER),
#endif
#ifdef HAVE_MFX // Media SDK
    DECLARE_BACKEND(CAP_INTEL_MFX, "INTEL_MFX", MODE_CAPTURE_BY_FILENAME | MODE_WRITER),
#endif


    // Apple platform
#if defined(HAVE_QUICKTIME) || defined(HAVE_QTKIT)
    DECLARE_BACKEND(CAP_QT, "QUICKTIME", MODE_CAPTURE_ALL | MODE_WRITER),
#endif
#ifdef HAVE_AVFOUNDATION
    DECLARE_BACKEND(CAP_AVFOUNDATION, "AVFOUNDATION", MODE_CAPTURE_ALL | MODE_WRITER),
#endif

    // Windows
#ifdef WINRT_VIDEO
    DECLARE_BACKEND(CAP_WINRT, "WINRT", MODE_CAPTURE_BY_INDEX),
#endif
#ifdef HAVE_MSMF
    DECLARE_BACKEND(CAP_MSMF, "MSMF", MODE_CAPTURE_ALL | MODE_WRITER),
#endif
#ifdef HAVE_DSHOW
    DECLARE_BACKEND(CAP_DSHOW, "DSHOW", MODE_CAPTURE_BY_INDEX),
#endif
#ifdef HAVE_VFW
    DECLARE_BACKEND(CAP_VFW, "VFW", MODE_CAPTURE_ALL | MODE_WRITER),
#endif

    // Linux, some Unix
#if defined HAVE_CAMV4L2
    DECLARE_BACKEND(CAP_V4L2, "V4L2", MODE_CAPTURE_ALL),
#elif defined HAVE_LIBV4L || defined HAVE_CAMV4L
    DECLARE_BACKEND(CAP_V4L, "V4L", MODE_CAPTURE_ALL),
#endif


    // RGB-D universal
#ifdef HAVE_OPENNI
    DECLARE_BACKEND(CAP_OPENNI, "OPENNI", MODE_CAPTURE_ALL),
#endif
#ifdef HAVE_OPENNI2
    DECLARE_BACKEND(CAP_OPENNI2, "OPENNI2", MODE_CAPTURE_ALL),
#endif
#ifdef HAVE_INTELPERC
    DECLARE_BACKEND(CAP_INTELPERC, "INTEL_PERC", MODE_CAPTURE_BY_INDEX),
#endif

    // OpenCV file-based only
    DECLARE_BACKEND(CAP_IMAGES, "CV_IMAGES", MODE_CAPTURE_BY_FILENAME | MODE_WRITER),
    DECLARE_BACKEND(CAP_OPENCV_MJPEG, "CV_MJPEG", MODE_CAPTURE_BY_FILENAME | MODE_WRITER),

    // special interfaces / stereo cameras / other SDKs
#if defined(HAVE_DC1394_2) || defined(HAVE_DC1394) || defined(HAVE_CMU1394)
    DECLARE_BACKEND(CAP_FIREWIRE, "FIREWIRE", MODE_CAPTURE_BY_INDEX),
#endif
    // GigE
#ifdef HAVE_PVAPI
    DECLARE_BACKEND(CAP_PVAPI, "PVAPI", MODE_CAPTURE_BY_INDEX),
#endif
#ifdef HAVE_XIMEA
    DECLARE_BACKEND(CAP_XIAPI, "XIMEA", MODE_CAPTURE_ALL),
#endif
#ifdef HAVE_GIGE_API
    DECLARE_BACKEND(CAP_GIGANETIX, "GIGANETIX", MODE_CAPTURE_BY_INDEX),
#endif
#ifdef HAVE_ARAVIS_API
    DECLARE_BACKEND(CAP_ARAVIS, "ARAVIS", MODE_CAPTURE_BY_INDEX),
#endif
#ifdef HAVE_UNICAP
    DECLARE_BACKEND(CAP_UNICAP, "UNICAP", MODE_CAPTURE_BY_INDEX),
#endif

#ifdef HAVE_GPHOTO2
    DECLARE_BACKEND(CAP_GPHOTO2, "GPHOTO2", MODE_CAPTURE_ALL),
#endif
#ifdef HAVE_XINE
    DECLARE_BACKEND(CAP_XINE, "XINE", MODE_CAPTURE_BY_FILENAME),
#endif

    // dropped backends: MIL, TYZX, Android
};

bool sortByPriority(const VideoBackendInfo &lhs, const VideoBackendInfo &rhs)
{
    return lhs.priority > rhs.priority;
}

/** @brief Manages list of enabled backends
 */
class VideoBackendRegistry
{
protected:
    std::vector<VideoBackendInfo> enabledBackends;
    VideoBackendRegistry()
    {
        const int N = sizeof(builtin_backends)/sizeof(builtin_backends[0]);
        enabledBackends.assign(builtin_backends, builtin_backends + N);
        for (int i = 0; i < N; i++)
        {
            VideoBackendInfo& info = enabledBackends[i];
            info.priority = 1000 - i * 10;
        }
        CV_LOG_DEBUG(NULL, "VIDEOIO: Builtin backends(" << N << "): " << dumpBackends());
        if (readPrioritySettings())
        {
            CV_LOG_INFO(NULL, "VIDEOIO: Updated backends priorities: " << dumpBackends());
        }
        int enabled = 0;
        for (int i = 0; i < N; i++)
        {
            VideoBackendInfo& info = enabledBackends[enabled];
            if (enabled != i)
                info = enabledBackends[i];
            size_t param_priority = utils::getConfigurationParameterSizeT(cv::format("OPENCV_VIDEOIO_PRIORITY_%s", info.name).c_str(), (size_t)info.priority);
            CV_Assert(param_priority == (size_t)(int)param_priority); // overflow check
            if (param_priority > 0)
            {
                info.priority = (int)param_priority;
                enabled++;
            }
            else
            {
                CV_LOG_INFO(NULL, "VIDEOIO: Disable backend: " << info.name);
            }
        }
        enabledBackends.resize(enabled);
        CV_LOG_DEBUG(NULL, "VIDEOIO: Available backends(" << enabled << "): " << dumpBackends());
        std::sort(enabledBackends.begin(), enabledBackends.end(), sortByPriority);
        CV_LOG_INFO(NULL, "VIDEOIO: Enabled backends(" << enabled << ", sorted by priority): " << dumpBackends());
    }

    static std::vector<std::string> tokenize_string(const std::string& input, char token)
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
    bool readPrioritySettings()
    {
        bool hasChanges = false;
        cv::String prioritized_backends = utils::getConfigurationParameterString("OPENCV_VIDEOIO_PRIORITY_LIST", NULL);
        if (prioritized_backends.empty())
            return hasChanges;
        CV_LOG_INFO(NULL, "VIDEOIO: Configured priority list (OPENCV_VIDEOIO_PRIORITY_LIST): " << prioritized_backends);
        const std::vector<std::string> names = tokenize_string(prioritized_backends, ',');
        for (size_t i = 0; i < names.size(); i++)
        {
            const std::string& name = names[i];
            bool found = false;
            for (size_t k = 0; k < enabledBackends.size(); k++)
            {
                VideoBackendInfo& info = enabledBackends[k];
                if (name == info.name)
                {
                    info.priority = (int)(100000 + (names.size() - i) * 1000);
                    CV_LOG_DEBUG(NULL, "VIDEOIO: New backend priority: '" << name << "' => " << info.priority);
                    found = true;
                    hasChanges = true;
                    break;
                }
            }
            if (!found)
            {
                CV_LOG_WARNING(NULL, "VIDEOIO: Can't prioritize unknown/unavailable backend: '" << name << "'");
            }
        }
        return hasChanges;
    }
public:
    std::string dumpBackends() const
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

    static VideoBackendRegistry& getInstance()
    {
        static VideoBackendRegistry g_instance;
        return g_instance;
    }

    inline std::vector<VideoBackendInfo> getEnabledBackends() const { return enabledBackends; }

    inline std::vector<VideoBackendInfo> getAvailableBackends_CaptureByIndex() const
    {
        std::vector<VideoBackendInfo> result;
        for (size_t i = 0; i < enabledBackends.size(); i++)
        {
            const VideoBackendInfo& info = enabledBackends[i];
            if (info.mode & MODE_CAPTURE_BY_INDEX)
                result.push_back(info);
        }
        return result;
    }
    inline std::vector<VideoBackendInfo> getAvailableBackends_CaptureByFilename() const
    {
        std::vector<VideoBackendInfo> result;
        for (size_t i = 0; i < enabledBackends.size(); i++)
        {
            const VideoBackendInfo& info = enabledBackends[i];
            if (info.mode & MODE_CAPTURE_BY_FILENAME)
                result.push_back(info);
        }
        return result;
    }
    inline std::vector<VideoBackendInfo> getAvailableBackends_Writer() const
    {
        std::vector<VideoBackendInfo> result;
        for (size_t i = 0; i < enabledBackends.size(); i++)
        {
            const VideoBackendInfo& info = enabledBackends[i];
            if (info.mode & MODE_WRITER)
                result.push_back(info);
        }
        return result;
    }
};

} // namespace

namespace videoio_registry {

std::vector<VideoBackendInfo> getAvailableBackends_CaptureByIndex()
{
    const std::vector<VideoBackendInfo> result = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByIndex();
    return result;
}
std::vector<VideoBackendInfo> getAvailableBackends_CaptureByFilename()
{
    const std::vector<VideoBackendInfo> result = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByFilename();
    return result;
}
std::vector<VideoBackendInfo> getAvailableBackends_Writer()
{
    const std::vector<VideoBackendInfo> result = VideoBackendRegistry::getInstance().getAvailableBackends_Writer();
    return result;
}

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

std::vector<VideoCaptureAPIs> getBackends()
{
    std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getEnabledBackends();
    std::vector<VideoCaptureAPIs> result;
    for (size_t i = 0; i < backends.size(); i++)
        result.push_back((VideoCaptureAPIs)backends[i].id);
    return result;
}

std::vector<VideoCaptureAPIs> getCameraBackends()
{
    const std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByIndex();
    std::vector<VideoCaptureAPIs> result;
    for (size_t i = 0; i < backends.size(); i++)
        result.push_back((VideoCaptureAPIs)backends[i].id);
    return result;

}

std::vector<VideoCaptureAPIs> getStreamBackends()
{
    const std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getAvailableBackends_CaptureByFilename();
    std::vector<VideoCaptureAPIs> result;
    for (size_t i = 0; i < backends.size(); i++)
        result.push_back((VideoCaptureAPIs)backends[i].id);
    return result;

}

std::vector<VideoCaptureAPIs> getWriterBackends()
{
    const std::vector<VideoBackendInfo> backends = VideoBackendRegistry::getInstance().getAvailableBackends_Writer();
    std::vector<VideoCaptureAPIs> result;
    for (size_t i = 0; i < backends.size(); i++)
        result.push_back((VideoCaptureAPIs)backends[i].id);
    return result;
}

} // namespace registry

#define TRY_OPEN(backend_func) \
{ \
    try { \
        if (param_VIDEOIO_DEBUG || param_VIDEOCAPTURE_DEBUG) \
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): trying ...\n", #backend_func)); \
        icap = backend_func; \
        if (param_VIDEOIO_DEBUG ||param_VIDEOCAPTURE_DEBUG) \
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): result=%p isOpened=%d ...\n", \
                #backend_func, icap.empty() ? NULL : icap.get(), icap.empty() ? -1: icap->isOpened())); \
    } catch(const cv::Exception& e) { \
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised OpenCV exception:\n\n%s\n", #backend_func, e.what())); \
    } catch (const std::exception& e) { \
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised C++ exception:\n\n%s\n", #backend_func, e.what())); \
    } catch(...) { \
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised unknown C++ exception!\n\n", #backend_func)); \
    } \
    break; \
}

#define TRY_OPEN_LEGACY(backend_func) \
{ \
    try { \
        if (param_VIDEOIO_DEBUG || param_VIDEOCAPTURE_DEBUG) \
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): trying ...\n", #backend_func)); \
        capture = backend_func; \
        if (param_VIDEOIO_DEBUG || param_VIDEOCAPTURE_DEBUG) \
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): result=%p ...\n", #backend_func, capture)); \
    } catch(const cv::Exception& e) { \
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised OpenCV exception:\n\n%s\n", #backend_func, e.what())); \
    } catch (const std::exception& e) { \
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised C++ exception:\n\n%s\n", #backend_func, e.what())); \
    } catch(...) { \
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised unknown C++ exception!\n\n", #backend_func)); \
    } \
    break; \
}


void VideoCapture_create(CvCapture*& capture, Ptr<IVideoCapture>& icap, VideoCaptureAPIs api, int index)
{
    CV_UNUSED(capture); CV_UNUSED(icap);
    switch (api)
    {
    default:
        CV_LOG_WARNING(NULL, "VideoCapture(index=" << index << ") was built without support of requested backendID=" << (int)api);
        break;
#ifdef HAVE_GSTREAMER
    case CAP_GSTREAMER:
        TRY_OPEN(createGStreamerCapture(index));
        break;
#endif
#ifdef HAVE_MSMF
    case CAP_MSMF:
        TRY_OPEN(cvCreateCapture_MSMF(index));
        break;
#endif
#ifdef HAVE_DSHOW
    case CAP_DSHOW:
        TRY_OPEN(makePtr<VideoCapture_DShow>(index));
        break;
#endif
#ifdef HAVE_INTELPERC
    case CAP_INTELPERC:
        TRY_OPEN(makePtr<VideoCapture_IntelPerC>());
        break;
#endif
#ifdef WINRT_VIDEO
    case CAP_WINRT:
        TRY_OPEN(makePtr<cv::VideoCapture_WinRT>(index));
        break;
#endif
#ifdef HAVE_GPHOTO2
    case CAP_GPHOTO2:
        TRY_OPEN(createGPhoto2Capture(index));
        break;
#endif
    case CAP_VFW: // or CAP_V4L or CAP_V4L2
#ifdef HAVE_VFW
        TRY_OPEN_LEGACY(cvCreateCameraCapture_VFW(index))
#endif
#if defined HAVE_LIBV4L || defined HAVE_CAMV4L || defined HAVE_CAMV4L2 || defined HAVE_VIDEOIO
        TRY_OPEN_LEGACY(cvCreateCameraCapture_V4L(index))
#endif
        break;
    case CAP_FIREWIRE:
#ifdef HAVE_DC1394_2
        TRY_OPEN_LEGACY(cvCreateCameraCapture_DC1394_2(index))
#endif
#ifdef HAVE_DC1394
        TRY_OPEN_LEGACY(cvCreateCameraCapture_DC1394(index))
#endif
#ifdef HAVE_CMU1394
        TRY_OPEN_LEGACY(cvCreateCameraCapture_CMU(index))
#endif
        break; // CAP_FIREWIRE
#ifdef HAVE_MIL
    case CAP_MIL:
        TRY_OPEN_LEGACY(cvCreateCameraCapture_MIL(index))
        break;
#endif
#if defined(HAVE_QUICKTIME) || defined(HAVE_QTKIT)
    case CAP_QT:
        TRY_OPEN_LEGACY(cvCreateCameraCapture_QT(index))
        break;
#endif
#ifdef HAVE_UNICAP
    case CAP_UNICAP:
        TRY_OPEN_LEGACY(cvCreateCameraCapture_Unicap(index))
        break;
#endif
#ifdef HAVE_PVAPI
    case CAP_PVAPI:
        TRY_OPEN_LEGACY(cvCreateCameraCapture_PvAPI(index))
        break;
#endif
#ifdef HAVE_OPENNI
    case CAP_OPENNI:
        TRY_OPEN_LEGACY(cvCreateCameraCapture_OpenNI(index))
        break;
#endif
#ifdef HAVE_OPENNI2
    case CAP_OPENNI2:
        TRY_OPEN_LEGACY(cvCreateCameraCapture_OpenNI2(index))
        break;
#endif
#ifdef HAVE_XIMEA
    case CAP_XIAPI:
        TRY_OPEN_LEGACY(cvCreateCameraCapture_XIMEA(index))
        break;
#endif

#ifdef HAVE_AVFOUNDATION
    case CAP_AVFOUNDATION:
        TRY_OPEN_LEGACY(cvCreateCameraCapture_AVFoundation(index))
        break;
#endif

#ifdef HAVE_GIGE_API
    case CAP_GIGANETIX:
        TRY_OPEN_LEGACY(cvCreateCameraCapture_Giganetix(index))
        break;
#endif

#ifdef HAVE_ARAVIS_API
    case CAP_ARAVIS:
        TRY_OPEN_LEGACY(cvCreateCameraCapture_Aravis(index))
        break;
#endif
    } // switch (api)
}

void VideoCapture_create(CvCapture*& capture, Ptr<IVideoCapture>& icap, VideoCaptureAPIs api, const cv::String& filename)
{
    switch (api)
    {
    default:
        CV_LOG_WARNING(NULL, "VideoCapture(filename=" << filename << ") was built without support of requested backendID=" << (int)api);
        break;
#if defined HAVE_LIBV4L || defined HAVE_CAMV4L || defined HAVE_CAMV4L2 || defined HAVE_VIDEOIO
    case CAP_V4L:
        TRY_OPEN_LEGACY(cvCreateCameraCapture_V4L(filename.c_str()))
        break;
#endif

#ifdef HAVE_VFW
    case CAP_VFW:
        TRY_OPEN_LEGACY(cvCreateFileCapture_VFW(filename.c_str()))
        break;
#endif

#if defined(HAVE_QUICKTIME) || defined(HAVE_QTKIT)
    case CAP_QT:
        TRY_OPEN_LEGACY(cvCreateFileCapture_QT(filename.c_str()))
        break;
#endif

#ifdef HAVE_AVFOUNDATION
    case CAP_AVFOUNDATION:
        TRY_OPEN_LEGACY(cvCreateFileCapture_AVFoundation(filename.c_str()))
        break;
#endif

#ifdef HAVE_OPENNI
    case CAP_OPENNI:
        TRY_OPEN_LEGACY(cvCreateFileCapture_OpenNI(filename.c_str()))
        break;
#endif

#ifdef HAVE_OPENNI2
    case CAP_OPENNI2:
        TRY_OPEN_LEGACY(cvCreateFileCapture_OpenNI2(filename.c_str()))
        break;
#endif
#ifdef HAVE_XIMEA
    case CAP_XIAPI:
        TRY_OPEN_LEGACY(cvCreateCameraCapture_XIMEA(filename.c_str()))
        break;
#endif
    case CAP_IMAGES:
        TRY_OPEN_LEGACY(cvCreateFileCapture_Images(filename.c_str()))
        break;
#ifdef HAVE_FFMPEG
    case CAP_FFMPEG:
        TRY_OPEN(cvCreateFileCapture_FFMPEG_proxy(filename))
        break;
#endif
#ifdef HAVE_GSTREAMER
    case CAP_GSTREAMER:
        TRY_OPEN(createGStreamerCapture(filename))
        break;
#endif
#ifdef HAVE_XINE
    case CAP_XINE:
        TRY_OPEN(createXINECapture(filename.c_str()))
        break;
#endif
#ifdef HAVE_MSMF
    case CAP_MSMF:
        TRY_OPEN(cvCreateCapture_MSMF(filename))
        break;
#endif
#ifdef HAVE_GPHOTO2
    case CAP_GPHOTO2:
        TRY_OPEN(createGPhoto2Capture(filename))
        break;
#endif
#ifdef HAVE_MFX
    case CAP_INTEL_MFX:
        TRY_OPEN(makePtr<VideoCapture_IntelMFX>(filename))
        break;
#endif
    case CAP_OPENCV_MJPEG:
        TRY_OPEN(createMotionJpegCapture(filename))
        break;
    } // switch
}


void VideoWriter_create(CvVideoWriter*& writer, Ptr<IVideoWriter>& iwriter, VideoCaptureAPIs api,
        const String& filename, int fourcc, double fps, const Size& frameSize, bool isColor)
{
#define CREATE_WRITER(backend_func) \
{ \
    try { \
        if (param_VIDEOIO_DEBUG || param_VIDEOWRITER_DEBUG) \
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): trying ...\n", #backend_func)); \
        iwriter = backend_func; \
        if (param_VIDEOIO_DEBUG || param_VIDEOWRITER_DEBUG) \
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): result=%p  isOpened=%d...\n", #backend_func, iwriter.empty() ? NULL : iwriter.get(), iwriter.empty() ? -1 : iwriter->isOpened())); \
    } catch(const cv::Exception& e) { \
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised OpenCV exception:\n\n%s\n", #backend_func, e.what())); \
    } catch (const std::exception& e) { \
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised C++ exception:\n\n%s\n", #backend_func, e.what())); \
    } catch(...) { \
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised unknown C++ exception!\n\n", #backend_func)); \
    } \
    break; \
}

#define CREATE_WRITER_LEGACY(backend_func) \
{ \
    try { \
        if (param_VIDEOIO_DEBUG || param_VIDEOWRITER_DEBUG) \
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): trying ...\n", #backend_func)); \
        writer = backend_func; \
        if (param_VIDEOIO_DEBUG || param_VIDEOWRITER_DEBUG) \
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): result=%p...\n", #backend_func, writer)); \
    } catch(const cv::Exception& e) { \
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised OpenCV exception:\n\n%s\n", #backend_func, e.what())); \
    } catch (const std::exception& e) { \
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised C++ exception:\n\n%s\n", #backend_func, e.what())); \
    } catch(...) { \
        CV_LOG_ERROR(NULL, cv::format("VIDEOIO(%s): raised unknown C++ exception!\n\n", #backend_func)); \
    } \
    break; \
}

    switch (api)
    {
    default:
        CV_LOG_ERROR(NULL, "Unknown VideoWriter backend (check getBuildInformation()): " << (int)api);
        break;
#ifdef HAVE_FFMPEG
    case CAP_FFMPEG:
        CREATE_WRITER(cvCreateVideoWriter_FFMPEG_proxy(filename, fourcc, fps, frameSize, isColor));
        break;
#endif
#ifdef HAVE_MSMF
    case CAP_MSMF:
        CREATE_WRITER(cvCreateVideoWriter_MSMF(filename, fourcc, fps, frameSize, isColor));
        break;
#endif
#ifdef HAVE_MFX
    case CAP_INTEL_MFX:
        CREATE_WRITER(VideoWriter_IntelMFX::create(filename, fourcc, fps, frameSize, isColor));
        break;
#endif
#ifdef HAVE_VFW
    case CAP_VFW:
        CREATE_WRITER_LEGACY(cvCreateVideoWriter_VFW(filename.c_str(), fourcc, fps, cvSize(frameSize), isColor))
        break;
#endif
#ifdef HAVE_AVFOUNDATION
    case CAP_AVFOUNDATION:
        CREATE_WRITER_LEGACY(cvCreateVideoWriter_AVFoundation(filename.c_str(), fourcc, fps, cvSize(frameSize), isColor))
        break;
#endif
#if defined(HAVE_QUICKTIME) || defined(HAVE_QTKIT)
    case(CAP_QT):
        CREATE_WRITER_LEGACY(cvCreateVideoWriter_QT(filename.c_str(), fourcc, fps, cvSize(frameSize), isColor))
        break;
#endif
#ifdef HAVE_GSTREAMER
case CAP_GSTREAMER:
        CREATE_WRITER_LEGACY(cvCreateVideoWriter_GStreamer (filename.c_str(), fourcc, fps, cvSize(frameSize), isColor))
        break;
#endif
    case CAP_OPENCV_MJPEG:
        CREATE_WRITER(createMotionJpegWriter(filename, fourcc, fps, frameSize, isColor));
        break;
    case CAP_IMAGES:
        if(!fourcc || !fps)
        {
            CREATE_WRITER_LEGACY(cvCreateVideoWriter_Images(filename.c_str()));
        }
        break;
    } // switch(api)
}


} // namespace
