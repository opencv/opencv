/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2008, 2011, Nils Hasler, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*!
 * \file cap_gstreamer.cpp
 * \author Nils Hasler <hasler@mpi-inf.mpg.de>
 *         Max-Planck-Institut Informatik
 * \author Dirk Van Haerenborgh <vhdirk@gmail.com>
 *
 * \brief Use GStreamer to read/write video
 */
#include "precomp.hpp"

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <iostream>
#include <string.h>

#include <gst/gst.h>
#include <gst/gstbuffer.h>
#include <gst/video/video.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/riff/riff-media.h>
#include <gst/pbutils/missing-plugins.h>

#define VERSION_NUM(major, minor, micro) (major * 1000000 + minor * 1000 + micro)
#define FULL_GST_VERSION VERSION_NUM(GST_VERSION_MAJOR, GST_VERSION_MINOR, GST_VERSION_MICRO)

#if FULL_GST_VERSION >= VERSION_NUM(0,10,32)
#include <gst/pbutils/encoding-profile.h>
//#include <gst/base/gsttypefindhelper.h>
#endif


#define CV_WARN(...) CV_LOG_WARNING(NULL, "OpenCV | GStreamer warning: " << __VA_ARGS__)

#if GST_VERSION_MAJOR == 0
#define COLOR_ELEM "ffmpegcolorspace"
#define COLOR_ELEM_NAME "ffmpegcsp"
#else
#define COLOR_ELEM "videoconvert"
#define COLOR_ELEM_NAME COLOR_ELEM
#endif

#if GST_VERSION_MAJOR == 0
#define CV_GST_FORMAT(format) &(format)
#else
#define CV_GST_FORMAT(format) (format)
#endif


namespace cv {

static void toFraction(double decimal, CV_OUT int& numerator, CV_OUT int& denominator);
static void handleMessage(GstElement * pipeline);


namespace {

#if defined __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wunused-function"
#endif

template<typename T> static inline void GSafePtr_addref(T* ptr)
{
    if (ptr)
        g_object_ref_sink(ptr);
}

template<typename T> static inline void GSafePtr_release(T** pPtr);

template<> inline void GSafePtr_release<GError>(GError** pPtr) { g_clear_error(pPtr); }
template<> inline void GSafePtr_release<GstElement>(GstElement** pPtr) { if (pPtr) { gst_object_unref(G_OBJECT(*pPtr)); *pPtr = NULL; } }
template<> inline void GSafePtr_release<GstElementFactory>(GstElementFactory** pPtr) { if (pPtr) { gst_object_unref(G_OBJECT(*pPtr)); *pPtr = NULL; } }
template<> inline void GSafePtr_release<GstPad>(GstPad** pPtr) { if (pPtr) { gst_object_unref(G_OBJECT(*pPtr)); *pPtr = NULL; } }
template<> inline void GSafePtr_release<GstCaps>(GstCaps** pPtr) { if (pPtr) { gst_caps_unref(*pPtr); *pPtr = NULL; } }
template<> inline void GSafePtr_release<GstBuffer>(GstBuffer** pPtr) { if (pPtr) { gst_buffer_unref(*pPtr); *pPtr = NULL; } }
#if GST_VERSION_MAJOR > 0
template<> inline void GSafePtr_release<GstSample>(GstSample** pPtr) { if (pPtr) { gst_sample_unref(*pPtr); *pPtr = NULL; } }
#endif
template<> inline void GSafePtr_release<GstBus>(GstBus** pPtr) { if (pPtr) { gst_object_unref(G_OBJECT(*pPtr)); *pPtr = NULL; } }
template<> inline void GSafePtr_release<GstMessage>(GstMessage** pPtr) { if (pPtr) { gst_message_unref(*pPtr); *pPtr = NULL; } }

#if FULL_GST_VERSION >= VERSION_NUM(0,10,32)
template<> inline void GSafePtr_release<GstEncodingVideoProfile>(GstEncodingVideoProfile** pPtr) { if (pPtr) { gst_encoding_profile_unref(*pPtr); *pPtr = NULL; } }
template<> inline void GSafePtr_release<GstEncodingContainerProfile>(GstEncodingContainerProfile** pPtr) { if (pPtr) { gst_object_unref(G_OBJECT(*pPtr)); *pPtr = NULL; } }
#endif

template<> inline void GSafePtr_addref<char>(char* pPtr);  // declaration only. not defined. should not be used
template<> inline void GSafePtr_release<char>(char** pPtr) { if (pPtr) { g_free(*pPtr); *pPtr = NULL; } }

#if defined __clang__
# pragma clang diagnostic pop
#endif

template <typename T>
class GSafePtr
{
protected:
    T* ptr;
public:
    inline GSafePtr() CV_NOEXCEPT : ptr(NULL) { }
    inline ~GSafePtr() CV_NOEXCEPT { release(); }
    inline void release() CV_NOEXCEPT
    {
#if 0
        printf("release: %s:%d: %p\n", CV__TRACE_FUNCTION, __LINE__, ptr);
        if (ptr) {
            printf("    refcount: %d\n", (int)GST_OBJECT_REFCOUNT_VALUE(ptr)); \
        }
#endif
        if (ptr)
            GSafePtr_release<T>(&ptr);
    }

    inline operator T* () CV_NOEXCEPT { return ptr; }
    inline operator /*const*/ T* () const CV_NOEXCEPT { return (T*)ptr; }  // there is no const correctness in Gst C API

    T* get() { CV_Assert(ptr); return ptr; }
    /*const*/ T* get() const { CV_Assert(ptr); return (T*)ptr; }  // there is no const correctness in Gst C API

    const T* operator -> () const { CV_Assert(ptr); return ptr; }
    inline operator bool () const CV_NOEXCEPT { return ptr != NULL; }
    inline bool operator ! () const CV_NOEXCEPT { return ptr == NULL; }

    T** getRef() { CV_Assert(ptr == NULL); return &ptr; }

    inline GSafePtr& reset(T* p) CV_NOEXCEPT // pass result of functions with "transfer floating" ownership
    {
        //printf("reset: %s:%d: %p\n", CV__TRACE_FUNCTION, __LINE__, p);
        release();
        if (p)
        {
            GSafePtr_addref<T>(p);
            ptr = p;
        }
        return *this;
    }

    inline GSafePtr& attach(T* p) CV_NOEXCEPT  // pass result of functions with "transfer full" ownership
    {
        //printf("attach: %s:%d: %p\n", CV__TRACE_FUNCTION, __LINE__, p);
        release(); ptr = p; return *this;
    }
    inline T* detach() CV_NOEXCEPT { T* p = ptr; ptr = NULL; return p; }

    inline void swap(GSafePtr& o) CV_NOEXCEPT { std::swap(ptr, o.ptr); }
private:
    GSafePtr(const GSafePtr&); // = disabled
    GSafePtr& operator=(const T*); // = disabled
};

} // namespace

/*!
 * \brief The gst_initializer class
 * Initializes gstreamer once in the whole process
 */
class gst_initializer
{
public:
    static gst_initializer& init()
    {
        static gst_initializer g_init;
        if (g_init.isFailed)
            CV_Error(Error::StsError, "Can't initialize GStreamer");
        return g_init;
    }
private:
    bool isFailed;
    bool call_deinit;
    gst_initializer() :
        isFailed(false)
    {
        call_deinit = utils::getConfigurationParameterBool("OPENCV_VIDEOIO_GSTREAMER_CALL_DEINIT", false);

        GSafePtr<GError> err;
        gboolean gst_init_res = gst_init_check(NULL, NULL, err.getRef());
        if (!gst_init_res)
        {
            CV_WARN("Can't initialize GStreamer: " << (err ? err->message : "<unknown reason>"));
            isFailed = true;
            return;
        }
        guint major, minor, micro, nano;
        gst_version(&major, &minor, &micro, &nano);
        if (GST_VERSION_MAJOR != major)
        {
            CV_WARN("incompatible GStreamer version");
            isFailed = true;
            return;
        }
    }
    ~gst_initializer()
    {
        if (call_deinit)
        {
            // Debug leaks: GST_LEAKS_TRACER_STACK_TRACE=1 GST_DEBUG="GST_TRACER:7" GST_TRACERS="leaks"
            gst_deinit();
        }
    }
};

inline static
std::string get_gst_propname(int propId)
{
    switch (propId)
    {
    case CV_CAP_PROP_BRIGHTNESS: return "brightness";
    case CV_CAP_PROP_CONTRAST: return "contrast";
    case CV_CAP_PROP_SATURATION: return "saturation";
    case CV_CAP_PROP_HUE: return "hue";
    default: return std::string();
    }
}

inline static
bool is_gst_element_exists(const std::string& name)
{
    GSafePtr<GstElementFactory> testfac; testfac.attach(gst_element_factory_find(name.c_str()));
    return (bool)testfac;
}

//==================================================================================================

class GStreamerCapture CV_FINAL : public IVideoCapture
{
private:
    GSafePtr<GstElement> pipeline;
    GSafePtr<GstElement> v4l2src;
    GSafePtr<GstElement> sink;
#if GST_VERSION_MAJOR == 0
    GSafePtr<GstBuffer> buffer;
#else
    GSafePtr<GstSample> sample;
#endif
    GSafePtr<GstCaps> caps;

    gint64        duration;
    gint          width;
    gint          height;
    double        fps;
    bool          isPosFramesSupported;
    bool          isPosFramesEmulated;
    gint64        emulatedFrameNumber;

public:
    GStreamerCapture();
    virtual ~GStreamerCapture() CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual bool retrieveFrame(int /*unused*/, OutputArray dst) CV_OVERRIDE;
    virtual double getProperty(int propId) const CV_OVERRIDE;
    virtual bool setProperty(int propId, double value) CV_OVERRIDE;
    virtual bool isOpened() const CV_OVERRIDE { return (bool)pipeline; }
    virtual int getCaptureDomain() CV_OVERRIDE { return cv::CAP_GSTREAMER; }
    bool open(int id);
    bool open(const String &filename_);
    static void newPad(GstElement * /*elem*/, GstPad     *pad, gpointer    data);

protected:
    bool determineFrameDims(CV_OUT Size& sz, CV_OUT gint& channels, CV_OUT bool& isOutputByteBuffer);
    bool isPipelinePlaying();
    void startPipeline();
    void stopPipeline();
    void restartPipeline();
    void setFilter(const char *prop, int type, int v1, int v2);
    void removeFilter(const char *filter);
};

GStreamerCapture::GStreamerCapture() :
    duration(-1), width(-1), height(-1), fps(-1),
    isPosFramesSupported(false),
    isPosFramesEmulated(false),
    emulatedFrameNumber(-1)
{
}

/*!
 * \brief CvCapture_GStreamer::close
 * Closes the pipeline and destroys all instances
 */
GStreamerCapture::~GStreamerCapture()
{
    if (isPipelinePlaying())
        stopPipeline();
    if (pipeline && GST_IS_ELEMENT(pipeline.get()))
    {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        pipeline.release();
    }
}

/*!
 * \brief CvCapture_GStreamer::grabFrame
 * \return
 * Grabs a sample from the pipeline, awaiting consumation by retrieveFrame.
 * The pipeline is started if it was not running yet
 */
bool GStreamerCapture::grabFrame()
{
    if (!pipeline || !GST_IS_ELEMENT(pipeline.get()))
        return false;

    // start the pipeline if it was not in playing state yet
    if (!this->isPipelinePlaying())
        this->startPipeline();

    // bail out if EOS
    if (gst_app_sink_is_eos(GST_APP_SINK(sink.get())))
        return false;

#if GST_VERSION_MAJOR == 0
    buffer.attach(gst_app_sink_pull_buffer(GST_APP_SINK(sink.get())));
    if (!buffer)
        return false;
#else
    sample.attach(gst_app_sink_pull_sample(GST_APP_SINK(sink.get())));
    if (!sample)
        return false;
#endif

    if (isPosFramesEmulated)
        emulatedFrameNumber++;

    return true;
}

/*!
 * \brief CvCapture_GStreamer::retrieveFrame
 * \return IplImage pointer. [Transfer Full]
 *  Retrieve the previously grabbed buffer, and wrap it in an IPLImage structure
 */
bool GStreamerCapture::retrieveFrame(int, OutputArray dst)
{
#if GST_VERSION_MAJOR == 0
    if (!buffer)
        return false;
#else
    if (!sample)
        return false;
#endif
    Size sz;
    gint channels = 0;
    bool isOutputByteBuffer = false;
    if (!determineFrameDims(sz, channels, isOutputByteBuffer))
        return false;

    // gstreamer expects us to handle the memory at this point
    // so we can just wrap the raw buffer and be done with it
#if GST_VERSION_MAJOR == 0
    Mat src(sz, CV_MAKETYPE(CV_8U, channels), (uchar*)GST_BUFFER_DATA(buffer.get()));
    src.copyTo(dst);
#else
    GstBuffer* buf = gst_sample_get_buffer(sample);  // no lifetime transfer
    if (!buf)
        return false;
    GstMapInfo info = {};
    if (!gst_buffer_map(buf, &info, GST_MAP_READ))
    {
        //something weird went wrong here. abort. abort.
        CV_WARN("Failed to map GStreamer buffer to system memory");
        return false;
    }

    try
    {
        Mat src;
        if (isOutputByteBuffer)
            src = Mat(Size(info.size, 1), CV_8UC1, info.data);
        else
            src = Mat(sz, CV_MAKETYPE(CV_8U, channels), info.data);
        CV_Assert(src.isContinuous());
        src.copyTo(dst);
    }
    catch (...)
    {
        gst_buffer_unmap(buf, &info);
        throw;
    }
    gst_buffer_unmap(buf, &info);
#endif

    return true;
}

bool GStreamerCapture::determineFrameDims(Size &sz, gint& channels, bool& isOutputByteBuffer)
{
#if GST_VERSION_MAJOR == 0
    GstCaps * frame_caps = gst_buffer_get_caps(buffer);  // no lifetime transfer
#else
    GstCaps * frame_caps = gst_sample_get_caps(sample);  // no lifetime transfer
#endif
    // bail out in no caps
    if (!GST_CAPS_IS_SIMPLE(frame_caps))
        return false;

    GstStructure* structure = gst_caps_get_structure(frame_caps, 0);  // no lifetime transfer

    // bail out if width or height are 0
    if (!gst_structure_get_int(structure, "width", &width)
        || !gst_structure_get_int(structure, "height", &height))
    {
        CV_WARN("Can't query frame size from GStreeamer buffer");
        return false;
    }

    sz = Size(width, height);

#if GST_VERSION_MAJOR > 0
    const gchar* name_ = gst_structure_get_name(structure);
    if (!name_)
        return false;
    std::string name = toLowerCase(std::string(name_));

    // we support 11 types of data:
    //     video/x-raw, format=BGR   -> 8bit, 3 channels
    //     video/x-raw, format=GRAY8 -> 8bit, 1 channel
    //     video/x-raw, format=UYVY  -> 8bit, 2 channel
    //     video/x-raw, format=YUY2  -> 8bit, 2 channel
    //     video/x-raw, format=YVYU  -> 8bit, 2 channel
    //     video/x-raw, format=NV12  -> 8bit, 1 channel (height is 1.5x larger than true height)
    //     video/x-raw, format=NV21  -> 8bit, 1 channel (height is 1.5x larger than true height)
    //     video/x-raw, format=YV12  -> 8bit, 1 channel (height is 1.5x larger than true height)
    //     video/x-raw, format=I420  -> 8bit, 1 channel (height is 1.5x larger than true height)
    //     video/x-bayer             -> 8bit, 1 channel
    //     image/jpeg                -> 8bit, mjpeg: buffer_size x 1 x 1
    // bayer data is never decoded, the user is responsible for that
    // everything is 8 bit, so we just test the caps for bit depth
    if (name == "video/x-raw")
    {
        const gchar* format_ = gst_structure_get_string(structure, "format");
        if (!format_)
            return false;
        std::string format = toUpperCase(std::string(format_));

        if (format == "BGR")
        {
            channels = 3;
        }
        else if (format == "UYVY" || format == "YUY2" || format == "YVYU")
        {
            channels = 2;
        }
        else if (format == "NV12" || format == "NV21" || format == "YV12" || format == "I420")
        {
            channels = 1;
            sz.height = sz.height * 3 / 2;
        }
        else if (format == "GRAY8")
        {
            channels = 1;
        }
        else
        {
            CV_Error_(Error::StsNotImplemented, ("Unsupported GStreamer format: %s", format.c_str()));
        }
    }
    else if (name == "video/x-bayer")
    {
        channels = 1;
    }
    else if (name == "image/jpeg")
    {
        // the correct size will be set once the first frame arrives
        channels = 1;
        isOutputByteBuffer = true;
    }
    else
    {
        CV_Error_(Error::StsNotImplemented, ("Unsupported GStreamer layer type: %s", name.c_str()));
    }
#else
    CV_UNUSED(isOutputByteBuffer);
    // we support only video/x-raw, format=BGR -> 8bit, 3 channels
    channels = 3;
#endif
    return true;
}

bool GStreamerCapture::isPipelinePlaying()
{
    if (!pipeline || !GST_IS_ELEMENT(pipeline.get()))
    {
        CV_WARN("GStreamer: pipeline have not been created");
        return false;
    }
    GstState current, pending;
    GstClockTime timeout = 5*GST_SECOND;
    GstStateChangeReturn ret = gst_element_get_state(pipeline, &current, &pending, timeout);
    if (!ret)
    {
        CV_WARN("unable to query pipeline state");
        return false;
    }
    return current == GST_STATE_PLAYING;
}

/*!
 * \brief CvCapture_GStreamer::startPipeline
 * Start the pipeline by setting it to the playing state
 */
void GStreamerCapture::startPipeline()
{
    if (!pipeline || !GST_IS_ELEMENT(pipeline.get()))
    {
        CV_WARN("GStreamer: pipeline have not been created");
        return;
    }
    GstStateChangeReturn status = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (status == GST_STATE_CHANGE_ASYNC)
    {
        // wait for status update
        status = gst_element_get_state(pipeline, NULL, NULL, GST_CLOCK_TIME_NONE);
    }
    if (status == GST_STATE_CHANGE_FAILURE)
    {
        handleMessage(pipeline);
        pipeline.release();
        CV_WARN("unable to start pipeline");
        return;
    }

    if (isPosFramesEmulated)
        emulatedFrameNumber = 0;

    handleMessage(pipeline);
}

void GStreamerCapture::stopPipeline()
{
    if (!pipeline || !GST_IS_ELEMENT(pipeline.get()))
    {
        CV_WARN("GStreamer: pipeline have not been created");
        return;
    }
    if (gst_element_set_state(pipeline, GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE)
    {
        CV_WARN("unable to stop pipeline");
        pipeline.release();
    }
}

/*!
 * \brief CvCapture_GStreamer::restartPipeline
 * Restart the pipeline
 */
void GStreamerCapture::restartPipeline()
{
    handleMessage(pipeline);

    this->stopPipeline();
    this->startPipeline();
}

/*!
 * \brief CvCapture_GStreamer::setFilter
 * \param prop the property name
 * \param type glib property type
 * \param v1 the value
 * \param v2 second value of property type requires it, else NULL
 * Filter the output formats by setting appsink caps properties
 */
void GStreamerCapture::setFilter(const char *prop, int type, int v1, int v2)
{
    if (!caps || !(GST_IS_CAPS(caps.get())))
    {
        if (type == G_TYPE_INT)
        {
#if GST_VERSION_MAJOR == 0
            caps.attach(gst_caps_new_simple("video/x-raw-rgb", prop, type, v1, NULL));
#else
            caps.attach(gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "BGR", prop, type, v1, NULL));
#endif
        }
        else
        {
#if GST_VERSION_MAJOR == 0
            caps.attach(gst_caps_new_simple("video/x-raw-rgb", prop, type, v1, v2, NULL));
#else
            caps.attach(gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "BGR", prop, type, v1, v2, NULL));
#endif
        }
    }
    else
    {
#if GST_VERSION_MAJOR > 0
        if (!gst_caps_is_writable(caps.get()))
            caps.attach(gst_caps_make_writable(caps.detach()));
#endif
        if (type == G_TYPE_INT)
        {
            gst_caps_set_simple(caps, prop, type, v1, NULL);
        }
        else
        {
            gst_caps_set_simple(caps, prop, type, v1, v2, NULL);
        }
    }

#if GST_VERSION_MAJOR > 0
    caps.attach(gst_caps_fixate(caps.detach()));
#endif

    gst_app_sink_set_caps(GST_APP_SINK(sink.get()), caps);
    GST_LOG("filtering with caps: %" GST_PTR_FORMAT, caps.get());
}

/*!
 * \brief CvCapture_GStreamer::removeFilter
 * \param filter filter to remove
 * remove the specified filter from the appsink template caps
 */
void GStreamerCapture::removeFilter(const char *filter)
{
    if(!caps)
        return;

#if GST_VERSION_MAJOR > 0
    if (!gst_caps_is_writable(caps.get()))
        caps.attach(gst_caps_make_writable(caps.detach()));
#endif

    GstStructure *s = gst_caps_get_structure(caps, 0);  // no lifetime transfer
    gst_structure_remove_field(s, filter);

#if GST_VERSION_MAJOR > 0
    caps.attach(gst_caps_fixate(caps.detach()));
#endif

    gst_app_sink_set_caps(GST_APP_SINK(sink.get()), caps);
}

/*!
 * \brief CvCapture_GStreamer::newPad link dynamic padd
 * \param pad
 * \param data
 * decodebin creates pads based on stream information, which is not known upfront
 * on receiving the pad-added signal, we connect it to the colorspace conversion element
 */
void GStreamerCapture::newPad(GstElement *, GstPad *pad, gpointer data)
{
    GSafePtr<GstPad> sinkpad;
    GstElement* color = (GstElement*)data;

    sinkpad.attach(gst_element_get_static_pad(color, "sink"));
    if (!sinkpad) {
        CV_WARN("no pad named sink");
        return;
    }

    gst_pad_link(pad, sinkpad.get());
}

/*!
 * \brief Create GStreamer pipeline
 * \param filename Filename to open in case of CV_CAP_GSTREAMER_FILE
 * \return boolean. Specifies if opening was successful.
 *
 * In case of camera 'index', a pipeline is constructed as follows:
 *    v4l2src ! autoconvert ! appsink
 *
 *
 * The 'filename' parameter is not limited to filesystem paths, and may be one of the following:
 *
 *  - a normal filesystem path:
 *        e.g. video.avi or /path/to/video.avi or C:\\video.avi
 *  - an uri:
 *        e.g. file:///path/to/video.avi or rtsp:///path/to/stream.asf
 *  - a gstreamer pipeline description:
 *        e.g. videotestsrc ! videoconvert ! appsink
 *        the appsink name should be either 'appsink0' (the default) or 'opencvsink'
 *
 *  GStreamer will not drop frames if the grabbing interval larger than the framerate period.
 *  To support dropping for live streams add appsink 'drop' parameter into your custom pipeline.
 *
 *  The pipeline will only be started whenever the first frame is grabbed. Setting pipeline properties
 *  is really slow if we need to restart the pipeline over and over again.
 *
 */
bool GStreamerCapture::open(int id)
{
    gst_initializer::init();

    if (!is_gst_element_exists("v4l2src"))
        return false;
    std::ostringstream desc;
    desc << "v4l2src device=/dev/video" << id
             << " ! " << COLOR_ELEM
             << " ! appsink drop=true";
    return open(desc.str());
}

bool GStreamerCapture::open(const String &filename_)
{
    gst_initializer::init();

    const gchar* filename = filename_.c_str();

    bool file = false;
    bool manualpipeline = false;
    GSafePtr<char> uri;
    GSafePtr<GstElement> uridecodebin;
    GSafePtr<GstElement> color;
    GstStateChangeReturn status;

    // test if we have a valid uri. If so, open it with an uridecodebin
    // else, we might have a file or a manual pipeline.
    // if gstreamer cannot parse the manual pipeline, we assume we were given and
    // ordinary file path.
    CV_LOG_INFO(NULL, "OpenCV | GStreamer: " << filename);
    if (!gst_uri_is_valid(filename))
    {
        if (utils::fs::exists(filename_))
        {
            GSafePtr<GError> err;
            uri.attach(gst_filename_to_uri(filename, err.getRef()));
            if (uri)
            {
                file = true;
            }
            else
            {
                CV_WARN("Error opening file: " << filename << " (" << (err ? err->message : "<unknown reason>") << ")");
                return false;
            }
        }
        else
        {
            GSafePtr<GError> err;
            uridecodebin.attach(gst_parse_launch(filename, err.getRef()));
            if (!uridecodebin)
            {
                CV_WARN("Error opening bin: " << (err ? err->message : "<unknown reason>"));
                return false;
            }
            manualpipeline = true;
        }
    }
    else
    {
        uri.attach(g_strdup(filename));
    }
    CV_LOG_INFO(NULL, "OpenCV | GStreamer: mode - " << (file ? "FILE" : manualpipeline ? "MANUAL" : "URI"));
    bool element_from_uri = false;
    if (!uridecodebin)
    {
        // At this writing, the v4l2 element (and maybe others too) does not support caps renegotiation.
        // This means that we cannot use an uridecodebin when dealing with v4l2, since setting
        // capture properties will not work.
        // The solution (probably only until gstreamer 1.2) is to make an element from uri when dealing with v4l2.
        GSafePtr<gchar> protocol_; protocol_.attach(gst_uri_get_protocol(uri));
        CV_Assert(protocol_);
        std::string protocol = toLowerCase(std::string(protocol_.get()));
        if (protocol == "v4l2")
        {
#if GST_VERSION_MAJOR == 0
            uridecodebin.reset(gst_element_make_from_uri(GST_URI_SRC, uri.get(), "src"));
#else
            uridecodebin.reset(gst_element_make_from_uri(GST_URI_SRC, uri.get(), "src", NULL));
#endif
            CV_Assert(uridecodebin);
            element_from_uri = true;
        }
        else
        {
            uridecodebin.reset(gst_element_factory_make("uridecodebin", NULL));
            CV_Assert(uridecodebin);
            g_object_set(G_OBJECT(uridecodebin.get()), "uri", uri.get(), NULL);
        }

        if (!uridecodebin)
        {
            CV_WARN("Can not parse GStreamer URI bin");
            return false;
        }
    }

    if (manualpipeline)
    {
        GstIterator *it = gst_bin_iterate_elements(GST_BIN(uridecodebin.get()));

        gboolean done = false;
#if GST_VERSION_MAJOR > 0
        GValue value = G_VALUE_INIT;
#endif

        while (!done)
        {
            GstElement *element = NULL;
            GSafePtr<gchar> name;
#if GST_VERSION_MAJOR > 0
            switch (gst_iterator_next (it, &value))
            {
            case GST_ITERATOR_OK:
                element = GST_ELEMENT (g_value_get_object (&value));
#else
            switch (gst_iterator_next (it, (gpointer *)&element))
            {
            case GST_ITERATOR_OK:
#endif
                name.attach(gst_element_get_name(element));
                if (name)
                {
                    if (strstr(name, "opencvsink") != NULL || strstr(name, "appsink") != NULL)
                    {
                        sink.attach(GST_ELEMENT(gst_object_ref(element)));
                    }
                    else if (strstr(name, COLOR_ELEM_NAME) != NULL)
                    {
                        color.attach(GST_ELEMENT(gst_object_ref(element)));
                    }
                    else if (strstr(name, "v4l") != NULL)
                    {
                        v4l2src.attach(GST_ELEMENT(gst_object_ref(element)));
                    }
                    name.release();

                    done = sink && color && v4l2src;
                }
#if GST_VERSION_MAJOR > 0
                g_value_unset (&value);
#endif
                break;
            case GST_ITERATOR_RESYNC:
                gst_iterator_resync (it);
                break;
            case GST_ITERATOR_ERROR:
            case GST_ITERATOR_DONE:
                done = TRUE;
                break;
            }
        }
        gst_iterator_free (it);

        if (!sink)
        {
            CV_WARN("cannot find appsink in manual pipeline");
            return false;
        }

        pipeline.swap(uridecodebin);
    }
    else
    {
        pipeline.reset(gst_pipeline_new(NULL));
        CV_Assert(pipeline);

        // videoconvert (in 0.10: ffmpegcolorspace, in 1.x autovideoconvert)
        //automatically selects the correct colorspace conversion based on caps.
        color.reset(gst_element_factory_make(COLOR_ELEM, NULL));
        CV_Assert(color);

        sink.reset(gst_element_factory_make("appsink", NULL));
        CV_Assert(sink);

        gst_bin_add_many(GST_BIN(pipeline.get()), uridecodebin.get(), color.get(), sink.get(), NULL);

        if (element_from_uri)
        {
            if(!gst_element_link(uridecodebin, color.get()))
            {
                CV_WARN("cannot link color -> sink");
                pipeline.release();
                return false;
            }
        }
        else
        {
            g_signal_connect(uridecodebin, "pad-added", G_CALLBACK(newPad), color.get());
        }

        if (!gst_element_link(color.get(), sink.get()))
        {
            CV_WARN("GStreamer: cannot link color -> sink");
            pipeline.release();
            return false;
        }
    }

    if (!manualpipeline || strstr(filename, " max-buffers=") == NULL)
    {
        //TODO: is 1 single buffer really high enough?
        gst_app_sink_set_max_buffers(GST_APP_SINK(sink.get()), 1);
    }

    if (!manualpipeline)
    {
        gst_base_sink_set_sync(GST_BASE_SINK(sink.get()), FALSE);
    }

    //do not emit signals: all calls will be synchronous and blocking
    gst_app_sink_set_emit_signals (GST_APP_SINK(sink.get()), FALSE);


#if GST_VERSION_MAJOR == 0
    caps.attach(gst_caps_new_simple("video/x-raw-rgb",
                                    "bpp",        G_TYPE_INT, 24,
                                    "red_mask",   G_TYPE_INT, 0x0000FF,
                                    "green_mask", G_TYPE_INT, 0x00FF00,
                                    "blue_mask",  G_TYPE_INT, 0xFF0000,
                                    NULL));
#else

    caps.attach(gst_caps_from_string("video/x-raw, format=(string){BGR, GRAY8}; video/x-bayer,format=(string){rggb,bggr,grbg,gbrg}; image/jpeg"));

    if (manualpipeline)
    {
        GSafePtr<GstCaps> peer_caps;
        GSafePtr<GstPad> sink_pad;
        sink_pad.attach(gst_element_get_static_pad(sink, "sink"));
        peer_caps.attach(gst_pad_peer_query_caps(sink_pad, NULL));
        if (!gst_caps_can_intersect(caps, peer_caps)) {
            caps.attach(gst_caps_from_string("video/x-raw, format=(string){UYVY,YUY2,YVYU,NV12,NV21,YV12,I420}"));
            CV_Assert(caps);
        }
    }

#endif
    gst_app_sink_set_caps(GST_APP_SINK(sink.get()), caps);
    caps.release();

    {
        GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline.get()), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline-init");

        status = gst_element_set_state(GST_ELEMENT(pipeline.get()),
                                       file ? GST_STATE_PAUSED : GST_STATE_PLAYING);
        if (status == GST_STATE_CHANGE_ASYNC)
        {
            // wait for status update
            status = gst_element_get_state(pipeline, NULL, NULL, GST_CLOCK_TIME_NONE);
        }
        if (status == GST_STATE_CHANGE_FAILURE)
        {
            GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline.get()), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline-error");
            handleMessage(pipeline);
            pipeline.release();
            CV_WARN("unable to start pipeline");
            return false;
        }

        GstFormat format;

        format = GST_FORMAT_DEFAULT;
#if GST_VERSION_MAJOR == 0
        if(!gst_element_query_duration(sink, &format, &duration))
#else
        if(!gst_element_query_duration(sink, format, &duration))
#endif
        {
            handleMessage(pipeline);
            CV_WARN("unable to query duration of stream");
            duration = -1;
        }

        handleMessage(pipeline);

        GSafePtr<GstPad> pad;
        pad.attach(gst_element_get_static_pad(sink, "sink"));

        GSafePtr<GstCaps> buffer_caps;
#if GST_VERSION_MAJOR == 0
        buffer_caps.attach(gst_pad_get_caps(pad));
#else
        buffer_caps.attach(gst_pad_get_current_caps(pad));
#endif
        const GstStructure *structure = gst_caps_get_structure(buffer_caps, 0);  // no lifetime transfer
        if (!gst_structure_get_int (structure, "width", &width) ||
            !gst_structure_get_int (structure, "height", &height))
        {
            CV_WARN("cannot query video width/height");
        }

        gint num = 0, denom=1;
        if (!gst_structure_get_fraction(structure, "framerate", &num, &denom))
        {
            CV_WARN("cannot query video fps");
        }

        fps = (double)num/(double)denom;

        {
            GstFormat format_;
            gint64 value_ = -1;
            gboolean status_;

            format_ = GST_FORMAT_DEFAULT;

            status_ = gst_element_query_position(sink, CV_GST_FORMAT(format_), &value_);
            if (!status_ || value_ != 0 || duration < 0)
            {
                CV_WARN("Cannot query video position: status=" << status_ << ", value=" << value_ << ", duration=" << duration);
                isPosFramesSupported = false;
                isPosFramesEmulated = true;
                emulatedFrameNumber = 0;
            }
            else
                isPosFramesSupported = true;
        }

        GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline.get()), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");
    }

    return true;
}

/*!
 * \brief CvCapture_GStreamer::getProperty retrieve the requested property from the pipeline
 * \param propId requested property
 * \return property value
 *
 * There are two ways the properties can be retrieved. For seek-based properties we can query the pipeline.
 * For frame-based properties, we use the caps of the last receivef sample. This means that some properties
 * are not available until a first frame was received
 */
double GStreamerCapture::getProperty(int propId) const
{
    GstFormat format;
    gint64 value;
    gboolean status;

    if(!pipeline) {
        CV_WARN("GStreamer: no pipeline");
        return -1.0;
    }

    switch(propId)
    {
    case CV_CAP_PROP_POS_MSEC:
        CV_LOG_ONCE_WARNING(NULL, "OpenCV | GStreamer: CAP_PROP_POS_MSEC property result may be unrealiable: "
                                  "https://github.com/opencv/opencv/issues/19025");
        format = GST_FORMAT_TIME;
        status = gst_element_query_position(sink.get(), CV_GST_FORMAT(format), &value);
        if(!status) {
            handleMessage(pipeline);
            CV_WARN("GStreamer: unable to query position of stream");
            return -1.0;
        }
        return value * 1e-6; // nano seconds to milli seconds
    case CV_CAP_PROP_POS_FRAMES:
        if (!isPosFramesSupported)
        {
            if (isPosFramesEmulated)
                return emulatedFrameNumber;
            return -1.0;
        }
        format = GST_FORMAT_DEFAULT;
        status = gst_element_query_position(sink.get(), CV_GST_FORMAT(format), &value);
        if(!status) {
            handleMessage(pipeline);
            CV_WARN("GStreamer: unable to query position of stream");
            return -1.0;
        }
        return value;
    case CV_CAP_PROP_POS_AVI_RATIO:
        format = GST_FORMAT_PERCENT;
        status = gst_element_query_position(sink.get(), CV_GST_FORMAT(format), &value);
        if(!status) {
            handleMessage(pipeline);
            CV_WARN("GStreamer: unable to query position of stream");
            return -1.0;
        }
        return ((double) value) / GST_FORMAT_PERCENT_MAX;
    case CV_CAP_PROP_FRAME_WIDTH:
        return width;
    case CV_CAP_PROP_FRAME_HEIGHT:
        return height;
    case CV_CAP_PROP_FPS:
        return fps;
    case CV_CAP_PROP_FRAME_COUNT:
        return duration;
    case CV_CAP_PROP_BRIGHTNESS:
    case CV_CAP_PROP_CONTRAST:
    case CV_CAP_PROP_SATURATION:
    case CV_CAP_PROP_HUE:
        if (v4l2src)
        {
            std::string propName = get_gst_propname(propId);
            if (!propName.empty())
            {
                gint32 val = 0;
                g_object_get(G_OBJECT(v4l2src.get()), propName.c_str(), &val, NULL);
                return static_cast<double>(val);
            }
        }
        break;
    case CV_CAP_GSTREAMER_QUEUE_LENGTH:
        if(!sink)
        {
            CV_WARN("there is no sink yet");
            return -1.0;
        }
        return gst_app_sink_get_max_buffers(GST_APP_SINK(sink.get()));
    default:
        CV_WARN("unhandled property: " << propId);
        break;
    }

    return -1.0;
}

/*!
 * \brief CvCapture_GStreamer::setProperty
 * \param propId
 * \param value
 * \return success
 * Sets the desired property id with val. If the pipeline is running,
 * it is briefly stopped and started again after the property was set
 */
bool GStreamerCapture::setProperty(int propId, double value)
{
    const GstSeekFlags flags = (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_ACCURATE);

    if(!pipeline)
    {
        CV_WARN("no pipeline");
        return false;
    }

    bool wasPlaying = this->isPipelinePlaying();
    if (wasPlaying)
        this->stopPipeline();

    switch(propId)
    {
    case CV_CAP_PROP_POS_MSEC:
        if(!gst_element_seek_simple(GST_ELEMENT(pipeline.get()), GST_FORMAT_TIME,
                                    flags, (gint64) (value * GST_MSECOND))) {
            handleMessage(pipeline);
            CV_WARN("GStreamer: unable to seek");
        }
        else
        {
            if (isPosFramesEmulated)
            {
                if (value == 0)
                {
                    emulatedFrameNumber = 0;
                    return true;
                }
                else
                {
                    isPosFramesEmulated = false; // reset frame counter emulation
                }
            }
        }
        break;
    case CV_CAP_PROP_POS_FRAMES:
    {
        if (!isPosFramesSupported)
        {
            if (isPosFramesEmulated)
            {
                if (value == 0)
                {
                    restartPipeline();
                    emulatedFrameNumber = 0;
                    return true;
                }
            }
            return false;
            CV_WARN("unable to seek");
        }
        if(!gst_element_seek_simple(GST_ELEMENT(pipeline.get()), GST_FORMAT_DEFAULT,
                                    flags, (gint64) value)) {
            handleMessage(pipeline);
            CV_WARN("GStreamer: unable to seek");
            break;
        }
        // wait for status update
        gst_element_get_state(pipeline, NULL, NULL, GST_CLOCK_TIME_NONE);
        return true;
    }
    case CV_CAP_PROP_POS_AVI_RATIO:
        if(!gst_element_seek_simple(GST_ELEMENT(pipeline.get()), GST_FORMAT_PERCENT,
                                    flags, (gint64) (value * GST_FORMAT_PERCENT_MAX))) {
            handleMessage(pipeline);
            CV_WARN("GStreamer: unable to seek");
        }
        else
        {
            if (isPosFramesEmulated)
            {
                if (value == 0)
                {
                    emulatedFrameNumber = 0;
                    return true;
                }
                else
                {
                    isPosFramesEmulated = false; // reset frame counter emulation
                }
            }
        }
        break;
    case CV_CAP_PROP_FRAME_WIDTH:
        if(value > 0)
            setFilter("width", G_TYPE_INT, (int) value, 0);
        else
            removeFilter("width");
        break;
    case CV_CAP_PROP_FRAME_HEIGHT:
        if(value > 0)
            setFilter("height", G_TYPE_INT, (int) value, 0);
        else
            removeFilter("height");
        break;
    case CV_CAP_PROP_FPS:
        if(value > 0) {
            int num = 0, denom = 1;
            toFraction(value, num, denom);
            setFilter("framerate", GST_TYPE_FRACTION, value, denom);
        } else
            removeFilter("framerate");
        break;
    case CV_CAP_PROP_BRIGHTNESS:
    case CV_CAP_PROP_CONTRAST:
    case CV_CAP_PROP_SATURATION:
    case CV_CAP_PROP_HUE:
        if (v4l2src)
        {
            std::string propName = get_gst_propname(propId);
            if (!propName.empty())
            {
                gint32 val = cv::saturate_cast<gint32>(value);
                g_object_set(G_OBJECT(v4l2src.get()), propName.c_str(), &val, NULL);
                return true;
            }
        }
        return false;
    case CV_CAP_PROP_GAIN:
    case CV_CAP_PROP_CONVERT_RGB:
        break;
    case CV_CAP_GSTREAMER_QUEUE_LENGTH:
    {
        if(!sink)
        {
            CV_WARN("there is no sink yet");
            return false;
        }
        gst_app_sink_set_max_buffers(GST_APP_SINK(sink.get()), (guint) value);
        return true;
    }
    default:
        CV_WARN("GStreamer: unhandled property");
    }

    if (wasPlaying)
        this->startPipeline();

    return false;
}


Ptr<IVideoCapture> createGStreamerCapture(const String& filename)
{
    Ptr<GStreamerCapture> cap = makePtr<GStreamerCapture>();
    if (cap && cap->open(filename))
        return cap;
    return Ptr<IVideoCapture>();
}

Ptr<IVideoCapture> createGStreamerCapture(int index)
{
    Ptr<GStreamerCapture> cap = makePtr<GStreamerCapture>();
    if (cap && cap->open(index))
        return cap;
    return Ptr<IVideoCapture>();
}

//==================================================================================================

/*!
 * \brief The CvVideoWriter_GStreamer class
 * Use GStreamer to write video
 */
class CvVideoWriter_GStreamer : public CvVideoWriter
{
public:
    CvVideoWriter_GStreamer()
        : input_pix_fmt(0),
          num_frames(0), framerate(0)
    {
    }
    virtual ~CvVideoWriter_GStreamer() CV_OVERRIDE
    {
        try
        {
            close();
        }
        catch (const std::exception& e)
        {
            CV_WARN("C++ exception in writer destructor: " << e.what());
        }
        catch (...)
        {
            CV_WARN("Unknown exception in writer destructor. Ignore");
        }
    }

    int getCaptureDomain() const CV_OVERRIDE { return cv::CAP_GSTREAMER; }

    virtual bool open( const char* filename, int fourcc,
                       double fps, CvSize frameSize, bool isColor );
    virtual void close();
    virtual bool writeFrame( const IplImage* image ) CV_OVERRIDE;
protected:
    const char* filenameToMimetype(const char* filename);
    GSafePtr<GstElement> pipeline;
    GSafePtr<GstElement> source;

    int input_pix_fmt;
    int num_frames;
    double framerate;

    void close_();
};

/*!
 * \brief CvVideoWriter_GStreamer::close
 * ends the pipeline by sending EOS and destroys the pipeline and all
 * elements afterwards
 */
void CvVideoWriter_GStreamer::close_()
{
    GstStateChangeReturn status;
    if (pipeline)
    {
        handleMessage(pipeline);

        if (!(bool)source)
        {
            CV_WARN("No source in GStreamer pipeline. Ignore");
        }
        else if (gst_app_src_end_of_stream(GST_APP_SRC(source.get())) != GST_FLOW_OK)
        {
            CV_WARN("Cannot send EOS to GStreamer pipeline");
        }
        else
        {
            //wait for EOS to trickle down the pipeline. This will let all elements finish properly
            GSafePtr<GstBus> bus; bus.attach(gst_element_get_bus(pipeline));
            if (bus)
            {
                GSafePtr<GstMessage> msg; msg.attach(gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS)));
                if (!msg || GST_MESSAGE_TYPE(msg.get()) == GST_MESSAGE_ERROR)
                {
                    CV_WARN("Error during VideoWriter finalization");
                    handleMessage(pipeline);
                }
            }
            else
            {
                CV_WARN("can't get GstBus");
            }
        }

        status = gst_element_set_state (pipeline, GST_STATE_NULL);
        if (status == GST_STATE_CHANGE_ASYNC)
        {
            // wait for status update
            GstState st1;
            GstState st2;
            status = gst_element_get_state(pipeline, &st1, &st2, GST_CLOCK_TIME_NONE);
        }
        if (status == GST_STATE_CHANGE_FAILURE)
        {
            handleMessage (pipeline);
            CV_WARN("Unable to stop writer pipeline");
        }
    }
}

void CvVideoWriter_GStreamer::close()
{
    close_();
    source.release();
    pipeline.release();
}

/*!
 * \brief filenameToMimetype
 * \param filename
 * \return mimetype
 * Returns a container mime type for a given filename by looking at it's extension
 */
const char* CvVideoWriter_GStreamer::filenameToMimetype(const char *filename)
{
    //get extension
    const char *ext_ = strrchr(filename, '.');
    if (!ext_ || ext_ == filename)
        return NULL;
    ext_ += 1; //exclude the dot

    std::string ext(ext_);
    ext = toLowerCase(ext);

    // return a container mime based on the given extension.
    // gstreamer's function returns too much possibilities, which is not useful to us

    //return the appropriate mime
    if (ext == "avi")
        return "video/x-msvideo";

    if (ext == "mkv" || ext == "mk3d" || ext == "webm")
        return "video/x-matroska";

    if (ext == "wmv")
        return "video/x-ms-asf";

    if (ext == "mov")
        return "video/x-quicktime";

    if (ext == "ogg" || ext == "ogv")
        return "application/ogg";

    if (ext == "rm")
        return "vnd.rn-realmedia";

    if (ext == "swf")
        return "application/x-shockwave-flash";

    if (ext == "mp4")
        return "video/x-quicktime, variant=(string)iso";

    //default to avi
    return "video/x-msvideo";
}

/*!
 * \brief CvVideoWriter_GStreamer::open
 * \param filename filename to output to
 * \param fourcc desired codec fourcc
 * \param fps desired framerate
 * \param frameSize the size of the expected frames
 * \param is_color color or grayscale
 * \return success
 *
 * We support 2 modes of operation. Either the user enters a filename and a fourcc
 * code, or enters a manual pipeline description like in CvVideoCapture_GStreamer.
 * In the latter case, we just push frames on the appsink with appropriate caps.
 * In the former case, we try to deduce the correct container from the filename,
 * and the correct encoder from the fourcc profile.
 *
 * If the file extension did was not recognize, an avi container is used
 *
 */
bool CvVideoWriter_GStreamer::open( const char * filename, int fourcc,
                                    double fps, CvSize frameSize, bool is_color )
{
    // check arguments
    CV_Assert(filename);
    CV_Assert(fps > 0);
    CV_Assert(frameSize.width > 0 && frameSize.height > 0);

    // init gstreamer
    gst_initializer::init();

    // init vars
    GSafePtr<GstElement> file;
    GSafePtr<GstElement> encodebin;

    bool manualpipeline = true;
    int  bufsize = 0;
    GSafePtr<GError> err;
    GstStateChangeReturn stateret;

    GSafePtr<GstCaps> caps;

    GstIterator* it = NULL;
    gboolean done = FALSE;

#if GST_VERSION_MAJOR == 0
    GstElement* splitter = NULL;
    GstElement* combiner = NULL;
#endif

    // we first try to construct a pipeline from the given string.
    // if that fails, we assume it is an ordinary filename

    encodebin.attach(gst_parse_launch(filename, err.getRef()));
    manualpipeline = (bool)encodebin;

    if (manualpipeline)
    {
        if (err)
        {
            CV_WARN("error opening writer pipeline: " << err->message);
            if (encodebin)
            {
                gst_element_set_state(encodebin, GST_STATE_NULL);
            }
            handleMessage(encodebin);
            encodebin.release();
            return false;
        }
#if GST_VERSION_MAJOR == 0
        it = gst_bin_iterate_sources(GST_BIN(encodebin.get()));
        if (gst_iterator_next(it, (gpointer *)source.getRef()) != GST_ITERATOR_OK) {
            CV_WARN("GStreamer: cannot find appsink in manual pipeline\n");
            return false;
        }
#else
        it = gst_bin_iterate_sources (GST_BIN(encodebin.get()));

        while (!done)
        {
          GValue value = G_VALUE_INIT;
          GSafePtr<gchar> name;
          GstElement* element = NULL;
          switch (gst_iterator_next (it, &value)) {
            case GST_ITERATOR_OK:
              element = GST_ELEMENT (g_value_get_object (&value));  // no lifetime transfer
              name.attach(gst_element_get_name(element));
              if (name)
              {
                if (strstr(name.get(), "opencvsrc") != NULL || strstr(name.get(), "appsrc") != NULL)
                {
                  source.attach(GST_ELEMENT(gst_object_ref(element)));
                  done = TRUE;
                }
              }
              g_value_unset(&value);

              break;
            case GST_ITERATOR_RESYNC:
              gst_iterator_resync (it);
              break;
            case GST_ITERATOR_ERROR:
            case GST_ITERATOR_DONE:
              done = TRUE;
              break;
          }
        }
        gst_iterator_free (it);

        if (!source){
            CV_WARN("GStreamer: cannot find appsrc in manual pipeline\n");
            return false;
        }
#endif
        pipeline.swap(encodebin);
    }
    else
    {
        err.release();
        pipeline.reset(gst_pipeline_new(NULL));

        // we just got a filename and a fourcc code.
        // first, try to guess the container from the filename

        //proxy old non existing fourcc ids. These were used in previous opencv versions,
        //but do not even exist in gstreamer any more
        if (fourcc == CV_FOURCC('M','P','1','V')) fourcc = CV_FOURCC('M', 'P', 'G' ,'1');
        if (fourcc == CV_FOURCC('M','P','2','V')) fourcc = CV_FOURCC('M', 'P', 'G' ,'2');
        if (fourcc == CV_FOURCC('D','R','A','C')) fourcc = CV_FOURCC('d', 'r', 'a' ,'c');


        //create encoder caps from fourcc
        GSafePtr<GstCaps> videocaps;
        videocaps.attach(gst_riff_create_video_caps(fourcc, NULL, NULL, NULL, NULL, NULL));
        if (!videocaps)
        {
            CV_WARN("OpenCV backend does not support passed FOURCC value");
            return false;
        }

        //create container caps from file extension
        const char* mime = filenameToMimetype(filename);
        if (!mime)
        {
            CV_WARN("OpenCV backend does not support this file type (extension): " << filename);
            return false;
        }

        //create pipeline elements
        encodebin.reset(gst_element_factory_make("encodebin", NULL));

#if FULL_GST_VERSION >= VERSION_NUM(0,10,32)
        GSafePtr<GstCaps> containercaps;
        GSafePtr<GstEncodingContainerProfile> containerprofile;
        GSafePtr<GstEncodingVideoProfile> videoprofile;

        containercaps.attach(gst_caps_from_string(mime));

        //create encodebin profile
        containerprofile.attach(gst_encoding_container_profile_new("container", "container", containercaps.get(), NULL));
        videoprofile.reset(gst_encoding_video_profile_new(videocaps.get(), NULL, NULL, 1));
        gst_encoding_container_profile_add_profile(containerprofile.get(), (GstEncodingProfile*)videoprofile.get());

        g_object_set(G_OBJECT(encodebin.get()), "profile", containerprofile.get(), NULL);
#endif

        source.reset(gst_element_factory_make("appsrc", NULL));
        file.reset(gst_element_factory_make("filesink", NULL));
        g_object_set(G_OBJECT(file.get()), "location", (const char*)filename, NULL);
    }

    int fps_num = 0, fps_denom = 1;
    toFraction(fps, fps_num, fps_denom);

    if (fourcc == CV_FOURCC('M','J','P','G') && frameSize.height == 1)
    {
#if GST_VERSION_MAJOR > 0
        input_pix_fmt = GST_VIDEO_FORMAT_ENCODED;
        caps.attach(gst_caps_new_simple("image/jpeg",
                                        "framerate", GST_TYPE_FRACTION, int(fps_num), int(fps_denom),
                                        NULL));
        caps.attach(gst_caps_fixate(caps.detach()));
#else
        CV_WARN("GStreamer 0.10 OpenCV backend does not support writing encoded MJPEG data.");
        return false;
#endif
    }
    else if (is_color)
    {
        input_pix_fmt = GST_VIDEO_FORMAT_BGR;
        bufsize = frameSize.width * frameSize.height * 3;

#if GST_VERSION_MAJOR == 0
        caps.attach(gst_video_format_new_caps(GST_VIDEO_FORMAT_BGR,
                                              frameSize.width,
                                              frameSize.height,
                                              gint(fps_num), gint(fps_denom),
                                              1, 1));
#else
        caps.attach(gst_caps_new_simple("video/x-raw",
                                        "format", G_TYPE_STRING, "BGR",
                                        "width", G_TYPE_INT, frameSize.width,
                                        "height", G_TYPE_INT, frameSize.height,
                                        "framerate", GST_TYPE_FRACTION, gint(fps_num), gint(fps_denom),
                                        NULL));
        CV_Assert(caps);
        caps.attach(gst_caps_fixate(caps.detach()));
#endif
        CV_Assert(caps);
    }
    else
    {
#if FULL_GST_VERSION >= VERSION_NUM(0,10,29)
        input_pix_fmt = GST_VIDEO_FORMAT_GRAY8;
        bufsize = frameSize.width * frameSize.height;

#if GST_VERSION_MAJOR == 0
        caps.attach(gst_video_format_new_caps(GST_VIDEO_FORMAT_GRAY8,
                                              frameSize.width,
                                              frameSize.height,
                                              gint(fps_num), gint(fps_denom),
                                              1, 1));
#else
        caps.attach(gst_caps_new_simple("video/x-raw",
                                        "format", G_TYPE_STRING, "GRAY8",
                                        "width", G_TYPE_INT, frameSize.width,
                                        "height", G_TYPE_INT, frameSize.height,
                                        "framerate", GST_TYPE_FRACTION, gint(fps_num), gint(fps_denom),
                                        NULL));
        caps.attach(gst_caps_fixate(caps.detach()));
#endif
#else
        CV_Error(Error::StsError,
                 "GStreamer 0.10.29 or newer is required for grayscale input");
#endif
    }

    gst_app_src_set_caps(GST_APP_SRC(source.get()), caps);
    gst_app_src_set_stream_type(GST_APP_SRC(source.get()), GST_APP_STREAM_TYPE_STREAM);
    gst_app_src_set_size (GST_APP_SRC(source.get()), -1);

    g_object_set(G_OBJECT(source.get()), "format", GST_FORMAT_TIME, NULL);
    g_object_set(G_OBJECT(source.get()), "block", 1, NULL);
    g_object_set(G_OBJECT(source.get()), "is-live", 0, NULL);


    if (!manualpipeline)
    {
        g_object_set(G_OBJECT(file.get()), "buffer-size", bufsize, NULL);
        gst_bin_add_many(GST_BIN(pipeline.get()), source.get(), encodebin.get(), file.get(), NULL);
        if (!gst_element_link_many(source.get(), encodebin.get(), file.get(), NULL))
        {
            CV_WARN("cannot link elements");
            pipeline.release();
            return false;
        }
    }

#if GST_VERSION_MAJOR == 0
    // HACK: remove streamsplitter and streamcombiner from
    // encodebin pipeline to prevent early EOF event handling
    // We always fetch BGR or gray-scale frames, so combiner->spliter
    // endge in graph is useless.
    it = gst_bin_iterate_recurse (GST_BIN(encodebin.get()));
    while (!done) {
      GSafePtr<gchar> name;
      GstElement* element = NULL;
      switch (gst_iterator_next (it, (void**)&element)) {
        case GST_ITERATOR_OK:
          name.attach(gst_element_get_name(element));
          if (strstr(name, "streamsplitter"))
            splitter = element;
          else if (strstr(name, "streamcombiner"))
            combiner = element;
          break;
        case GST_ITERATOR_RESYNC:
          gst_iterator_resync (it);
          break;
        case GST_ITERATOR_ERROR:
          done = true;
          break;
        case GST_ITERATOR_DONE:
          done = true;
          break;
      }
    }

    gst_iterator_free (it);

    if (splitter && combiner)
    {
        gst_element_unlink(splitter, combiner);

        GstPad* src  = gst_element_get_pad(combiner, "src");
        GstPad* sink = gst_element_get_pad(combiner, "encodingsink");

        GstPad* srcPeer = gst_pad_get_peer(src);
        GstPad* sinkPeer = gst_pad_get_peer(sink);

        gst_pad_unlink(sinkPeer, sink);
        gst_pad_unlink(src, srcPeer);

        gst_pad_link(sinkPeer, srcPeer);

        src = gst_element_get_pad(splitter, "encodingsrc");
        sink = gst_element_get_pad(splitter, "sink");

        srcPeer = gst_pad_get_peer(src);
        sinkPeer = gst_pad_get_peer(sink);

        gst_pad_unlink(sinkPeer, sink);
        gst_pad_unlink(src, srcPeer);

        gst_pad_link(sinkPeer, srcPeer);
    }
#endif

    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline.get()), GST_DEBUG_GRAPH_SHOW_ALL, "write-pipeline");

    stateret = gst_element_set_state(GST_ELEMENT(pipeline.get()), GST_STATE_PLAYING);
    if (stateret == GST_STATE_CHANGE_FAILURE)
    {
        handleMessage(pipeline);
        CV_WARN("GStreamer: cannot put pipeline to play\n");
        pipeline.release();
        return false;
    }

    framerate = fps;
    num_frames = 0;

    handleMessage(pipeline);

    return true;
}


/*!
 * \brief CvVideoWriter_GStreamer::writeFrame
 * \param image
 * \return
 * Pushes the given frame on the pipeline.
 * The timestamp for the buffer is generated from the framerate set in open
 * and ensures a smooth video
 */
bool CvVideoWriter_GStreamer::writeFrame( const IplImage * image )
{
    GstClockTime duration, timestamp;
    GstFlowReturn ret;
    int size;

    handleMessage(pipeline);

#if GST_VERSION_MAJOR > 0
    if (input_pix_fmt == GST_VIDEO_FORMAT_ENCODED) {
        if (image->nChannels != 1 || image->depth != IPL_DEPTH_8U || image->height != 1) {
            CV_WARN("cvWriteFrame() needs images with depth = IPL_DEPTH_8U, nChannels = 1 and height = 1.");
            return false;
        }
    }
    else
#endif
    if(input_pix_fmt == GST_VIDEO_FORMAT_BGR) {
        if (image->nChannels != 3 || image->depth != IPL_DEPTH_8U) {
            CV_WARN("cvWriteFrame() needs images with depth = IPL_DEPTH_8U and nChannels = 3.");
            return false;
        }
    }
#if FULL_GST_VERSION >= VERSION_NUM(0,10,29)
    else if (input_pix_fmt == GST_VIDEO_FORMAT_GRAY8) {
        if (image->nChannels != 1 || image->depth != IPL_DEPTH_8U) {
            CV_WARN("cvWriteFrame() needs images with depth = IPL_DEPTH_8U and nChannels = 1.");
            return false;
        }
    }
#endif
    else {
        CV_WARN("cvWriteFrame() needs BGR or grayscale images\n");
        return false;
    }

    size = image->imageSize;
    duration = ((double)1/framerate) * GST_SECOND;
    timestamp = num_frames * duration;

    //gst_app_src_push_buffer takes ownership of the buffer, so we need to supply it a copy
#if GST_VERSION_MAJOR == 0
    GstBuffer *buffer = gst_buffer_try_new_and_alloc (size);
    if (!buffer)
    {
        CV_WARN("Cannot create GStreamer buffer");
    }

    memcpy(GST_BUFFER_DATA (buffer), (guint8*)image->imageData, size);
    GST_BUFFER_DURATION(buffer) = duration;
    GST_BUFFER_TIMESTAMP(buffer) = timestamp;
#else
    GstBuffer *buffer = gst_buffer_new_allocate(NULL, size, NULL);
    GstMapInfo info;
    gst_buffer_map(buffer, &info, (GstMapFlags)GST_MAP_READ);
    memcpy(info.data, (guint8*)image->imageData, size);
    gst_buffer_unmap(buffer, &info);
    GST_BUFFER_DURATION(buffer) = duration;
    GST_BUFFER_PTS(buffer) = timestamp;
    GST_BUFFER_DTS(buffer) = timestamp;
#endif
    //set the current number in the frame
    GST_BUFFER_OFFSET(buffer) = num_frames;

    ret = gst_app_src_push_buffer(GST_APP_SRC(source.get()), buffer);
    if (ret != GST_FLOW_OK)
    {
        CV_WARN("Error pushing buffer to GStreamer pipeline");
        return false;
    }

    //GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");

    ++num_frames;

    return true;
}

CvVideoWriter* cvCreateVideoWriter_GStreamer(const char* filename, int fourcc, double fps,
                                             CvSize frameSize, int isColor )
{
    CvVideoWriter_GStreamer* wrt = new CvVideoWriter_GStreamer;
    try
    {
        if (wrt->open(filename, fourcc, fps, frameSize, isColor))
            return wrt;
        delete wrt;
    }
    catch (...)
    {
        delete wrt;
        throw;
    }

    return 0;
}

// utility functions

void toFraction(const double decimal, int &numerator_i, int &denominator_i)
{
    double err = 1.0;
    int denominator = 1;
    double numerator = 0;
    for (int check_denominator = 1; ; check_denominator++)
    {
        double check_numerator = (double)check_denominator * decimal;
        double dummy;
        double check_err = modf(check_numerator, &dummy);
        if (check_err < err)
        {
            err = check_err;
            denominator = check_denominator;
            numerator = check_numerator;
            if (err < FLT_EPSILON)
                break;
        }
        if (check_denominator == 100)  // limit
            break;
    }
    numerator_i = cvRound(numerator);
    denominator_i = denominator;
    //printf("%g: %d/%d    (err=%g)\n", decimal, numerator_i, denominator_i, err);
}


/*!
 * \brief handleMessage
 * Handles gstreamer bus messages. Mainly for debugging purposes and ensuring clean shutdown on error
 */
void handleMessage(GstElement * pipeline)
{
    GSafePtr<GstBus> bus;
    GstStreamStatusType tp;
    GstElement * elem = NULL;

    bus.attach(gst_element_get_bus(pipeline));

    while (gst_bus_have_pending(bus))
    {
        GSafePtr<GstMessage> msg;
        msg.attach(gst_bus_pop(bus));
        if (!msg || !GST_IS_MESSAGE(msg.get()))
            continue;

        //printf("\t\tGot %s message\n", GST_MESSAGE_TYPE_NAME(msg));

        if (gst_is_missing_plugin_message(msg))
        {
            CV_WARN("your GStreamer installation is missing a required plugin");
        }
        else
        {
            switch (GST_MESSAGE_TYPE (msg)) {
            case GST_MESSAGE_STATE_CHANGED:
                GstState oldstate, newstate, pendstate;
                gst_message_parse_state_changed(msg, &oldstate, &newstate, &pendstate);
                //fprintf(stderr, "\t\t%s: state changed from %s to %s (pending: %s)\n",
                //                gst_element_get_name(GST_MESSAGE_SRC (msg)),
                //                gst_element_state_get_name(oldstate),
                //                gst_element_state_get_name(newstate), gst_element_state_get_name(pendstate));
                break;
            case GST_MESSAGE_ERROR:
            {
                GSafePtr<GError> err;
                GSafePtr<gchar> debug;
                gst_message_parse_error(msg, err.getRef(), debug.getRef());
                GSafePtr<gchar> name; name.attach(gst_element_get_name(GST_MESSAGE_SRC (msg)));
                CV_WARN("Embedded video playback halted; module " << name.get() <<
                        " reported: " << (err ? err->message : "<unknown reason>"));
                CV_LOG_DEBUG(NULL, "GStreamer debug: " << debug.get());

                gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
                break;
            }
            case GST_MESSAGE_EOS:
                //fprintf(stderr, "\t\treached the end of the stream.");
                break;
            case GST_MESSAGE_STREAM_STATUS:
                gst_message_parse_stream_status(msg,&tp,&elem);
                //fprintf(stderr, "\t\tstream status: elem %s, %i\n", GST_ELEMENT_NAME(elem), tp);
                break;
            default:
                //fprintf(stderr, "\t\tunhandled message %s\n",GST_MESSAGE_TYPE_NAME(msg));
                break;
            }
        }
    }
}


} // namespace cv
