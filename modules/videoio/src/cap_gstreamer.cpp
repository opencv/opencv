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
#ifndef _MSC_VER
#include <unistd.h>
#endif
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


#ifdef NDEBUG
#define CV_WARN(message)
#else
#define CV_WARN(message) fprintf(stderr, "warning: %s (%s:%d)\n", message, __FILE__, __LINE__)
#endif

#if GST_VERSION_MAJOR == 0
#define COLOR_ELEM "ffmpegcolorspace"
#define COLOR_ELEM_NAME "ffmpegcsp"
#else
#define COLOR_ELEM "videoconvert"
#define COLOR_ELEM_NAME COLOR_ELEM
#endif

#if defined(_WIN32) || defined(_WIN64)
#if defined(__MINGW32__)
inline char *realpath(const char *path, char *resolved_path)
{
    return _fullpath(resolved_path,path,PATH_MAX);
}
#endif
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#include <sys/stat.h>
#endif

void toFraction(double decimal, double &numerator, double &denominator);
void handleMessage(GstElement * pipeline);


static cv::Mutex gst_initializer_mutex;

/*!
 * \brief The gst_initializer class
 * Initializes gstreamer once in the whole process
 */
class gst_initializer
{
public:
    static void init()
    {
        gst_initializer_mutex.lock();
        static gst_initializer init;
        gst_initializer_mutex.unlock();
    }
private:
    gst_initializer()
    {
        gst_init(NULL, NULL);
//        gst_debug_set_active(1);
//        gst_debug_set_colored(1);
//        gst_debug_set_default_threshold(GST_LEVEL_INFO);
    }
};

/*!
 * \brief The CvCapture_GStreamer class
 * Use GStreamer to capture video
 */
class CvCapture_GStreamer : public CvCapture
{
public:
    CvCapture_GStreamer() { init(); }
    virtual ~CvCapture_GStreamer() { close(); }

    virtual bool open( int type, const char* filename );
    virtual void close();

    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);

protected:
    void init();
    bool reopen();
    bool isPipelinePlaying();
    void startPipeline();
    void stopPipeline();
    void restartPipeline();
    void setFilter(const char* prop, int type, int v1, int v2 = 0);
    void removeFilter(const char *filter);
    static void newPad(GstElement *myelement,
                       GstPad     *pad,
                       gpointer    data);
    GstElement*   pipeline;
    GstElement*   uridecodebin;
    GstElement*   v4l2src;
    GstElement*   color;
    GstElement*   sink;
#if GST_VERSION_MAJOR > 0
    GstSample*    sample;
#endif
    GstBuffer*    buffer;
    GstCaps*      caps;
    IplImage*     frame;
    gint64        duration;
    gint          width;
    gint          height;
    double        fps;

    bool          isPosFramesSupported;
    bool          isPosFramesEmulated;
    gint64        emulatedFrameNumber;

    bool          isOutputByteBuffer;
};

/*!
 * \brief CvCapture_GStreamer::init
 * inits the class
 */
void CvCapture_GStreamer::init()
{
    pipeline = NULL;
    uridecodebin = NULL;
    v4l2src = NULL;
    color = NULL;
    sink = NULL;
#if GST_VERSION_MAJOR > 0
    sample = NULL;
#endif
    buffer = NULL;
    caps = NULL;
    frame = NULL;
    duration = -1;
    width = -1;
    height = -1;
    fps = -1;

    isPosFramesSupported = false;
    isPosFramesEmulated = false;
    emulatedFrameNumber = -1;

    isOutputByteBuffer = false;
}

/*!
 * \brief CvCapture_GStreamer::close
 * Closes the pipeline and destroys all instances
 */
void CvCapture_GStreamer::close()
{
    if (isPipelinePlaying())
        this->stopPipeline();

    if(pipeline) {
        gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        pipeline = NULL;
    }

    duration = -1;
    width = -1;
    height = -1;
    fps = -1;
    isPosFramesSupported = false;
    isPosFramesEmulated = false;
    emulatedFrameNumber = -1;
}

/*!
 * \brief CvCapture_GStreamer::grabFrame
 * \return
 * Grabs a sample from the pipeline, awaiting consumation by retreiveFrame.
 * The pipeline is started if it was not running yet
 */
bool CvCapture_GStreamer::grabFrame()
{
    if(!pipeline)
        return false;

    // start the pipeline if it was not in playing state yet
    if(!this->isPipelinePlaying())
        this->startPipeline();

    // bail out if EOS
    if(gst_app_sink_is_eos(GST_APP_SINK(sink)))
        return false;

#if GST_VERSION_MAJOR == 0
    if(buffer)
        gst_buffer_unref(buffer);

    buffer = gst_app_sink_pull_buffer(GST_APP_SINK(sink));
#else
    if(sample)
        gst_sample_unref(sample);

    sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));

    if(!sample)
        return false;

    buffer = gst_sample_get_buffer(sample);
#endif

    if(!buffer)
        return false;

    if (isPosFramesEmulated)
        emulatedFrameNumber++;

    return true;
}

/*!
 * \brief CvCapture_GStreamer::retrieveFrame
 * \return IplImage pointer. [Transfer Full]
 *  Retreive the previously grabbed buffer, and wrap it in an IPLImage structure
 */
IplImage * CvCapture_GStreamer::retrieveFrame(int)
{
    if(!buffer)
        return 0;

    //construct a frame header if we did not have any yet
    if(!frame)
    {
#if GST_VERSION_MAJOR == 0
        GstCaps* buffer_caps = gst_buffer_get_caps(buffer);
#else
        GstCaps* buffer_caps = gst_sample_get_caps(sample);
#endif
        // bail out in no caps
        assert(gst_caps_get_size(buffer_caps) == 1);
        GstStructure* structure = gst_caps_get_structure(buffer_caps, 0);

        // bail out if width or height are 0
        if(!gst_structure_get_int(structure, "width", &width) ||
                !gst_structure_get_int(structure, "height", &height))
        {
            gst_caps_unref(buffer_caps);
            return 0;
        }

        int depth = 3;
        bool height_extend = false;

#if GST_VERSION_MAJOR > 0
        depth = 0;
        const gchar* name = gst_structure_get_name(structure);
        const gchar* format = gst_structure_get_string(structure, "format");

        if (!name)
            return 0;

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

        if (strcasecmp(name, "video/x-raw") == 0)
        {
            if (!format)
                return 0;

            if (strcasecmp(format, "BGR") == 0) {
                depth = 3;
            }
            else if( (strcasecmp(format, "UYVY") == 0) || (strcasecmp(format, "YUY2") == 0) || (strcasecmp(format, "YVYU") == 0) ){
                depth = 2;
            }
            else if( (strcasecmp(format, "NV12") == 0) || (strcasecmp(format, "NV21") == 0) || (strcasecmp(format, "YV12") == 0) || (strcasecmp(format, "I420") == 0) ){
                depth = 1;
                height_extend = true;
            }
            else if(strcasecmp(format, "GRAY8") == 0){
                depth = 1;
            }
        }
        else if (strcasecmp(name, "video/x-bayer") == 0)
        {
            depth = 1;
        } else if(strcasecmp(name, "image/jpeg") == 0) {
            depth = 1;
            // the correct size will be set once the first frame arrives
            isOutputByteBuffer = true;
        }
#endif
        if (depth > 0) {
            if(height_extend){
                frame = cvCreateImageHeader(cvSize(width, height*3/2), IPL_DEPTH_8U, depth);
            }else{
                frame = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_8U, depth);
            }
        } else {
            gst_caps_unref(buffer_caps);
            return 0;
        }

        gst_caps_unref(buffer_caps);
    }

    // gstreamer expects us to handle the memory at this point
    // so we can just wrap the raw buffer and be done with it
#if GST_VERSION_MAJOR == 0
    frame->imageData = (char *)GST_BUFFER_DATA(buffer);
#else
    // info.data ptr is valid until next grabFrame where the associated sample is unref'd
    GstMapInfo info = GstMapInfo();
    gboolean success = gst_buffer_map(buffer,&info, (GstMapFlags)GST_MAP_READ);

    // with MJPEG streams frame size can change arbitrarily
    if (isOutputByteBuffer && (size_t)info.size != (size_t)frame->imageSize)
    {
        cvReleaseImageHeader(&frame);
        frame = cvCreateImageHeader(cvSize(info.size, 1), IPL_DEPTH_8U, 1);
    }

    if (!success){
        //something weird went wrong here. abort. abort.
        //fprintf(stderr,"GStreamer: unable to map buffer");
        return 0;
    }
    frame->imageData = (char*)info.data;
    gst_buffer_unmap(buffer,&info);
#endif

    return frame;
}


/*!
 * \brief CvCapture_GStreamer::isPipelinePlaying
 * \return if the pipeline is currently playing.
 */
bool CvCapture_GStreamer::isPipelinePlaying()
{
    GstState current, pending;
    GstClockTime timeout = 5*GST_SECOND;
    if(!GST_IS_ELEMENT(pipeline)){
        return false;
    }

    GstStateChangeReturn ret = gst_element_get_state(GST_ELEMENT(pipeline),&current, &pending, timeout);
    if (!ret){
        //fprintf(stderr, "GStreamer: unable to query pipeline state\n");
        return false;
    }

    return current == GST_STATE_PLAYING;
}

/*!
 * \brief CvCapture_GStreamer::startPipeline
 * Start the pipeline by setting it to the playing state
 */
void CvCapture_GStreamer::startPipeline()
{
    CV_FUNCNAME("icvStartPipeline");

    __BEGIN__;

    //fprintf(stderr, "relinked, pausing\n");
    GstStateChangeReturn status = gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING);
    if (status == GST_STATE_CHANGE_ASYNC)
    {
        // wait for status update
        status = gst_element_get_state(pipeline, NULL, NULL, GST_CLOCK_TIME_NONE);
    }
    if (status == GST_STATE_CHANGE_FAILURE)
    {
        handleMessage(pipeline);
        gst_object_unref(pipeline);
        pipeline = NULL;
        CV_ERROR(CV_StsError, "GStreamer: unable to start pipeline\n");
        return;
    }

    if (isPosFramesEmulated)
        emulatedFrameNumber = 0;

    //printf("state now playing\n");
    handleMessage(pipeline);
    __END__;
}


/*!
 * \brief CvCapture_GStreamer::stopPipeline
 * Stop the pipeline by setting it to NULL
 */
void CvCapture_GStreamer::stopPipeline()
{
    CV_FUNCNAME("icvStopPipeline");

    __BEGIN__;

    //fprintf(stderr, "restarting pipeline, going to ready\n");
    if(gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL) ==
            GST_STATE_CHANGE_FAILURE) {
        CV_ERROR(CV_StsError, "GStreamer: unable to stop pipeline\n");
        gst_object_unref(pipeline);
        pipeline = NULL;
        return;
    }
    __END__;
}

/*!
 * \brief CvCapture_GStreamer::restartPipeline
 * Restart the pipeline
 */
void CvCapture_GStreamer::restartPipeline()
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
void CvCapture_GStreamer::setFilter(const char *prop, int type, int v1, int v2)
{
    //printf("GStreamer: setFilter \n");
    if(!caps || !( GST_IS_CAPS (caps) ))
    {
        if(type == G_TYPE_INT)
        {
#if GST_VERSION_MAJOR == 0
            caps = gst_caps_new_simple("video/x-raw-rgb", prop, type, v1, NULL);
#else
            caps = gst_caps_new_simple("video/x-raw","format",G_TYPE_STRING,"BGR", prop, type, v1, NULL);
#endif
        }
        else
        {
#if GST_VERSION_MAJOR == 0
            caps = gst_caps_new_simple("video/x-raw-rgb", prop, type, v1, v2, NULL);
#else
            caps = gst_caps_new_simple("video/x-raw","format",G_TYPE_STRING,"BGR", prop, type, v1, v2, NULL);
#endif
        }
    }
    else
    {
#if GST_VERSION_MAJOR > 0
        if (! gst_caps_is_writable(caps))
            caps = gst_caps_make_writable (caps);
#endif
        if(type == G_TYPE_INT){
            gst_caps_set_simple(caps, prop, type, v1, NULL);
        }else{
            gst_caps_set_simple(caps, prop, type, v1, v2, NULL);
        }
    }

#if GST_VERSION_MAJOR > 0
    caps = gst_caps_fixate(caps);
#endif

    gst_app_sink_set_caps(GST_APP_SINK(sink), caps);
    //printf("filtering with %s\n", gst_caps_to_string(caps));
}


/*!
 * \brief CvCapture_GStreamer::removeFilter
 * \param filter filter to remove
 * remove the specified filter from the appsink template caps
 */
void CvCapture_GStreamer::removeFilter(const char *filter)
{
    if(!caps)
        return;

#if GST_VERSION_MAJOR > 0
    if (! gst_caps_is_writable(caps))
        caps = gst_caps_make_writable (caps);
#endif

    GstStructure *s = gst_caps_get_structure(caps, 0);
    gst_structure_remove_field(s, filter);

    gst_app_sink_set_caps(GST_APP_SINK(sink), caps);
}

/*!
 * \brief CvCapture_GStreamer::newPad link dynamic padd
 * \param pad
 * \param data
 * decodebin creates pads based on stream information, which is not known upfront
 * on receiving the pad-added signal, we connect it to the colorspace conversion element
 */
void CvCapture_GStreamer::newPad(GstElement * /*elem*/,
                                 GstPad     *pad,
                                 gpointer    data)
{
    GstPad *sinkpad;
    GstElement *color = (GstElement *) data;

    sinkpad = gst_element_get_static_pad (color, "sink");
    if (!sinkpad){
        //fprintf(stderr, "Gstreamer: no pad named sink\n");
        return;
    }

    gst_pad_link (pad, sinkpad);
    gst_object_unref (sinkpad);
}

/*!
 * \brief CvCapture_GStreamer::open Open the given file with gstreamer
 * \param type CvCapture type. One of CV_CAP_GSTREAMER_*
 * \param filename Filename to open in case of CV_CAP_GSTREAMER_FILE
 * \return boolean. Specifies if opening was successful.
 *
 * In case of CV_CAP_GSTREAMER_V4L(2), a pipelin is constructed as follows:
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
 *  When dealing with a file, CvCapture_GStreamer will not drop frames if the grabbing interval
 *  larger than the framerate period. (Unlike the uri or manual pipeline description, which assume
 *  a live source)
 *
 *  The pipeline will only be started whenever the first frame is grabbed. Setting pipeline properties
 *  is really slow if we need to restart the pipeline over and over again.
 *
 *  TODO: the 'type' parameter is imo unneeded. for v4l2, filename 'v4l2:///dev/video0' can be used.
 *  I expect this to be the same for CV_CAP_GSTREAMER_1394. Is anyone actually still using v4l (v1)?
 *
 */
bool CvCapture_GStreamer::open( int type, const char* filename )
{
    CV_FUNCNAME("cvCaptureFromCAM_GStreamer");

    __BEGIN__;

    gst_initializer::init();

    bool file = false;
    bool stream = false;
    bool manualpipeline = false;
    char *uri = NULL;
    uridecodebin = NULL;
    GstElementFactory * testfac;
    GstStateChangeReturn status;

    if (type == CV_CAP_GSTREAMER_V4L){
        testfac = gst_element_factory_find("v4lsrc");
        if (!testfac){
            return false;
        }
        g_object_unref(G_OBJECT(testfac));
        filename = "v4lsrc ! " COLOR_ELEM " ! appsink";
    }
    if (type == CV_CAP_GSTREAMER_V4L2){
        testfac = gst_element_factory_find("v4l2src");
        if (!testfac){
            return false;
        }
        g_object_unref(G_OBJECT(testfac));
        filename = "v4l2src ! " COLOR_ELEM " ! appsink";
    }


    // test if we have a valid uri. If so, open it with an uridecodebin
    // else, we might have a file or a manual pipeline.
    // if gstreamer cannot parse the manual pipeline, we assume we were given and
    // ordinary file path.
    if (!gst_uri_is_valid(filename))
    {
#ifdef _MSC_VER
        uri = new char[2048];
        DWORD pathSize = GetFullPathName(filename, 2048, uri, NULL);
        struct stat buf;
        if (pathSize == 0 || stat(uri, &buf) != 0)
        {
            delete[] uri;
            uri = NULL;
        }
#else
        uri = realpath(filename, NULL);
#endif
        stream = false;
        if(uri)
        {
            uri = g_filename_to_uri(uri, NULL, NULL);
            if(uri)
            {
                file = true;
            }
            else
            {
                CV_WARN("GStreamer: Error opening file\n");
                CV_WARN(filename);
                CV_WARN(uri);
                close();
                return false;
            }
        }
        else
        {
            GError *err = NULL;
            uridecodebin = gst_parse_launch(filename, &err);
            if(!uridecodebin)
            {
                fprintf(stderr, "GStreamer: Error opening bin: %s\n", err->message);
                return false;
            }
            stream = true;
            manualpipeline = true;
        }
    }
    else
    {
        stream = true;
        uri = g_strdup(filename);
    }

    bool element_from_uri = false;
    if(!uridecodebin)
    {
        // At this writing, the v4l2 element (and maybe others too) does not support caps renegotiation.
        // This means that we cannot use an uridecodebin when dealing with v4l2, since setting
        // capture properties will not work.
        // The solution (probably only until gstreamer 1.2) is to make an element from uri when dealing with v4l2.
        gchar * protocol = gst_uri_get_protocol(uri);
        if (!strcasecmp(protocol , "v4l2"))
        {
#if GST_VERSION_MAJOR == 0
            uridecodebin = gst_element_make_from_uri(GST_URI_SRC, uri, "src");
#else
            uridecodebin = gst_element_make_from_uri(GST_URI_SRC, uri, "src", NULL);
#endif
            element_from_uri = true;
        }
        else
        {
            uridecodebin = gst_element_factory_make("uridecodebin", NULL);
            g_object_set(G_OBJECT(uridecodebin), "uri", uri, NULL);
        }
        g_free(protocol);

        if(!uridecodebin)
        {
            //fprintf(stderr, "GStreamer: Error opening bin: %s\n", err->message);
            close();
            return false;
        }
    }

    if (manualpipeline)
    {
        GstIterator *it = gst_bin_iterate_elements(GST_BIN(uridecodebin));

        GstElement *element = NULL;
        gboolean done = false;
        gchar* name = NULL;
#if GST_VERSION_MAJOR > 0
        GValue value = G_VALUE_INIT;
#endif

        while (!done)
        {
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
                name = gst_element_get_name(element);
                if (name)
                {
                    if (strstr(name, "opencvsink") != NULL || strstr(name, "appsink") != NULL)
                    {
                        sink = GST_ELEMENT ( gst_object_ref (element) );
                    }
                    else if (strstr(name, COLOR_ELEM_NAME) != NULL)
                    {
                        color = GST_ELEMENT ( gst_object_ref (element) );
                    }
                    else if (strstr(name, "v4l") != NULL)
                    {
                        v4l2src = GST_ELEMENT ( gst_object_ref (element) );
                    }
                    g_free(name);

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
            CV_ERROR(CV_StsError, "GStreamer: cannot find appsink in manual pipeline\n");
            return false;
        }

        pipeline = uridecodebin;
    }
    else
    {
        pipeline = gst_pipeline_new(NULL);
        // videoconvert (in 0.10: ffmpegcolorspace, in 1.x autovideoconvert)
        //automatically selects the correct colorspace conversion based on caps.
        color = gst_element_factory_make(COLOR_ELEM, NULL);
        sink = gst_element_factory_make("appsink", NULL);

        gst_bin_add_many(GST_BIN(pipeline), uridecodebin, color, sink, NULL);

        if(element_from_uri)
        {
            if(!gst_element_link(uridecodebin, color))
            {
                CV_ERROR(CV_StsError, "GStreamer: cannot link color -> sink\n");
                gst_object_unref(pipeline);
                pipeline = NULL;
                return false;
            }
        }
        else
        {
            g_signal_connect(uridecodebin, "pad-added", G_CALLBACK(newPad), color);
        }

        if(!gst_element_link(color, sink))
        {
            CV_ERROR(CV_StsError, "GStreamer: cannot link color -> sink\n");
            gst_object_unref(pipeline);
            pipeline = NULL;
            return false;
        }
    }

    //TODO: is 1 single buffer really high enough?
    gst_app_sink_set_max_buffers (GST_APP_SINK(sink), 1);
    gst_app_sink_set_drop (GST_APP_SINK(sink), stream);
    //do not emit signals: all calls will be synchronous and blocking
    gst_app_sink_set_emit_signals (GST_APP_SINK(sink), 0);

#if GST_VERSION_MAJOR == 0
    caps = gst_caps_new_simple("video/x-raw-rgb",
                               "bpp",        G_TYPE_INT, 24,
                               "red_mask",   G_TYPE_INT, 0x0000FF,
                               "green_mask", G_TYPE_INT, 0x00FF00,
                               "blue_mask",  G_TYPE_INT, 0xFF0000,
                               NULL);
#else

    caps = gst_caps_from_string("video/x-raw, format=(string){BGR, GRAY8}; video/x-bayer,format=(string){rggb,bggr,grbg,gbrg}; image/jpeg");

    if(manualpipeline){
        GstPad* sink_pad = gst_element_get_static_pad(sink, "sink");
        GstCaps* peer_caps = gst_pad_peer_query_caps(sink_pad,NULL);
        if (!gst_caps_can_intersect(caps, peer_caps)) {
            gst_caps_unref(caps);
            caps = gst_caps_from_string("video/x-raw, format=(string){UYVY,YUY2,YVYU,NV12,NV21,YV12,I420}");
        }
        gst_object_unref(sink_pad);
        gst_caps_unref(peer_caps);
    }

#endif
    gst_app_sink_set_caps(GST_APP_SINK(sink), caps);
    gst_caps_unref(caps);

    {
        status = gst_element_set_state(GST_ELEMENT(pipeline),
                                       file ? GST_STATE_PAUSED : GST_STATE_PLAYING);
        if (status == GST_STATE_CHANGE_ASYNC)
        {
            // wait for status update
            status = gst_element_get_state(pipeline, NULL, NULL, GST_CLOCK_TIME_NONE);
        }
        if (status == GST_STATE_CHANGE_FAILURE)
        {
            handleMessage(pipeline);
            gst_object_unref(pipeline);
            pipeline = NULL;
            CV_ERROR(CV_StsError, "GStreamer: unable to start pipeline\n");
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
            CV_WARN("GStreamer: unable to query duration of stream");
            duration = -1;
        }

        handleMessage(pipeline);

        GstPad* pad = gst_element_get_static_pad(sink, "sink");
#if GST_VERSION_MAJOR == 0
        GstCaps* buffer_caps = gst_pad_get_caps(pad);
#else
        GstCaps* buffer_caps = gst_pad_get_current_caps(pad);
#endif
        const GstStructure *structure = gst_caps_get_structure (buffer_caps, 0);

        if (!gst_structure_get_int (structure, "width", &width))
        {
            CV_WARN("Cannot query video width\n");
        }

        if (!gst_structure_get_int (structure, "height", &height))
        {
            CV_WARN("Cannot query video heigth\n");
        }

        gint num = 0, denom=1;
        if(!gst_structure_get_fraction(structure, "framerate", &num, &denom))
        {
            CV_WARN("Cannot query video fps\n");
        }

        fps = (double)num/(double)denom;

        {
            GstFormat format_;
            gint64 value_ = -1;
            gboolean status_;

            format_ = GST_FORMAT_DEFAULT;
#if GST_VERSION_MAJOR == 0
#define FORMAT &format_
#else
#define FORMAT format_
#endif
            status_ = gst_element_query_position(pipeline, FORMAT, &value_);
#undef FORMAT
            if (!status_ || value_ != 0 || duration < 0)
            {
                CV_WARN(cv::format("Cannot query video position: status=%d value=%lld duration=%lld\n",
                        (int)status_, (long long int)value_, (long long int)duration).c_str());
                isPosFramesSupported = false;
                isPosFramesEmulated = true;
                emulatedFrameNumber = 0;
            }
            else
                isPosFramesSupported = true;
        }

        GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");
    }

    __END__;

    return true;
}

/*!
 * \brief CvCapture_GStreamer::getProperty retreive the requested property from the pipeline
 * \param propId requested property
 * \return property value
 *
 * There are two ways the properties can be retreived. For seek-based properties we can query the pipeline.
 * For frame-based properties, we use the caps of the lasst receivef sample. This means that some properties
 * are not available until a first frame was received
 */
double CvCapture_GStreamer::getProperty( int propId ) const
{
    GstFormat format;
    gint64 value;
    gboolean status;

#if GST_VERSION_MAJOR == 0
#define FORMAT &format
#else
#define FORMAT format
#endif

    if(!pipeline) {
        CV_WARN("GStreamer: no pipeline");
        return 0;
    }

    switch(propId) {
    case CV_CAP_PROP_POS_MSEC:
        format = GST_FORMAT_TIME;
        status = gst_element_query_position(sink, FORMAT, &value);
        if(!status) {
            handleMessage(pipeline);
            CV_WARN("GStreamer: unable to query position of stream");
            return 0;
        }
        return value * 1e-6; // nano seconds to milli seconds
    case CV_CAP_PROP_POS_FRAMES:
        if (!isPosFramesSupported)
        {
            if (isPosFramesEmulated)
                return emulatedFrameNumber;
            return 0; // TODO getProperty() "unsupported" value should be changed
        }
        format = GST_FORMAT_DEFAULT;
        status = gst_element_query_position(sink, FORMAT, &value);
        if(!status) {
            handleMessage(pipeline);
            CV_WARN("GStreamer: unable to query position of stream");
            return 0;
        }
        return value;
    case CV_CAP_PROP_POS_AVI_RATIO:
        format = GST_FORMAT_PERCENT;
        status = gst_element_query_position(sink, FORMAT, &value);
        if(!status) {
            handleMessage(pipeline);
            CV_WARN("GStreamer: unable to query position of stream");
            return 0;
        }
        return ((double) value) / GST_FORMAT_PERCENT_MAX;
    case CV_CAP_PROP_FRAME_WIDTH:
        return width;
    case CV_CAP_PROP_FRAME_HEIGHT:
        return height;
    case CV_CAP_PROP_FPS:
        return fps;
    case CV_CAP_PROP_FOURCC:
        break;
    case CV_CAP_PROP_FRAME_COUNT:
        return duration;
    case CV_CAP_PROP_FORMAT:
    case CV_CAP_PROP_MODE:
    case CV_CAP_PROP_BRIGHTNESS:
    case CV_CAP_PROP_CONTRAST:
    case CV_CAP_PROP_SATURATION:
    case CV_CAP_PROP_HUE:
        if (v4l2src)
        {
            const gchar * propName =
                    propId == CV_CAP_PROP_BRIGHTNESS ? "brightness" :
                    propId == CV_CAP_PROP_CONTRAST ? "contrast" :
                    propId == CV_CAP_PROP_SATURATION ? "saturation" :
                    propId == CV_CAP_PROP_HUE ? "hue" : NULL;

            if (propName)
            {
                gint32 value32 = 0;
                g_object_get(G_OBJECT(v4l2src), propName, &value32, NULL);
                return value32;
            }
        }
    case CV_CAP_PROP_GAIN:
    case CV_CAP_PROP_CONVERT_RGB:
        break;
    case CV_CAP_GSTREAMER_QUEUE_LENGTH:
        if(!sink) {
            CV_WARN("GStreamer: there is no sink yet");
            return false;
        }
        return gst_app_sink_get_max_buffers(GST_APP_SINK(sink));
    default:
        CV_WARN("GStreamer: unhandled property");
        break;
    }

#undef FORMAT

    return 0;
}

/*!
 * \brief CvCapture_GStreamer::setProperty
 * \param propId
 * \param value
 * \return success
 * Sets the desired property id with val. If the pipeline is running,
 * it is briefly stopped and started again after the property was set
 */
bool CvCapture_GStreamer::setProperty( int propId, double value )
{
    GstFormat format;
    GstSeekFlags flags;

    if(!pipeline) {
        CV_WARN("GStreamer: no pipeline");
        return false;
    }

    bool wasPlaying = this->isPipelinePlaying();
    if (wasPlaying)
        this->stopPipeline();


    switch(propId) {
    case CV_CAP_PROP_POS_MSEC:
        format = GST_FORMAT_TIME;
        flags = (GstSeekFlags) (GST_SEEK_FLAG_FLUSH|GST_SEEK_FLAG_ACCURATE);
        if(!gst_element_seek_simple(GST_ELEMENT(pipeline), format,
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
        }
        format = GST_FORMAT_DEFAULT;
        flags = (GstSeekFlags) (GST_SEEK_FLAG_FLUSH|GST_SEEK_FLAG_ACCURATE);
        if(!gst_element_seek_simple(GST_ELEMENT(pipeline), format,
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
        format = GST_FORMAT_PERCENT;
        flags = (GstSeekFlags) (GST_SEEK_FLAG_FLUSH|GST_SEEK_FLAG_ACCURATE);
        if(!gst_element_seek_simple(GST_ELEMENT(pipeline), format,
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
            double num=0, denom = 1;
            toFraction(value, num,  denom);
            setFilter("framerate", GST_TYPE_FRACTION, value, denom);
        } else
            removeFilter("framerate");
        break;
    case CV_CAP_PROP_FOURCC:
    case CV_CAP_PROP_FRAME_COUNT:
    case CV_CAP_PROP_FORMAT:
    case CV_CAP_PROP_MODE:
    case CV_CAP_PROP_BRIGHTNESS:
    case CV_CAP_PROP_CONTRAST:
    case CV_CAP_PROP_SATURATION:
    case CV_CAP_PROP_HUE:
        if (v4l2src)
        {
            const gchar * propName =
                    propId == CV_CAP_PROP_BRIGHTNESS ? "brightness" :
                    propId == CV_CAP_PROP_CONTRAST ? "contrast" :
                    propId == CV_CAP_PROP_SATURATION ? "saturation" :
                    propId == CV_CAP_PROP_HUE ? "hue" : NULL;

            if (propName)
            {
                gint32 value32 = cv::saturate_cast<gint32>(value);
                g_object_set(G_OBJECT(v4l2src), propName, &value32, NULL);
                return true;
            }
        }
    case CV_CAP_PROP_GAIN:
    case CV_CAP_PROP_CONVERT_RGB:
        break;
    case CV_CAP_GSTREAMER_QUEUE_LENGTH:
        if(!sink)
            break;
        gst_app_sink_set_max_buffers(GST_APP_SINK(sink), (guint) value);
        break;
    default:
        CV_WARN("GStreamer: unhandled property");
    }

    if (wasPlaying)
        this->startPipeline();

    return false;
}

/*!
 * \brief cvCreateCapture_GStreamer
 * \param type
 * \param filename
 * \return
 */
CvCapture* cvCreateCapture_GStreamer(int type, const char* filename )
{
    CvCapture_GStreamer* capture = new CvCapture_GStreamer;

    if( capture->open( type, filename ))
        return capture;

    delete capture;
    return 0;
}


/*!
 * \brief The CvVideoWriter_GStreamer class
 * Use Gstreamer to write video
 */
class CvVideoWriter_GStreamer : public CvVideoWriter
{
public:
    CvVideoWriter_GStreamer() { init(); }
    virtual ~CvVideoWriter_GStreamer() { close(); }

    virtual bool open( const char* filename, int fourcc,
                       double fps, CvSize frameSize, bool isColor );
    virtual void close();
    virtual bool writeFrame( const IplImage* image );
protected:
    void init();
    const char* filenameToMimetype(const char* filename);
    GstElement* pipeline;
    GstElement* source;
    GstElement* encodebin;
    GstElement* file;

    GstBuffer* buffer;
    int input_pix_fmt;
    int num_frames;
    double framerate;
};

/*!
 * \brief CvVideoWriter_GStreamer::init
 * initialise all variables
 */
void CvVideoWriter_GStreamer::init()
{
    pipeline = NULL;
    source = NULL;
    encodebin = NULL;
    file = NULL;
    buffer = NULL;

    num_frames = 0;
    framerate = 0;
}

/*!
 * \brief CvVideoWriter_GStreamer::close
 * ends the pipeline by sending EOS and destroys the pipeline and all
 * elements afterwards
 */
void CvVideoWriter_GStreamer::close()
{
    GstStateChangeReturn status;
    if (pipeline)
    {
        handleMessage(pipeline);

        if (gst_app_src_end_of_stream(GST_APP_SRC(source)) != GST_FLOW_OK)
        {
            CV_WARN("Cannot send EOS to GStreamer pipeline\n");
            return;
        }

        //wait for EOS to trickle down the pipeline. This will let all elements finish properly
        GstBus* bus = gst_element_get_bus(pipeline);
        GstMessage *msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR)
        {
            CV_WARN("Error during VideoWriter finalization\n");
            return;
        }

        if(msg != NULL)
        {
            gst_message_unref(msg);
            g_object_unref(G_OBJECT(bus));
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
            gst_object_unref (GST_OBJECT (pipeline));
            pipeline = NULL;
            CV_WARN("Unable to stop gstreamer pipeline\n");
            return;
        }

        gst_object_unref (GST_OBJECT (pipeline));
        pipeline = NULL;
    }
}


/*!
 * \brief CvVideoWriter_GStreamer::filenameToMimetype
 * \param filename
 * \return mimetype
 * Resturns a container mime type for a given filename by looking at it's extension
 */
const char* CvVideoWriter_GStreamer::filenameToMimetype(const char *filename)
{
    //get extension
    const char *ext = strrchr(filename, '.');
    if(!ext || ext == filename) return NULL;
    ext += 1; //exclude the dot

    // return a container mime based on the given extension.
    // gstreamer's function returns too much possibilities, which is not useful to us

    //return the appropriate mime
    if (strncasecmp(ext,"avi", 3) == 0)
        return (const char*)"video/x-msvideo";

    if (strncasecmp(ext,"mkv", 3) == 0 || strncasecmp(ext,"mk3d",4) == 0  || strncasecmp(ext,"webm",4) == 0 )
        return (const char*)"video/x-matroska";

    if (strncasecmp(ext,"wmv", 3) == 0)
        return (const char*)"video/x-ms-asf";

    if (strncasecmp(ext,"mov", 3) == 0)
        return (const char*)"video/x-quicktime";

    if (strncasecmp(ext,"ogg", 3) == 0 || strncasecmp(ext,"ogv", 3) == 0)
        return (const char*)"application/ogg";

    if (strncasecmp(ext,"rm", 3) == 0)
        return (const char*)"vnd.rn-realmedia";

    if (strncasecmp(ext,"swf", 3) == 0)
        return (const char*)"application/x-shockwave-flash";

    if (strncasecmp(ext,"mp4", 3) == 0)
        return (const char*)"video/x-quicktime, variant=(string)iso";

    //default to avi
    return (const char*)"video/x-msvideo";
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
 * code, or enters a manual pipeline description like in CvVideoCapture_Gstreamer.
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
    CV_FUNCNAME("CvVideoWriter_GStreamer::open");

    // check arguments
    assert (filename);
    assert (fps > 0);
    assert (frameSize.width > 0  &&  frameSize.height > 0);

    // init gstreamer
    gst_initializer::init();

    // init vars
    bool manualpipeline = true;
    int  bufsize = 0;
    GError *err = NULL;
    const char* mime = NULL;
    GstStateChangeReturn stateret;

    GstCaps* caps = NULL;
    GstCaps* videocaps = NULL;

#if FULL_GST_VERSION >= VERSION_NUM(0,10,32)
    GstCaps* containercaps = NULL;
    GstEncodingContainerProfile* containerprofile = NULL;
    GstEncodingVideoProfile* videoprofile = NULL;
#endif

    GstIterator* it = NULL;
    gboolean done = FALSE;
    GstElement *element = NULL;
    gchar* name = NULL;

#if GST_VERSION_MAJOR == 0
    GstElement* splitter = NULL;
    GstElement* combiner = NULL;
#endif

    // we first try to construct a pipeline from the given string.
    // if that fails, we assume it is an ordinary filename

    __BEGIN__;

    encodebin = gst_parse_launch(filename, &err);
    manualpipeline = (encodebin != NULL);

    if(manualpipeline)
    {
#if GST_VERSION_MAJOR == 0
        it = gst_bin_iterate_sources(GST_BIN(encodebin));
        if(gst_iterator_next(it, (gpointer *)&source) != GST_ITERATOR_OK) {
            CV_ERROR(CV_StsError, "GStreamer: cannot find appsink in manual pipeline\n");
            return false;
        }
#else
        it = gst_bin_iterate_sources (GST_BIN(encodebin));
        GValue value = G_VALUE_INIT;

        while (!done) {
          switch (gst_iterator_next (it, &value)) {
            case GST_ITERATOR_OK:
              element = GST_ELEMENT (g_value_get_object (&value));
              name = gst_element_get_name(element);
              if (name){
                if(strstr(name, "opencvsrc") != NULL || strstr(name, "appsrc") != NULL) {
                  source = GST_ELEMENT ( gst_object_ref (element) );
                  done = TRUE;
                }
                g_free(name);
              }
              g_value_unset (&value);

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
            CV_ERROR(CV_StsError, "GStreamer: cannot find appsrc in manual pipeline\n");
            return false;
        }
#endif
        pipeline = encodebin;
    }
    else
    {
        pipeline = gst_pipeline_new (NULL);

        // we just got a filename and a fourcc code.
        // first, try to guess the container from the filename
        //encodebin = gst_element_factory_make("encodebin", NULL);

        //proxy old non existing fourcc ids. These were used in previous opencv versions,
        //but do not even exist in gstreamer any more
        if (fourcc == CV_FOURCC('M','P','1','V')) fourcc = CV_FOURCC('M', 'P', 'G' ,'1');
        if (fourcc == CV_FOURCC('M','P','2','V')) fourcc = CV_FOURCC('M', 'P', 'G' ,'2');
        if (fourcc == CV_FOURCC('D','R','A','C')) fourcc = CV_FOURCC('d', 'r', 'a' ,'c');


        //create encoder caps from fourcc

        videocaps = gst_riff_create_video_caps(fourcc, NULL, NULL, NULL, NULL, NULL);
        if (!videocaps){
            CV_ERROR( CV_StsUnsupportedFormat, "Gstreamer Opencv backend does not support this codec.");
        }

        //create container caps from file extension
        mime = filenameToMimetype(filename);
        if (!mime) {
            CV_ERROR( CV_StsUnsupportedFormat, "Gstreamer Opencv backend does not support this file type.");
        }

#if FULL_GST_VERSION >= VERSION_NUM(0,10,32)
        containercaps = gst_caps_from_string(mime);

        //create encodebin profile
        containerprofile = gst_encoding_container_profile_new("container", "container", containercaps, NULL);
        videoprofile = gst_encoding_video_profile_new(videocaps, NULL, NULL, 1);
        gst_encoding_container_profile_add_profile(containerprofile, (GstEncodingProfile *) videoprofile);
#endif

        //create pipeline elements
        encodebin = gst_element_factory_make("encodebin", NULL);

#if FULL_GST_VERSION >= VERSION_NUM(0,10,32)
        g_object_set(G_OBJECT(encodebin), "profile", containerprofile, NULL);
#endif
        source = gst_element_factory_make("appsrc", NULL);
        file = gst_element_factory_make("filesink", NULL);
        g_object_set(G_OBJECT(file), "location", filename, NULL);
    }

    if (fourcc == CV_FOURCC('M','J','P','G') && frameSize.height == 1)
    {
#if GST_VERSION_MAJOR > 0
        input_pix_fmt = GST_VIDEO_FORMAT_ENCODED;
        caps = gst_caps_new_simple("image/jpeg",
                                   "framerate", GST_TYPE_FRACTION, int(fps), 1,
                                   NULL);
        caps = gst_caps_fixate(caps);
#else
        CV_ERROR( CV_StsUnsupportedFormat, "Gstreamer 0.10 Opencv backend does not support writing encoded MJPEG data.");
#endif
    }
    else if(is_color)
    {
        input_pix_fmt = GST_VIDEO_FORMAT_BGR;
        bufsize = frameSize.width * frameSize.height * 3;

#if GST_VERSION_MAJOR == 0
        caps = gst_video_format_new_caps(GST_VIDEO_FORMAT_BGR,
                                         frameSize.width,
                                         frameSize.height,
                                         int(fps), 1,
                                         1, 1);
#else
        caps = gst_caps_new_simple("video/x-raw",
                                   "format", G_TYPE_STRING, "BGR",
                                   "width", G_TYPE_INT, frameSize.width,
                                   "height", G_TYPE_INT, frameSize.height,
                                   "framerate", GST_TYPE_FRACTION, int(fps), 1,
                                   NULL);
        caps = gst_caps_fixate(caps);

#endif

    }
    else
    {
#if FULL_GST_VERSION >= VERSION_NUM(0,10,29)
        input_pix_fmt = GST_VIDEO_FORMAT_GRAY8;
        bufsize = frameSize.width * frameSize.height;

#if GST_VERSION_MAJOR == 0
        caps = gst_video_format_new_caps(GST_VIDEO_FORMAT_GRAY8,
                                         frameSize.width,
                                         frameSize.height,
                                         int(fps), 1,
                                         1, 1);
#else
        caps = gst_caps_new_simple("video/x-raw",
                                   "format", G_TYPE_STRING, "GRAY8",
                                   "width", G_TYPE_INT, frameSize.width,
                                   "height", G_TYPE_INT, frameSize.height,
                                   "framerate", GST_TYPE_FRACTION, int(fps), 1,
                                   NULL);
        caps = gst_caps_fixate(caps);
#endif
#else
        CV_Assert(!"Gstreamer 0.10.29 or newer is required for grayscale input");
#endif
    }

    gst_app_src_set_caps(GST_APP_SRC(source), caps);
    gst_app_src_set_stream_type(GST_APP_SRC(source), GST_APP_STREAM_TYPE_STREAM);
    gst_app_src_set_size (GST_APP_SRC(source), -1);

    g_object_set(G_OBJECT(source), "format", GST_FORMAT_TIME, NULL);
    g_object_set(G_OBJECT(source), "block", 1, NULL);
    g_object_set(G_OBJECT(source), "is-live", 0, NULL);


    if(!manualpipeline)
    {
        g_object_set(G_OBJECT(file), "buffer-size", bufsize, NULL);
        gst_bin_add_many(GST_BIN(pipeline), source, encodebin, file, NULL);
        if(!gst_element_link_many(source, encodebin, file, NULL)) {
            CV_ERROR(CV_StsError, "GStreamer: cannot link elements\n");
        }
    }

#if GST_VERSION_MAJOR == 0
    // HACK: remove streamsplitter and streamcombiner from
    // encodebin pipeline to prevent early EOF event handling
    // We always fetch BGR or gray-scale frames, so combiner->spliter
    // endge in graph is useless.
    it = gst_bin_iterate_recurse (GST_BIN(encodebin));
    while (!done) {
      switch (gst_iterator_next (it, (void**)&element)) {
        case GST_ITERATOR_OK:
          name = gst_element_get_name(element);
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

    stateret = gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING);
    if(stateret  == GST_STATE_CHANGE_FAILURE) {
        handleMessage(pipeline);
        CV_ERROR(CV_StsError, "GStreamer: cannot put pipeline to play\n");
    }

    framerate = fps;
    num_frames = 0;

    handleMessage(pipeline);

    __END__;

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
    CV_FUNCNAME("CvVideoWriter_GStreamer::writerFrame");

    GstClockTime duration, timestamp;
    GstFlowReturn ret;
    int size;

    __BEGIN__;

    handleMessage(pipeline);

#if GST_VERSION_MAJOR > 0
    if (input_pix_fmt == GST_VIDEO_FORMAT_ENCODED) {
        if (image->nChannels != 1 || image->depth != IPL_DEPTH_8U || image->height != 1) {
            CV_ERROR(CV_StsUnsupportedFormat, "cvWriteFrame() needs images with depth = IPL_DEPTH_8U, nChannels = 1 and height = 1.");
        }
    }
    else
#endif
    if(input_pix_fmt == GST_VIDEO_FORMAT_BGR) {
        if (image->nChannels != 3 || image->depth != IPL_DEPTH_8U) {
            CV_ERROR(CV_StsUnsupportedFormat, "cvWriteFrame() needs images with depth = IPL_DEPTH_8U and nChannels = 3.");
        }
    }
#if FULL_GST_VERSION >= VERSION_NUM(0,10,29)
    else if (input_pix_fmt == GST_VIDEO_FORMAT_GRAY8) {
        if (image->nChannels != 1 || image->depth != IPL_DEPTH_8U) {
            CV_ERROR(CV_StsUnsupportedFormat, "cvWriteFrame() needs images with depth = IPL_DEPTH_8U and nChannels = 1.");
        }
    }
#endif
    else {
        CV_ERROR(CV_StsUnsupportedFormat, "cvWriteFrame() needs BGR or grayscale images\n");
        return false;
    }

    size = image->imageSize;
    duration = ((double)1/framerate) * GST_SECOND;
    timestamp = num_frames * duration;

    //gst_app_src_push_buffer takes ownership of the buffer, so we need to supply it a copy
#if GST_VERSION_MAJOR == 0
    buffer = gst_buffer_try_new_and_alloc (size);
    if (!buffer)
    {
        CV_ERROR(CV_StsBadSize, "Cannot create GStreamer buffer");
    }

    memcpy(GST_BUFFER_DATA (buffer), (guint8*)image->imageData, size);
    GST_BUFFER_DURATION(buffer) = duration;
    GST_BUFFER_TIMESTAMP(buffer) = timestamp;
#else
    buffer = gst_buffer_new_allocate (NULL, size, NULL);
    GstMapInfo info;
    gst_buffer_map(buffer, &info, (GstMapFlags)GST_MAP_READ);
    memcpy(info.data, (guint8*)image->imageData, size);
    gst_buffer_unmap(buffer, &info);
    GST_BUFFER_DURATION(buffer) = duration;
    GST_BUFFER_PTS(buffer) = timestamp;
    GST_BUFFER_DTS(buffer) = timestamp;
#endif
    //set the current number in the frame
    GST_BUFFER_OFFSET(buffer) =  num_frames;

    ret = gst_app_src_push_buffer(GST_APP_SRC(source), buffer);
    if (ret != GST_FLOW_OK) {
        CV_WARN("Error pushing buffer to GStreamer pipeline");
        return false;
    }

    //GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");

    ++num_frames;

    __END__;

    return true;
}

/*!
 * \brief cvCreateVideoWriter_GStreamer
 * \param filename
 * \param fourcc
 * \param fps
 * \param frameSize
 * \param isColor
 * \return
 * Constructor
 */
CvVideoWriter* cvCreateVideoWriter_GStreamer(const char* filename, int fourcc, double fps,
                                             CvSize frameSize, int isColor )
{
    CvVideoWriter_GStreamer* wrt = new CvVideoWriter_GStreamer;
    if( wrt->open(filename, fourcc, fps,frameSize, isColor))
        return wrt;

    delete wrt;
    return 0;
}

// utility functions

/*!
 * \brief toFraction
 * \param decimal
 * \param numerator
 * \param denominator
 * Split a floating point value into numerator and denominator
 */
void toFraction(double decimal, double &numerator, double &denominator)
{
    double dummy;
    double whole;
    decimal = modf (decimal, &whole);
    for (denominator = 1; denominator<=100; denominator++){
        if (modf(denominator * decimal, &dummy) < 0.001f)
            break;
    }
    numerator = denominator * decimal;
}


/*!
 * \brief handleMessage
 * Handles gstreamer bus messages. Mainly for debugging purposes and ensuring clean shutdown on error
 */
void handleMessage(GstElement * pipeline)
{
    CV_FUNCNAME("handlemessage");

    GError *err = NULL;
    gchar *debug = NULL;
    GstBus* bus = NULL;
    GstStreamStatusType tp;
    GstElement * elem = NULL;
    GstMessage* msg  = NULL;

    __BEGIN__;
    bus = gst_element_get_bus(pipeline);

    while(gst_bus_have_pending(bus)) {
        msg = gst_bus_pop(bus);

        //printf("\t\tGot %s message\n", GST_MESSAGE_TYPE_NAME(msg));

        if(gst_is_missing_plugin_message(msg))
        {
            CV_ERROR(CV_StsError, "GStreamer: your gstreamer installation is missing a required plugin\n");
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
                gst_message_parse_error(msg, &err, &debug);
                //fprintf(stderr, "\t\tGStreamer Plugin: Embedded video playback halted; module %s reported: %s\n",
                //                gst_element_get_name(GST_MESSAGE_SRC (msg)), err->message);

                g_error_free(err);
                g_free(debug);

                gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
                break;
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
        gst_message_unref(msg);
    }

    gst_object_unref(GST_OBJECT(bus));

    __END__
}
