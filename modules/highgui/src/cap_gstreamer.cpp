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

// Author: Nils Hasler <hasler@mpi-inf.mpg.de>
//
//         Max-Planck-Institut Informatik
//
// this implementation was inspired by gnash's gstreamer interface

//
// use GStreamer to read a video
//

#include "precomp.hpp"
#include <unistd.h>
#include <string.h>
#include <map>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/riff/riff-media.h>

#ifdef NDEBUG
#define CV_WARN(message)
#else
#define CV_WARN(message) fprintf(stderr, "warning: %s (%s:%d)\n", message, __FILE__, __LINE__)
#endif

static bool isInited = false;
class CvCapture_GStreamer : public CvCapture
{
public:
    CvCapture_GStreamer() { init(); }
    virtual ~CvCapture_GStreamer() { close(); }

    virtual bool open( int type, const char* filename );
    virtual void close();

    virtual double getProperty(int);
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);

protected:
    void init();
    bool reopen();
    void handleMessage();
    void restartPipeline();
    void setFilter(const char*, int, int, int);
    void removeFilter(const char *filter);
    void static newPad(GstElement *myelement,
                             GstPad     *pad,
                             gpointer    data);
    GstElement           *pipeline;
    GstElement           *uridecodebin;
    GstElement           *color;
    GstElement           *sink;

    GstBuffer           *buffer;
    GstCaps            *caps;
    IplImage           *frame;
};

void CvCapture_GStreamer::init()
{
    pipeline=0;
    frame=0;
    buffer=0;
    frame=0;

}

void CvCapture_GStreamer::handleMessage()
{
    GstBus* bus = gst_element_get_bus(pipeline);

    while(gst_bus_have_pending(bus)) {
        GstMessage* msg = gst_bus_pop(bus);

//        printf("Got %s message\n", GST_MESSAGE_TYPE_NAME(msg));

        switch (GST_MESSAGE_TYPE (msg)) {
        case GST_MESSAGE_STATE_CHANGED:
            GstState oldstate, newstate, pendstate;
            gst_message_parse_state_changed(msg, &oldstate, &newstate, &pendstate);
//            printf("state changed from %d to %d (%d)\n", oldstate, newstate, pendstate);
            break;
        case GST_MESSAGE_ERROR: {
            GError *err;
            gchar *debug;
            gst_message_parse_error(msg, &err, &debug);

            fprintf(stderr, "GStreamer Plugin: Embedded video playback halted; module %s reported: %s\n",
                  gst_element_get_name(GST_MESSAGE_SRC (msg)), err->message);

            g_error_free(err);
            g_free(debug);

            gst_element_set_state(pipeline, GST_STATE_NULL);

            break;
            }
        case GST_MESSAGE_EOS:
//            CV_WARN("NetStream has reached the end of the stream.");

            break;
        default:
//            CV_WARN("unhandled message\n");
            break;
        }

        gst_message_unref(msg);
    }

    gst_object_unref(GST_OBJECT(bus));
}

//
// start the pipeline, grab a buffer, and pause again
//
bool CvCapture_GStreamer::grabFrame()
{

    if(!pipeline)
        return false;

    if(gst_app_sink_is_eos(GST_APP_SINK(sink)))
        return false;

    if(buffer)
        gst_buffer_unref(buffer);
    handleMessage();

    buffer = gst_app_sink_pull_buffer(GST_APP_SINK(sink));
    if(!buffer)
        return false;

    return true;
}

//
// decode buffer
//
IplImage * CvCapture_GStreamer::retrieveFrame(int)
{
    if(!buffer)
        return false;

    if(!frame) {
        gint height, width;
        GstCaps *buff_caps = gst_buffer_get_caps(buffer);
        assert(gst_caps_get_size(buff_caps) == 1);
        GstStructure* structure = gst_caps_get_structure(buff_caps, 0);

        if(!gst_structure_get_int(structure, "width", &width) ||
           !gst_structure_get_int(structure, "height", &height))
            return false;

        frame = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_8U, 3);
        gst_caps_unref(buff_caps);
    }

    // no need to memcpy, just use gstreamer's buffer :-)
    frame->imageData = (char *)GST_BUFFER_DATA(buffer);
    //memcpy (frame->imageData, GST_BUFFER_DATA(buffer), GST_BUFFER_SIZE (buffer));
    //gst_buffer_unref(buffer);
    //buffer = 0;
    return frame;
}

void CvCapture_GStreamer::restartPipeline()
{
    CV_FUNCNAME("icvRestartPipeline");

    __BEGIN__;

    printf("restarting pipeline, going to ready\n");

    if(gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_READY) ==
       GST_STATE_CHANGE_FAILURE) {
        CV_ERROR(CV_StsError, "GStreamer: unable to start pipeline\n");
        return;
    }

    printf("ready, relinking\n");

    gst_element_unlink(uridecodebin, color);
    printf("filtering with %s\n", gst_caps_to_string(caps));
    gst_element_link_filtered(uridecodebin, color, caps);

    printf("relinked, pausing\n");

    if(gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING) ==
       GST_STATE_CHANGE_FAILURE) {
        CV_ERROR(CV_StsError, "GStreamer: unable to start pipeline\n");
        return;
    }

    printf("state now paused\n");

     __END__;
}

void CvCapture_GStreamer::setFilter(const char *property, int type, int v1, int v2)
{

    if(!caps) {
        if(type == G_TYPE_INT)
            caps = gst_caps_new_simple("video/x-raw-rgb", property, type, v1, NULL);
        else
            caps = gst_caps_new_simple("video/x-raw-rgb", property, type, v1, v2, NULL);
    } else {
        //printf("caps before setting %s\n", gst_caps_to_string(caps));
        if(type == G_TYPE_INT)
            gst_caps_set_simple(caps, "video/x-raw-rgb", property, type, v1, NULL);
        else
            gst_caps_set_simple(caps, "video/x-raw-rgb", property, type, v1, v2, NULL);
    }

    restartPipeline();
}

void CvCapture_GStreamer::removeFilter(const char *filter)
{
    if(!caps)
        return;

    GstStructure *s = gst_caps_get_structure(caps, 0);
    gst_structure_remove_field(s, filter);

    restartPipeline();
}


//
// connect uridecodebin dynamically created source pads to colourconverter
//
void CvCapture_GStreamer::newPad(GstElement *uridecodebin,
                             GstPad     *pad,
                             gpointer    data)
{
  GstPad *sinkpad;
  GstElement *color = (GstElement *) data;


  sinkpad = gst_element_get_static_pad (color, "sink");
  
//  printf("linking dynamic pad to colourconverter %p %p\n", uridecodebin, pad);

  gst_pad_link (pad, sinkpad);

  gst_object_unref (sinkpad);
}

bool CvCapture_GStreamer::open( int type, const char* filename )
{
    close();
    CV_FUNCNAME("cvCaptureFromCAM_GStreamer");

    __BEGIN__;

    if(!isInited) {
//        printf("gst_init\n");
        gst_init (NULL, NULL);

//        gst_debug_set_active(TRUE);
//        gst_debug_set_colored(TRUE);
//        gst_debug_set_default_threshold(GST_LEVEL_WARNING);

        isInited = true;
    }
    bool stream = false;
    bool manualpipeline = false;
    char *uri = NULL;
    uridecodebin = NULL;
    if(type != CV_CAP_GSTREAMER_FILE) {
        close();
        return false;
    }

    if(!gst_uri_is_valid(filename)) {
        uri = realpath(filename, NULL);
        stream=false;
        if(uri) {
            uri = g_filename_to_uri(uri, NULL, NULL);
            if(!uri) {
                CV_WARN("GStreamer: Error opening file\n");
                close();
                return false;
            }
        } else {
            GError *err = NULL;
            //uridecodebin = gst_parse_bin_from_description(filename, FALSE, &err);
            uridecodebin = gst_parse_launch(filename, &err);
            if(!uridecodebin) {
                CV_WARN("GStreamer: Error opening bin\n");
                close();
                return false;
            }
            stream = true;
            manualpipeline = true;
        }
    } else {
        stream = true;
        uri = g_strdup(filename);
    }

    if(!uridecodebin) {
        uridecodebin = gst_element_factory_make ("uridecodebin", NULL);
        g_object_set(G_OBJECT(uridecodebin),"uri",uri, NULL);
    }
    if(!uridecodebin) {
        CV_WARN("GStreamer: Failed to create uridecodebin\n");
        close();
        return false;
    }

    if(manualpipeline) {
        GstIterator *it = gst_bin_iterate_sinks(GST_BIN(uridecodebin));
        if(gst_iterator_next(it, (gpointer *)&sink) != GST_ITERATOR_OK) {
	    CV_ERROR(CV_StsError, "GStreamer: cannot find appsink in manual pipeline\n");
	    return false;
        }

	pipeline = uridecodebin;
    } else {
	pipeline = gst_pipeline_new (NULL);

        color = gst_element_factory_make("ffmpegcolorspace", NULL);
        sink = gst_element_factory_make("appsink", NULL);

        gst_bin_add_many(GST_BIN(pipeline), uridecodebin, color, sink, NULL);
        g_signal_connect(uridecodebin, "pad-added", G_CALLBACK(newPad), color);

        if(!gst_element_link(color, sink)) {
            CV_ERROR(CV_StsError, "GStreamer: cannot link color -> sink\n");
            gst_object_unref(pipeline);
            return false;
        }
    }

    gst_app_sink_set_max_buffers (GST_APP_SINK(sink), 1);
    gst_app_sink_set_drop (GST_APP_SINK(sink), stream);

    {
    GstCaps* caps;
    caps = gst_caps_new_simple("video/x-raw-rgb",
                               "red_mask",   G_TYPE_INT, 0x0000FF,
                               "green_mask", G_TYPE_INT, 0x00FF00,
                               "blue_mask",  G_TYPE_INT, 0xFF0000,
                               NULL);
    gst_app_sink_set_caps(GST_APP_SINK(sink), caps);
    gst_caps_unref(caps);
    }

    if(gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_READY) ==
       GST_STATE_CHANGE_FAILURE) {
        CV_WARN("GStreamer: unable to set pipeline to ready\n");
        gst_object_unref(pipeline);
        return false;
    }

    if(gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING) ==
       GST_STATE_CHANGE_FAILURE) {
        CV_WARN("GStreamer: unable to set pipeline to playing\n");
        gst_object_unref(pipeline);
        return false;
    }



    handleMessage();

    __END__;

    return true;
}

//
//
// gstreamer image sequence writer
//
//
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
    std::map<int, char*> encs;
    GstElement* source;
    GstElement* file;
    GstElement* enc;
    GstElement* mux;
    GstElement* color;
    GstBuffer* buffer;
    GstElement* pipeline;
    int input_pix_fmt;
};

void CvVideoWriter_GStreamer::init()
{
    encs[CV_FOURCC('H','F','Y','U')]=(char*)"ffenc_huffyuv";
    encs[CV_FOURCC('D','R','A','C')]=(char*)"diracenc";
    encs[CV_FOURCC('X','V','I','D')]=(char*)"xvidenc";
    encs[CV_FOURCC('X','2','6','4')]=(char*)"x264enc";
    encs[CV_FOURCC('M','P','1','V')]=(char*)"mpeg2enc";
    //encs[CV_FOURCC('M','P','2','V')]=(char*)"mpeg2enc";
    pipeline=0;
    buffer=0;
}
void CvVideoWriter_GStreamer::close()
{
    if (pipeline) {
        gst_app_src_end_of_stream(GST_APP_SRC(source));
        gst_element_set_state (pipeline, GST_STATE_NULL);
        gst_object_unref (GST_OBJECT (pipeline));
    }
}
bool CvVideoWriter_GStreamer::open( const char * filename, int fourcc,
        double fps, CvSize frameSize, bool is_color )
{
    CV_FUNCNAME("CvVideoWriter_GStreamer::open");

    __BEGIN__;
    //actually doesn't support fourcc parameter and encode an avi with jpegenc
    //we need to find a common api between backend to support fourcc for avi
    //but also to choose in a common way codec and container format (ogg,dirac,matroska)
    // check arguments

    assert (filename);
    assert (fps > 0);
    assert (frameSize.width > 0  &&  frameSize.height > 0);
    std::map<int,char*>::iterator encit;
    encit=encs.find(fourcc);
    if (encit==encs.end())
        CV_ERROR( CV_StsUnsupportedFormat,"Gstreamer Opencv backend doesn't support this codec acutally.");
    if(!isInited) {
        gst_init (NULL, NULL);
        isInited = true;
    }
    close();
    source=gst_element_factory_make("appsrc",NULL);
    file=gst_element_factory_make("filesink", NULL);
    enc=gst_element_factory_make(encit->second, NULL);
    mux=gst_element_factory_make("avimux", NULL);
    color = gst_element_factory_make("ffmpegcolorspace", NULL);
    if (!enc)
        CV_ERROR( CV_StsUnsupportedFormat, "Your version of Gstreamer doesn't support this codec acutally or needed plugin missing.");
    g_object_set(G_OBJECT(file), "location", filename, NULL);
    pipeline = gst_pipeline_new (NULL);
    GstCaps* caps;
    if (is_color) {
        input_pix_fmt=1;
        caps= gst_video_format_new_caps(GST_VIDEO_FORMAT_BGR,
                                        frameSize.width,
                                        frameSize.height,
                                        (int) (fps * 1000),
                                        1000,
                                        1,
                                        1);
    }
    else  {
        input_pix_fmt=0;
        caps= gst_caps_new_simple("video/x-raw-gray",
                                  "width", G_TYPE_INT, frameSize.width,
                                  "height", G_TYPE_INT, frameSize.height,
                                  "framerate", GST_TYPE_FRACTION, int(fps),1,
                                  "bpp",G_TYPE_INT,8,
                                  "depth",G_TYPE_INT,8,
                                  NULL);
    }
    gst_app_src_set_caps(GST_APP_SRC(source), caps);
    if (fourcc==CV_FOURCC_DEFAULT) {
        gst_bin_add_many(GST_BIN(pipeline), source, color,mux, file, NULL);
        if(!gst_element_link_many(source,color,enc,mux,file,NULL)) {
            CV_ERROR(CV_StsError, "GStreamer: cannot link elements\n");
        }
    }
    else {
        gst_bin_add_many(GST_BIN(pipeline), source, color,enc,mux, file, NULL);
        if(!gst_element_link_many(source,color,enc,mux,file,NULL)) {
            CV_ERROR(CV_StsError, "GStreamer: cannot link elements\n");
        }
    }


    if(gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING) ==
        GST_STATE_CHANGE_FAILURE) {
            CV_ERROR(CV_StsError, "GStreamer: cannot put pipeline to play\n");
    }
    __END__;
    return true;
}
bool CvVideoWriter_GStreamer::writeFrame( const IplImage * image )
{

    CV_FUNCNAME("CvVideoWriter_GStreamer::writerFrame");

    __BEGIN__;
    if (input_pix_fmt == 1) {
        if (image->nChannels != 3 || image->depth != IPL_DEPTH_8U) {
            CV_ERROR(CV_StsUnsupportedFormat, "cvWriteFrame() needs images with depth = IPL_DEPTH_8U and nChannels = 3.");
        }
    }
    else if (input_pix_fmt == 0) {
        if (image->nChannels != 1 || image->depth != IPL_DEPTH_8U) {
            CV_ERROR(CV_StsUnsupportedFormat, "cvWriteFrame() needs images with depth = IPL_DEPTH_8U and nChannels = 1.");
        }
    }
    else {
        assert(false);
    }
    int size;
    size = image->imageSize;
    buffer = gst_buffer_new_and_alloc (size);
    //gst_buffer_set_data (buffer,(guint8*)image->imageData, size);
    memcpy (GST_BUFFER_DATA(buffer),image->imageData, size);
    gst_app_src_push_buffer(GST_APP_SRC(source),buffer);
    //gst_buffer_unref(buffer);
    //buffer = 0;
    __END__;
    return true;
}
CvVideoWriter* cvCreateVideoWriter_GStreamer(const char* filename, int fourcc, double fps,
                                           CvSize frameSize, int isColor )
{
    CvVideoWriter_GStreamer* wrt = new CvVideoWriter_GStreamer;
    if( wrt->open(filename, fourcc, fps,frameSize, isColor))
        return wrt;

    delete wrt;
    return false;
}

void CvCapture_GStreamer::close()
{
    if(pipeline) {
        gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
    }
    if(buffer)
        gst_buffer_unref(buffer);
    if(frame) {
        frame->imageData = 0;
        cvReleaseImage(&frame);
    }
}

double CvCapture_GStreamer::getProperty( int propId )
{
    GstFormat format;
    //GstQuery q;
    gint64 value;

    if(!pipeline) {
        CV_WARN("GStreamer: no pipeline");
        return false;
    }

    switch(propId) {
    case CV_CAP_PROP_POS_MSEC:
        format = GST_FORMAT_TIME;
        if(!gst_element_query_position(sink, &format, &value)) {
            CV_WARN("GStreamer: unable to query position of stream");
            return false;
        }
        return value * 1e-6; // nano seconds to milli seconds
    case CV_CAP_PROP_POS_FRAMES:
        format = GST_FORMAT_DEFAULT;
        if(!gst_element_query_position(sink, &format, &value)) {
            CV_WARN("GStreamer: unable to query position of stream");
            return false;
        }
        return value;
    case CV_CAP_PROP_POS_AVI_RATIO:
        format = GST_FORMAT_PERCENT;
        if(!gst_element_query_position(pipeline, &format, &value)) {
            CV_WARN("GStreamer: unable to query position of stream");
            return false;
        }
        return ((double) value) / GST_FORMAT_PERCENT_MAX;
    case CV_CAP_PROP_FRAME_WIDTH:
    case CV_CAP_PROP_FRAME_HEIGHT:
    case CV_CAP_PROP_FPS:
    case CV_CAP_PROP_FOURCC:
        break;
    case CV_CAP_PROP_FRAME_COUNT:
        format = GST_FORMAT_DEFAULT;
        if(!gst_element_query_duration(pipeline, &format, &value)) {
            CV_WARN("GStreamer: unable to query position of stream");
            return false;
        }
        return value;
    case CV_CAP_PROP_FORMAT:
    case CV_CAP_PROP_MODE:
    case CV_CAP_PROP_BRIGHTNESS:
    case CV_CAP_PROP_CONTRAST:
    case CV_CAP_PROP_SATURATION:
    case CV_CAP_PROP_HUE:
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
    return false;
}

bool CvCapture_GStreamer::setProperty( int propId, double value )
{
       GstFormat format;
    GstSeekFlags flags;

    if(!pipeline) {
        CV_WARN("GStreamer: no pipeline");
        return false;
    }

    switch(propId) {
    case CV_CAP_PROP_POS_MSEC:
        format = GST_FORMAT_TIME;
        flags = (GstSeekFlags) (GST_SEEK_FLAG_FLUSH|GST_SEEK_FLAG_ACCURATE);
        if(!gst_element_seek_simple(GST_ELEMENT(pipeline), format,
                        flags, (gint64) (value * GST_MSECOND))) {
            CV_WARN("GStreamer: unable to seek");
        }
        break;
    case CV_CAP_PROP_POS_FRAMES:
        format = GST_FORMAT_DEFAULT;
        flags = (GstSeekFlags) (GST_SEEK_FLAG_FLUSH|GST_SEEK_FLAG_ACCURATE);
        if(!gst_element_seek_simple(GST_ELEMENT(pipeline), format,
                        flags, (gint64) value)) {
            CV_WARN("GStreamer: unable to seek");
        }
        break;
    case CV_CAP_PROP_POS_AVI_RATIO:
        format = GST_FORMAT_PERCENT;
        flags = (GstSeekFlags) (GST_SEEK_FLAG_FLUSH|GST_SEEK_FLAG_ACCURATE);
        if(!gst_element_seek_simple(GST_ELEMENT(pipeline), format,
                        flags, (gint64) (value * GST_FORMAT_PERCENT_MAX))) {
            CV_WARN("GStreamer: unable to seek");
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
            int num, denom;
            num = (int) value;
            if(value != num) { // FIXME this supports only fractions x/1 and x/2
                num = (int) (value * 2);
                denom = 2;
            } else
                denom = 1;

            setFilter("framerate", GST_TYPE_FRACTION, num, denom);
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
    return false;
}
CvCapture* cvCreateCapture_GStreamer(int type, const char* filename )
{
    CvCapture_GStreamer* capture = new CvCapture_GStreamer;

    if( capture->open( type, filename ))
        return capture;

    delete capture;
    return false;
}
