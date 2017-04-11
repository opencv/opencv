////////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//

//
// The code has been contributed by Arkadiusz Raj on 2016 Oct
//

#include "precomp.hpp"

#ifdef HAVE_ARAVIS_API

#include <arv.h>

//
// This file provides wrapper for using Aravis SDK library to access GigE Vision cameras.
// Aravis library (version 0.4 or 0.6) shall be installed else this code will not be included in build.
//
// To include this module invoke cmake with -DWITH_ARAVIS=ON
//
// Please obvserve, that jumbo frames are required when high fps & 16bit data is selected.
// (camera, switches/routers and the computer this software is running on)
//
// Basic usage: VideoCapture cap(CAP_ARAVIS + <camera id>);
//
// Supported properties:
//  read/write
//      CAP_PROP_AUTO_EXPOSURE(0|1)
//      CAP_PROP_EXPOSURE(t), t in seconds
//      CAP_PROP_GAIN(g), g >=0 or -1 for automatic control if CAP_PROP_AUTO_EXPOSURE is true
//      CAP_PROP_FPS(f)
//      CAP_PROP_FOURCC(type)
//      CAP_PROP_BUFFERSIZE(n)
//  read only:
//      CAP_PROP_POS_MSEC
//      CAP_PROP_FRAME_WIDTH
//      CAP_PROP_FRAME_HEIGHT
//
//  Supported types of data:
//      video/x-raw, fourcc:'GREY'  -> 8bit, 1 channel
//      video/x-raw, fourcc:'Y800'  -> 8bit, 1 channel
//      video/x-raw, fourcc:'Y12 '  -> 12bit, 1 channel
//

#define MODE_GREY   CV_FOURCC_MACRO('G','R','E','Y')
#define MODE_Y800   CV_FOURCC_MACRO('Y','8','0','0')
#define MODE_Y12    CV_FOURCC_MACRO('Y','1','2',' ')

#define CLIP(a,b,c) (cv::max(cv::min((a),(c)),(b)))

/********************* Capturing video from camera via Aravis *********************/

class CvCaptureCAM_Aravis : public CvCapture
{
public:
    CvCaptureCAM_Aravis();
    virtual ~CvCaptureCAM_Aravis()
    {
        close();
    }

    virtual bool open(int);
    virtual void close();
    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
    virtual int getCaptureDomain()
    {
        return CV_CAP_ARAVIS;
    }

protected:
    bool create(int);
    bool init_buffers();

    void stopCapture();
    bool startCapture();

    bool getDeviceNameById(int id, std::string &device);

    void autoExposureControl(IplImage*);

    ArvCamera       *camera;                // Camera to control.
    ArvStream       *stream;                // Object for video stream reception.
    void            *framebuffer;           //

    unsigned int    payload;                // Width x height x Pixel width.

    int             widthMin;               // Camera sensor minium width.
    int             widthMax;               // Camera sensor maximum width.
    int             heightMin;              // Camera sensor minium height.
    int             heightMax;              // Camera sensor maximum height.
    bool            fpsAvailable;
    double          fpsMin;                 // Camera minium fps.
    double          fpsMax;                 // Camera maximum fps.
    bool            gainAvailable;
    double          gainMin;                // Camera minimum gain.
    double          gainMax;                // Camera maximum gain.
    bool            exposureAvailable;
    double          exposureMin;            // Camera's minimum exposure time.
    double          exposureMax;            // Camera's maximum exposure time.

    bool            controlExposure;        // Flag if automatic exposure shall be done by this SW
    bool            autoGain;
    double          targetGrey;             // Target grey value (mid grey))

    gint64          *pixelFormats;
    guint           pixelFormatsCnt;


    int             num_buffers;            // number of payload transmission buffers

    ArvPixelFormat  pixelFormat;            // pixel format

    int             xoffset;                // current frame region x offset
    int             yoffset;                // current frame region y offset
    int             width;                  // current frame width of frame
    int             height;                 // current frame height of image

    double          fps;                    // current value of fps
    double          exposure;               // current value of exposure time
    double          gain;                   // current value of gain
    double          midGrey;                // current value of mid grey (brightness)

    unsigned        frameID;                // current frame id
    unsigned        prevFrameID;

    IplImage        *frame;                 // local frame copy
};


CvCaptureCAM_Aravis::CvCaptureCAM_Aravis()
{
    camera = NULL;
    stream = NULL;
    framebuffer = NULL;

    payload = 0;

    widthMin = widthMax = heightMin = heightMax = 0;
    xoffset = yoffset = width = height = 0;
    fpsMin = fpsMax = gainMin = gainMax = exposureMin = exposureMax = 0;
    controlExposure = false;
    targetGrey = 0;
    frameID = prevFrameID = 0;

    num_buffers = 50;
    frame = NULL;
}

void CvCaptureCAM_Aravis::close()
{
    if(camera) {
        stopCapture();

        g_object_unref(camera);
        camera = NULL;
    }
}

bool CvCaptureCAM_Aravis::getDeviceNameById(int id, std::string &device)
{
    arv_update_device_list();

    if((id >= 0) && (id < (int)arv_get_n_devices())) {
        device = arv_get_device_id(id);
        return true;
    }

    return false;
}

bool CvCaptureCAM_Aravis::create( int index )
{
    std::string deviceName;
    if(!getDeviceNameById(index, deviceName))
        return false;

    return NULL != (camera = arv_camera_new(deviceName.c_str()));
}

bool CvCaptureCAM_Aravis::init_buffers()
{
    if(stream) {
        g_object_unref(stream);
        stream = NULL;
    }
    if( (stream = arv_camera_create_stream(camera, NULL, NULL)) ) {
        g_object_set(stream,
            "socket-buffer", ARV_GV_STREAM_SOCKET_BUFFER_AUTO,
            "socket-buffer-size", 0, NULL);
        g_object_set(stream,
            "packet-resend", ARV_GV_STREAM_PACKET_RESEND_NEVER, NULL);
        g_object_set(stream,
            "packet-timeout", (unsigned) 40000,
            "frame-retention", (unsigned) 200000, NULL);

        payload = arv_camera_get_payload (camera);

        for (int i = 0; i < num_buffers; i++)
            arv_stream_push_buffer(stream, arv_buffer_new(payload, NULL));

        return true;
    }

    return false;
}

bool CvCaptureCAM_Aravis::open( int index )
{
    if(create(index)) {
        // fetch properties bounds
        pixelFormats = arv_camera_get_available_pixel_formats(camera, &pixelFormatsCnt);

        arv_camera_get_width_bounds(camera, &widthMin, &widthMax);
        arv_camera_get_height_bounds(camera, &heightMin, &heightMax);
        arv_camera_set_region(camera, 0, 0, widthMax, heightMax);

        if( (fpsAvailable = arv_camera_is_frame_rate_available(camera)) )
            arv_camera_get_frame_rate_bounds(camera, &fpsMin, &fpsMax);
        if( (gainAvailable = arv_camera_is_gain_available(camera)) )
            arv_camera_get_gain_bounds (camera, &gainMin, &gainMax);
        if( (exposureAvailable = arv_camera_is_exposure_time_available(camera)) )
            arv_camera_get_exposure_time_bounds (camera, &exposureMin, &exposureMax);

        // get initial values
        pixelFormat = arv_camera_get_pixel_format(camera);
        exposure = exposureAvailable ? arv_camera_get_exposure_time(camera) : 0;
        gain = gainAvailable ? arv_camera_get_gain(camera) : 0;
        fps = arv_camera_get_frame_rate(camera);

        return startCapture();
    }
    return false;
}

bool CvCaptureCAM_Aravis::grabFrame()
{
    // remove content of previous frame
    framebuffer = NULL;

    if(stream) {
        ArvBuffer *arv_buffer = NULL;
        int max_tries = 10;
        int tries = 0;
        for(; tries < max_tries; tries ++) {
            arv_buffer = arv_stream_timeout_pop_buffer (stream, 200000);
            if (arv_buffer != NULL && arv_buffer_get_status (arv_buffer) != ARV_BUFFER_STATUS_SUCCESS) {
                arv_stream_push_buffer (stream, arv_buffer);
            } else break;
        }
        if(arv_buffer != NULL && tries < max_tries) {
            size_t buffer_size;
            framebuffer = (void*)arv_buffer_get_data (arv_buffer, &buffer_size);

            // retieve image size properites
            arv_buffer_get_image_region (arv_buffer, &xoffset, &yoffset, &width, &height);

            // retieve image ID set by camera
            frameID = arv_buffer_get_frame_id(arv_buffer);

            arv_stream_push_buffer(stream, arv_buffer);
            return true;
        }
    }
    return false;
}

IplImage* CvCaptureCAM_Aravis::retrieveFrame(int)
{
    if(framebuffer) {
        int depth = 0, channels = 0;
        switch(pixelFormat) {
            case ARV_PIXEL_FORMAT_MONO_8:
                depth = IPL_DEPTH_8U;
                channels = 1;
                break;
            case ARV_PIXEL_FORMAT_MONO_12:
                depth = IPL_DEPTH_16U;
                channels = 1;
                break;
        }
        if(depth && channels) {
            IplImage src;
            cvInitImageHeader( &src, cvSize( width, height ), depth, channels, IPL_ORIGIN_TL, 4 );

            cvSetData( &src, framebuffer, src.widthStep );
            if( !frame ||
                 frame->width != src.width ||
                 frame->height != src.height ||
                 frame->depth != src.depth ||
                 frame->nChannels != src.nChannels) {

                cvReleaseImage( &frame );
                frame = cvCreateImage( cvGetSize(&src), src.depth, channels );
            }
            cvCopy(&src, frame);

            if(controlExposure && ((frameID - prevFrameID) > 1)) {
                // control exposure every second frame
                // i.e. skip frame taken with previous exposure setup
                autoExposureControl(frame);
            }

            return frame;
        }
    }
    return NULL;
}

void CvCaptureCAM_Aravis::autoExposureControl(IplImage* image)
{
    // Software control of exposure parameters utilizing
    // automatic change of exposure time & gain

    // Priority is set as follows:
    // - to increase brightness, first increase time then gain
    // - to decrease brightness, first decrease gain then time

    cv::Mat m = cv::cvarrToMat(image);

    // calc mean value for luminance or green channel
    double brightness = cv::mean(m)[image->nChannels > 1 ? 1 : 0];
    if(brightness < 1) brightness = 1;

    // mid point - 100 % means no change
    static const double dmid = 100;

    // distance from optimal value as a percentage
    double d = (targetGrey * dmid) / brightness;
    if(d >= dmid) d = ( d + (dmid * 2) ) / 3;

    prevFrameID = frameID;
    midGrey = brightness;

    double maxe = 1e6 / fps;
    double ne = CLIP( ( exposure * d ) / dmid, exposureMin, maxe);

    // if change of value requires intervention
    if(fabs(d-dmid) > 5) {
        double ev, ng = 0;

        if(gainAvailable && autoGain) {
            ev = log( d / dmid ) / log(2);
            ng = CLIP( gain + ev, gainMin, gainMax);

            if( ng < gain ) {
                // piority 1 - reduce gain
                arv_camera_set_gain(camera, (gain = ng));
                return;
            }
        }

        if(exposureAvailable) {
            if(abs(exposure - ne) > 2) {
                // priority 2 - control of exposure time
                arv_camera_set_exposure_time(camera, (exposure = ne) );
                return;
            }
        }

        if(gainAvailable && autoGain) {
            if(exposureAvailable) {
                // exposure at maximum - increase gain if possible
                if(ng > gain && ng < gainMax && ne >= maxe) {
                    arv_camera_set_gain(camera, (gain = ng));
                    return;
                }
            } else {
                // priority 3 - increase gain
                arv_camera_set_gain(camera, (gain = ng));
                return;
            }
        }
    }

    // if gain can be reduced - do it
    if(gainAvailable && autoGain && exposureAvailable) {
        if(gain > gainMin && exposure < maxe) {
            exposure = CLIP( ne * 1.05, exposureMin, maxe);
            arv_camera_set_exposure_time(camera, exposure );
        }
    }
}

double CvCaptureCAM_Aravis::getProperty( int property_id ) const
{
    switch(property_id) {
        case CV_CAP_PROP_POS_MSEC:
            return (double)frameID/fps;

        case CV_CAP_PROP_FRAME_WIDTH:
            return width;

        case CV_CAP_PROP_FRAME_HEIGHT:
            return height;

        case CV_CAP_PROP_AUTO_EXPOSURE:
            return (controlExposure ? 1 : 0);

        case CV_CAP_PROP_EXPOSURE:
            if(exposureAvailable) {
                /* exposure time in seconds, like 1/100 s */
                return arv_camera_get_exposure_time(camera) / 1e6;
            }
            break;

        case CV_CAP_PROP_FPS:
            if(fpsAvailable) {
                return arv_camera_get_frame_rate(camera);
            }
            break;

        case CV_CAP_PROP_GAIN:
            if(gainAvailable) {
                return arv_camera_get_gain(camera);
            }
            break;

        case CV_CAP_PROP_FOURCC:
            {
                ArvPixelFormat currFormat = arv_camera_get_pixel_format(camera);
                switch( currFormat ) {
                    case ARV_PIXEL_FORMAT_MONO_8:
                        return MODE_Y800;
                    case ARV_PIXEL_FORMAT_MONO_12:
                        return MODE_Y12;
                }
            }
            break;

        case CV_CAP_PROP_BUFFERSIZE:
            if(stream) {
                int in, out;
                arv_stream_get_n_buffers(stream, &in, &out);
                // return number of available buffers in Aravis output queue
                return out;
            }
            break;
    }
    return -1.0;
}

bool CvCaptureCAM_Aravis::setProperty( int property_id, double value )
{
    switch(property_id) {
        case CV_CAP_PROP_AUTO_EXPOSURE:
            if(exposureAvailable || gainAvailable) {
                if( (controlExposure = (bool)(int)value) ) {
                    exposure = exposureAvailable ? arv_camera_get_exposure_time(camera) : 0;
                    gain = gainAvailable ? arv_camera_get_gain(camera) : 0;
                }
            }
            break;

        case CV_CAP_PROP_EXPOSURE:
            if(exposureAvailable) {
                /* exposure time in seconds, like 1/100 s */
                value *= 1e6; // -> from s to us

                arv_camera_set_exposure_time(camera, exposure = CLIP(value, exposureMin, exposureMax));
                break;
            } else return false;

        case CV_CAP_PROP_FPS:
            if(fpsAvailable) {
                arv_camera_set_frame_rate(camera, fps = CLIP(value, fpsMin, fpsMax));
                break;
            } else return false;

        case CV_CAP_PROP_GAIN:
            if(gainAvailable) {
                if ( (autoGain = (-1 == value) ) )
                    break;

                arv_camera_set_gain(camera, gain = CLIP(value, gainMin, gainMax));
                break;
            } else return false;

        case CV_CAP_PROP_FOURCC:
            {
                ArvPixelFormat newFormat = pixelFormat;
                switch((int)value) {
                    case MODE_GREY:
                    case MODE_Y800:
                        newFormat = ARV_PIXEL_FORMAT_MONO_8;
                        targetGrey = 128;
                        break;
                    case MODE_Y12:
                        newFormat = ARV_PIXEL_FORMAT_MONO_12;
                        targetGrey = 2048;
                        break;
                }
                if(newFormat != pixelFormat) {
                    stopCapture();
                    arv_camera_set_pixel_format(camera, pixelFormat = newFormat);
                    startCapture();
                }
            }
            break;

        case CV_CAP_PROP_BUFFERSIZE:
            {
                int x = (int)value;
                if((x > 0) && (x != num_buffers)) {
                    stopCapture();
                    num_buffers = x;
                    startCapture();
                }
            }
            break;


        default:
            return false;
    }

    return true;
}

void CvCaptureCAM_Aravis::stopCapture()
{
    arv_camera_stop_acquisition(camera);

    if(stream) {
        g_object_unref(stream);
        stream = NULL;
    }
}

bool CvCaptureCAM_Aravis::startCapture()
{
    if(init_buffers() ) {
        arv_camera_set_acquisition_mode(camera, ARV_ACQUISITION_MODE_CONTINUOUS);
        arv_device_set_string_feature_value(arv_camera_get_device (camera), "TriggerMode" , "Off");
        arv_camera_start_acquisition(camera);

        return true;
    }
    return false;
}

CvCapture* cvCreateCameraCapture_Aravis( int index )
{
    CvCaptureCAM_Aravis* capture = new CvCaptureCAM_Aravis;

    if(capture->open(index)) {
        return capture;
    }

    delete capture;
    return NULL;
}
#endif
