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
#include "cap_interface.hpp"

using namespace cv;

#ifdef HAVE_ARAVIS_API

#include <arv.h>

//
// This file provides wrapper for using Aravis SDK library to access GigE and USB 3 Vision cameras.
// Aravis library (version 0.8) shall be installed else this code will not be included in build.
//
// To include this module invoke cmake with -DWITH_ARAVIS=ON
//
// Please obvserve, that jumbo frames are required when high fps & 16bit data is selected.
// (camera, switches/routers and the computer this software is running on)
//
// Basic usage: VideoCapture cap(<camera id>, CAP_ARAVIS);
//
// Supported properties:
//  read/write
//      CAP_PROP_AUTO_EXPOSURE(0|1)
//      CAP_PROP_EXPOSURE(t), t in seconds
//      CAP_PROP_BRIGHTNESS (ev), exposure compensation in EV for auto exposure algorithm
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
//      video/x-raw, fourcc:'Y16 '  -> 16bit, 1 channel
//      video/x-raw, fourcc:'GRBG'  -> 8bit, 1 channel
//

#define MODE_GREY   CV_FOURCC_MACRO('G','R','E','Y')
#define MODE_Y800   CV_FOURCC_MACRO('Y','8','0','0')
#define MODE_Y12    CV_FOURCC_MACRO('Y','1','2',' ')
#define MODE_Y16    CV_FOURCC_MACRO('Y','1','6',' ')
#define MODE_GRBG   CV_FOURCC_MACRO('G','R','B','G')

#define CLIP(a,b,c) (cv::max(cv::min((a),(c)),(b)))

/********************* Capturing video from camera via Aravis *********************/

class CvCaptureCAM_Aravis : public IVideoCapture
{
public:
    CvCaptureCAM_Aravis();
    ~CvCaptureCAM_Aravis()
    {
        close();
    }

    bool open(int);
    void close();
    double getProperty(int) const CV_OVERRIDE;
    bool setProperty(int, double) CV_OVERRIDE;
    bool grabFrame() CV_OVERRIDE;
    bool retrieveFrame(int, OutputArray) CV_OVERRIDE;
    int getCaptureDomain() CV_OVERRIDE
    {
        return cv::CAP_ARAVIS;
    }
    bool isOpened() const CV_OVERRIDE { return stream != NULL; }

protected:
    bool create(int);
    bool init_buffers();

    void stopCapture();
    bool startCapture();

    bool getDeviceNameById(int id, std::string &device);

    void autoExposureControl(const Mat &);

    ArvCamera       *camera;                // Camera to control.
    ArvStream       *stream;                // Object for video stream reception.
    void            *framebuffer;           //

    unsigned int    payload;                // Width x height x Pixel width.

    int             widthMin;               // Camera sensor minimum width.
    int             widthMax;               // Camera sensor maximum width.
    int             heightMin;              // Camera sensor minimum height.
    int             heightMax;              // Camera sensor maximum height.
    bool            fpsAvailable;
    double          fpsMin;                 // Camera minimum fps.
    double          fpsMax;                 // Camera maximum fps.
    bool            gainAvailable;
    double          gainMin;                // Camera minimum gain.
    double          gainMax;                // Camera maximum gain.
    bool            exposureAvailable;
    double          exposureMin;            // Camera's minimum exposure time.
    double          exposureMax;            // Camera's maximum exposure time.

    bool            controlExposure;        // Flag if automatic exposure shall be done by this SW
    double          exposureCompensation;
    bool            autoGain;
    double          targetGrey;             // Target grey value (mid grey))
    bool            softwareTriggered;      // Flag if the camera is software triggered
    bool            allowAutoTrigger;       // Flag that user allowed to trigger software triggered cameras automatically

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
    exposureCompensation = 0;
    targetGrey = 0;
    frameID = prevFrameID = 0;
    allowAutoTrigger = false;

    num_buffers = 10;
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

    return NULL != (camera = arv_camera_new(deviceName.c_str(), NULL));
}

bool CvCaptureCAM_Aravis::init_buffers()
{
    if(stream) {
        g_object_unref(stream);
        stream = NULL;
    }
    if( (stream = arv_camera_create_stream(camera, NULL, NULL, NULL)) ) {
        if( arv_camera_is_gv_device(camera) ) {
            g_object_set(stream,
                "socket-buffer", ARV_GV_STREAM_SOCKET_BUFFER_AUTO,
                "socket-buffer-size", 0, NULL);
            g_object_set(stream,
                "packet-resend", ARV_GV_STREAM_PACKET_RESEND_NEVER, NULL);
            g_object_set(stream,
                "packet-timeout", (unsigned) 40000,
                "frame-retention", (unsigned) 200000, NULL);
        }
        payload = arv_camera_get_payload (camera, NULL);

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
        arv_camera_get_width_bounds(camera, &widthMin, &widthMax, NULL);
        arv_camera_get_height_bounds(camera, &heightMin, &heightMax, NULL);
        arv_camera_set_region(camera, 0, 0, widthMax, heightMax, NULL);

        if( (fpsAvailable = arv_camera_is_frame_rate_available(camera, NULL)) )
            arv_camera_get_frame_rate_bounds(camera, &fpsMin, &fpsMax, NULL);
        if( (gainAvailable = arv_camera_is_gain_available(camera, NULL)) )
            arv_camera_get_gain_bounds (camera, &gainMin, &gainMax, NULL);
        if( (exposureAvailable = arv_camera_is_exposure_time_available(camera, NULL)) )
            arv_camera_get_exposure_time_bounds (camera, &exposureMin, &exposureMax, NULL);

        // get initial values
        pixelFormat = arv_camera_get_pixel_format(camera, NULL);
        exposure = exposureAvailable ? arv_camera_get_exposure_time(camera, NULL) : 0;
        gain = gainAvailable ? arv_camera_get_gain(camera, NULL) : 0;
        fps = arv_camera_get_frame_rate(camera, NULL);
        softwareTriggered = (strcmp(arv_camera_get_trigger_source(camera, NULL), "Software") == 0);

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
        if (softwareTriggered && allowAutoTrigger) {
            arv_camera_software_trigger (camera, NULL);
        }
        for(; tries < max_tries; tries ++) {
            arv_buffer = arv_stream_timeout_pop_buffer (stream, 200000);
            if (arv_buffer != NULL && arv_buffer_get_status (arv_buffer) != ARV_BUFFER_STATUS_SUCCESS) {
                arv_stream_push_buffer (stream, arv_buffer);
            } else break;
        }
        if(arv_buffer != NULL && tries < max_tries) {
            size_t buffer_size;
            framebuffer = (void*)arv_buffer_get_data (arv_buffer, &buffer_size);

            // retrieve image size properties
            arv_buffer_get_image_region (arv_buffer, &xoffset, &yoffset, &width, &height);

            // retrieve image ID set by camera
            frameID = arv_buffer_get_frame_id(arv_buffer);

            arv_stream_push_buffer(stream, arv_buffer);
            return true;
        }
    }
    return false;
}

bool CvCaptureCAM_Aravis::retrieveFrame(int, OutputArray arr)
{
    if(framebuffer) {
        int depth = 0, channels = 0;
        switch(pixelFormat) {
            case ARV_PIXEL_FORMAT_MONO_8:
            case ARV_PIXEL_FORMAT_BAYER_GR_8:
                depth = CV_8U;
                channels = 1;
                break;
            case ARV_PIXEL_FORMAT_MONO_12:
            case ARV_PIXEL_FORMAT_MONO_16:
                depth = CV_16U;
                channels = 1;
                break;
            default:
                return false;
        }
        Mat src(Size( width, height ), CV_MAKE_TYPE(depth, channels), framebuffer);
        if(controlExposure && ((frameID - prevFrameID) >= 3)) {
            // control exposure every third frame
            // i.e. skip frame taken with previous exposure setup
            autoExposureControl(src);
        }
        src.copyTo(arr);
        return true;
    }
    return false;
}

void CvCaptureCAM_Aravis::autoExposureControl(const Mat & image)
{
    // Software control of exposure parameters utilizing
    // automatic change of exposure time & gain

    // Priority is set as follows:
    // - to increase brightness, first increase time then gain
    // - to decrease brightness, first decrease gain then time

    // calc mean value for luminance or green channel
    double brightness = cv::mean(image)[image.channels() > 1 ? 1 : 0];
    if(brightness < 1) brightness = 1;

    // mid point - 100 % means no change
    static const double dmid = 100;

    // distance from optimal value as a percentage
    double d = (targetGrey * dmid) / brightness;
    if(d >= dmid) d = ( d + (dmid * 2) ) / 3;

    prevFrameID = frameID;
    midGrey = brightness;

    double maxe = 1e6 / fps;
    double ne = CLIP( ( exposure * d ) / ( dmid * pow(sqrt(2), -2 * exposureCompensation) ), exposureMin, maxe);

    // if change of value requires intervention
    if(std::fabs(d-dmid) > 5) {
        double ev, ng = 0;

        if(gainAvailable && autoGain) {
            ev = log( d / dmid ) / log(2);
            ng = CLIP( gain + ev + exposureCompensation, gainMin, gainMax);

            if( ng < gain ) {
                // priority 1 - reduce gain
                arv_camera_set_gain(camera, (gain = ng), NULL);
                return;
            }
        }

        if(exposureAvailable) {
            // priority 2 - control of exposure time
            if(std::fabs(exposure - ne) > 2) {
                // we have not yet reach the max-e level
                arv_camera_set_exposure_time(camera, (exposure = ne), NULL);
                return;
            }
        }

        if(gainAvailable && autoGain) {
            if(exposureAvailable) {
                // exposure at maximum - increase gain if possible
                if(ng > gain && ng < gainMax && ne >= maxe) {
                    arv_camera_set_gain(camera, (gain = ng), NULL);
                    return;
                }
            } else {
                // priority 3 - increase gain
                arv_camera_set_gain(camera, (gain = ng), NULL);
                return;
            }
        }
    }

    // if gain can be reduced - do it
    if(gainAvailable && autoGain && exposureAvailable) {
        if(gain > gainMin && exposure < maxe) {
            exposure = CLIP( ne * 1.05, exposureMin, maxe);
            arv_camera_set_exposure_time(camera, exposure, NULL);
        }
    }
}

double CvCaptureCAM_Aravis::getProperty( int property_id ) const
{
    switch(property_id) {
        case CAP_PROP_POS_MSEC:
            return (double)frameID/fps;

        case CAP_PROP_FRAME_WIDTH:
            return width;

        case CAP_PROP_FRAME_HEIGHT:
            return height;

        case CAP_PROP_AUTO_EXPOSURE:
            return (controlExposure ? 1 : 0);

    case CAP_PROP_BRIGHTNESS:
        return exposureCompensation;

        case CAP_PROP_EXPOSURE:
            if(exposureAvailable) {
                /* exposure time in seconds, like 1/100 s */
                return arv_camera_get_exposure_time(camera, NULL) / 1e6;
            }
            break;

        case CAP_PROP_FPS:
            if(fpsAvailable) {
                return arv_camera_get_frame_rate(camera, NULL);
            }
            break;

        case CAP_PROP_GAIN:
            if(gainAvailable) {
                return arv_camera_get_gain(camera, NULL);
            }
            break;

        case CAP_PROP_FOURCC:
            {
                ArvPixelFormat currFormat = arv_camera_get_pixel_format(camera, NULL);
                switch( currFormat ) {
                    case ARV_PIXEL_FORMAT_MONO_8:
                        return MODE_Y800;
                    case ARV_PIXEL_FORMAT_MONO_12:
                        return MODE_Y12;
                    case ARV_PIXEL_FORMAT_MONO_16:
                        return MODE_Y16;
                    case ARV_PIXEL_FORMAT_BAYER_GR_8:
                        return MODE_GRBG;
                }
            }
            break;

        case CAP_PROP_BUFFERSIZE:
            if(stream) {
                int in, out;
                arv_stream_get_n_buffers(stream, &in, &out);
                // return number of available buffers in Aravis output queue
                return out;
            }
            break;

        case cv::CAP_PROP_ARAVIS_AUTOTRIGGER:
        {
            return allowAutoTrigger ? 1. : 0.;
        }
        break;
    }
    return -1.0;
}

bool CvCaptureCAM_Aravis::setProperty( int property_id, double value )
{
    switch(property_id) {
        case CAP_PROP_AUTO_EXPOSURE:
            if(exposureAvailable || gainAvailable) {
                if( (controlExposure = (bool)(int)value) ) {
                    exposure = exposureAvailable ? arv_camera_get_exposure_time(camera, NULL) : 0;
                    gain = gainAvailable ? arv_camera_get_gain(camera, NULL) : 0;
                }
            }
            break;
    case CAP_PROP_BRIGHTNESS:
       exposureCompensation = CLIP(value, -3., 3.);
       break;

        case CAP_PROP_EXPOSURE:
            if(exposureAvailable) {
                /* exposure time in seconds, like 1/100 s */
                value *= 1e6; // -> from s to us

                arv_camera_set_exposure_time(camera, exposure = CLIP(value, exposureMin, exposureMax), NULL);
                break;
            } else return false;

        case CAP_PROP_FPS:
            if(fpsAvailable) {
                arv_camera_set_frame_rate(camera, fps = CLIP(value, fpsMin, fpsMax), NULL);
                break;
            } else return false;

        case CAP_PROP_GAIN:
            if(gainAvailable) {
                if ( (autoGain = (-1 == value) ) )
                    break;

                arv_camera_set_gain(camera, gain = CLIP(value, gainMin, gainMax), NULL);
                break;
            } else return false;

        case CAP_PROP_FOURCC:
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
                    case MODE_Y16:
                        newFormat = ARV_PIXEL_FORMAT_MONO_16;
                        targetGrey = 32768;
                        break;
                    case MODE_GRBG:
                        newFormat = ARV_PIXEL_FORMAT_BAYER_GR_8;
                        targetGrey = 128;
                        break;
                }
                if(newFormat != pixelFormat) {
                    stopCapture();
                    arv_camera_set_pixel_format(camera, pixelFormat = newFormat, NULL);
                    startCapture();
                }
            }
            break;

        case CAP_PROP_BUFFERSIZE:
            {
                int x = (int)value;
                if((x > 0) && (x != num_buffers)) {
                    stopCapture();
                    num_buffers = x;
                    startCapture();
                }
            }
            break;

        case cv::CAP_PROP_ARAVIS_AUTOTRIGGER:
            {
                allowAutoTrigger = (bool) value;
            }
            break;

        default:
            return false;
    }

    return true;
}

void CvCaptureCAM_Aravis::stopCapture()
{
    arv_camera_stop_acquisition(camera, NULL);

    if(stream) {
        g_object_unref(stream);
        stream = NULL;
    }
}

bool CvCaptureCAM_Aravis::startCapture()
{
    if(init_buffers() ) {
        arv_camera_set_acquisition_mode(camera, ARV_ACQUISITION_MODE_CONTINUOUS, NULL);
        arv_camera_start_acquisition(camera, NULL);

        return true;
    }
    return false;
}

cv::Ptr<cv::IVideoCapture> cv::create_Aravis_capture( int index )
{
    Ptr<CvCaptureCAM_Aravis> capture = makePtr<CvCaptureCAM_Aravis>();
    if(capture->open(index)) {
        return capture;
    }
    return NULL;
}
#endif
