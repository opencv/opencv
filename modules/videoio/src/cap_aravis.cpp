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
//      CAP_PROP_FRAME_WIDTH(w)
//      CAP_PROP_FRAME_HEIGHT(h)
//  read only:
//      CAP_PROP_POS_MSEC
//
// The capture is set up for 640x480 @ 30 Hz on open(), clipped to the range the camera
// supports. Both the region and the frame rate can be changed afterwards with
// CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT and CAP_PROP_FPS.
// On open() the highest priority pixel format supported by the camera is selected:
//      1. true color: BGR/RGB and BGRa/RGBa, 8 or 16 bit per component,
//      2. Bayer CFA: BayerRG/BayerBG/BayerGR/BayerGB, 8 or 16 bit per component,
//      3. grayscale: Mono8/Mono10/Mono12/Mono14/Mono16.
//  The format can be overridden afterwards with CAP_PROP_FOURCC.
//
//  Whatever the camera sends, retrieveFrame() always returns a BGR CV_8UC3 image:
//  Bayer data is demosaiced, RGB data is swapped to BGR, grayscale is replicated to
//  three channels, and formats deeper than 8 bit are scaled down to 8 bit.
//
//  Supported fourcc codes for CAP_PROP_FOURCC:
//      'GREY', 'Y800'  -> Mono8
//      'Y12 '          -> Mono12
//      'Y16 '          -> Mono16
//      'GRBG'          -> BayerGR8
//      'RGGB'          -> BayerRG8
//      'GBRG'          -> BayerGB8
//      'BGGR'          -> BayerBG8
//      'BGR3', 'RGB3'  -> BGR8, RGB8
//      'BGR4', 'RGB4'  -> BGRa8, RGBa8
//

#define MODE_GREY   CV_FOURCC_MACRO('G','R','E','Y')
#define MODE_Y800   CV_FOURCC_MACRO('Y','8','0','0')
#define MODE_Y12    CV_FOURCC_MACRO('Y','1','2',' ')
#define MODE_Y16    CV_FOURCC_MACRO('Y','1','6',' ')
#define MODE_GRBG   CV_FOURCC_MACRO('G','R','B','G')
#define MODE_RGGB   CV_FOURCC_MACRO('R','G','G','B')
#define MODE_GBRG   CV_FOURCC_MACRO('G','B','R','G')
#define MODE_BGGR   CV_FOURCC_MACRO('B','G','G','R')
#define MODE_BGR3   CV_FOURCC_MACRO('B','G','R','3')
#define MODE_RGB3   CV_FOURCC_MACRO('R','G','B','3')
#define MODE_BGR4   CV_FOURCC_MACRO('B','G','R','4')
#define MODE_RGB4   CV_FOURCC_MACRO('R','G','B','4')

#define CLIP(a,b,c) (cv::max(cv::min((a),(c)),(b)))

// Capture setup applied on open(). Cameras usually come up with the full sensor
// region and whatever frame rate was left over from the previous session, which
// makes the default VideoCapture behaviour depend on the camera state. Request a
// commonly supported VGA @ 30 Hz instead, clipped to what the camera can do.
static const int    DEFAULT_FRAME_WIDTH  = 640;
static const int    DEFAULT_FRAME_HEIGHT = 480;
static const double DEFAULT_FPS          = 30.;

namespace {

// The data is BGR already, no color conversion is needed.
const int CONVERSION_NONE = -1;

// Description of a pixel format the backend is able to decode.
struct PixelFormatInfo
{
    ArvPixelFormat  format;
    int             fourcc;         // CAP_PROP_FOURCC representation, 0 if there is no common one
    int             cvType;         // type of the Mat mapped over the raw frame buffer
    int             bits;           // significant bits per component
    int             conversion;     // cvtColor()/demosaicing() code producing BGR, see CONVERSION_NONE
};

// Note on the Bayer codes: Aravis follows the GenICam convention and names the pattern after
// the top left 2x2 tile, while OpenCV names it after the second and third component of the
// second row. The two namings are related by an R <-> B swap, hence BayerRG -> COLOR_BayerBG2BGR.
//
// The order of the entries defines the selection priority in selectPixelFormat():
// color first, then Bayer, then grayscale, the least deep format first within each group.
// Bit packed formats (Mono12Packed, BayerRG12p, ...) are intentionally not listed here,
// their payload cannot be mapped to a Mat without unpacking it first.
const PixelFormatInfo supportedPixelFormats[] =
{
    // 1st priority - true color
    { ARV_PIXEL_FORMAT_BGR_8_PACKED,    MODE_BGR3,  CV_8UC3,   8, CONVERSION_NONE       },
    { ARV_PIXEL_FORMAT_RGB_8_PACKED,    MODE_RGB3,  CV_8UC3,   8, COLOR_RGB2BGR         },
    { ARV_PIXEL_FORMAT_BGRA_8_PACKED,   MODE_BGR4,  CV_8UC4,   8, COLOR_BGRA2BGR        },
    { ARV_PIXEL_FORMAT_RGBA_8_PACKED,   MODE_RGB4,  CV_8UC4,   8, COLOR_RGBA2BGR        },
    { ARV_PIXEL_FORMAT_BGR_10_PACKED,   0,          CV_16UC3, 10, CONVERSION_NONE       },
    { ARV_PIXEL_FORMAT_RGB_10_PACKED,   0,          CV_16UC3, 10, COLOR_RGB2BGR         },
    { ARV_PIXEL_FORMAT_BGR_12_PACKED,   0,          CV_16UC3, 12, CONVERSION_NONE       },
    { ARV_PIXEL_FORMAT_RGB_12_PACKED,   0,          CV_16UC3, 12, COLOR_RGB2BGR         },

    // 2nd priority - Bayer CFA
    { ARV_PIXEL_FORMAT_BAYER_GR_8,      MODE_GRBG,  CV_8UC1,   8, COLOR_BayerGB2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_RG_8,      MODE_RGGB,  CV_8UC1,   8, COLOR_BayerBG2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_GB_8,      MODE_GBRG,  CV_8UC1,   8, COLOR_BayerGR2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_BG_8,      MODE_BGGR,  CV_8UC1,   8, COLOR_BayerRG2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_GR_10,     0,          CV_16UC1, 10, COLOR_BayerGB2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_RG_10,     0,          CV_16UC1, 10, COLOR_BayerBG2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_GB_10,     0,          CV_16UC1, 10, COLOR_BayerGR2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_BG_10,     0,          CV_16UC1, 10, COLOR_BayerRG2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_GR_12,     0,          CV_16UC1, 12, COLOR_BayerGB2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_RG_12,     0,          CV_16UC1, 12, COLOR_BayerBG2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_GB_12,     0,          CV_16UC1, 12, COLOR_BayerGR2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_BG_12,     0,          CV_16UC1, 12, COLOR_BayerRG2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_GR_16,     0,          CV_16UC1, 16, COLOR_BayerGB2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_RG_16,     0,          CV_16UC1, 16, COLOR_BayerBG2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_GB_16,     0,          CV_16UC1, 16, COLOR_BayerGR2BGR     },
    { ARV_PIXEL_FORMAT_BAYER_BG_16,     0,          CV_16UC1, 16, COLOR_BayerRG2BGR     },

    // 3rd priority - grayscale
    { ARV_PIXEL_FORMAT_MONO_8,          MODE_Y800,  CV_8UC1,   8, COLOR_GRAY2BGR        },
    { ARV_PIXEL_FORMAT_MONO_10,         0,          CV_16UC1, 10, COLOR_GRAY2BGR        },
    { ARV_PIXEL_FORMAT_MONO_12,         MODE_Y12,   CV_16UC1, 12, COLOR_GRAY2BGR        },
    { ARV_PIXEL_FORMAT_MONO_14,         0,          CV_16UC1, 14, COLOR_GRAY2BGR        },
    { ARV_PIXEL_FORMAT_MONO_16,         MODE_Y16,   CV_16UC1, 16, COLOR_GRAY2BGR        },
};

const PixelFormatInfo* getPixelFormatInfo(ArvPixelFormat format)
{
    for(size_t i = 0; i < sizeof(supportedPixelFormats) / sizeof(supportedPixelFormats[0]); i++) {
        if(supportedPixelFormats[i].format == format)
            return &supportedPixelFormats[i];
    }
    return NULL;
}

const PixelFormatInfo* getPixelFormatInfoByFourcc(int fourcc)
{
    // 'GREY' is an alias of 'Y800' kept for backward compatibility
    if(fourcc == MODE_GREY)
        fourcc = MODE_Y800;

    for(size_t i = 0; i < sizeof(supportedPixelFormats) / sizeof(supportedPixelFormats[0]); i++) {
        if(supportedPixelFormats[i].fourcc != 0 && supportedPixelFormats[i].fourcc == fourcc)
            return &supportedPixelFormats[i];
    }
    return NULL;
}

} // namespace

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
    bool open(const std::string&);
    void configure();
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
    bool create(const std::string&);
    bool init_buffers();

    void stopCapture();
    bool startCapture();

    bool selectPixelFormat();
    bool applyPixelFormat(ArvPixelFormat format);
    void updatePixelFormatInfo();
    bool setRegionSize(int newWidth, int newHeight);
    bool setFrameRate(double newFps);

    bool getDeviceNameById(int id, std::string &device);

    void autoExposureControl(const Mat &);

    double getExpectedMidGrey(ArvPixelFormat fmt) const;

    ArvCamera       *camera;                // Camera to control.
    ArvStream       *stream;                // Object for video stream reception.
    void            *framebuffer;           //
    size_t          framebufferSize;        // Size of the payload of the last grabbed frame.

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
    bool            pixelFormatSupported;   // true if the backend is able to decode pixelFormat
    int             srcType;                // OpenCV type of the raw frame buffer
    int             srcBits;                // significant bits per component in the raw frame
    int             conversionCode;         // color conversion producing BGR, see CONVERSION_NONE

    int             xoffset;                // current frame region x offset
    int             yoffset;                // current frame region y offset
    int             width;                  // current frame width of frame
    int             height;                 // current frame height of image
    int             widthSet;               // last frame width set by user
    int             heightSet;              // last frame height set by user

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
    framebufferSize = 0;

    payload = 0;

    pixelFormat = ARV_PIXEL_FORMAT_MONO_8;
    pixelFormatSupported = true;
    srcType = CV_8UC1;
    srcBits = 8;
    conversionCode = COLOR_GRAY2BGR;

    widthMin = widthMax = heightMin = heightMax = 0;
    xoffset = yoffset = width = height = 0;
    fpsMin = fpsMax = gainMin = gainMax = exposureMin = exposureMax = 0;
    fpsAvailable = gainAvailable = exposureAvailable = false;
    fps = exposure = gain = midGrey = 0;
    autoGain = false;
    softwareTriggered = false;
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

bool CvCaptureCAM_Aravis::create( const std::string &deviceName )
{
    GError *error = NULL;

    // NULL name asks Aravis for the first device found
    camera = arv_camera_new(deviceName.empty() ? NULL : deviceName.c_str(), &error);
    if(error) {
        CV_LOG_WARNING(NULL, cv::format("Aravis: failed to open camera '%s': %s",
                                        deviceName.c_str(), error->message));
        g_clear_error(&error);
    }

    return camera != NULL;
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

// Refresh the cached description of the pixel format the camera is currently set to.
void CvCaptureCAM_Aravis::updatePixelFormatInfo()
{
    pixelFormat = arv_camera_get_pixel_format(camera, NULL);

    const PixelFormatInfo *info = getPixelFormatInfo(pixelFormat);
    pixelFormatSupported = (info != NULL);
    if(info) {
        srcType = info->cvType;
        srcBits = info->bits;
        conversionCode = info->conversion;
    } else {
        // retrieveFrame() has no way to decode this payload
        CV_LOG_WARNING(NULL, cv::format("Aravis: pixel format '%s' is not supported by the backend.",
                                        arv_camera_get_pixel_format_as_string(camera, NULL)));
    }
}

bool CvCaptureCAM_Aravis::applyPixelFormat(ArvPixelFormat format)
{
    if(format != arv_camera_get_pixel_format(camera, NULL)) {
        GError *error = NULL;
        arv_camera_set_pixel_format(camera, format, &error);
        if(error) {
            CV_LOG_WARNING(NULL, cv::format("Aravis: failed to set pixel format: %s", error->message));
            g_clear_error(&error);
        }
    }

    updatePixelFormatInfo();

    return pixelFormatSupported && pixelFormat == format;
}

// Query the pixel formats the camera offers and switch it to the most preferred one
// this backend is able to convert to BGR, see supportedPixelFormats[] for the priorities.
bool CvCaptureCAM_Aravis::selectPixelFormat()
{
    GError *error = NULL;
    guint n_formats = 0;
    gint64 *formats = arv_camera_dup_available_pixel_formats(camera, &n_formats, &error);
    if(error) {
        CV_LOG_WARNING(NULL, cv::format("Aravis: failed to enumerate pixel formats: %s", error->message));
        g_clear_error(&error);
    }

    const PixelFormatInfo *selected = NULL;
    if(formats) {
        for(size_t i = 0; !selected && i < sizeof(supportedPixelFormats) / sizeof(supportedPixelFormats[0]); i++) {
            for(guint j = 0; j < n_formats; j++) {
                if((ArvPixelFormat)formats[j] == supportedPixelFormats[i].format) {
                    selected = &supportedPixelFormats[i];
                    break;
                }
            }
        }
        g_free(formats);
    }

    if(!selected) {
        // the camera did not report anything usable, keep whatever it is set to
        updatePixelFormatInfo();
        if(!pixelFormatSupported) {
            CV_LOG_WARNING(NULL, "Aravis: no supported pixel format found, falling back to Mono8.");
            return applyPixelFormat(ARV_PIXEL_FORMAT_MONO_8);
        }
        return true;
    }

    return applyPixelFormat(selected->format);
}

// Change the size of the captured region. The payload size changes with it, so the
// stream has to be recreated if the acquisition is already running.
bool CvCaptureCAM_Aravis::setRegionSize(int newWidth, int newHeight)
{
    if (newWidth > 0)
        widthSet = newWidth;

    if (newHeight > 0)
        heightSet = newHeight;

    /* two subsequent calls setting WIDTH and HEIGHT will change
     *      the video size */
    if (widthSet <= 0 || heightSet <= 0)
    {
        return true;
    }

    newWidth = CLIP(widthSet, widthMin, widthMax);
    newHeight = CLIP(heightSet, heightMin, heightMax);

    widthSet = heightSet = 0;

    const bool capturing = (stream != NULL);
    if(capturing)
        stopCapture();

    GError *error = NULL;
    // the offset is reset to 0 first, a large one left over from a previous region
    // would not leave enough room for the requested width or height
    arv_camera_set_region(camera, 0, 0, newWidth, newHeight, &error);
    if(error) {
        CV_LOG_WARNING(NULL, cv::format("Aravis: failed to set region to %dx%d: %s",
                                        newWidth, newHeight, error->message));
        g_clear_error(&error);
    }

    // the camera is free to round the request to a size it supports
    arv_camera_get_region(camera, &xoffset, &yoffset, &width, &height, NULL);
    if(width != newWidth || height != newHeight) {
        CV_LOG_INFO(NULL, cv::format("Aravis: %dx%d requested, camera applied %dx%d.",
                                     newWidth, newHeight, width, height));
    }

    // both the achievable frame rate range and the current frame rate depend on
    // the region size, the camera clamps the latter on its own
    if(fpsAvailable) {
        arv_camera_get_frame_rate_bounds(camera, &fpsMin, &fpsMax, NULL);
        fps = arv_camera_get_frame_rate(camera, NULL);
    }

    if(capturing && !startCapture())
        return false;

    return width == newWidth && height == newHeight;
}

bool CvCaptureCAM_Aravis::setFrameRate(double newFps)
{
    if(!fpsAvailable)
        return false;

    GError *error = NULL;
    arv_camera_set_frame_rate(camera, CLIP(newFps, fpsMin, fpsMax), &error);

    const bool ok = (error == NULL);
    if(error) {
        CV_LOG_WARNING(NULL, cv::format("Aravis: failed to set frame rate to %g: %s",
                                        newFps, error->message));
        g_clear_error(&error);
    }

    // the camera may quantize the request, keep the value actually in effect
    fps = arv_camera_get_frame_rate(camera, NULL);

    return ok;
}

void CvCaptureCAM_Aravis::configure()
{
    widthSet = heightSet = 0;

    // fetch properties bounds
    arv_camera_get_width_bounds(camera, &widthMin, &widthMax, NULL);
    arv_camera_get_height_bounds(camera, &heightMin, &heightMax, NULL);
    setRegionSize(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT);

    // the frame rate bounds depend on the region, query them once it is set
    if( (fpsAvailable = arv_camera_is_frame_rate_available(camera, NULL)) )
        arv_camera_get_frame_rate_bounds(camera, &fpsMin, &fpsMax, NULL);
    if( (gainAvailable = arv_camera_is_gain_available(camera, NULL)) )
        arv_camera_get_gain_bounds (camera, &gainMin, &gainMax, NULL);
    if( (exposureAvailable = arv_camera_is_exposure_time_available(camera, NULL)) )
        arv_camera_get_exposure_time_bounds (camera, &exposureMin, &exposureMax, NULL);

    // pick the best pixel format the camera and this backend have in common
    selectPixelFormat();

    midGrey = getExpectedMidGrey(pixelFormat);

    exposure = exposureAvailable ? arv_camera_get_exposure_time(camera, NULL) : 0;
    gain = gainAvailable ? arv_camera_get_gain(camera, NULL) : 0;

    setFrameRate(DEFAULT_FPS);
    fps = arv_camera_get_frame_rate(camera, NULL);

    // arv_camera_set_frame_rate() may have switched the trigger off, so the trigger
    // source has to be read after it. It is not implemented by every camera.
    const char *triggerSource = arv_camera_get_trigger_source(camera, NULL);
    softwareTriggered = (triggerSource != NULL) && (strcmp(triggerSource, "Software") == 0);
}

bool CvCaptureCAM_Aravis::open( int index )
{
    if(create(index)) {
        configure();
        return startCapture();
    }
    return false;
}

bool CvCaptureCAM_Aravis::open( const std::string& deviceName)
{
    if(create(deviceName)) {
        configure();
        return startCapture();
    }
    return false;
}

bool CvCaptureCAM_Aravis::grabFrame()
{
    // remove content of previous frame
    framebuffer = NULL;
    framebufferSize = 0;

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
            framebuffer = (void*)arv_buffer_get_data (arv_buffer, &framebufferSize);

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
    if(!framebuffer || !pixelFormatSupported)
        return false;

    const size_t expectedSize = (size_t)width * (size_t)height * CV_ELEM_SIZE(srcType);
    if(width <= 0 || height <= 0 || framebufferSize < expectedSize) {
        CV_LOG_WARNING(NULL, "Aravis: payload is too small for the current pixel format.");
        return false;
    }

    Mat src(Size(width, height), srcType, framebuffer);
    if(controlExposure && ((frameID - prevFrameID) >= 3)) {
        // control exposure every third frame
        // i.e. skip frame taken with previous exposure setup
        autoExposureControl(src);
    }

    // Scale the deeper formats down to 8 bit. GenICam stores them right aligned in a
    // 16 bit container, so the significant bits are the srcBits least significant ones.
    Mat src8;
    if(src.depth() != CV_8U)
        src.convertTo(src8, CV_8U, 255. / ((1 << srcBits) - 1));
    else
        src8 = src;

    if(conversionCode == CONVERSION_NONE)
        src8.copyTo(arr);           // already BGR
    else
        cvtColor(src8, arr, conversionCode, 3);

    return true;
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
    double ne = CLIP( ( exposure * d ) / ( dmid * std::pow(sqrt(2), -2 * exposureCompensation) ), exposureMin, maxe);

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
                const PixelFormatInfo *info =
                    getPixelFormatInfo(arv_camera_get_pixel_format(camera, NULL));
                if(info && info->fourcc != 0)
                    return info->fourcc;
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

double CvCaptureCAM_Aravis::getExpectedMidGrey(ArvPixelFormat fmt) const
{
    // half of the range of the raw samples, i.e. 128 for 8 bit, 2048 for 12 bit, ...
    const PixelFormatInfo *info = getPixelFormatInfo(fmt);

    return info ? (double)(1 << (info->bits - 1)) : 0.;
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

        case CAP_PROP_FRAME_WIDTH:
        {
            return setRegionSize((int)value, 0);
        }

        case CAP_PROP_FRAME_HEIGHT:
        {
            return setRegionSize(0, (int)value);
        }

        case CAP_PROP_FPS:
            return setFrameRate(value);

        case CAP_PROP_GAIN:
            if(gainAvailable) {
                if ( (autoGain = (-1 == value) ) )
                    break;

                arv_camera_set_gain(camera, gain = CLIP(value, gainMin, gainMax), NULL);
                break;
            } else return false;

        case CAP_PROP_FOURCC:
            {
                const PixelFormatInfo *info = getPixelFormatInfoByFourcc((int)value);
                if(!info)
                    return false;

                if(info->format != pixelFormat) {
                    stopCapture();
                    bool ok = applyPixelFormat(info->format);
                    midGrey = getExpectedMidGrey(pixelFormat);
                    startCapture();
                    if(!ok)
                        return false;
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

cv::Ptr<cv::IVideoCapture> cv::create_Aravis_capture_by_name( const std::string &deviceName )
{
    Ptr<CvCaptureCAM_Aravis> capture = makePtr<CvCaptureCAM_Aravis>();
    if(capture->open(deviceName)) {
        return capture;
    }
    return NULL;
}

#endif
