/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#ifndef __OPENCV_VIDEOIO_HPP__
#define __OPENCV_VIDEOIO_HPP__

#include "opencv2/core.hpp"

/**
  @defgroup videoio Media I/O
  @{
    @defgroup videoio_c C API
    @defgroup videoio_ios iOS glue
  @}
*/

////////////////////////////////// video io /////////////////////////////////

typedef struct CvCapture CvCapture;
typedef struct CvVideoWriter CvVideoWriter;

namespace cv
{

//! @addtogroup videoio
//! @{

// Camera API
enum { CAP_ANY          = 0,     // autodetect
       CAP_VFW          = 200,   // platform native
       CAP_V4L          = 200,
       CAP_V4L2         = CAP_V4L,
       CAP_FIREWARE     = 300,   // IEEE 1394 drivers
       CAP_FIREWIRE     = CAP_FIREWARE,
       CAP_IEEE1394     = CAP_FIREWARE,
       CAP_DC1394       = CAP_FIREWARE,
       CAP_CMU1394      = CAP_FIREWARE,
       CAP_QT           = 500,   // QuickTime
       CAP_UNICAP       = 600,   // Unicap drivers
       CAP_DSHOW        = 700,   // DirectShow (via videoInput)
       CAP_PVAPI        = 800,   // PvAPI, Prosilica GigE SDK
       CAP_OPENNI       = 900,   // OpenNI (for Kinect)
       CAP_OPENNI_ASUS  = 910,   // OpenNI (for Asus Xtion)
       CAP_ANDROID      = 1000,  // Android
       CAP_XIAPI        = 1100,  // XIMEA Camera API
       CAP_AVFOUNDATION = 1200,  // AVFoundation framework for iOS (OS X Lion will have the same API)
       CAP_GIGANETIX    = 1300,  // Smartek Giganetix GigEVisionSDK
       CAP_MSMF         = 1400,  // Microsoft Media Foundation (via videoInput)
       CAP_INTELPERC    = 1500,   // Intel Perceptual Computing SDK
       CAP_OPENNI2      = 1600,   // OpenNI2 (for Kinect)
       CAP_OPENNI2_ASUS = 1610   // OpenNI2 (for Asus Xtion and Occipital Structure sensors)
     };

// generic properties (based on DC1394 properties)
enum { CAP_PROP_POS_MSEC       =0,
       CAP_PROP_POS_FRAMES     =1,
       CAP_PROP_POS_AVI_RATIO  =2,
       CAP_PROP_FRAME_WIDTH    =3,
       CAP_PROP_FRAME_HEIGHT   =4,
       CAP_PROP_FPS            =5,
       CAP_PROP_FOURCC         =6,
       CAP_PROP_FRAME_COUNT    =7,
       CAP_PROP_FORMAT         =8,
       CAP_PROP_MODE           =9,
       CAP_PROP_BRIGHTNESS    =10,
       CAP_PROP_CONTRAST      =11,
       CAP_PROP_SATURATION    =12,
       CAP_PROP_HUE           =13,
       CAP_PROP_GAIN          =14,
       CAP_PROP_EXPOSURE      =15,
       CAP_PROP_CONVERT_RGB   =16,
       CAP_PROP_WHITE_BALANCE_BLUE_U =17,
       CAP_PROP_RECTIFICATION =18,
       CAP_PROP_MONOCHROME    =19,
       CAP_PROP_SHARPNESS     =20,
       CAP_PROP_AUTO_EXPOSURE =21, // DC1394: exposure control done by camera, user can adjust refernce level using this feature
       CAP_PROP_GAMMA         =22,
       CAP_PROP_TEMPERATURE   =23,
       CAP_PROP_TRIGGER       =24,
       CAP_PROP_TRIGGER_DELAY =25,
       CAP_PROP_WHITE_BALANCE_RED_V =26,
       CAP_PROP_ZOOM          =27,
       CAP_PROP_FOCUS         =28,
       CAP_PROP_GUID          =29,
       CAP_PROP_ISO_SPEED     =30,
       CAP_PROP_BACKLIGHT     =32,
       CAP_PROP_PAN           =33,
       CAP_PROP_TILT          =34,
       CAP_PROP_ROLL          =35,
       CAP_PROP_IRIS          =36,
       CAP_PROP_SETTINGS      =37
     };


// Generic camera output modes.
// Currently, these are supported through the libv4l interface only.
enum { CAP_MODE_BGR  = 0, // BGR24 (default)
       CAP_MODE_RGB  = 1, // RGB24
       CAP_MODE_GRAY = 2, // Y8
       CAP_MODE_YUYV = 3  // YUYV
     };


// DC1394 only
// modes of the controlling registers (can be: auto, manual, auto single push, absolute Latter allowed with any other mode)
// every feature can have only one mode turned on at a time
enum { CAP_PROP_DC1394_OFF                = -4, //turn the feature off (not controlled manually nor automatically)
       CAP_PROP_DC1394_MODE_MANUAL        = -3, //set automatically when a value of the feature is set by the user
       CAP_PROP_DC1394_MODE_AUTO          = -2,
       CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO = -1,
       CAP_PROP_DC1394_MAX                = 31
     };


// OpenNI map generators
enum { CAP_OPENNI_DEPTH_GENERATOR = 1 << 31,
       CAP_OPENNI_IMAGE_GENERATOR = 1 << 30,
       CAP_OPENNI_GENERATORS_MASK = CAP_OPENNI_DEPTH_GENERATOR + CAP_OPENNI_IMAGE_GENERATOR
     };

// Properties of cameras available through OpenNI interfaces
enum { CAP_PROP_OPENNI_OUTPUT_MODE       = 100,
       CAP_PROP_OPENNI_FRAME_MAX_DEPTH   = 101, // in mm
       CAP_PROP_OPENNI_BASELINE          = 102, // in mm
       CAP_PROP_OPENNI_FOCAL_LENGTH      = 103, // in pixels
       CAP_PROP_OPENNI_REGISTRATION      = 104, // flag that synchronizes the remapping depth map to image map
                                                // by changing depth generator's view point (if the flag is "on") or
                                                // sets this view point to its normal one (if the flag is "off").
       CAP_PROP_OPENNI_REGISTRATION_ON   = CAP_PROP_OPENNI_REGISTRATION,
       CAP_PROP_OPENNI_APPROX_FRAME_SYNC = 105,
       CAP_PROP_OPENNI_MAX_BUFFER_SIZE   = 106,
       CAP_PROP_OPENNI_CIRCLE_BUFFER     = 107,
       CAP_PROP_OPENNI_MAX_TIME_DURATION = 108,
       CAP_PROP_OPENNI_GENERATOR_PRESENT = 109,
       CAP_PROP_OPENNI2_SYNC             = 110,
       CAP_PROP_OPENNI2_MIRROR           = 111
     };

// OpenNI shortcats
enum { CAP_OPENNI_IMAGE_GENERATOR_PRESENT         = CAP_OPENNI_IMAGE_GENERATOR + CAP_PROP_OPENNI_GENERATOR_PRESENT,
       CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE     = CAP_OPENNI_IMAGE_GENERATOR + CAP_PROP_OPENNI_OUTPUT_MODE,
       CAP_OPENNI_DEPTH_GENERATOR_BASELINE        = CAP_OPENNI_DEPTH_GENERATOR + CAP_PROP_OPENNI_BASELINE,
       CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH    = CAP_OPENNI_DEPTH_GENERATOR + CAP_PROP_OPENNI_FOCAL_LENGTH,
       CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION    = CAP_OPENNI_DEPTH_GENERATOR + CAP_PROP_OPENNI_REGISTRATION,
       CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION_ON = CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION
     };

// OpenNI data given from depth generator
enum { CAP_OPENNI_DEPTH_MAP         = 0, // Depth values in mm (CV_16UC1)
       CAP_OPENNI_POINT_CLOUD_MAP   = 1, // XYZ in meters (CV_32FC3)
       CAP_OPENNI_DISPARITY_MAP     = 2, // Disparity in pixels (CV_8UC1)
       CAP_OPENNI_DISPARITY_MAP_32F = 3, // Disparity in pixels (CV_32FC1)
       CAP_OPENNI_VALID_DEPTH_MASK  = 4, // CV_8UC1

       // Data given from RGB image generator
       CAP_OPENNI_BGR_IMAGE         = 5,
       CAP_OPENNI_GRAY_IMAGE        = 6
     };

// Supported output modes of OpenNI image generator
enum { CAP_OPENNI_VGA_30HZ  = 0,
       CAP_OPENNI_SXGA_15HZ = 1,
       CAP_OPENNI_SXGA_30HZ = 2,
       CAP_OPENNI_QVGA_30HZ = 3,
       CAP_OPENNI_QVGA_60HZ = 4
     };


// GStreamer
enum { CAP_PROP_GSTREAMER_QUEUE_LENGTH = 200 // default is 1
     };


// PVAPI
enum { CAP_PROP_PVAPI_MULTICASTIP           = 300, // ip for anable multicast master mode. 0 for disable multicast
       CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE = 301, // FrameStartTriggerMode: Determines how a frame is initiated
       CAP_PROP_PVAPI_DECIMATIONHORIZONTAL  = 302, // Horizontal sub-sampling of the image
       CAP_PROP_PVAPI_DECIMATIONVERTICAL    = 303, // Vertical sub-sampling of the image
       CAP_PROP_PVAPI_BINNINGX              = 304, // Horizontal binning factor
       CAP_PROP_PVAPI_BINNINGY              = 305, // Vertical binning factor
       CAP_PROP_PVAPI_PIXELFORMAT           = 306  // Pixel format
     };

// PVAPI: FrameStartTriggerMode
enum { CAP_PVAPI_FSTRIGMODE_FREERUN     = 0,    // Freerun
       CAP_PVAPI_FSTRIGMODE_SYNCIN1     = 1,    // SyncIn1
       CAP_PVAPI_FSTRIGMODE_SYNCIN2     = 2,    // SyncIn2
       CAP_PVAPI_FSTRIGMODE_FIXEDRATE   = 3,    // FixedRate
       CAP_PVAPI_FSTRIGMODE_SOFTWARE    = 4     // Software
     };

// PVAPI: DecimationHorizontal, DecimationVertical
enum { CAP_PVAPI_DECIMATION_OFF       = 1,    // Off
       CAP_PVAPI_DECIMATION_2OUTOF4   = 2,    // 2 out of 4 decimation
       CAP_PVAPI_DECIMATION_2OUTOF8   = 4,    // 2 out of 8 decimation
       CAP_PVAPI_DECIMATION_2OUTOF16  = 8     // 2 out of 16 decimation
     };

// PVAPI: PixelFormat
enum { CAP_PVAPI_PIXELFORMAT_MONO8    = 1,    // Mono8
       CAP_PVAPI_PIXELFORMAT_MONO16   = 2,    // Mono16
       CAP_PVAPI_PIXELFORMAT_BAYER8   = 3,    // Bayer8
       CAP_PVAPI_PIXELFORMAT_BAYER16  = 4,    // Bayer16
       CAP_PVAPI_PIXELFORMAT_RGB24    = 5,    // Rgb24
       CAP_PVAPI_PIXELFORMAT_BGR24    = 6,    // Bgr24
       CAP_PVAPI_PIXELFORMAT_RGBA32   = 7,    // Rgba32
       CAP_PVAPI_PIXELFORMAT_BGRA32   = 8,    // Bgra32
     };

// Properties of cameras available through XIMEA SDK interface
enum { CAP_PROP_XI_DOWNSAMPLING  = 400, // Change image resolution by binning or skipping.
       CAP_PROP_XI_DATA_FORMAT   = 401, // Output data format.
       CAP_PROP_XI_OFFSET_X      = 402, // Horizontal offset from the origin to the area of interest (in pixels).
       CAP_PROP_XI_OFFSET_Y      = 403, // Vertical offset from the origin to the area of interest (in pixels).
       CAP_PROP_XI_TRG_SOURCE    = 404, // Defines source of trigger.
       CAP_PROP_XI_TRG_SOFTWARE  = 405, // Generates an internal trigger. PRM_TRG_SOURCE must be set to TRG_SOFTWARE.
       CAP_PROP_XI_GPI_SELECTOR  = 406, // Selects general purpose input
       CAP_PROP_XI_GPI_MODE      = 407, // Set general purpose input mode
       CAP_PROP_XI_GPI_LEVEL     = 408, // Get general purpose level
       CAP_PROP_XI_GPO_SELECTOR  = 409, // Selects general purpose output
       CAP_PROP_XI_GPO_MODE      = 410, // Set general purpose output mode
       CAP_PROP_XI_LED_SELECTOR  = 411, // Selects camera signalling LED
       CAP_PROP_XI_LED_MODE      = 412, // Define camera signalling LED functionality
       CAP_PROP_XI_MANUAL_WB     = 413, // Calculates White Balance(must be called during acquisition)
       CAP_PROP_XI_AUTO_WB       = 414, // Automatic white balance
       CAP_PROP_XI_AEAG          = 415, // Automatic exposure/gain
       CAP_PROP_XI_EXP_PRIORITY  = 416, // Exposure priority (0.5 - exposure 50%, gain 50%).
       CAP_PROP_XI_AE_MAX_LIMIT  = 417, // Maximum limit of exposure in AEAG procedure
       CAP_PROP_XI_AG_MAX_LIMIT  = 418, // Maximum limit of gain in AEAG procedure
       CAP_PROP_XI_AEAG_LEVEL    = 419, // Average intensity of output signal AEAG should achieve(in %)
       CAP_PROP_XI_TIMEOUT       = 420  // Image capture timeout in milliseconds
     };


// Properties for Android cameras
enum { CAP_PROP_ANDROID_AUTOGRAB               = 1024,
       CAP_PROP_ANDROID_PREVIEW_SIZES_STRING   = 1025, // readonly, tricky property, returns const char* indeed
       CAP_PROP_ANDROID_PREVIEW_FORMAT         = 1026, // readonly, tricky property, returns const char* indeed
       CAP_PROP_ANDROID_FLASH_MODE             = 8001,
       CAP_PROP_ANDROID_FOCUS_MODE             = 8002,
       CAP_PROP_ANDROID_WHITE_BALANCE          = 8003,
       CAP_PROP_ANDROID_ANTIBANDING            = 8004,
       CAP_PROP_ANDROID_FOCAL_LENGTH           = 8005,
       CAP_PROP_ANDROID_FOCUS_DISTANCE_NEAR    = 8006,
       CAP_PROP_ANDROID_FOCUS_DISTANCE_OPTIMAL = 8007,
       CAP_PROP_ANDROID_FOCUS_DISTANCE_FAR     = 8008
     };


// Android camera output formats
enum { CAP_ANDROID_COLOR_FRAME_BGR  = 0, //BGR
       CAP_ANDROID_COLOR_FRAME      = CAP_ANDROID_COLOR_FRAME_BGR,
       CAP_ANDROID_GREY_FRAME       = 1,  //Y
       CAP_ANDROID_GRAY_FRAME       = CAP_ANDROID_GREY_FRAME,
       CAP_ANDROID_COLOR_FRAME_RGB  = 2,
       CAP_ANDROID_COLOR_FRAME_BGRA = 3,
       CAP_ANDROID_COLOR_FRAME_RGBA = 4
     };


// Android camera flash modes
enum { CAP_ANDROID_FLASH_MODE_AUTO     = 0,
       CAP_ANDROID_FLASH_MODE_OFF      = 1,
       CAP_ANDROID_FLASH_MODE_ON       = 2,
       CAP_ANDROID_FLASH_MODE_RED_EYE  = 3,
       CAP_ANDROID_FLASH_MODE_TORCH    = 4
     };


// Android camera focus modes
enum { CAP_ANDROID_FOCUS_MODE_AUTO             = 0,
       CAP_ANDROID_FOCUS_MODE_CONTINUOUS_VIDEO = 1,
       CAP_ANDROID_FOCUS_MODE_EDOF             = 2,
       CAP_ANDROID_FOCUS_MODE_FIXED            = 3,
       CAP_ANDROID_FOCUS_MODE_INFINITY         = 4,
       CAP_ANDROID_FOCUS_MODE_MACRO            = 5
     };


// Android camera white balance modes
enum { CAP_ANDROID_WHITE_BALANCE_AUTO             = 0,
       CAP_ANDROID_WHITE_BALANCE_CLOUDY_DAYLIGHT  = 1,
       CAP_ANDROID_WHITE_BALANCE_DAYLIGHT         = 2,
       CAP_ANDROID_WHITE_BALANCE_FLUORESCENT      = 3,
       CAP_ANDROID_WHITE_BALANCE_INCANDESCENT     = 4,
       CAP_ANDROID_WHITE_BALANCE_SHADE            = 5,
       CAP_ANDROID_WHITE_BALANCE_TWILIGHT         = 6,
       CAP_ANDROID_WHITE_BALANCE_WARM_FLUORESCENT = 7
     };


// Android camera antibanding modes
enum { CAP_ANDROID_ANTIBANDING_50HZ = 0,
       CAP_ANDROID_ANTIBANDING_60HZ = 1,
       CAP_ANDROID_ANTIBANDING_AUTO = 2,
       CAP_ANDROID_ANTIBANDING_OFF  = 3
     };


// Properties of cameras available through AVFOUNDATION interface
enum { CAP_PROP_IOS_DEVICE_FOCUS        = 9001,
       CAP_PROP_IOS_DEVICE_EXPOSURE     = 9002,
       CAP_PROP_IOS_DEVICE_FLASH        = 9003,
       CAP_PROP_IOS_DEVICE_WHITEBALANCE = 9004,
       CAP_PROP_IOS_DEVICE_TORCH        = 9005
     };


// Properties of cameras available through Smartek Giganetix Ethernet Vision interface
/* --- Vladimir Litvinenko (litvinenko.vladimir@gmail.com) --- */
enum { CAP_PROP_GIGA_FRAME_OFFSET_X   = 10001,
       CAP_PROP_GIGA_FRAME_OFFSET_Y   = 10002,
       CAP_PROP_GIGA_FRAME_WIDTH_MAX  = 10003,
       CAP_PROP_GIGA_FRAME_HEIGH_MAX  = 10004,
       CAP_PROP_GIGA_FRAME_SENS_WIDTH = 10005,
       CAP_PROP_GIGA_FRAME_SENS_HEIGH = 10006
     };

enum { CAP_PROP_INTELPERC_PROFILE_COUNT               = 11001,
       CAP_PROP_INTELPERC_PROFILE_IDX                 = 11002,
       CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE  = 11003,
       CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE      = 11004,
       CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD  = 11005,
       CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ     = 11006,
       CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT     = 11007
     };

// Intel PerC streams
enum { CAP_INTELPERC_DEPTH_GENERATOR = 1 << 29,
       CAP_INTELPERC_IMAGE_GENERATOR = 1 << 28,
       CAP_INTELPERC_GENERATORS_MASK = CAP_INTELPERC_DEPTH_GENERATOR + CAP_INTELPERC_IMAGE_GENERATOR
     };

enum { CAP_INTELPERC_DEPTH_MAP              = 0, // Each pixel is a 16-bit integer. The value indicates the distance from an object to the camera's XY plane or the Cartesian depth.
       CAP_INTELPERC_UVDEPTH_MAP            = 1, // Each pixel contains two 32-bit floating point values in the range of 0-1, representing the mapping of depth coordinates to the color coordinates.
       CAP_INTELPERC_IR_MAP                 = 2, // Each pixel is a 16-bit integer. The value indicates the intensity of the reflected laser beam.
       CAP_INTELPERC_IMAGE                  = 3
     };

enum { VIDEOWRITER_PROP_QUALITY = 1,    // Quality (0..100%) of the videostream encoded
       VIDEOWRITER_PROP_FRAMEBYTES = 2, // (Read-only): Size of just encoded video frame
     };

class IVideoCapture;

/** @brief Class for video capturing from video files, image sequences or cameras. The class provides C++ API
for capturing video from cameras or for reading video files and image sequences. Here is how the
class can be used: :
@code
    #include "opencv2/opencv.hpp"

    using namespace cv;

    int main(int, char**)
    {
        VideoCapture cap(0); // open the default camera
        if(!cap.isOpened())  // check if we succeeded
            return -1;

        Mat edges;
        namedWindow("edges",1);
        for(;;)
        {
            Mat frame;
            cap >> frame; // get a new frame from camera
            cvtColor(frame, edges, COLOR_BGR2GRAY);
            GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
            Canny(edges, edges, 0, 30, 3);
            imshow("edges", edges);
            if(waitKey(30) >= 0) break;
        }
        // the camera will be deinitialized automatically in VideoCapture destructor
        return 0;
    }
@endcode
@note In C API the black-box structure CvCapture is used instead of VideoCapture.

@note
-   A basic sample on using the VideoCapture interface can be found at
    opencv_source_code/samples/cpp/starter_video.cpp
-   Another basic video processing sample can be found at
    opencv_source_code/samples/cpp/video_dmtx.cpp
-   (Python) A basic sample on using the VideoCapture interface can be found at
    opencv_source_code/samples/python2/video.py
-   (Python) Another basic video processing sample can be found at
    opencv_source_code/samples/python2/video_dmtx.py
-   (Python) A multi threaded video processing sample can be found at
    opencv_source_code/samples/python2/video_threaded.py
 */
class CV_EXPORTS_W VideoCapture
{
public:
    /** @brief
    @note In C API, when you finished working with video, release CvCapture structure with
    cvReleaseCapture(), or use Ptr\<CvCapture\> that calls cvReleaseCapture() automatically in the
    destructor.
     */
    CV_WRAP VideoCapture();

    /** @overload
    @param filename name of the opened video file (eg. video.avi) or image sequence (eg.
    img_%02d.jpg, which will read samples like img_00.jpg, img_01.jpg, img_02.jpg, ...)
    */
    CV_WRAP VideoCapture(const String& filename);

    /** @overload
    @param device id of the opened video capturing device (i.e. a camera index). If there is a single
    camera connected, just pass 0.
    */
    CV_WRAP VideoCapture(int device);

    virtual ~VideoCapture();

    /** @brief Open video file or a capturing device for video capturing

    @param filename name of the opened video file (eg. video.avi) or image sequence (eg.
    img_%02d.jpg, which will read samples like img_00.jpg, img_01.jpg, img_02.jpg, ...)

    The methods first call VideoCapture::release to close the already opened file or camera.
     */
    CV_WRAP virtual bool open(const String& filename);

    /** @overload
    @param device id of the opened video capturing device (i.e. a camera index).
    */
    CV_WRAP virtual bool open(int device);

    /** @brief Returns true if video capturing has been initialized already.

    If the previous call to VideoCapture constructor or VideoCapture::open succeeded, the method returns
    true.
     */
    CV_WRAP virtual bool isOpened() const;

    /** @brief Closes video file or capturing device.

    The methods are automatically called by subsequent VideoCapture::open and by VideoCapture
    destructor.

    The C function also deallocates memory and clears \*capture pointer.
     */
    CV_WRAP virtual void release();

    /** @brief Grabs the next frame from video file or capturing device.

    The methods/functions grab the next frame from video file or camera and return true (non-zero) in
    the case of success.

    The primary use of the function is in multi-camera environments, especially when the cameras do not
    have hardware synchronization. That is, you call VideoCapture::grab() for each camera and after that
    call the slower method VideoCapture::retrieve() to decode and get frame from each camera. This way
    the overhead on demosaicing or motion jpeg decompression etc. is eliminated and the retrieved frames
    from different cameras will be closer in time.

    Also, when a connected camera is multi-head (for example, a stereo camera or a Kinect device), the
    correct way of retrieving data from it is to call VideoCapture::grab first and then call
    VideoCapture::retrieve one or more times with different values of the channel parameter. See
    <https://github.com/Itseez/opencv/tree/master/samples/cpp/openni_capture.cpp>
     */
    CV_WRAP virtual bool grab();

    /** @brief Decodes and returns the grabbed video frame.

    The methods/functions decode and return the just grabbed frame. If no frames has been grabbed
    (camera has been disconnected, or there are no more frames in video file), the methods return false
    and the functions return NULL pointer.

    @note OpenCV 1.x functions cvRetrieveFrame and cv.RetrieveFrame return image stored inside the video
    capturing structure. It is not allowed to modify or release the image! You can copy the frame using
    :ocvcvCloneImage and then do whatever you want with the copy.
     */
    CV_WRAP virtual bool retrieve(OutputArray image, int flag = 0);
    virtual VideoCapture& operator >> (CV_OUT Mat& image);
    virtual VideoCapture& operator >> (CV_OUT UMat& image);

    /** @brief Grabs, decodes and returns the next video frame.

    The methods/functions combine VideoCapture::grab and VideoCapture::retrieve in one call. This is the
    most convenient method for reading video files or capturing data from decode and return the just
    grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more
    frames in video file), the methods return false and the functions return NULL pointer.

    @note OpenCV 1.x functions cvRetrieveFrame and cv.RetrieveFrame return image stored inside the video
    capturing structure. It is not allowed to modify or release the image! You can copy the frame using
    :ocvcvCloneImage and then do whatever you want with the copy.
     */
    CV_WRAP virtual bool read(OutputArray image);

    /** @brief Sets a property in the VideoCapture.

    @param propId Property identifier. It can be one of the following:
     -   **CAP_PROP_POS_MSEC** Current position of the video file in milliseconds.
     -   **CAP_PROP_POS_FRAMES** 0-based index of the frame to be decoded/captured next.
     -   **CAP_PROP_POS_AVI_RATIO** Relative position of the video file: 0 - start of the
         film, 1 - end of the film.
     -   **CAP_PROP_FRAME_WIDTH** Width of the frames in the video stream.
     -   **CAP_PROP_FRAME_HEIGHT** Height of the frames in the video stream.
     -   **CAP_PROP_FPS** Frame rate.
     -   **CAP_PROP_FOURCC** 4-character code of codec.
     -   **CAP_PROP_FRAME_COUNT** Number of frames in the video file.
     -   **CAP_PROP_FORMAT** Format of the Mat objects returned by retrieve() .
     -   **CAP_PROP_MODE** Backend-specific value indicating the current capture mode.
     -   **CAP_PROP_BRIGHTNESS** Brightness of the image (only for cameras).
     -   **CAP_PROP_CONTRAST** Contrast of the image (only for cameras).
     -   **CAP_PROP_SATURATION** Saturation of the image (only for cameras).
     -   **CAP_PROP_HUE** Hue of the image (only for cameras).
     -   **CAP_PROP_GAIN** Gain of the image (only for cameras).
     -   **CAP_PROP_EXPOSURE** Exposure (only for cameras).
     -   **CAP_PROP_CONVERT_RGB** Boolean flags indicating whether images should be converted
         to RGB.
     -   **CAP_PROP_WHITE_BALANCE** Currently unsupported
     -   **CAP_PROP_RECTIFICATION** Rectification flag for stereo cameras (note: only supported
         by DC1394 v 2.x backend currently)
    @param value Value of the property.
     */
    CV_WRAP virtual bool set(int propId, double value);

    /** @brief Returns the specified VideoCapture property

    @param propId Property identifier. It can be one of the following:
     -   **CAP_PROP_POS_MSEC** Current position of the video file in milliseconds or video
         capture timestamp.
     -   **CAP_PROP_POS_FRAMES** 0-based index of the frame to be decoded/captured next.
     -   **CAP_PROP_POS_AVI_RATIO** Relative position of the video file: 0 - start of the
         film, 1 - end of the film.
     -   **CAP_PROP_FRAME_WIDTH** Width of the frames in the video stream.
     -   **CAP_PROP_FRAME_HEIGHT** Height of the frames in the video stream.
     -   **CAP_PROP_FPS** Frame rate.
     -   **CAP_PROP_FOURCC** 4-character code of codec.
     -   **CAP_PROP_FRAME_COUNT** Number of frames in the video file.
     -   **CAP_PROP_FORMAT** Format of the Mat objects returned by retrieve() .
     -   **CAP_PROP_MODE** Backend-specific value indicating the current capture mode.
     -   **CAP_PROP_BRIGHTNESS** Brightness of the image (only for cameras).
     -   **CAP_PROP_CONTRAST** Contrast of the image (only for cameras).
     -   **CAP_PROP_SATURATION** Saturation of the image (only for cameras).
     -   **CAP_PROP_HUE** Hue of the image (only for cameras).
     -   **CAP_PROP_GAIN** Gain of the image (only for cameras).
     -   **CAP_PROP_EXPOSURE** Exposure (only for cameras).
     -   **CAP_PROP_CONVERT_RGB** Boolean flags indicating whether images should be converted
         to RGB.
     -   **CAP_PROP_WHITE_BALANCE** Currently not supported
     -   **CAP_PROP_RECTIFICATION** Rectification flag for stereo cameras (note: only supported
         by DC1394 v 2.x backend currently)

    @note When querying a property that is not supported by the backend used by the VideoCapture
    class, value 0 is returned.
     */
    CV_WRAP virtual double get(int propId) const;

protected:
    Ptr<CvCapture> cap;
    Ptr<IVideoCapture> icap;
};

class IVideoWriter;

/** @brief Video writer class.
 */
class CV_EXPORTS_W VideoWriter
{
public:
    /** @brief VideoWriter constructors

    The constructors/functions initialize video writers. On Linux FFMPEG is used to write videos; on
    Windows FFMPEG or VFW is used; on MacOSX QTKit is used.
     */
    CV_WRAP VideoWriter();

    /** @overload
    @param filename Name of the output video file.
    @param fourcc 4-character code of codec used to compress the frames. For example,
    VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec, VideoWriter::fourcc('M','J','P','G') is a
    motion-jpeg codec etc. List of codes can be obtained at [Video Codecs by
    FOURCC](http://www.fourcc.org/codecs.php) page.
    @param fps Framerate of the created video stream.
    @param frameSize Size of the video frames.
    @param isColor If it is not zero, the encoder will expect and encode color frames, otherwise it
    will work with grayscale frames (the flag is currently supported on Windows only).
    */
    CV_WRAP VideoWriter(const String& filename, int fourcc, double fps,
                Size frameSize, bool isColor = true);

    virtual ~VideoWriter();

    /** @brief Initializes or reinitializes video writer.

    The method opens video writer. Parameters are the same as in the constructor
    VideoWriter::VideoWriter.

     */
    CV_WRAP virtual bool open(const String& filename, int fourcc, double fps,
                      Size frameSize, bool isColor = true);

    /** @brief Returns true if video writer has been successfully initialized.
    */
    CV_WRAP virtual bool isOpened() const;
    CV_WRAP virtual void release();
    virtual VideoWriter& operator << (const Mat& image);

    /** @brief Writes the next video frame

    @param image The written frame

    The functions/methods write the specified image to video file. It must have the same size as has
    been specified when opening the video writer.
     */
    CV_WRAP virtual void write(const Mat& image);

    /** @brief Sets a property in the VideoWriter.

     @param propId Property identifier. It can be one of the following:
     -   **VIDEOWRITER_PROP_QUALITY** Quality (0..100%) of the videostream encoded. Can be adjusted dynamically in some codecs.
     @param value Value of the property.
     */
    CV_WRAP virtual bool set(int propId, double value);

    /** @brief Returns the specified VideoWriter property

     @param propId Property identifier. It can be one of the following:
     -   **VIDEOWRITER_PROP_QUALITY** Current quality of the encoded videostream.
     -   **VIDEOWRITER_PROP_FRAMEBYTES** (Read-only) Size of just encoded video frame; note that the encoding order may be different from representation order.

     @note When querying a property that is not supported by the backend used by the VideoWriter
     class, value 0 is returned.
     */
    CV_WRAP virtual double get(int propId) const;

    /** @brief Concatenates 4 chars to a fourcc code

    This static method constructs the fourcc code of the codec to be used in the constructor
    VideoWriter::VideoWriter or VideoWriter::open.
     */
    CV_WRAP static int fourcc(char c1, char c2, char c3, char c4);

protected:
    Ptr<CvVideoWriter> writer;
    Ptr<IVideoWriter> iwriter;

    static Ptr<IVideoWriter> create(const String& filename, int fourcc, double fps,
                                    Size frameSize, bool isColor = true);
};

template<> CV_EXPORTS void DefaultDeleter<CvCapture>::operator ()(CvCapture* obj) const;
template<> CV_EXPORTS void DefaultDeleter<CvVideoWriter>::operator ()(CvVideoWriter* obj) const;

//! @} videoio

} // cv

#endif //__OPENCV_VIDEOIO_HPP__
