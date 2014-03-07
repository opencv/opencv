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

#ifndef __OPENCV_HIGHGUI_HPP__
#define __OPENCV_HIGHGUI_HPP__

#include "opencv2/core.hpp"


///////////////////////// graphical user interface //////////////////////////
namespace cv
{

// Flags for namedWindow
enum { WINDOW_NORMAL     = 0x00000000, // the user can resize the window (no constraint) / also use to switch a fullscreen window to a normal size
       WINDOW_AUTOSIZE   = 0x00000001, // the user cannot resize the window, the size is constrainted by the image displayed
       WINDOW_OPENGL     = 0x00001000, // window with opengl support

       WINDOW_FULLSCREEN = 1,          // change the window to fullscreen
       WINDOW_FREERATIO  = 0x00000100, // the image expends as much as it can (no ratio constraint)
       WINDOW_KEEPRATIO  = 0x00000000  // the ratio of the image is respected
     };

// Flags for set / getWindowProperty
enum { WND_PROP_FULLSCREEN   = 0, // fullscreen property    (can be WINDOW_NORMAL or WINDOW_FULLSCREEN)
       WND_PROP_AUTOSIZE     = 1, // autosize property      (can be WINDOW_NORMAL or WINDOW_AUTOSIZE)
       WND_PROP_ASPECT_RATIO = 2, // window's aspect ration (can be set to WINDOW_FREERATIO or WINDOW_KEEPRATIO);
       WND_PROP_OPENGL       = 3  // opengl support
     };

enum { EVENT_MOUSEMOVE      = 0,
       EVENT_LBUTTONDOWN    = 1,
       EVENT_RBUTTONDOWN    = 2,
       EVENT_MBUTTONDOWN    = 3,
       EVENT_LBUTTONUP      = 4,
       EVENT_RBUTTONUP      = 5,
       EVENT_MBUTTONUP      = 6,
       EVENT_LBUTTONDBLCLK  = 7,
       EVENT_RBUTTONDBLCLK  = 8,
       EVENT_MBUTTONDBLCLK  = 9,
       EVENT_MOUSEWHEEL     = 10,
       EVENT_MOUSEHWHEEL    = 11
     };

enum { EVENT_FLAG_LBUTTON   = 1,
       EVENT_FLAG_RBUTTON   = 2,
       EVENT_FLAG_MBUTTON   = 4,
       EVENT_FLAG_CTRLKEY   = 8,
       EVENT_FLAG_SHIFTKEY  = 16,
       EVENT_FLAG_ALTKEY    = 32
     };

// Qt font
enum {  QT_FONT_LIGHT           = 25, //QFont::Light,
        QT_FONT_NORMAL          = 50, //QFont::Normal,
        QT_FONT_DEMIBOLD        = 63, //QFont::DemiBold,
        QT_FONT_BOLD            = 75, //QFont::Bold,
        QT_FONT_BLACK           = 87  //QFont::Black
     };

// Qt font style
enum {  QT_STYLE_NORMAL         = 0, //QFont::StyleNormal,
        QT_STYLE_ITALIC         = 1, //QFont::StyleItalic,
        QT_STYLE_OBLIQUE        = 2  //QFont::StyleOblique
     };

// Qt "button" type
enum { QT_PUSH_BUTTON = 0,
       QT_CHECKBOX    = 1,
       QT_RADIOBOX    = 2
     };


typedef void (*MouseCallback)(int event, int x, int y, int flags, void* userdata);
typedef void (*TrackbarCallback)(int pos, void* userdata);
typedef void (*OpenGlDrawCallback)(void* userdata);
typedef void (*ButtonCallback)(int state, void* userdata);


CV_EXPORTS_W void namedWindow(const String& winname, int flags = WINDOW_AUTOSIZE);

CV_EXPORTS_W void destroyWindow(const String& winname);

CV_EXPORTS_W void destroyAllWindows();

CV_EXPORTS_W int startWindowThread();

CV_EXPORTS_W int waitKey(int delay = 0);

CV_EXPORTS_W void imshow(const String& winname, InputArray mat);

CV_EXPORTS_W void resizeWindow(const String& winname, int width, int height);

CV_EXPORTS_W void moveWindow(const String& winname, int x, int y);

CV_EXPORTS_W void setWindowProperty(const String& winname, int prop_id, double prop_value);

CV_EXPORTS_W double getWindowProperty(const String& winname, int prop_id);

//! assigns callback for mouse events
CV_EXPORTS void setMouseCallback(const String& winname, MouseCallback onMouse, void* userdata = 0);

CV_EXPORTS int getMouseWheelDelta(int flags);

CV_EXPORTS int createTrackbar(const String& trackbarname, const String& winname,
                              int* value, int count,
                              TrackbarCallback onChange = 0,
                              void* userdata = 0);

CV_EXPORTS_W int getTrackbarPos(const String& trackbarname, const String& winname);

CV_EXPORTS_W void setTrackbarPos(const String& trackbarname, const String& winname, int pos);


// OpenGL support
CV_EXPORTS void imshow(const String& winname, const ogl::Texture2D& tex);

CV_EXPORTS void setOpenGlDrawCallback(const String& winname, OpenGlDrawCallback onOpenGlDraw, void* userdata = 0);

CV_EXPORTS void setOpenGlContext(const String& winname);

CV_EXPORTS void updateWindow(const String& winname);


// Only for Qt

struct QtFont
{
    const char* nameFont;  // Qt: nameFont
    Scalar      color;     // Qt: ColorFont -> cvScalar(blue_component, green_component, red\_component[, alpha_component])
    int         font_face; // Qt: bool italic
    const int*  ascii;     // font data and metrics
    const int*  greek;
    const int*  cyrillic;
    float       hscale, vscale;
    float       shear;     // slope coefficient: 0 - normal, >0 - italic
    int         thickness; // Qt: weight
    float       dx;        // horizontal interval between letters
    int         line_type; // Qt: PointSize
};

CV_EXPORTS QtFont fontQt(const String& nameFont, int pointSize = -1,
                         Scalar color = Scalar::all(0), int weight = QT_FONT_NORMAL,
                         int style = QT_STYLE_NORMAL, int spacing = 0);

CV_EXPORTS void addText( const Mat& img, const String& text, Point org, const QtFont& font);

CV_EXPORTS void displayOverlay(const String& winname, const String& text, int delayms = 0);

CV_EXPORTS void displayStatusBar(const String& winname, const String& text, int delayms = 0);

CV_EXPORTS void saveWindowParameters(const String& windowName);

CV_EXPORTS void loadWindowParameters(const String& windowName);

CV_EXPORTS  int startLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[]);

CV_EXPORTS  void stopLoop();

CV_EXPORTS int createButton( const String& bar_name, ButtonCallback on_change,
                             void* userdata = 0, int type = QT_PUSH_BUTTON,
                             bool initial_button_state = false);

} // cv



//////////////////////////////// image codec ////////////////////////////////
namespace cv
{

enum { IMREAD_UNCHANGED  = -1, // 8bit, color or not
       IMREAD_GRAYSCALE  = 0,  // 8bit, gray
       IMREAD_COLOR      = 1,  // ?, color
       IMREAD_ANYDEPTH   = 2,  // any depth, ?
       IMREAD_ANYCOLOR   = 4   // ?, any color
     };

enum { IMWRITE_JPEG_QUALITY    = 1,
       IMWRITE_PNG_COMPRESSION = 16,
       IMWRITE_PNG_STRATEGY    = 17,
       IMWRITE_PNG_BILEVEL     = 18,
       IMWRITE_PXM_BINARY      = 32,
       IMWRITE_WEBP_QUALITY    = 64
     };

enum { IMWRITE_PNG_STRATEGY_DEFAULT      = 0,
       IMWRITE_PNG_STRATEGY_FILTERED     = 1,
       IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2,
       IMWRITE_PNG_STRATEGY_RLE          = 3,
       IMWRITE_PNG_STRATEGY_FIXED        = 4
     };

CV_EXPORTS_W Mat imread( const String& filename, int flags = IMREAD_COLOR );

CV_EXPORTS_W bool imwrite( const String& filename, InputArray img,
              const std::vector<int>& params = std::vector<int>());

CV_EXPORTS_W Mat imdecode( InputArray buf, int flags );

CV_EXPORTS Mat imdecode( InputArray buf, int flags, Mat* dst);

CV_EXPORTS_W bool imencode( const String& ext, InputArray img,
                            CV_OUT std::vector<uchar>& buf,
                            const std::vector<int>& params = std::vector<int>());

} // cv



////////////////////////////////// video io /////////////////////////////////

typedef struct CvCapture CvCapture;
typedef struct CvVideoWriter CvVideoWriter;

namespace cv
{

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
       CAP_INTELPERC    = 1500   // Intel Perceptual Computing SDK
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
       CAP_PROP_MONOCROME     =19,
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
       CAP_PROP_OPENNI_GENERATOR_PRESENT = 109
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
enum { CAP_PROP_PVAPI_MULTICASTIP               = 300, // ip for anable multicast master mode. 0 for disable multicast
       CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE     = 301  // FrameStartTriggerMode: Determines how a frame is initiated
     };

// PVAPI: FrameStartTriggerMode
enum { CAP_PVAPI_FSTRIGMODE_FREERUN     = 0,    // Freerun
       CAP_PVAPI_FSTRIGMODE_SYNCIN1     = 1,    // SyncIn1
       CAP_PVAPI_FSTRIGMODE_SYNCIN2     = 2,    // SyncIn2
       CAP_PVAPI_FSTRIGMODE_FIXEDRATE   = 3,    // FixedRate
       CAP_PVAPI_FSTRIGMODE_SOFTWARE    = 4     // Software
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


class IVideoCapture;
class CV_EXPORTS_W VideoCapture
{
public:
    CV_WRAP VideoCapture();
    CV_WRAP VideoCapture(const String& filename);
    CV_WRAP VideoCapture(int device);

    virtual ~VideoCapture();
    CV_WRAP virtual bool open(const String& filename);
    CV_WRAP virtual bool open(int device);
    CV_WRAP virtual bool isOpened() const;
    CV_WRAP virtual void release();

    CV_WRAP virtual bool grab();
    CV_WRAP virtual bool retrieve(OutputArray image, int flag = 0);
    virtual VideoCapture& operator >> (CV_OUT Mat& image);
    virtual VideoCapture& operator >> (CV_OUT UMat& image);
    CV_WRAP virtual bool read(OutputArray image);

    CV_WRAP virtual bool set(int propId, double value);
    CV_WRAP virtual double get(int propId);

protected:
    Ptr<CvCapture> cap;
    Ptr<IVideoCapture> icap;
private:
    static Ptr<IVideoCapture> createCameraCapture(int index);
};

class CV_EXPORTS_W VideoWriter
{
public:
    CV_WRAP VideoWriter();
    CV_WRAP VideoWriter(const String& filename, int fourcc, double fps,
                Size frameSize, bool isColor = true);

    virtual ~VideoWriter();
    CV_WRAP virtual bool open(const String& filename, int fourcc, double fps,
                      Size frameSize, bool isColor = true);
    CV_WRAP virtual bool isOpened() const;
    CV_WRAP virtual void release();
    virtual VideoWriter& operator << (const Mat& image);
    CV_WRAP virtual void write(const Mat& image);

    CV_WRAP static int fourcc(char c1, char c2, char c3, char c4);

protected:
    Ptr<CvVideoWriter> writer;
};

template<> CV_EXPORTS void DefaultDeleter<CvCapture>::operator ()(CvCapture* obj) const;
template<> CV_EXPORTS void DefaultDeleter<CvVideoWriter>::operator ()(CvVideoWriter* obj) const;

} // cv

#endif
