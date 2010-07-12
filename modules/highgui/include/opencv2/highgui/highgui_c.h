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
//M*/

#ifndef __OPENCV_HIGHGUI_H__
#define __OPENCV_HIGHGUI_H__

#include "opencv2/core/core_c.h"
#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
   	#include <windows.h>
	#undef min
	#undef max
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/****************************************************************************************\
*                                  Basic GUI functions                                   *
\****************************************************************************************/
//YV
//-----------New for Qt
/* For font */
enum {	CV_FONT_LIGHT 			= 25,//QFont::Light,
		CV_FONT_NORMAL 			= 50,//QFont::Normal,
		CV_FONT_DEMIBOLD 		= 63,//QFont::DemiBold,
		CV_FONT_BOLD 			= 75,//QFont::Bold,
		CV_FONT_BLACK 			= 87 //QFont::Black
};

enum {	CV_STYLE_NORMAL			= 0,//QFont::StyleNormal,
		CV_STYLE_ITALIC 		= 1,//QFont::StyleItalic,
		CV_STYLE_OBLIQUE 		= 2 //QFont::StyleOblique
};
/* ---------*/

//for color cvScalar(blue_component, green_component, red\_component[, alpha_component])
//and alpha= 0 <-> 0xFF (not transparent <-> transparent)
CVAPI(CvFont) cvFont_Qt(const char* nameFont, int pointSize CV_DEFAULT(-1), CvScalar color CV_DEFAULT(cvScalarAll(0)), int weight CV_DEFAULT(CV_FONT_NORMAL),  int style CV_DEFAULT(CV_STYLE_NORMAL), int spacing CV_DEFAULT(0));

CVAPI(void) cvAddText( CvArr* img, const char* text, CvPoint org, CvFont *arg2);

CVAPI(void) cvDisplayOverlay(const char* name, const char* text, int delayms);
CVAPI(void) cvDisplayStatusBar(const char* name, const char* text, int delayms);

typedef void (CV_CDECL *CvOpenGLCallback)(void* userdata);
CVAPI(void) cvCreateOpenGLCallback( const char* window_name, CvOpenGLCallback callbackOpenGL, void* userdata CV_DEFAULT(NULL));

CVAPI(void) cvSaveWindowParameters(const char* name);
CVAPI(void) cvLoadWindowParameters(const char* name);
CVAPI(int) cvStartLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[]);
CVAPI(void) cvStopLoop();
//----------------------


/* this function is used to set some external parameters in case of X Window */
CVAPI(int) cvInitSystem( int argc, char** argv );

CVAPI(int) cvStartWindowThread();

enum { CV_WINDOW_AUTOSIZE = 1 };

/* create window */
CVAPI(int) cvNamedWindow( const char* name, int flags CV_DEFAULT(CV_WINDOW_AUTOSIZE) );

// ---------  YV ---------
enum
{
	CV_WND_PROP_FULLSCREEN = 0,
	CV_WND_PROP_AUTOSIZE   = 1,
	CV_WND_PROP_ASPECTRATIO= 2,
	CV_WINDOW_NORMAL       = 0,
	CV_WINDOW_FULLSCREEN   = 1,
	CV_WINDOW_FREERATIO	   = 0,
	CV_WINDOW_KEEPRATIO    = 1
};


/* Set and Get Property of the window */
CVAPI(void) cvSetWindowProperty(const char* name, int prop_id, double prop_value);
CVAPI(double) cvGetWindowProperty(const char* name, int prop_id);

/* display image within window (highgui windows remember their content) */
CVAPI(void) cvShowImage( const char* name, const CvArr* image );

/* resize/move window */
CVAPI(void) cvResizeWindow( const char* name, int width, int height );
CVAPI(void) cvMoveWindow( const char* name, int x, int y );


/* destroy window and all the trackers associated with it */
CVAPI(void) cvDestroyWindow( const char* name );

CVAPI(void) cvDestroyAllWindows(void);

/* get native window handle (HWND in case of Win32 and Widget in case of X Window) */
CVAPI(void*) cvGetWindowHandle( const char* name );

/* get name of highgui window given its native handle */
CVAPI(const char*) cvGetWindowName( void* window_handle );


typedef void (CV_CDECL *CvTrackbarCallback)(int pos);

/* create trackbar and display it on top of given window, set callback */
CVAPI(int) cvCreateTrackbar( const char* trackbar_name, const char* window_name,
                             int* value, int count, CvTrackbarCallback on_change);

typedef void (CV_CDECL *CvTrackbarCallback2)(int pos, void* userdata);

CVAPI(int) cvCreateTrackbar2( const char* trackbar_name, const char* window_name,
                              int* value, int count, CvTrackbarCallback2 on_change,
                              void* userdata CV_DEFAULT(0));

/* retrieve or set trackbar position */
CVAPI(int) cvGetTrackbarPos( const char* trackbar_name, const char* window_name );
CVAPI(void) cvSetTrackbarPos( const char* trackbar_name, const char* window_name, int pos );

enum
{
	CV_EVENT_MOUSEMOVE      =0,
	CV_EVENT_LBUTTONDOWN    =1,
	CV_EVENT_RBUTTONDOWN    =2,
	CV_EVENT_MBUTTONDOWN    =3,
	CV_EVENT_LBUTTONUP      =4,
	CV_EVENT_RBUTTONUP      =5,
	CV_EVENT_MBUTTONUP      =6,
	CV_EVENT_LBUTTONDBLCLK  =7,
	CV_EVENT_RBUTTONDBLCLK  =8,
	CV_EVENT_MBUTTONDBLCLK  =9
};

enum
{
	CV_EVENT_FLAG_LBUTTON   =1,
	CV_EVENT_FLAG_RBUTTON   =2,
	CV_EVENT_FLAG_MBUTTON   =4,
	CV_EVENT_FLAG_CTRLKEY   =8,
	CV_EVENT_FLAG_SHIFTKEY  =16,
	CV_EVENT_FLAG_ALTKEY    =32
};

typedef void (CV_CDECL *CvMouseCallback )(int event, int x, int y, int flags, void* param);

/* assign callback for mouse events */
CVAPI(void) cvSetMouseCallback( const char* window_name, CvMouseCallback on_mouse,
                                void* param CV_DEFAULT(NULL));

enum
{
/* 8bit, color or not */
	CV_LOAD_IMAGE_UNCHANGED  =-1,
/* 8bit, gray */
	CV_LOAD_IMAGE_GRAYSCALE  =0,
/* ?, color */
	CV_LOAD_IMAGE_COLOR      =1,
/* any depth, ? */
	CV_LOAD_IMAGE_ANYDEPTH   =2,
/* ?, any color */
	CV_LOAD_IMAGE_ANYCOLOR   =4
};

/* load image from file
  iscolor can be a combination of above flags where CV_LOAD_IMAGE_UNCHANGED
  overrides the other flags
  using CV_LOAD_IMAGE_ANYCOLOR alone is equivalent to CV_LOAD_IMAGE_UNCHANGED
  unless CV_LOAD_IMAGE_ANYDEPTH is specified images are converted to 8bit
*/
CVAPI(IplImage*) cvLoadImage( const char* filename, int iscolor CV_DEFAULT(CV_LOAD_IMAGE_COLOR));
CVAPI(CvMat*) cvLoadImageM( const char* filename, int iscolor CV_DEFAULT(CV_LOAD_IMAGE_COLOR));

enum
{
	CV_IMWRITE_JPEG_QUALITY =1,
	CV_IMWRITE_PNG_COMPRESSION =16,
	CV_IMWRITE_PXM_BINARY =32
};

/* save image to file */
CVAPI(int) cvSaveImage( const char* filename, const CvArr* image,
                        const int* params CV_DEFAULT(0) );

/* decode image stored in the buffer */
CVAPI(IplImage*) cvDecodeImage( const CvMat* buf, int iscolor CV_DEFAULT(CV_LOAD_IMAGE_COLOR));
CVAPI(CvMat*) cvDecodeImageM( const CvMat* buf, int iscolor CV_DEFAULT(CV_LOAD_IMAGE_COLOR));

/* encode image and store the result as a byte vector (single-row 8uC1 matrix) */
CVAPI(CvMat*) cvEncodeImage( const char* ext, const CvArr* image,
                             const int* params CV_DEFAULT(0) );

enum
{
	CV_CVTIMG_FLIP      =1,
	CV_CVTIMG_SWAP_RB   =2
};

/* utility function: convert one image to another with optional vertical flip */
CVAPI(void) cvConvertImage( const CvArr* src, CvArr* dst, int flags CV_DEFAULT(0));

/* wait for key event infinitely (delay<=0) or for "delay" milliseconds */
CVAPI(int) cvWaitKey(int delay CV_DEFAULT(0));

/****************************************************************************************\
*                         Working with Video Files and Cameras                           *
\****************************************************************************************/

/* "black box" capture structure */
typedef struct CvCapture CvCapture;

/* start capturing frames from video file */
CVAPI(CvCapture*) cvCreateFileCapture( const char* filename );

enum
{
	CV_CAP_ANY      =0,     // autodetect

	CV_CAP_MIL      =100,   // MIL proprietary drivers

	CV_CAP_VFW      =200,   // platform native
	CV_CAP_V4L      =200,
	CV_CAP_V4L2     =200,

	CV_CAP_FIREWARE =300,   // IEEE 1394 drivers
	CV_CAP_FIREWIRE =300,
	CV_CAP_IEEE1394 =300,
	CV_CAP_DC1394   =300,
	CV_CAP_CMU1394  =300,

	CV_CAP_STEREO   =400,   // TYZX proprietary drivers
	CV_CAP_TYZX     =400,
	CV_TYZX_LEFT    =400,
	CV_TYZX_RIGHT   =401,
	CV_TYZX_COLOR   =402,
	CV_TYZX_Z       =403,

	CV_CAP_QT       =500,   // QuickTime

	CV_CAP_UNICAP   =600,   // Unicap drivers

	CV_CAP_DSHOW    =700,   // DirectShow (via videoInput)

	CV_CAP_PVAPI    =800   // PvAPI, Prosilica GigE SDK
};

/* start capturing frames from camera: index = camera_index + domain_offset (CV_CAP_*) */
CVAPI(CvCapture*) cvCreateCameraCapture( int index );

/* grab a frame, return 1 on success, 0 on fail.
  this function is thought to be fast               */
CVAPI(int) cvGrabFrame( CvCapture* capture );

/* get the frame grabbed with cvGrabFrame(..)
  This function may apply some frame processing like
  frame decompression, flipping etc.
  !!!DO NOT RELEASE or MODIFY the retrieved frame!!! */
CVAPI(IplImage*) cvRetrieveFrame( CvCapture* capture, int streamIdx CV_DEFAULT(0) );

/* Just a combination of cvGrabFrame and cvRetrieveFrame
   !!!DO NOT RELEASE or MODIFY the retrieved frame!!!      */
CVAPI(IplImage*) cvQueryFrame( CvCapture* capture );

/* stop capturing/reading and free resources */
CVAPI(void) cvReleaseCapture( CvCapture** capture );

enum
{
	CV_CAP_PROP_POS_MSEC       =0,
	CV_CAP_PROP_POS_FRAMES     =1,
	CV_CAP_PROP_POS_AVI_RATIO  =2,
	CV_CAP_PROP_FRAME_WIDTH    =3,
	CV_CAP_PROP_FRAME_HEIGHT   =4,
	CV_CAP_PROP_FPS            =5,
	CV_CAP_PROP_FOURCC         =6,
	CV_CAP_PROP_FRAME_COUNT    =7,
	CV_CAP_PROP_FORMAT         =8,
	CV_CAP_PROP_MODE           =9,
	CV_CAP_PROP_BRIGHTNESS    =10,
	CV_CAP_PROP_CONTRAST      =11,
	CV_CAP_PROP_SATURATION    =12,
	CV_CAP_PROP_HUE           =13,
	CV_CAP_PROP_GAIN          =14,
	CV_CAP_PROP_EXPOSURE      =15,
	CV_CAP_PROP_CONVERT_RGB   =16,
	CV_CAP_PROP_WHITE_BALANCE =17,
	CV_CAP_PROP_RECTIFICATION =18,
	CV_CAP_PROP_MONOCROME	  =19
};

/* retrieve or set capture properties */
CVAPI(double) cvGetCaptureProperty( CvCapture* capture, int property_id );
CVAPI(int)    cvSetCaptureProperty( CvCapture* capture, int property_id, double value );

// Return the type of the capturer (eg, CV_CAP_V4W, CV_CAP_UNICAP), which is unknown if created with CV_CAP_ANY
CVAPI(int)    cvGetCaptureDomain( CvCapture* capture);  

/* "black box" video file writer structure */
typedef struct CvVideoWriter CvVideoWriter;

CV_INLINE int CV_FOURCC(char c1, char c2, char c3, char c4)
{
	return (c1 & 255) + ((c2 & 255) << 8) + ((c3 &255) << 16) + ((c4 & 255) << 24);
}

#define CV_FOURCC_PROMPT -1  /* Open Codec Selection Dialog (Windows only) */
#define CV_FOURCC_DEFAULT CV_FOURCC('I', 'Y', 'U', 'V') /* Use default codec for specified filename (Linux only) */

/* initialize video file writer */
CVAPI(CvVideoWriter*) cvCreateVideoWriter( const char* filename, int fourcc,
                                           double fps, CvSize frame_size,
                                           int is_color CV_DEFAULT(1));

//CVAPI(CvVideoWriter*) cvCreateImageSequenceWriter( const char* filename,
//                                                   int is_color CV_DEFAULT(1));

/* write frame to video file */
CVAPI(int) cvWriteFrame( CvVideoWriter* writer, const IplImage* image );

/* close video file writer */
CVAPI(void) cvReleaseVideoWriter( CvVideoWriter** writer );

/****************************************************************************************\
*                              Obsolete functions/synonyms                               *
\****************************************************************************************/

#define cvCaptureFromFile cvCreateFileCapture
#define cvCaptureFromCAM cvCreateCameraCapture
#define cvCaptureFromAVI cvCaptureFromFile
#define cvCreateAVIWriter cvCreateVideoWriter
#define cvWriteToAVI cvWriteFrame
#define cvAddSearchPath(path)
#define cvvInitSystem cvInitSystem
#define cvvNamedWindow cvNamedWindow
#define cvvShowImage cvShowImage
#define cvvResizeWindow cvResizeWindow
#define cvvDestroyWindow cvDestroyWindow
#define cvvCreateTrackbar cvCreateTrackbar
#define cvvLoadImage(name) cvLoadImage((name),1)
#define cvvSaveImage cvSaveImage
#define cvvAddSearchPath cvAddSearchPath
#define cvvWaitKey(name) cvWaitKey(0)
#define cvvWaitKeyEx(name,delay) cvWaitKey(delay)
#define cvvConvertImage cvConvertImage
#define HG_AUTOSIZE CV_WINDOW_AUTOSIZE
#define set_preprocess_func cvSetPreprocessFuncWin32
#define set_postprocess_func cvSetPostprocessFuncWin32

#if defined WIN32 || defined _WIN32

typedef int (CV_CDECL * CvWin32WindowCallback)(HWND, UINT, WPARAM, LPARAM, int*);
CVAPI(void) cvSetPreprocessFuncWin32( CvWin32WindowCallback on_preprocess );
CVAPI(void) cvSetPostprocessFuncWin32( CvWin32WindowCallback on_postprocess );

#endif

#ifdef __cplusplus
}
#endif

#endif
