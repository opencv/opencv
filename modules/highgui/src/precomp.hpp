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

#ifndef __HIGHGUI_H_
#define __HIGHGUI_H_

#if _MSC_VER >= 1200
#pragma warning( disable: 4251 )
#endif

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>
#include <assert.h>

#if !defined WIN32 && !defined _WIN32
#include "cvconfig.h"
#else
void  FillBitmapInfo( BITMAPINFO* bmi, int width, int height, int bpp, int origin );
#endif

/* Errors */
#define HG_OK          0 /* Don't bet on it! */
#define HG_BADNAME    -1 /* Bad window or file name */
#define HG_INITFAILED -2 /* Can't initialize HigHGUI */
#define HG_WCFAILED   -3 /* Can't create a window */
#define HG_NULLPTR    -4 /* The null pointer where it should not appear */
#define HG_BADPARAM   -5

#define __BEGIN__ __CV_BEGIN__
#define __END__  __CV_END__
#define EXIT __CV_EXIT__

#define CV_WINDOW_MAGIC_VAL     0x00420042
#define CV_TRACKBAR_MAGIC_VAL   0x00420043

/***************************** CvCapture structure ******************************/

struct CvCapture
{
    virtual ~CvCapture() {}
    virtual double getProperty(int) { return 0; }
    virtual bool setProperty(int, double) { return 0; }
    virtual bool grabFrame() { return true; }
    virtual IplImage* retrieveFrame(int) { return 0; }
    virtual IplImage* queryFrame() { return grabFrame() ? retrieveFrame(0) : 0; }
	virtual int getCaptureDomain() { return CV_CAP_ANY; } // Return the type of the capture object: CV_CAP_VFW, etc...
};

/*************************** CvVideoWriter structure ****************************/

struct CvVideoWriter
{
    virtual ~CvVideoWriter() {}
    virtual bool writeFrame(const IplImage*) { return false; }
};

#if defined WIN32 || defined _WIN32
#define HAVE_VFW 1

#if (_MSC_VER >= 1400 || defined __GNUC__) && !defined WIN64 && !defined _WIN64
#define HAVE_VIDEOINPUT 1
#endif

/* uncomment to enable OpenEXR codec (will not compile under MSVC6) */
//#define HAVE_ILMIMF 1

/* uncomment to enable CMUCamera1394 fireware camera module */
//#define HAVE_CMU1394 1
#endif


CvCapture * cvCreateCameraCapture_V4L( int index );
CvCapture * cvCreateCameraCapture_DC1394( int index );
CvCapture * cvCreateCameraCapture_DC1394_2( int index );
CvCapture* cvCreateCameraCapture_MIL( int index );
CvCapture * cvCreateCameraCapture_CMU( int index );
CV_IMPL CvCapture * cvCreateCameraCapture_TYZX( int index );
CvCapture* cvCreateFileCapture_Win32( const char* filename );
CvCapture* cvCreateCameraCapture_VFW( int index );
CvCapture* cvCreateFileCapture_VFW( const char* filename );
CvVideoWriter* cvCreateVideoWriter_Win32( const char* filename, int fourcc,
                                          double fps, CvSize frameSize, int is_color );
CvVideoWriter* cvCreateVideoWriter_VFW( const char* filename, int fourcc,
                                        double fps, CvSize frameSize, int is_color );
CvCapture* cvCreateCameraCapture_DShow( int index );

CVAPI(int) cvHaveImageReader(const char* filename);
CVAPI(int) cvHaveImageWriter(const char* filename);

CvCapture* cvCreateFileCapture_Images(const char* filename);
CvVideoWriter* cvCreateVideoWriter_Images(const char* filename);

CvCapture* cvCreateFileCapture_XINE (const char* filename);

#define CV_CAP_GSTREAMER_1394		0
#define CV_CAP_GSTREAMER_V4L		1
#define CV_CAP_GSTREAMER_V4L2		2
#define CV_CAP_GSTREAMER_FILE		3

CvCapture * cvCreateCapture_GStreamer(int type, const char *filename);
CVAPI(CvCapture*) cvCreateFileCapture_FFMPEG (const char* filename);


CVAPI(CvVideoWriter*) cvCreateVideoWriter_FFMPEG ( const char* filename, int fourcc,
                                            double fps, CvSize frameSize, int is_color );

CvCapture * cvCreateFileCapture_QT (const char  * filename);
CvCapture * cvCreateCameraCapture_QT  (const int     index);

CvVideoWriter* cvCreateVideoWriter_QT ( const char* filename, int fourcc,
                                        double fps, CvSize frameSize, int is_color );

CvCapture * cvCreateCameraCapture_Unicap  (const int     index);
CvCapture * cvCreateCameraCapture_PvAPI  (const int     index);
CvVideoWriter* cvCreateVideoWriter_GStreamer( const char* filename, int fourcc,
                                            double fps, CvSize frameSize, int is_color );

//Yannick Verdie 2010                                 
double cvGetModeWindow_W32(const char* name);
double cvGetModeWindow_GTK(const char* name);
double cvGetModeWindow_CARBON(const char* name);

void cvSetModeWindow_W32(const char* name, double prop_value);
void cvSetModeWindow_GTK(const char* name, double prop_value);
void cvSetModeWindow_CARBON(const char* name, double prop_value);


//for QT
#if defined (HAVE_QT)
double cvGetModeWindow_QT(const char* name);
void cvSetModeWindow_QT(const char* name, double prop_value);
double cvGetPropWindow_QT(const char* name);
void cvSetPropWindow_QT(const char* name,double prop_value);
double cvGetRatioWindow_QT(const char* name);
void cvSetRatioWindow_QT(const char* name,double prop_value);
#endif

/*namespace cv
{

class CV_EXPORTS BaseWindow
{
public:
    BaseWindow(const String& name, int flags=0);
    virtual ~BaseWindow();
    virtual void close();
    virtual void show(const Mat& mat);
    virtual void resize(Size size);
    virtual void move(Point topleft);
    virtual Size size() const;
    virtual Point topLeft() const;
    virtual void setGeometry(Point topLeft, Size size);
    virtual void getGeometry(Point& topLeft, Size& size) const;
    virtual String getTitle() const;
    virtual void setTitle(const String& str);
    virtual String getName() const;
    virtual void setScaleMode(int mode);
    virtual int getScaleMode();
    virtual void setScrollPos(double pos);
    virtual double getScrollPos() const;
    virtual void setScale(double scale);
    virtual double getScale() const;
    virtual Point getImageCoords(Point pos) const;
    virtual Scalar getPixelValue(Point pos, const String& colorspace=String()) const;

    virtual void addTrackbar( const String& trackbar, int low, int high, int step );
};

typedef Ptr<BaseWindow> Window;

}*/

#endif /* __HIGHGUI_H_ */
