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

#ifndef __VIDEOIO_H_
#define __VIDEOIO_H_

#include "opencv2/videoio.hpp"

#include "opencv2/core/utility.hpp"
#include "opencv2/core/private.hpp"

#include "opencv2/imgcodecs.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgcodecs/imgcodecs_c.h"
#include "opencv2/videoio/videoio_c.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>
#include <assert.h>

#if defined WIN32 || defined WINCE
    #if !defined _WIN32_WINNT
        #ifdef HAVE_MSMF
            #define _WIN32_WINNT 0x0600 // Windows Vista
        #else
            #define _WIN32_WINNT 0x0500 // Windows 2000
        #endif
    #endif

    #include <windows.h>
    #undef small
    #undef min
    #undef max
    #undef abs
#endif

#define __BEGIN__ __CV_BEGIN__
#define __END__  __CV_END__
#define EXIT __CV_EXIT__

/***************************** CvCapture structure ******************************/

struct CvCapture
{
    virtual ~CvCapture() {}
    virtual double getProperty(int) const { return 0; }
    virtual bool setProperty(int, double) { return 0; }
    virtual bool grabFrame() { return true; }
    virtual IplImage* retrieveFrame(int) { return 0; }
    virtual int getCaptureDomain() { return CV_CAP_ANY; } // Return the type of the capture object: CV_CAP_VFW, etc...
};

/*************************** CvVideoWriter structure ****************************/

struct CvVideoWriter
{
    virtual ~CvVideoWriter() {}
    virtual bool writeFrame(const IplImage*) { return false; }
};

CvCapture * cvCreateCameraCapture_V4L( int index );
CvCapture * cvCreateCameraCapture_DC1394( int index );
CvCapture * cvCreateCameraCapture_DC1394_2( int index );
CvCapture* cvCreateCameraCapture_MIL( int index );
CvCapture* cvCreateCameraCapture_Giganetix( int index );
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
CvCapture* cvCreateCameraCapture_MSMF( int index );
CvCapture* cvCreateFileCapture_MSMF (const char* filename);
CvVideoWriter* cvCreateVideoWriter_MSMF( const char* filename, int fourcc,
                                        double fps, CvSize frameSize, int is_color );
CvCapture* cvCreateCameraCapture_OpenNI( int index );
CvCapture* cvCreateCameraCapture_OpenNI2( int index );
CvCapture* cvCreateFileCapture_OpenNI( const char* filename );
CvCapture* cvCreateCameraCapture_Android( int index );
CvCapture* cvCreateCameraCapture_XIMEA( int index );
CvCapture* cvCreateCameraCapture_AVFoundation(int index);

CvCapture* cvCreateFileCapture_Images(const char* filename);
CvVideoWriter* cvCreateVideoWriter_Images(const char* filename);

CvCapture* cvCreateFileCapture_XINE (const char* filename);


#define CV_CAP_GSTREAMER_1394		0
#define CV_CAP_GSTREAMER_V4L		1
#define CV_CAP_GSTREAMER_V4L2		2
#define CV_CAP_GSTREAMER_FILE		3

CvCapture* cvCreateCapture_GStreamer(int type, const char *filename);
CvCapture* cvCreateFileCapture_FFMPEG_proxy(const char* filename);


CvVideoWriter* cvCreateVideoWriter_FFMPEG_proxy( const char* filename, int fourcc,
                                            double fps, CvSize frameSize, int is_color );

CvCapture * cvCreateFileCapture_QT (const char  * filename);
CvCapture * cvCreateCameraCapture_QT  (const int     index);

CvVideoWriter* cvCreateVideoWriter_QT ( const char* filename, int fourcc,
                                        double fps, CvSize frameSize, int is_color );

CvCapture* cvCreateFileCapture_AVFoundation (const char * filename);
CvVideoWriter* cvCreateVideoWriter_AVFoundation( const char* filename, int fourcc,
                                                double fps, CvSize frameSize, int is_color );


CvCapture * cvCreateCameraCapture_Unicap  (const int     index);
CvCapture * cvCreateCameraCapture_PvAPI  (const int     index);
CvVideoWriter* cvCreateVideoWriter_GStreamer( const char* filename, int fourcc,
                                            double fps, CvSize frameSize, int is_color );

namespace cv
{
    class IVideoCapture
    {
    public:
        virtual ~IVideoCapture() {}
        virtual double getProperty(int) const { return 0; }
        virtual bool setProperty(int, double) { return false; }
        virtual bool grabFrame() = 0;
        virtual bool retrieveFrame(int, OutputArray) = 0;
        virtual bool isOpened() const = 0;
        virtual int getCaptureDomain() { return CAP_ANY; } // Return the type of the capture object: CAP_VFW, etc...
    };

    class IVideoWriter
    {
    public:
        virtual ~IVideoWriter() {}
        virtual double getProperty(int) const { return 0; }
        virtual bool setProperty(int, double) { return false; }

        virtual bool isOpened() const = 0;
        virtual void write(InputArray) = 0;
    };

    Ptr<IVideoCapture> createMotionJpegCapture(const String& filename);
    Ptr<IVideoWriter> createMotionJpegWriter( const String& filename, double fps, Size frameSize, bool iscolor );

    Ptr<IVideoCapture> createGPhoto2Capture(int index);
    Ptr<IVideoCapture> createGPhoto2Capture(const String& deviceName);
};

#endif /* __VIDEOIO_H_ */
