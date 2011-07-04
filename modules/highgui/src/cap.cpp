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

#include "precomp.hpp"

#if _MSC_VER >= 1200
#pragma warning( disable: 4711 )
#endif

#if defined _M_X64 && defined _MSC_VER && !defined CV_ICC
#pragma optimize("",off)
#endif

namespace cv
{

template<> void Ptr<CvCapture>::delete_obj()
{ cvReleaseCapture(&obj); }

template<> void Ptr<CvVideoWriter>::delete_obj()
{ cvReleaseVideoWriter(&obj); }

}

/************************* Reading AVIs & Camera data **************************/

CV_IMPL void cvReleaseCapture( CvCapture** pcapture )
{
    if( pcapture && *pcapture )
    {
        delete *pcapture;
        *pcapture = 0;
    }
}

CV_IMPL IplImage* cvQueryFrame( CvCapture* capture )
{
    if(!capture)
        return 0;
    if(!capture->grabFrame())
        return 0;
    return capture->retrieveFrame(0);
}


CV_IMPL int cvGrabFrame( CvCapture* capture )
{
    return capture ? capture->grabFrame() : 0;
}

CV_IMPL IplImage* cvRetrieveFrame( CvCapture* capture, int idx )
{
    return capture ? capture->retrieveFrame(idx) : 0;
}

CV_IMPL double cvGetCaptureProperty( CvCapture* capture, int id )
{
    return capture ? capture->getProperty(id) : 0;
}

CV_IMPL int cvSetCaptureProperty( CvCapture* capture, int id, double value )
{
    return capture ? capture->setProperty(id, value) : 0;
}

CV_IMPL int cvGetCaptureDomain( CvCapture* capture)
{
    return capture ? capture->getCaptureDomain() : 0;
}


/**
 * Camera dispatching method: index is the camera number.
 * If given an index from 0 to 99, it tries to find the first
 * API that can access a given camera index.
 * Add multiples of 100 to select an API.
 */
CV_IMPL CvCapture * cvCreateCameraCapture (int index)
{
    int  domains[] =
    {
#ifdef HAVE_VIDEOINPUT
        CV_CAP_DSHOW,
#endif
        CV_CAP_IEEE1394,   // identical to CV_CAP_DC1394
        CV_CAP_STEREO,
        CV_CAP_PVAPI,
        CV_CAP_VFW,        // identical to CV_CAP_V4L
        CV_CAP_MIL,
        CV_CAP_QT,
        CV_CAP_UNICAP,
#ifdef HAVE_OPENNI
        CV_CAP_OPENNI,
#endif
#ifdef HAVE_ANDROID_NATIVE_CAMERA
        CV_CAP_ANDROID,
#endif
        -1
    };

    // interpret preferred interface (0 = autodetect)
    int pref = (index / 100) * 100;
    if (pref)
    {
        domains[0]=pref;
        index %= 100;
        domains[1]=-1;
    }

    // try every possibly installed camera API
    for (int i = 0; domains[i] >= 0; i++)
    {
        #if defined(HAVE_VIDEOINPUT) || defined(HAVE_TYZX) || defined(HAVE_VFW) || \
        defined(HAVE_CAMV4L) || defined (HAVE_CAMV4L2) || defined(HAVE_GSTREAMER) || \
        defined(HAVE_DC1394_2) || defined(HAVE_DC1394) || defined(HAVE_CMU1394) || \
        defined(HAVE_GSTREAMER) || defined(HAVE_MIL) || defined(HAVE_QUICKTIME) || \
        defined(HAVE_UNICAP) || defined(HAVE_PVAPI) || defined(HAVE_OPENNI) || defined(HAVE_ANDROID_NATIVE_CAMERA)
        // local variable to memorize the captured device
        CvCapture *capture;
        #endif

        switch (domains[i])
        {
        #ifdef HAVE_VIDEOINPUT
        case CV_CAP_DSHOW:
            capture = cvCreateCameraCapture_DShow (index);
            if (capture)
                return capture;
            break;
        #endif

        #ifdef HAVE_TYZX
        case CV_CAP_STEREO:
            capture = cvCreateCameraCapture_TYZX (index);
            if (capture)
                return capture;
            break;
        #endif

        case CV_CAP_VFW:
        #ifdef HAVE_VFW
            capture = cvCreateCameraCapture_VFW (index);
            if (capture)
                return capture;
        #endif
        #if defined HAVE_LIBV4L || (defined (HAVE_CAMV4L) && defined (HAVE_CAMV4L2))
            capture = cvCreateCameraCapture_V4L (index);
            if (capture)
                return capture;
        #endif
        #ifdef HAVE_GSTREAMER
            capture = cvCreateCapture_GStreamer(CV_CAP_GSTREAMER_V4L2, 0);
            if (capture)
                return capture;
            capture = cvCreateCapture_GStreamer(CV_CAP_GSTREAMER_V4L, 0);
            if (capture)
                return capture;
        #endif
            break;

        case CV_CAP_FIREWIRE:
        #ifdef HAVE_DC1394_2
            capture = cvCreateCameraCapture_DC1394_2 (index);
            if (capture)
                return capture;
        #endif
        #ifdef HAVE_DC1394
            capture = cvCreateCameraCapture_DC1394 (index);
            if (capture)
                return capture;
        #endif
        #ifdef HAVE_CMU1394
            capture = cvCreateCameraCapture_CMU (index);
            if (capture)
                return capture;
        #endif
    /* Re-enable again when gstreamer 1394 support will land in the backend code	
        #ifdef HAVE_GSTREAMER
            capture = cvCreateCapture_GStreamer(CV_CAP_GSTREAMER_1394, 0);
            if (capture)
                return capture;
        #endif
    */ 
            break;
        #ifdef HAVE_MIL
        case CV_CAP_MIL:
            capture = cvCreateCameraCapture_MIL (index);
            if (capture)
                return capture;
            break;
        #endif

        #ifdef HAVE_QUICKTIME
        case CV_CAP_QT:
            capture = cvCreateCameraCapture_QT (index);
            if (capture)
                return capture;
            break;
        #endif

        #ifdef HAVE_UNICAP
        case CV_CAP_UNICAP:
        capture = cvCreateCameraCapture_Unicap (index);
        if (capture)
            return capture;
        break;
        #endif
        
        #ifdef HAVE_PVAPI
        case CV_CAP_PVAPI:
        capture = cvCreateCameraCapture_PvAPI (index);
        if (capture)
            return capture;
        break;
        #endif

        #ifdef HAVE_OPENNI
        case CV_CAP_OPENNI:
        capture = cvCreateCameraCapture_OpenNI (index);
        if (capture)
            return capture;
        break;
        #endif

		#ifdef HAVE_ANDROID_NATIVE_CAMERA
        case CV_CAP_ANDROID:
          capture = cvCreateCameraCapture_Android (index);
        if (capture)
            return capture;
        break;
        #endif

        }
    }

    // failed open a camera
    return 0;
}

/**
 * Videoreader dispatching method: it tries to find the first
 * API that can access a given filename.
 */
CV_IMPL CvCapture * cvCreateFileCapture (const char * filename)
{
    CvCapture * result = 0;

    if (! result)
        result = cvCreateFileCapture_FFMPEG_proxy (filename);

    #ifdef HAVE_XINE
    if (! result)
        result = cvCreateFileCapture_XINE (filename);
    #endif

    #ifdef HAVE_GSTREAMER
    if (! result)
        result = cvCreateCapture_GStreamer (CV_CAP_GSTREAMER_FILE, filename);
    #endif

    #ifdef HAVE_QUICKTIME
    if (! result)
        result = cvCreateFileCapture_QT (filename);
    #endif
    
    if (! result)
        result = cvCreateFileCapture_Images (filename);

    return result;
}

/**
 * Videowriter dispatching method: it tries to find the first
 * API that can write a given stream.
 */
CV_IMPL CvVideoWriter* cvCreateVideoWriter( const char* filename, int fourcc,
                                            double fps, CvSize frameSize, int is_color )
{
	//CV_FUNCNAME( "cvCreateVideoWriter" );

	CvVideoWriter *result = 0;

	if(!fourcc || !fps)
		result = cvCreateVideoWriter_Images(filename);

	if(!result)
		result = cvCreateVideoWriter_FFMPEG_proxy (filename, fourcc, fps, frameSize, is_color);

/*	#ifdef HAVE_XINE
	if(!result)
		result = cvCreateVideoWriter_XINE(filename, fourcc, fps, frameSize, is_color);
	#endif
*/

	#ifdef HAVE_QUICKTIME
	if(!result)
		result = cvCreateVideoWriter_QT(filename, fourcc, fps, frameSize, is_color);
	#endif
    #ifdef HAVE_GSTREAMER
    if (! result)
        result = cvCreateVideoWriter_GStreamer(filename, fourcc, fps, frameSize, is_color);
    #endif
	if(!result)
		result = cvCreateVideoWriter_Images(filename);

	return result;
}

CV_IMPL int cvWriteFrame( CvVideoWriter* writer, const IplImage* image )
{

    return writer ? writer->writeFrame(image) : 0;
}

CV_IMPL void cvReleaseVideoWriter( CvVideoWriter** pwriter )
{
    if( pwriter && *pwriter )
    {
        delete *pwriter;
        *pwriter = 0;
    }
}

namespace cv
{

VideoCapture::VideoCapture()
{}
        
VideoCapture::VideoCapture(const string& filename)
{
    open(filename);
}
    
VideoCapture::VideoCapture(int device)
{
    open(device);
}

VideoCapture::~VideoCapture()
{
    cap.release();
}
    
bool VideoCapture::open(const string& filename)
{
    cap = cvCreateFileCapture(filename.c_str());
    return isOpened();
}
    
bool VideoCapture::open(int device)
{
    cap = cvCreateCameraCapture(device);
    return isOpened();
}
    
bool VideoCapture::isOpened() const { return !cap.empty(); }
    
void VideoCapture::release()
{
    cap.release();
}

bool VideoCapture::grab()
{
    return cvGrabFrame(cap) != 0;
}
    
bool VideoCapture::retrieve(Mat& image, int channel)
{
    IplImage* _img = cvRetrieveFrame(cap, channel);
    if( !_img )
    {
        image.release();
        return false;
    }
    if(_img->origin == IPL_ORIGIN_TL)
        image = Mat(_img);
    else
    {
        Mat temp(_img);
        flip(temp, image, 0);
    }
    return true;
}

bool VideoCapture::read(Mat& image)
{
    if(!grab())
        image.release();
    else
        retrieve(image);
    return !image.empty();
}
    
VideoCapture& VideoCapture::operator >> (Mat& image)
{
    if(!grab())
        image.release();
    else
        retrieve(image);
    return *this;
}
    
bool VideoCapture::set(int propId, double value)
{
    return cvSetCaptureProperty(cap, propId, value) != 0;
}
    
double VideoCapture::get(int propId)
{
    return cvGetCaptureProperty(cap, propId);
}

VideoWriter::VideoWriter()
{}
    
VideoWriter::VideoWriter(const string& filename, int fourcc, double fps, Size frameSize, bool isColor)
{
    open(filename, fourcc, fps, frameSize, isColor);
}

VideoWriter::~VideoWriter()
{
    writer.release();
}
    
bool VideoWriter::open(const string& filename, int fourcc, double fps, Size frameSize, bool isColor)
{
    writer = cvCreateVideoWriter(filename.c_str(), fourcc, fps, frameSize, isColor);
    return isOpened();
}

bool VideoWriter::isOpened() const
{
    return !writer.empty();
}    

void VideoWriter::write(const Mat& image)
{
    IplImage _img = image;
    cvWriteFrame(writer, &_img);
}
    
VideoWriter& VideoWriter::operator << (const Mat& image)
{
    write(image);
    return *this;    
}

}
