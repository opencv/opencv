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
#include "cap_intelperc.hpp"

#if defined _M_X64 && defined _MSC_VER && !defined CV_ICC
#pragma optimize("",off)
#pragma warning(disable: 4748)
#endif

namespace cv
{

template<> void DefaultDeleter<CvCapture>::operator ()(CvCapture* obj) const
{ cvReleaseCapture(&obj); }

template<> void DefaultDeleter<CvVideoWriter>::operator ()(CvVideoWriter* obj) const
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
#ifdef HAVE_DSHOW
        CV_CAP_DSHOW,
#endif
#ifdef HAVE_MSMF
        CV_CAP_MSMF,
#endif
#if 1
        CV_CAP_IEEE1394,   // identical to CV_CAP_DC1394
#endif
#ifdef HAVE_TYZX
        CV_CAP_STEREO,
#endif
#ifdef HAVE_PVAPI
        CV_CAP_PVAPI,
#endif
#if 1
        CV_CAP_VFW,        // identical to CV_CAP_V4L
#endif
#ifdef HAVE_MIL
        CV_CAP_MIL,
#endif
#if defined(HAVE_QUICKTIME) || defined(HAVE_QTKIT)
        CV_CAP_QT,
#endif
#ifdef HAVE_UNICAP
        CV_CAP_UNICAP,
#endif
#ifdef HAVE_OPENNI
        CV_CAP_OPENNI,
#endif
#ifdef HAVE_ANDROID_NATIVE_CAMERA
        CV_CAP_ANDROID,
#endif
#ifdef HAVE_XIMEA
        CV_CAP_XIAPI,
#endif
#ifdef HAVE_AVFOUNDATION
        CV_CAP_AVFOUNDATION,
#endif
#ifdef HAVE_GIGE_API
        CV_CAP_GIGANETIX,
#endif
#ifdef HAVE_INTELPERC
        CV_CAP_INTELPERC,
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
#if defined(HAVE_DSHOW)        || \
    defined(HAVE_MSMF)         || \
    defined(HAVE_TYZX)         || \
    defined(HAVE_VFW)          || \
    defined(HAVE_LIBV4L)       || \
    defined(HAVE_CAMV4L)       || \
    defined(HAVE_CAMV4L2)      || \
    defined(HAVE_VIDEOIO)      || \
    defined(HAVE_GSTREAMER)    || \
    defined(HAVE_DC1394_2)     || \
    defined(HAVE_DC1394)       || \
    defined(HAVE_CMU1394)      || \
    defined(HAVE_MIL)          || \
    defined(HAVE_QUICKTIME)    || \
    defined(HAVE_QTKIT)        || \
    defined(HAVE_UNICAP)       || \
    defined(HAVE_PVAPI)        || \
    defined(HAVE_OPENNI)       || \
    defined(HAVE_XIMEA)        || \
    defined(HAVE_AVFOUNDATION) || \
    defined(HAVE_ANDROID_NATIVE_CAMERA) || \
    defined(HAVE_GIGE_API) || \
    defined(HAVE_INTELPERC)    || \
    (0)
        // local variable to memorize the captured device
        CvCapture *capture;
#endif

        switch (domains[i])
        {
#ifdef HAVE_DSHOW
        case CV_CAP_DSHOW:
             capture = cvCreateCameraCapture_DShow (index);
             if (capture)
                 return capture;
            break;
#endif
#ifdef HAVE_MSMF
        case CV_CAP_MSMF:
             capture = cvCreateCameraCapture_MSMF (index);
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
#if defined HAVE_LIBV4L || defined HAVE_CAMV4L || defined HAVE_CAMV4L2 || defined HAVE_VIDEOIO
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
            break; //CV_CAP_VFW

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

#if defined(HAVE_GSTREAMER) && 0
            //Re-enable again when gstreamer 1394 support will land in the backend code
            capture = cvCreateCapture_GStreamer(CV_CAP_GSTREAMER_1394, 0);
            if (capture)
                return capture;
#endif
            break; //CV_CAP_FIREWIRE

#ifdef HAVE_MIL
        case CV_CAP_MIL:
            capture = cvCreateCameraCapture_MIL (index);
            if (capture)
                return capture;
            break;
#endif

#if defined(HAVE_QUICKTIME) || defined(HAVE_QTKIT)
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

#ifdef HAVE_XIMEA
        case CV_CAP_XIAPI:
            capture = cvCreateCameraCapture_XIMEA (index);
            if (capture)
                return capture;
        break;
#endif

#ifdef HAVE_AVFOUNDATION
        case CV_CAP_AVFOUNDATION:
            capture = cvCreateCameraCapture_AVFoundation (index);
            if (capture)
                return capture;
        break;
#endif

#ifdef HAVE_GIGE_API
        case CV_CAP_GIGANETIX:
            capture = cvCreateCameraCapture_Giganetix (index);
            if (capture)
                return capture;
        break; // CV_CAP_GIGANETIX
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

#ifdef HAVE_VFW
    if (! result)
        result = cvCreateFileCapture_VFW (filename);
#endif

#ifdef HAVE_MSMF
    if (! result)
        result = cvCreateFileCapture_MSMF (filename);
#endif

#ifdef HAVE_XINE
    if (! result)
        result = cvCreateFileCapture_XINE (filename);
#endif

#ifdef HAVE_GSTREAMER
    if (! result)
        result = cvCreateCapture_GStreamer (CV_CAP_GSTREAMER_FILE, filename);
#endif

#if defined(HAVE_QUICKTIME) || defined(HAVE_QTKIT)
    if (! result)
        result = cvCreateFileCapture_QT (filename);
#endif

#ifdef HAVE_AVFOUNDATION
    if (! result)
        result = cvCreateFileCapture_AVFoundation (filename);
#endif

#ifdef HAVE_OPENNI
    if (! result)
        result = cvCreateFileCapture_OpenNI (filename);
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

#ifdef HAVE_VFW
    if(!result)
        result = cvCreateVideoWriter_VFW(filename, fourcc, fps, frameSize, is_color);
#endif

#ifdef HAVE_MSMF
    if (!result)
        result = cvCreateVideoWriter_MSMF(filename, fourcc, fps, frameSize, is_color);
#endif

/*  #ifdef HAVE_XINE
    if(!result)
        result = cvCreateVideoWriter_XINE(filename, fourcc, fps, frameSize, is_color);
    #endif
*/
#ifdef HAVE_AVFOUNDATION
    if (! result)
        result = cvCreateVideoWriter_AVFoundation(filename, fourcc, fps, frameSize, is_color);
#endif

#if defined(HAVE_QUICKTIME) || defined(HAVE_QTKIT)
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

VideoCapture::VideoCapture(const String& filename)
{
    open(filename);
}

VideoCapture::VideoCapture(int device)
{
    open(device);
}

VideoCapture::~VideoCapture()
{
    icap.release();
    cap.release();
}

bool VideoCapture::open(const String& filename)
{
    if (isOpened()) release();
    cap.reset(cvCreateFileCapture(filename.c_str()));
    return isOpened();
}

bool VideoCapture::open(int device)
{
    if (isOpened()) release();
    icap = createCameraCapture(device);
    if (!icap.empty())
        return true;
    cap.reset(cvCreateCameraCapture(device));
    return isOpened();
}

bool VideoCapture::isOpened() const
{
    return (!cap.empty() || !icap.empty());
}

void VideoCapture::release()
{
    icap.release();
    cap.release();
}

bool VideoCapture::grab()
{
    if (!icap.empty())
        return icap->grabFrame();
    return cvGrabFrame(cap) != 0;
}

bool VideoCapture::retrieve(OutputArray image, int channel)
{
    if (!icap.empty())
        return icap->retrieveFrame(channel, image);

    IplImage* _img = cvRetrieveFrame(cap, channel);
    if( !_img )
    {
        image.release();
        return false;
    }
    if(_img->origin == IPL_ORIGIN_TL)
        cv::cvarrToMat(_img).copyTo(image);
    else
    {
        Mat temp = cv::cvarrToMat(_img);
        flip(temp, image, 0);
    }
    return true;
}

bool VideoCapture::read(OutputArray image)
{
    if(grab())
        retrieve(image);
    else
        image.release();
    return !image.empty();
}

VideoCapture& VideoCapture::operator >> (Mat& image)
{
    read(image);
    return *this;
}

VideoCapture& VideoCapture::operator >> (UMat& image)
{
    read(image);
    return *this;
}

bool VideoCapture::set(int propId, double value)
{
    if (!icap.empty())
        return icap->setProperty(propId, value);
    return cvSetCaptureProperty(cap, propId, value) != 0;
}

double VideoCapture::get(int propId)
{
    if (!icap.empty())
        return icap->getProperty(propId);
    return cvGetCaptureProperty(cap, propId);
}

Ptr<IVideoCapture> VideoCapture::createCameraCapture(int index)
{
    int  domains[] =
    {
#ifdef HAVE_INTELPERC
        CV_CAP_INTELPERC,
#endif
        -1, -1
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
#if defined(HAVE_INTELPERC)    || \
    (0)
        Ptr<IVideoCapture> capture;

        switch (domains[i])
        {
#ifdef HAVE_INTELPERC
        case CV_CAP_INTELPERC:
            capture = Ptr<IVideoCapture>(new cv::VideoCapture_IntelPerC());
            if (capture)
                return capture;
        break; // CV_CAP_INTEL_PERC
#endif
        }
#endif
    }

    // failed open a camera
    return Ptr<IVideoCapture>();
}


VideoWriter::VideoWriter()
{}

VideoWriter::VideoWriter(const String& filename, int _fourcc, double fps, Size frameSize, bool isColor)
{
    open(filename, _fourcc, fps, frameSize, isColor);
}

void VideoWriter::release()
{
    writer.release();
}

VideoWriter::~VideoWriter()
{
    release();
}

bool VideoWriter::open(const String& filename, int _fourcc, double fps, Size frameSize, bool isColor)
{
    writer.reset(cvCreateVideoWriter(filename.c_str(), _fourcc, fps, frameSize, isColor));
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

int VideoWriter::fourcc(char c1, char c2, char c3, char c4)
{
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24);
}

}
