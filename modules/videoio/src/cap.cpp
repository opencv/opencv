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
#include "cap_dshow.hpp"

#ifdef HAVE_MFX
#include "cap_mfx_reader.hpp"
#include "cap_mfx_writer.hpp"
#endif

// All WinRT versions older than 8.0 should provide classes used for video support
#if defined(WINRT) && !defined(WINRT_8_0) && defined(__cplusplus_winrt)
#   include "cap_winrt_capture.hpp"
#   include "cap_winrt_bridge.hpp"
#   define WINRT_VIDEO
#endif

#if defined _M_X64 && defined _MSC_VER && !defined CV_ICC
#pragma optimize("",off)
#pragma warning(disable: 4748)
#endif

using namespace cv;

namespace cv
{

template<> void DefaultDeleter<CvCapture>::operator ()(CvCapture* obj) const
{ cvReleaseCapture(&obj); }

template<> void DefaultDeleter<CvVideoWriter>::operator ()(CvVideoWriter* obj) const
{ cvReleaseVideoWriter(&obj); }

}

/************************* Reading AVIs & Camera data **************************/

static inline double icvGetCaptureProperty( const CvCapture* capture, int id )
{
    return capture ? capture->getProperty(id) : 0;
}

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
    return icvGetCaptureProperty(capture, id);
}

CV_IMPL int cvSetCaptureProperty( CvCapture* capture, int id, double value )
{
    return capture ? capture->setProperty(id, value) : 0;
}

CV_IMPL int cvGetCaptureDomain( CvCapture* capture)
{
    return capture ? capture->getCaptureDomain() : 0;
}

static bool get_capture_debug_flag()
{
    static bool initialized = false;
    static bool flag = false;
    if (!initialized)
    {
#ifndef NO_GETENV
        flag = getenv("OPENCV_VIDEOCAPTURE_DEBUG") ? true : false; // TODO Use getBoolParameter
#endif
        initialized = true;
    }
    return flag;
}

#define TRY_OPEN(capture, backend_func) \
{ \
    if (!capture) \
        try { \
            if (get_capture_debug_flag()) fprintf(stderr, "VIDEOIO(%s): trying ...\n", #backend_func); \
            capture = backend_func; \
            if (get_capture_debug_flag()) fprintf(stderr, "VIDEOIO(%s): result=%p ...\n", #backend_func, capture); \
        } catch (const cv::Exception& e) { \
            fprintf(stderr, "VIDEOIO(%s): raised OpenCV exception:\n\n%s\n", #backend_func, e.what()); \
        } catch (const std::exception& e) { \
            fprintf(stderr, "VIDEOIO(%s): raised C++ exception:\n\n%s\n", #backend_func, e.what()); \
        } catch (...) { \
            fprintf(stderr, "VIDEOIO(%s): raised unknown C++ exception!\n\n", #backend_func); \
        } \
}


/**
 * Camera dispatching method: index is the camera number.
 * If given an index from 0 to 99, it tries to find the first
 * API that can access a given camera index.
 * Add multiples of 100 to select an API.
 */
CV_IMPL CvCapture * cvCreateCameraCapture (int index)
{
    // interpret preferred interface (0 = autodetect)
    int pref = (index / 100) * 100;

    // remove pref from index
    index -= pref;

    // local variable to memorize the captured device
    CvCapture *capture = 0;

    switch (pref)
    {
    default:
        // user specified an API we do not know
        // bail out to let the user know that it is not available
        if (pref) break;

#ifdef HAVE_MSMF
    case CAP_MSMF:
        TRY_OPEN(capture, cvCreateCameraCapture_MSMF(index))
        if (pref) break;
#endif
    case CAP_VFW: // or CAP_V4L or CAP_V4L2
#ifdef HAVE_VFW
        TRY_OPEN(capture, cvCreateCameraCapture_VFW(index))
#endif

#if defined HAVE_LIBV4L || defined HAVE_CAMV4L || defined HAVE_CAMV4L2 || defined HAVE_VIDEOIO
        TRY_OPEN(capture, cvCreateCameraCapture_V4L(index))
#endif

#ifdef HAVE_GSTREAMER
        TRY_OPEN(capture, cvCreateCapture_GStreamer(CV_CAP_GSTREAMER_V4L2, reinterpret_cast<char *>(index)))

        TRY_OPEN(capture, cvCreateCapture_GStreamer(CV_CAP_GSTREAMER_V4L, reinterpret_cast<char *>(index)))
#endif

        if (pref) break; // CAP_VFW or CAP_V4L or CAP_V4L2

    case CAP_FIREWIRE:
#ifdef HAVE_DC1394_2
        TRY_OPEN(capture, cvCreateCameraCapture_DC1394_2(index))
#endif

#ifdef HAVE_DC1394
        TRY_OPEN(capture, cvCreateCameraCapture_DC1394(index))
#endif

#ifdef HAVE_CMU1394
        TRY_OPEN(capture, cvCreateCameraCapture_CMU(index))
#endif

#if defined(HAVE_GSTREAMER) && 0
        // Re-enable again when gstreamer 1394 support will land in the backend code
        TRY_OPEN(capture, cvCreateCapture_GStreamer(CV_CAP_GSTREAMER_1394, 0))
#endif

        if (pref) break; // CAP_FIREWIRE

#ifdef HAVE_MIL
    case CAP_MIL:
        TRY_OPEN(capture, cvCreateCameraCapture_MIL(index))
        if (pref) break;
#endif

#if defined(HAVE_QUICKTIME) || defined(HAVE_QTKIT)
    case CAP_QT:
        TRY_OPEN(capture, cvCreateCameraCapture_QT(index))
        if (pref) break;
#endif

#ifdef HAVE_UNICAP
    case CAP_UNICAP:
        TRY_OPEN(capture, cvCreateCameraCapture_Unicap(index))
        if (pref) break;
#endif

#ifdef HAVE_PVAPI
    case CAP_PVAPI:
        TRY_OPEN(capture, cvCreateCameraCapture_PvAPI(index))
        if (pref) break;
#endif

#ifdef HAVE_OPENNI
    case CAP_OPENNI:
        TRY_OPEN(capture, cvCreateCameraCapture_OpenNI(index))
        if (pref) break;
#endif

#ifdef HAVE_OPENNI2
    case CAP_OPENNI2:
        TRY_OPEN(capture, cvCreateCameraCapture_OpenNI2(index))
        if (pref) break;
#endif

#ifdef HAVE_XIMEA
    case CAP_XIAPI:
        TRY_OPEN(capture, cvCreateCameraCapture_XIMEA(index))
        if (pref) break;
#endif

#ifdef HAVE_AVFOUNDATION
    case CAP_AVFOUNDATION:
        TRY_OPEN(capture, cvCreateCameraCapture_AVFoundation(index))
        if (pref) break;
#endif

#ifdef HAVE_GIGE_API
    case CAP_GIGANETIX:
        TRY_OPEN(capture, cvCreateCameraCapture_Giganetix(index))
        if (pref) break; // CAP_GIGANETIX
#endif

#ifdef HAVE_ARAVIS_API
    case CAP_ARAVIS:
        TRY_OPEN(capture, cvCreateCameraCapture_Aravis(index))
        if (pref) break;
#endif
    }

    return capture;
}

/**
 * Videoreader dispatching method: it tries to find the first
 * API that can access a given filename.
 */
CV_IMPL CvCapture * cvCreateFileCaptureWithPreference (const char * filename, int apiPreference)
{
    CvCapture * result = 0;

    switch(apiPreference) {
    default:
        // user specified an API we do not know
        // bail out to let the user know that it is not available
        if (apiPreference) break;

#ifdef HAVE_FFMPEG
    case CAP_FFMPEG:
        TRY_OPEN(result, cvCreateFileCapture_FFMPEG_proxy (filename))
        if (apiPreference) break;
#endif

    case CAP_VFW:
#ifdef HAVE_VFW
        TRY_OPEN(result, cvCreateFileCapture_VFW (filename))
#endif

#if defined HAVE_LIBV4L || defined HAVE_CAMV4L || defined HAVE_CAMV4L2 || defined HAVE_VIDEOIO
        TRY_OPEN(result, cvCreateCameraCapture_V4L(filename))
#endif
        if (apiPreference) break;

    case CAP_MSMF:
#ifdef HAVE_MSMF
        TRY_OPEN(result, cvCreateFileCapture_MSMF (filename))
#endif

#ifdef HAVE_XINE
        TRY_OPEN(result, cvCreateFileCapture_XINE (filename))
#endif
        if (apiPreference) break;

#ifdef HAVE_GSTREAMER
    case CAP_GSTREAMER:
        TRY_OPEN(result, cvCreateCapture_GStreamer (CV_CAP_GSTREAMER_FILE, filename))
        if (apiPreference) break;
#endif

#if defined(HAVE_QUICKTIME) || defined(HAVE_QTKIT)
    case CAP_QT:
        TRY_OPEN(result, cvCreateFileCapture_QT (filename))
        if (apiPreference) break;
#endif

#ifdef HAVE_AVFOUNDATION
    case CAP_AVFOUNDATION:
        TRY_OPEN(result, cvCreateFileCapture_AVFoundation (filename))
        if (apiPreference) break;
#endif

#ifdef HAVE_OPENNI
    case CAP_OPENNI:
        TRY_OPEN(result, cvCreateFileCapture_OpenNI (filename))
        if (apiPreference) break;
#endif

#ifdef HAVE_OPENNI2
    case CAP_OPENNI2:
        TRY_OPEN(result, cvCreateFileCapture_OpenNI2 (filename))
        if (apiPreference) break;
#endif
#ifdef HAVE_XIMEA
    case CAP_XIAPI:
        TRY_OPEN(result, cvCreateCameraCapture_XIMEA(filename))
        if (apiPreference) break;
#endif
    case CAP_IMAGES:
        TRY_OPEN(result, cvCreateFileCapture_Images (filename))
    }

    return result;
}

CV_IMPL CvCapture * cvCreateFileCapture (const char * filename)
{
    return cvCreateFileCaptureWithPreference(filename, CAP_ANY);
}

/**
 * Videowriter dispatching method: it tries to find the first
 * API that can write a given stream.
 */
static CvVideoWriter* cvCreateVideoWriterWithPreference(const char* filename, int apiPreference, int fourcc,
                                            double fps, CvSize frameSize, int is_color )
{
    CV_UNUSED(frameSize);
    CV_UNUSED(is_color);

    CvVideoWriter *result = 0;

    if(!fourcc || !fps)
        TRY_OPEN(result, cvCreateVideoWriter_Images(filename))

    CV_Assert(result || fps != 0);

    switch(apiPreference)
    {
        default:
            //exit if the specified API is unavaliable
            if (apiPreference != CAP_ANY) break;
        #ifdef HAVE_FFMPEG
        case CAP_FFMPEG:
            TRY_OPEN(result, cvCreateVideoWriter_FFMPEG_proxy (filename, fourcc, fps, frameSize, is_color))
            if (apiPreference != CAP_ANY) break;
        #endif
        #ifdef HAVE_VFW
        case CAP_VFW:
            TRY_OPEN(result, cvCreateVideoWriter_VFW(filename, fourcc, fps, frameSize, is_color))
            if (apiPreference != CAP_ANY) break;
        #endif
        #ifdef HAVE_MSMF
        case CAP_MSMF:
            TRY_OPEN(result, cvCreateVideoWriter_MSMF(filename, fourcc, fps, frameSize, is_color))
            if (apiPreference != CAP_ANY) break;
        #endif
        #ifdef HAVE_AVFOUNDATION
        case CAP_AVFOUNDATION:
            TRY_OPEN(result, cvCreateVideoWriter_AVFoundation(filename, fourcc, fps, frameSize, is_color))
            if (apiPreference != CAP_ANY) break;
        #endif
        #if defined(HAVE_QUICKTIME) || defined(HAVE_QTKIT)
        case(CAP_QT):
            TRY_OPEN(result, cvCreateVideoWriter_QT(filename, fourcc, fps, frameSize, is_color))
            if (apiPreference != CAP_ANY) break;
        #endif
        #ifdef HAVE_GSTREAMER
        case CAP_GSTREAMER:
            TRY_OPEN(result, cvCreateVideoWriter_GStreamer (filename, fourcc, fps, frameSize, is_color))
            if (apiPreference != CAP_ANY) break;
        #endif
        case CAP_IMAGES:
            TRY_OPEN(result, cvCreateVideoWriter_Images(filename))
            if (apiPreference != CAP_ANY) break;
    }

    return result;
}

CV_IMPL CvVideoWriter* cvCreateVideoWriter( const char* filename, int fourcc,
                                            double fps, CvSize frameSize, int is_color )
{
    return cvCreateVideoWriterWithPreference(filename, CAP_ANY, fourcc, fps, frameSize, is_color);
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

static Ptr<IVideoCapture> IVideoCapture_create(int index)
{
    int  domains[] =
    {
#ifdef HAVE_DSHOW
        CAP_DSHOW,
#endif
#ifdef HAVE_INTELPERC
        CAP_INTELPERC,
#endif
#ifdef WINRT_VIDEO
        CAP_WINRT,
#endif
#ifdef HAVE_GPHOTO2
        CAP_GPHOTO2,
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
#if defined(HAVE_DSHOW)        || \
    defined(HAVE_INTELPERC)    || \
    defined(WINRT_VIDEO)       || \
    defined(HAVE_GPHOTO2)      || \
    (0)
        Ptr<IVideoCapture> capture;

        switch (domains[i])
        {
#ifdef HAVE_DSHOW
            case CAP_DSHOW:
                capture = makePtr<VideoCapture_DShow>(index);
                break; // CAP_DSHOW
#endif
#ifdef HAVE_INTELPERC
            case CAP_INTELPERC:
                capture = makePtr<VideoCapture_IntelPerC>();
                break; // CAP_INTEL_PERC
#endif
#ifdef WINRT_VIDEO
        case CAP_WINRT:
            capture = Ptr<IVideoCapture>(new cv::VideoCapture_WinRT(index));
            if (capture)
                return capture;
            break; // CAP_WINRT
#endif
#ifdef HAVE_GPHOTO2
            case CAP_GPHOTO2:
                capture = createGPhoto2Capture(index);
                break;
#endif
        }
        if (capture && capture->isOpened())
            return capture;
#endif
    }

    // failed open a camera
    return Ptr<IVideoCapture>();
}


static Ptr<IVideoCapture> IVideoCapture_create(const String& filename)
{
    int  domains[] =
    {
        CAP_ANY,
#ifdef HAVE_GPHOTO2
        CAP_GPHOTO2,
#endif
#ifdef HAVE_MFX
        CAP_INTEL_MFX,
#endif
        -1, -1
    };

    // try every possibly installed camera API
    for (int i = 0; domains[i] >= 0; i++)
    {
        Ptr<IVideoCapture> capture;

        switch (domains[i])
        {
        case CAP_ANY:
            capture = createMotionJpegCapture(filename);
            break;
#ifdef HAVE_GPHOTO2
        case CAP_GPHOTO2:
            capture = createGPhoto2Capture(filename);
            break;
#endif
#ifdef HAVE_MFX
        case CAP_INTEL_MFX:
            capture = makePtr<VideoCapture_IntelMFX>(filename);
            break;
#endif
        }

        if (capture && capture->isOpened())
        {
            return capture;
        }
    }
    // failed open a camera
    return Ptr<IVideoCapture>();
}

static Ptr<IVideoWriter> IVideoWriter_create(const String& filename, int apiPreference, int _fourcc, double fps, Size frameSize, bool isColor)
{
    Ptr<IVideoWriter> iwriter;
#ifdef HAVE_MFX
    if (apiPreference == CAP_INTEL_MFX || apiPreference == CAP_ANY)
    {
        iwriter = VideoWriter_IntelMFX::create(filename, _fourcc, fps, frameSize, isColor);
        if (!iwriter.empty())
            return iwriter;
    }
#endif

    if( (apiPreference == CAP_OPENCV_MJPEG || apiPreference == CAP_ANY)
            && _fourcc == CV_FOURCC('M', 'J', 'P', 'G') )
        iwriter = createMotionJpegWriter(filename, fps, frameSize, isColor);

    return iwriter;
}

VideoCapture::VideoCapture()
{}

VideoCapture::VideoCapture(const String& filename, int apiPreference)
{
    CV_TRACE_FUNCTION();
    open(filename, apiPreference);
}

VideoCapture::VideoCapture(const String& filename)
{
    CV_TRACE_FUNCTION();
    open(filename, CAP_ANY);
}

VideoCapture::VideoCapture(int index)
{
    CV_TRACE_FUNCTION();
    open(index);
}

VideoCapture::~VideoCapture()
{
    CV_TRACE_FUNCTION();

    icap.release();
    cap.release();
}

bool VideoCapture::open(const String& filename, int apiPreference)
{
    CV_TRACE_FUNCTION();

    if (isOpened()) release();
    icap = IVideoCapture_create(filename);
    if (!icap.empty())
        return true;

    cap.reset(cvCreateFileCaptureWithPreference(filename.c_str(), apiPreference));
    return isOpened();
}

bool VideoCapture::open(const String& filename)
{
    CV_TRACE_FUNCTION();

    return open(filename, CAP_ANY);
}

bool VideoCapture::open(int index)
{
    CV_TRACE_FUNCTION();

    if (isOpened()) release();
    icap = IVideoCapture_create(index);
    if (!icap.empty())
        return true;
    cap.reset(cvCreateCameraCapture(index));
    return isOpened();
}
bool  VideoCapture::open(int cameraNum, int apiPreference)
{
    CV_TRACE_FUNCTION();

    cameraNum = cameraNum + apiPreference;
    return open(cameraNum);
}

bool VideoCapture::isOpened() const
{
    return (!cap.empty() || !icap.empty());
}

void VideoCapture::release()
{
    CV_TRACE_FUNCTION();
    icap.release();
    cap.release();
}

bool VideoCapture::grab()
{
    CV_INSTRUMENT_REGION()

    if (!icap.empty())
        return icap->grabFrame();
    return cvGrabFrame(cap) != 0;
}

bool VideoCapture::retrieve(OutputArray image, int channel)
{
    CV_INSTRUMENT_REGION()

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
    CV_INSTRUMENT_REGION()

    if(grab())
        retrieve(image);
    else
        image.release();
    return !image.empty();
}

VideoCapture& VideoCapture::operator >> (Mat& image)
{
#ifdef WINRT_VIDEO
    if (grab())
    {
        if (retrieve(image))
        {
            std::lock_guard<std::mutex> lock(VideoioBridge::getInstance().inputBufferMutex);
            VideoioBridge& bridge = VideoioBridge::getInstance();

            // double buffering
            bridge.swapInputBuffers();
            auto p = bridge.frontInputPtr;

            bridge.bIsFrameNew = false;

            // needed here because setting Mat 'image' is not allowed by OutputArray in read()
            Mat m(bridge.getHeight(), bridge.getWidth(), CV_8UC3, p);
            image = m;
        }
    }
#else
    read(image);
#endif

    return *this;
}

VideoCapture& VideoCapture::operator >> (UMat& image)
{
    CV_INSTRUMENT_REGION()

    read(image);
    return *this;
}

bool VideoCapture::set(int propId, double value)
{
    if (!icap.empty())
        return icap->setProperty(propId, value);
    return cvSetCaptureProperty(cap, propId, value) != 0;
}

double VideoCapture::get(int propId) const
{
    if (!icap.empty())
        return icap->getProperty(propId);
    return icvGetCaptureProperty(cap, propId);
}


VideoWriter::VideoWriter()
{}

VideoWriter::VideoWriter(const String& filename, int _fourcc, double fps, Size frameSize, bool isColor)
{
    open(filename, _fourcc, fps, frameSize, isColor);
}


VideoWriter::VideoWriter(const String& filename, int apiPreference, int _fourcc, double fps, Size frameSize, bool isColor)
{
    open(filename, apiPreference, _fourcc, fps, frameSize, isColor);
}

void VideoWriter::release()
{
    iwriter.release();
    writer.release();
}

VideoWriter::~VideoWriter()
{
    release();
}

bool VideoWriter::open(const String& filename, int _fourcc, double fps, Size frameSize, bool isColor)
{
    return open(filename, CAP_ANY, _fourcc, fps, frameSize, isColor);
}

bool VideoWriter::open(const String& filename, int apiPreference, int _fourcc, double fps, Size frameSize, bool isColor)
{
    CV_INSTRUMENT_REGION()

    if (isOpened()) release();
    iwriter = IVideoWriter_create(filename, apiPreference, _fourcc, fps, frameSize, isColor);
    if (!iwriter.empty())
        return true;
    writer.reset(cvCreateVideoWriterWithPreference(filename.c_str(), apiPreference, _fourcc, fps, frameSize, isColor));
    return isOpened();
}

bool VideoWriter::isOpened() const
{
    return !iwriter.empty() || !writer.empty();
}


bool VideoWriter::set(int propId, double value)
{
    if (!iwriter.empty())
        return iwriter->setProperty(propId, value);
    return false;
}

double VideoWriter::get(int propId) const
{
    if (!iwriter.empty())
        return iwriter->getProperty(propId);
    return 0.;
}

void VideoWriter::write(const Mat& image)
{
    CV_INSTRUMENT_REGION()

    if( iwriter )
        iwriter->write(image);
    else
    {
        IplImage _img = image;
        cvWriteFrame(writer, &_img);
    }
}

VideoWriter& VideoWriter::operator << (const Mat& image)
{
    CV_INSTRUMENT_REGION()

    write(image);
    return *this;
}

int VideoWriter::fourcc(char c1, char c2, char c3, char c4)
{
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24);
}

}
