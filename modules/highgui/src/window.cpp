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
#include <map>
#include "opencv2/core/opengl.hpp"

// in later times, use this file as a dispatcher to implementations like cvcap.cpp

CV_IMPL void cvSetWindowProperty(const char* name, int prop_id, double prop_value)
{
    switch(prop_id)
    {
    //change between fullscreen or not.
    case CV_WND_PROP_FULLSCREEN:

        if (!name || (prop_value!=CV_WINDOW_NORMAL && prop_value!=CV_WINDOW_FULLSCREEN))//bad argument
            break;

        #if defined (HAVE_QT)
            cvSetModeWindow_QT(name,prop_value);
        #elif defined(HAVE_WIN32UI)
            cvSetModeWindow_W32(name,prop_value);
        #elif defined (HAVE_GTK)
            cvSetModeWindow_GTK(name,prop_value);
        #elif defined (HAVE_CARBON)
            cvSetModeWindow_CARBON(name,prop_value);
        #elif defined (HAVE_COCOA)
            cvSetModeWindow_COCOA(name,prop_value);
        #elif defined (WINRT)
            cvSetModeWindow_WinRT(name, prop_value);
        #endif

    break;

    case CV_WND_PROP_AUTOSIZE:
        #if defined (HAVE_QT)
            cvSetPropWindow_QT(name,prop_value);
        #endif
    break;

    case CV_WND_PROP_ASPECTRATIO:
        #if defined (HAVE_QT)
            cvSetRatioWindow_QT(name,prop_value);
        #endif
    break;

    default:;
    }
}

/* return -1 if error */
CV_IMPL double cvGetWindowProperty(const char* name, int prop_id)
{
    if (!name)
        return -1;

    switch(prop_id)
    {
    case CV_WND_PROP_FULLSCREEN:

        #if defined (HAVE_QT)
            return cvGetModeWindow_QT(name);
        #elif defined(HAVE_WIN32UI)
            return cvGetModeWindow_W32(name);
        #elif defined (HAVE_GTK)
            return cvGetModeWindow_GTK(name);
        #elif defined (HAVE_CARBON)
            return cvGetModeWindow_CARBON(name);
        #elif defined (HAVE_COCOA)
            return cvGetModeWindow_COCOA(name);
        #elif defined (WINRT)
            return cvGetModeWindow_WinRT(name);
        #else
            return -1;
        #endif
    break;

    case CV_WND_PROP_AUTOSIZE:

        #if defined (HAVE_QT)
            return cvGetPropWindow_QT(name);
        #elif defined(HAVE_WIN32UI)
            return cvGetPropWindowAutoSize_W32(name);
        #elif defined (HAVE_GTK)
            return cvGetPropWindowAutoSize_GTK(name);
        #else
            return -1;
        #endif
    break;

    case CV_WND_PROP_ASPECTRATIO:

        #if defined (HAVE_QT)
            return cvGetRatioWindow_QT(name);
        #elif defined(HAVE_WIN32UI)
            return cvGetRatioWindow_W32(name);
        #elif defined (HAVE_GTK)
            return cvGetRatioWindow_GTK(name);
        #else
            return -1;
        #endif
    break;

    case CV_WND_PROP_OPENGL:

        #if defined (HAVE_QT)
            return cvGetOpenGlProp_QT(name);
        #elif defined(HAVE_WIN32UI)
            return cvGetOpenGlProp_W32(name);
        #elif defined (HAVE_GTK)
            return cvGetOpenGlProp_GTK(name);
        #else
            return -1;
        #endif
    break;

    case CV_WND_PROP_VISIBLE:
        #if defined (HAVE_QT)
            return cvGetPropVisible_QT(name);
        #else
            return -1;
        #endif
    break;

    default:
        return -1;
    }
}

void cv::namedWindow( const String& winname, int flags )
{
    CV_TRACE_FUNCTION();
    cvNamedWindow( winname.c_str(), flags );
}

void cv::destroyWindow( const String& winname )
{
    CV_TRACE_FUNCTION();
    cvDestroyWindow( winname.c_str() );
}

void cv::destroyAllWindows()
{
    CV_TRACE_FUNCTION();
    cvDestroyAllWindows();
}

void cv::resizeWindow( const String& winname, int width, int height )
{
    CV_TRACE_FUNCTION();
    cvResizeWindow( winname.c_str(), width, height );
}

void cv::moveWindow( const String& winname, int x, int y )
{
    CV_TRACE_FUNCTION();
    cvMoveWindow( winname.c_str(), x, y );
}

void cv::setWindowProperty(const String& winname, int prop_id, double prop_value)
{
    CV_TRACE_FUNCTION();
    cvSetWindowProperty( winname.c_str(), prop_id, prop_value);
}

double cv::getWindowProperty(const String& winname, int prop_id)
{
    CV_TRACE_FUNCTION();
    return cvGetWindowProperty(winname.c_str(), prop_id);
}

int cv::waitKeyEx(int delay)
{
    CV_TRACE_FUNCTION();
    return cvWaitKey(delay);
}

int cv::waitKey(int delay)
{
    CV_TRACE_FUNCTION();
    int code = waitKeyEx(delay);
#ifndef WINRT
    static int use_legacy = -1;
    if (use_legacy < 0)
    {
        use_legacy = getenv("OPENCV_LEGACY_WAITKEY") != NULL ? 1 : 0;
    }
    if (use_legacy > 0)
        return code;
#endif
    return (code != -1) ? (code & 0xff) : -1;
}

int cv::createTrackbar(const String& trackbarName, const String& winName,
                   int* value, int count, TrackbarCallback callback,
                   void* userdata)
{
    CV_TRACE_FUNCTION();
    return cvCreateTrackbar2(trackbarName.c_str(), winName.c_str(),
                             value, count, callback, userdata);
}

void cv::setTrackbarPos( const String& trackbarName, const String& winName, int value )
{
    CV_TRACE_FUNCTION();
    cvSetTrackbarPos(trackbarName.c_str(), winName.c_str(), value );
}

void cv::setTrackbarMax(const String& trackbarName, const String& winName, int maxval)
{
    CV_TRACE_FUNCTION();
    cvSetTrackbarMax(trackbarName.c_str(), winName.c_str(), maxval);
}

void cv::setTrackbarMin(const String& trackbarName, const String& winName, int minval)
{
    CV_TRACE_FUNCTION();
    cvSetTrackbarMin(trackbarName.c_str(), winName.c_str(), minval);
}

int cv::getTrackbarPos( const String& trackbarName, const String& winName )
{
    CV_TRACE_FUNCTION();
    return cvGetTrackbarPos(trackbarName.c_str(), winName.c_str());
}

void cv::setMouseCallback( const String& windowName, MouseCallback onMouse, void* param)
{
    CV_TRACE_FUNCTION();
    cvSetMouseCallback(windowName.c_str(), onMouse, param);
}

int cv::getMouseWheelDelta( int flags )
{
    CV_TRACE_FUNCTION();
    return CV_GET_WHEEL_DELTA(flags);
}

int cv::startWindowThread()
{
    CV_TRACE_FUNCTION();
    return cvStartWindowThread();
}

// OpenGL support

void cv::setOpenGlDrawCallback(const String& name, OpenGlDrawCallback callback, void* userdata)
{
    CV_TRACE_FUNCTION();
    cvSetOpenGlDrawCallback(name.c_str(), callback, userdata);
}

void cv::setOpenGlContext(const String& windowName)
{
    CV_TRACE_FUNCTION();
    cvSetOpenGlContext(windowName.c_str());
}

void cv::updateWindow(const String& windowName)
{
    CV_TRACE_FUNCTION();
    cvUpdateWindow(windowName.c_str());
}

#ifdef HAVE_OPENGL
namespace
{
    std::map<cv::String, cv::ogl::Texture2D> wndTexs;
    std::map<cv::String, cv::ogl::Texture2D> ownWndTexs;
    std::map<cv::String, cv::ogl::Buffer> ownWndBufs;

    void glDrawTextureCallback(void* userdata)
    {
        cv::ogl::Texture2D* texObj = static_cast<cv::ogl::Texture2D*>(userdata);

        cv::ogl::render(*texObj);
    }
}
#endif // HAVE_OPENGL

void cv::imshow( const String& winname, InputArray _img )
{
    CV_TRACE_FUNCTION();
    const Size size = _img.size();
#ifndef HAVE_OPENGL
    CV_Assert(size.width>0 && size.height>0);
    {
        Mat img = _img.getMat();
        CvMat c_img = img;
        cvShowImage(winname.c_str(), &c_img);
    }
#else
    const double useGl = getWindowProperty(winname, WND_PROP_OPENGL);
    CV_Assert(size.width>0 && size.height>0);

    if (useGl <= 0)
    {
        Mat img = _img.getMat();
        CvMat c_img = img;
        cvShowImage(winname.c_str(), &c_img);
    }
    else
    {
        const double autoSize = getWindowProperty(winname, WND_PROP_AUTOSIZE);

        if (autoSize > 0)
        {
            resizeWindow(winname, size.width, size.height);
        }

        setOpenGlContext(winname);

        cv::ogl::Texture2D& tex = ownWndTexs[winname];

        if (_img.kind() == _InputArray::CUDA_GPU_MAT)
        {
            cv::ogl::Buffer& buf = ownWndBufs[winname];
            buf.copyFrom(_img);
            buf.setAutoRelease(false);

            tex.copyFrom(buf);
            tex.setAutoRelease(false);
        }
        else
        {
            tex.copyFrom(_img);
        }

        tex.setAutoRelease(false);

        setOpenGlDrawCallback(winname, glDrawTextureCallback, &tex);

        updateWindow(winname);
    }
#endif
}

void cv::imshow(const String& winname, const ogl::Texture2D& _tex)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENGL
    (void) winname;
    (void) _tex;
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
#else
    const double useGl = getWindowProperty(winname, WND_PROP_OPENGL);

    if (useGl <= 0)
    {
        CV_Error(cv::Error::OpenGlNotSupported, "The window was created without OpenGL context");
    }
    else
    {
        const double autoSize = getWindowProperty(winname, WND_PROP_AUTOSIZE);

        if (autoSize > 0)
        {
            Size size = _tex.size();
            resizeWindow(winname, size.width, size.height);
        }

        setOpenGlContext(winname);

        cv::ogl::Texture2D& tex = wndTexs[winname];

        tex = _tex;

        tex.setAutoRelease(false);

        setOpenGlDrawCallback(winname, glDrawTextureCallback, &tex);

        updateWindow(winname);
    }
#endif
}

// Without OpenGL

#ifndef HAVE_OPENGL

CV_IMPL void cvSetOpenGlDrawCallback(const char*, CvOpenGlDrawCallback, void*)
{
    CV_Error(CV_OpenGlNotSupported, "The library is compiled without OpenGL support");
}

CV_IMPL void cvSetOpenGlContext(const char*)
{
    CV_Error(CV_OpenGlNotSupported, "The library is compiled without OpenGL support");
}

CV_IMPL void cvUpdateWindow(const char*)
{
    CV_Error(CV_OpenGlNotSupported, "The library is compiled without OpenGL support");
}

#endif // !HAVE_OPENGL

#if defined (HAVE_QT)

cv::QtFont cv::fontQt(const String& nameFont, int pointSize, Scalar color, int weight, int style, int spacing)
{
    CvFont f = cvFontQt(nameFont.c_str(), pointSize,color,weight, style, spacing);
    void* pf = &f; // to suppress strict-aliasing
    return *(cv::QtFont*)pf;
}

void cv::addText( const Mat& img, const String& text, Point org, const QtFont& font)
{
    CvMat _img = img;
    cvAddText( &_img, text.c_str(), org, (CvFont*)&font);
}

void cv::addText( const Mat& img, const String& text, Point org, const String& nameFont,
        int pointSize, Scalar color, int weight, int style, int spacing)
{
    CvFont f = cvFontQt(nameFont.c_str(), pointSize,color,weight, style, spacing);
    CvMat _img = img;
    cvAddText( &_img, text.c_str(), org, &f);
}

void cv::displayStatusBar(const String& name,  const String& text, int delayms)
{
    cvDisplayStatusBar(name.c_str(),text.c_str(), delayms);
}

void cv::displayOverlay(const String& name,  const String& text, int delayms)
{
    cvDisplayOverlay(name.c_str(),text.c_str(), delayms);
}

int cv::startLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[])
{
    return cvStartLoop(pt2Func, argc, argv);
}

void cv::stopLoop()
{
    cvStopLoop();
}

void cv::saveWindowParameters(const String& windowName)
{
    cvSaveWindowParameters(windowName.c_str());
}

void cv::loadWindowParameters(const String& windowName)
{
    cvLoadWindowParameters(windowName.c_str());
}

int cv::createButton(const String& button_name, ButtonCallback on_change, void* userdata, int button_type , bool initial_button_state  )
{
    return cvCreateButton(button_name.c_str(), on_change, userdata, button_type , initial_button_state );
}

#else

static const char* NO_QT_ERR_MSG = "The library is compiled without QT support";

cv::QtFont cv::fontQt(const String&, int, Scalar, int,  int, int)
{
    CV_Error(CV_StsNotImplemented, NO_QT_ERR_MSG);
    return QtFont();
}

void cv::addText( const Mat&, const String&, Point, const QtFont&)
{
    CV_Error(CV_StsNotImplemented, NO_QT_ERR_MSG);
}

void cv::addText(const Mat&, const String&, Point, const String&, int, Scalar, int, int, int)
{
    CV_Error(CV_StsNotImplemented, NO_QT_ERR_MSG);
}

void cv::displayStatusBar(const String&,  const String&, int)
{
    CV_Error(CV_StsNotImplemented, NO_QT_ERR_MSG);
}

void cv::displayOverlay(const String&,  const String&, int )
{
    CV_Error(CV_StsNotImplemented, NO_QT_ERR_MSG);
}

int cv::startLoop(int (*)(int argc, char *argv[]), int , char**)
{
    CV_Error(CV_StsNotImplemented, NO_QT_ERR_MSG);
    return 0;
}

void cv::stopLoop()
{
    CV_Error(CV_StsNotImplemented, NO_QT_ERR_MSG);
}

void cv::saveWindowParameters(const String&)
{
    CV_Error(CV_StsNotImplemented, NO_QT_ERR_MSG);
}

void cv::loadWindowParameters(const String&)
{
    CV_Error(CV_StsNotImplemented, NO_QT_ERR_MSG);
}

int cv::createButton(const String&, ButtonCallback, void*, int , bool )
{
    CV_Error(CV_StsNotImplemented, NO_QT_ERR_MSG);
    return 0;
}

#endif

#if   defined (HAVE_WIN32UI)  // see window_w32.cpp
#elif defined (HAVE_GTK)      // see window_gtk.cpp
#elif defined (HAVE_COCOA)    // see window_carbon.cpp
#elif defined (HAVE_CARBON)
#elif defined (HAVE_QT)       // see window_QT.cpp
#elif defined (WINRT) && !defined (WINRT_8_0) // see window_winrt.cpp

#else

// No windowing system present at compile time ;-(
//
// We will build place holders that don't break the API but give an error
// at runtime. This way people can choose to replace an installed HighGUI
// version with a more capable one without a need to recompile dependent
// applications or libraries.

void cv::setWindowTitle(const String&, const String&)
{
    CV_Error(Error::StsNotImplemented, "The function is not implemented. "
        "Rebuild the library with Windows, GTK+ 2.x or Carbon support. "
        "If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script");
}

#define CV_NO_GUI_ERROR(funcname) \
    cvError( CV_StsError, funcname, \
    "The function is not implemented. " \
    "Rebuild the library with Windows, GTK+ 2.x or Carbon support. "\
    "If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script", \
    __FILE__, __LINE__ )


CV_IMPL int cvNamedWindow( const char*, int )
{
    CV_NO_GUI_ERROR("cvNamedWindow");
    return -1;
}

CV_IMPL void cvDestroyWindow( const char* )
{
    CV_NO_GUI_ERROR( "cvDestroyWindow" );
}

CV_IMPL void
cvDestroyAllWindows( void )
{
    CV_NO_GUI_ERROR( "cvDestroyAllWindows" );
}

CV_IMPL void
cvShowImage( const char*, const CvArr* )
{
    CV_NO_GUI_ERROR( "cvShowImage" );
}

CV_IMPL void cvResizeWindow( const char*, int, int )
{
    CV_NO_GUI_ERROR( "cvResizeWindow" );
}

CV_IMPL void cvMoveWindow( const char*, int, int )
{
    CV_NO_GUI_ERROR( "cvMoveWindow" );
}

CV_IMPL int
cvCreateTrackbar( const char*, const char*,
                  int*, int, CvTrackbarCallback )
{
    CV_NO_GUI_ERROR( "cvCreateTrackbar" );
    return -1;
}

CV_IMPL int
cvCreateTrackbar2( const char* /*trackbar_name*/, const char* /*window_name*/,
                   int* /*val*/, int /*count*/, CvTrackbarCallback2 /*on_notify2*/,
                   void* /*userdata*/ )
{
    CV_NO_GUI_ERROR( "cvCreateTrackbar2" );
    return -1;
}

CV_IMPL void
cvSetMouseCallback( const char*, CvMouseCallback, void* )
{
    CV_NO_GUI_ERROR( "cvSetMouseCallback" );
}

CV_IMPL int cvGetTrackbarPos( const char*, const char* )
{
    CV_NO_GUI_ERROR( "cvGetTrackbarPos" );
    return -1;
}

CV_IMPL void cvSetTrackbarPos( const char*, const char*, int )
{
    CV_NO_GUI_ERROR( "cvSetTrackbarPos" );
}

CV_IMPL void cvSetTrackbarMax(const char*, const char*, int)
{
    CV_NO_GUI_ERROR( "cvSetTrackbarMax" );
}

CV_IMPL void cvSetTrackbarMin(const char*, const char*, int)
{
    CV_NO_GUI_ERROR( "cvSetTrackbarMin" );
}

CV_IMPL void* cvGetWindowHandle( const char* )
{
    CV_NO_GUI_ERROR( "cvGetWindowHandle" );
    return 0;
}

CV_IMPL const char* cvGetWindowName( void* )
{
    CV_NO_GUI_ERROR( "cvGetWindowName" );
    return 0;
}

CV_IMPL int cvWaitKey( int )
{
    CV_NO_GUI_ERROR( "cvWaitKey" );
    return -1;
}

CV_IMPL int cvInitSystem( int , char** )
{

    CV_NO_GUI_ERROR( "cvInitSystem" );
    return -1;
}

CV_IMPL int cvStartWindowThread()
{

    CV_NO_GUI_ERROR( "cvStartWindowThread" );
    return -1;
}

//-------- Qt ---------
CV_IMPL void cvAddText( const CvArr*, const char*, CvPoint , CvFont* )
{
    CV_NO_GUI_ERROR("cvAddText");
}

CV_IMPL void cvDisplayStatusBar(const char* , const char* , int )
{
    CV_NO_GUI_ERROR("cvDisplayStatusBar");
}

CV_IMPL void cvDisplayOverlay(const char* , const char* , int )
{
    CV_NO_GUI_ERROR("cvNamedWindow");
}

CV_IMPL int cvStartLoop(int (*)(int argc, char *argv[]), int , char* argv[])
{
    (void)argv;
    CV_NO_GUI_ERROR("cvStartLoop");
    return -1;
}

CV_IMPL void cvStopLoop()
{
    CV_NO_GUI_ERROR("cvStopLoop");
}

CV_IMPL void cvSaveWindowParameters(const char* )
{
    CV_NO_GUI_ERROR("cvSaveWindowParameters");
}

// CV_IMPL void cvLoadWindowParameterss(const char* name)
// {
//     CV_NO_GUI_ERROR("cvLoadWindowParameters");
// }

CV_IMPL int cvCreateButton(const char*, void (*)(int, void*), void*, int, int)
{
    CV_NO_GUI_ERROR("cvCreateButton");
    return -1;
}


#endif

/* End of file. */
