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
#include "backend.hpp"

#include "opencv2/core/opengl.hpp"
#include "opencv2/core/utils/logger.hpp"

// in later times, use this file as a dispatcher to implementations like cvcap.cpp


using namespace cv;
using namespace cv::highgui_backend;

namespace cv {

Mutex& getWindowMutex()
{
    static Mutex* g_window_mutex = new Mutex();
    return *g_window_mutex;
}

namespace impl {

typedef std::map<std::string, highgui_backend::UIWindowBase::Ptr> WindowsMap_t;
static WindowsMap_t& getWindowsMap()
{
    static WindowsMap_t g_windowsMap;
    return g_windowsMap;
}

static std::shared_ptr<UIWindow> findWindow_(const std::string& name)
{
    cv::AutoLock lock(cv::getWindowMutex());
    auto& windowsMap = getWindowsMap();
    auto i = windowsMap.find(name);
    if (i != windowsMap.end())
    {
        const auto& ui_base = i->second;
        if (ui_base)
        {
            if (!ui_base->isActive())
            {
                windowsMap.erase(i);
                return std::shared_ptr<UIWindow>();
            }
            auto window = std::dynamic_pointer_cast<UIWindow>(ui_base);
            return window;
        }
    }
    return std::shared_ptr<UIWindow>();
}

static void cleanupTrackbarCallbacksWithData_();  // forward declaration

static void cleanupClosedWindows_()
{
    cv::AutoLock lock(cv::getWindowMutex());
    auto& windowsMap = getWindowsMap();
    for (auto it = windowsMap.begin(); it != windowsMap.end();)
    {
        const auto& ui_base = it->second;
        bool erase = (!ui_base || !ui_base->isActive());
        if (erase)
        {
            it = windowsMap.erase(it);
        }
        else
        {
            ++it;
        }
    }

    cleanupTrackbarCallbacksWithData_();
}

// Just to support deprecated API, to be removed
struct TrackbarCallbackWithData
{
    std::weak_ptr<UITrackbar> trackbar_;
    int* data_;
    TrackbarCallback callback_;
    void* userdata_;

    TrackbarCallbackWithData(int* data, TrackbarCallback callback, void* userdata)
        : data_(data)
        , callback_(callback), userdata_(userdata)
    {
        // trackbar_ is initialized separatelly
    }

    ~TrackbarCallbackWithData()
    {
        CV_LOG_DEBUG(NULL, "UI/Trackbar: Cleanup deprecated TrackbarCallbackWithData");
    }

    void onChange(int pos)
    {
        if (data_)
            *data_ = pos;
        if (callback_)
            callback_(pos, userdata_);
    }

    static void onChangeCallback(int pos, void* userdata)
    {
        TrackbarCallbackWithData* thiz = (TrackbarCallbackWithData*)userdata;
        CV_Assert(thiz);
        return thiz->onChange(pos);
    }
};

typedef std::vector< std::shared_ptr<TrackbarCallbackWithData> > TrackbarCallbacksWithData_t;
static TrackbarCallbacksWithData_t& getTrackbarCallbacksWithData()
{
    static TrackbarCallbacksWithData_t g_trackbarCallbacksWithData;
    return g_trackbarCallbacksWithData;
}

static void cleanupTrackbarCallbacksWithData_()
{
    cv::AutoLock lock(cv::getWindowMutex());
    auto& callbacks = getTrackbarCallbacksWithData();
    for (auto it = callbacks.begin(); it != callbacks.end();)
    {
        const auto& cb = *it;
        bool erase = (!cb || cb->trackbar_.expired());
        if (erase)
        {
            it = callbacks.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

}}  // namespace cv::impl

using namespace cv::impl;

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
static void deprecateNotFoundNoOpBehavior()
{
    CV_LOG_ONCE_WARNING(NULL, "This no-op behavior is deprecated. Future versions of OpenCV will trigger exception in this case");
}
#define CV_NOT_FOUND_DEPRECATION deprecateNotFoundNoOpBehavior()
#endif

CV_IMPL void cvSetWindowProperty(const char* name, int prop_id, double prop_value)
{
    CV_TRACE_FUNCTION();
    CV_Assert(name);

    {
        auto window = findWindow_(name);
        if (window)
        {
            /*bool res = */window->setProperty(prop_id, prop_value);
            return;
        }
    }

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        CV_LOG_WARNING(NULL, "Can't find window with name: '" << name << "'. Do nothing");
        CV_NOT_FOUND_DEPRECATION;
    }
    else
    {
        CV_LOG_WARNING(NULL, "No UI backends available. Use OPENCV_LOG_LEVEL=DEBUG for investigation");
    }
    return;
#else
    switch(prop_id)
    {
    //change between fullscreen or not.
    case cv::WND_PROP_FULLSCREEN:

        if ((int)prop_value != cv::WINDOW_NORMAL && (int)prop_value != cv::WINDOW_FULLSCREEN)  // bad argument
            break;

        #if defined (HAVE_QT)
            cvSetModeWindow_QT(name,prop_value);
        #elif defined(HAVE_WIN32UI)
            cvSetModeWindow_W32(name,prop_value);
        #elif defined (HAVE_GTK)
            cvSetModeWindow_GTK(name,prop_value);
        #elif defined (HAVE_COCOA)
            cvSetModeWindow_COCOA(name,prop_value);
        #elif defined (WINRT)
            cvSetModeWindow_WinRT(name, prop_value);
        #endif

    break;

    case cv::WND_PROP_AUTOSIZE:
        #if defined (HAVE_QT)
            cvSetPropWindow_QT(name,prop_value);
        #endif
    break;

    case cv::WND_PROP_ASPECT_RATIO:
        #if defined (HAVE_QT)
            cvSetRatioWindow_QT(name,prop_value);
        #endif
    break;

    case cv::WND_PROP_TOPMOST:
        #if defined (HAVE_QT)
            // nothing
        #elif defined(HAVE_WIN32UI)
            cvSetPropTopmost_W32(name, (prop_value != 0 ? true : false));
        #elif defined(HAVE_COCOA)
            cvSetPropTopmost_COCOA(name, (prop_value != 0 ? true : false));
        #endif
    break;

    case cv::WND_PROP_VSYNC:
        #if defined (HAVE_QT)
            // nothing
        #elif defined (HAVE_WIN32UI)
            cvSetPropVsync_W32(name, (prop_value != 0));
        #else
            // not implemented yet for other toolkits
        #endif
    break;

    default:;
    }
#endif
}

/* return -1 if error */
CV_IMPL double cvGetWindowProperty(const char* name, int prop_id)
{
    CV_TRACE_FUNCTION();
    CV_Assert(name);

    {
        auto window = findWindow_(name);
        if (window)
        {
            double v = window->getProperty(prop_id);
            if (cvIsNaN(v))
                return -1;
            return v;
        }
    }

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        CV_LOG_WARNING(NULL, "Can't find window with name: '" << name << "'. Do nothing");
        CV_NOT_FOUND_DEPRECATION;
    }
    else
    {
        CV_LOG_WARNING(NULL, "No UI backends available. Use OPENCV_LOG_LEVEL=DEBUG for investigation");
    }
    return -1;
#else
    switch(prop_id)
    {
    case cv::WND_PROP_FULLSCREEN:

        #if defined (HAVE_QT)
            return cvGetModeWindow_QT(name);
        #elif defined(HAVE_WIN32UI)
            return cvGetModeWindow_W32(name);
        #elif defined (HAVE_GTK)
            return cvGetModeWindow_GTK(name);
        #elif defined (HAVE_COCOA)
            return cvGetModeWindow_COCOA(name);
        #elif defined (WINRT)
            return cvGetModeWindow_WinRT(name);
        #else
            return -1;
        #endif
    break;

    case cv::WND_PROP_AUTOSIZE:

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

    case cv::WND_PROP_ASPECT_RATIO:

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

    case cv::WND_PROP_OPENGL:

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

    case cv::WND_PROP_VISIBLE:
        #if defined (HAVE_QT)
            return cvGetPropVisible_QT(name);
        #elif defined(HAVE_WIN32UI)
            return cvGetPropVisible_W32(name);
        #elif defined(HAVE_COCOA)
            return cvGetPropVisible_COCOA(name);
        #else
            return -1;
        #endif
    break;

    case cv::WND_PROP_TOPMOST:
        #if defined (HAVE_QT)
            return -1;
        #elif defined(HAVE_WIN32UI)
            return cvGetPropTopmost_W32(name);
        #elif defined(HAVE_COCOA)
            return cvGetPropTopmost_COCOA(name);
        #else
            return -1;
        #endif
    break;

    case cv::WND_PROP_VSYNC:
        #if defined (HAVE_QT)
            return -1;
        #elif defined (HAVE_WIN32UI)
            return cvGetPropVsync_W32(name);
        #else
            return -1;
        #endif
    break;

    default:
        return -1;
    }
#endif
}

cv::Rect cv::getWindowImageRect(const String& winname)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!winname.empty());

    {
        auto window = findWindow_(winname);
        if (window)
        {
            return window->getImageRect();
        }
    }

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        CV_LOG_WARNING(NULL, "Can't find window with name: '" << winname << "'. Do nothing");
        CV_NOT_FOUND_DEPRECATION;
    }
    else
    {
        CV_LOG_WARNING(NULL, "No UI backends available. Use OPENCV_LOG_LEVEL=DEBUG for investigation");
    }
    return Rect(-1, -1, -1, -1);
#else

    #if defined (HAVE_QT)
        return cvGetWindowRect_QT(winname.c_str());
    #elif defined(HAVE_WIN32UI)
        return cvGetWindowRect_W32(winname.c_str());
    #elif defined (HAVE_GTK)
        return cvGetWindowRect_GTK(winname.c_str());
    #elif defined (HAVE_COCOA)
        return cvGetWindowRect_COCOA(winname.c_str());
    #elif defined (HAVE_WAYLAND)
        return cvGetWindowRect_WAYLAND(winname.c_str());
    #else
        return Rect(-1, -1, -1, -1);
    #endif

#endif
}

void cv::namedWindow( const String& winname, int flags )
{
    CV_TRACE_FUNCTION();
    CV_Assert(!winname.empty());

    {
        cv::AutoLock lock(cv::getWindowMutex());
        cleanupClosedWindows_();
        auto& windowsMap = getWindowsMap();
        auto i = windowsMap.find(winname);
        if (i != windowsMap.end())
        {
            auto ui_base = i->second;
            if (ui_base)
            {
                auto window = std::dynamic_pointer_cast<UIWindow>(ui_base);
                if (!window)
                {
                    CV_LOG_ERROR(NULL, "OpenCV/UI: Can't create window: '" << winname << "'");
                }
                return;
            }
        }
        auto backend = getCurrentUIBackend();
        if (backend)
        {
            auto window = backend->createWindow(winname, flags);
            if (!window)
            {
                CV_LOG_ERROR(NULL, "OpenCV/UI: Can't create window: '" << winname << "'");
                return;
            }
            windowsMap.emplace(winname, window);
            return;
        }
    }

    cvNamedWindow( winname.c_str(), flags );
}

void cv::destroyWindow( const String& winname )
{
    CV_TRACE_FUNCTION();

    {
        auto window = findWindow_(winname);
        if (window)
        {
            window->destroy();
            cleanupClosedWindows_();
            return;
        }
    }

    cvDestroyWindow( winname.c_str() );
}

void cv::destroyAllWindows()
{
    CV_TRACE_FUNCTION();

    {
        cv::AutoLock lock(cv::getWindowMutex());
        auto backend = getCurrentUIBackend();
        if (backend)
        {
            backend->destroyAllWindows();
            cleanupClosedWindows_();
            return;
        }
    }

    cvDestroyAllWindows();
}

void cv::resizeWindow( const String& winname, int width, int height )
{
    CV_TRACE_FUNCTION();

    {
        auto window = findWindow_(winname);
        if (window)
        {
            return window->resize(width, height);
        }
    }

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        CV_LOG_WARNING(NULL, "Can't find window with name: '" << winname << "'. Do nothing");
        CV_NOT_FOUND_DEPRECATION;
    }
    else
    {
        CV_LOG_WARNING(NULL, "No UI backends available. Use OPENCV_LOG_LEVEL=DEBUG for investigation");
    }
    return;
#else
    cvResizeWindow( winname.c_str(), width, height );
#endif
}

void cv::resizeWindow(const String& winname, const cv::Size& size)
{
   CV_TRACE_FUNCTION();
   cvResizeWindow(winname.c_str(), size.width, size.height);
}

void cv::moveWindow( const String& winname, int x, int y )
{
    CV_TRACE_FUNCTION();

    {
        auto window = findWindow_(winname);
        if (window)
        {
            return window->move(x, y);
        }
    }

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        CV_LOG_WARNING(NULL, "Can't find window with name: '" << winname << "'. Do nothing");
        CV_NOT_FOUND_DEPRECATION;
    }
    else
    {
        CV_LOG_WARNING(NULL, "No UI backends available. Use OPENCV_LOG_LEVEL=DEBUG for investigation");
    }
    return;
#else
    cvMoveWindow( winname.c_str(), x, y );
#endif
}

void cv::setWindowTitle(const String& winname, const String& title)
{
    CV_TRACE_FUNCTION();

    {
        cv::AutoLock lock(cv::getWindowMutex());
        auto window = findWindow_(winname);
        if (window)
        {
            return window->setTitle(title);
        }
    }

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        CV_LOG_WARNING(NULL, "Can't find window with name: '" << winname << "'. Do nothing");
        CV_NOT_FOUND_DEPRECATION;
    }
    else
    {
        CV_LOG_WARNING(NULL, "No UI backends available. Use OPENCV_LOG_LEVEL=DEBUG for investigation");
    }
    return;
#elif defined(HAVE_WIN32UI)
    return setWindowTitle_W32(winname, title);
#elif defined (HAVE_GTK)
    return setWindowTitle_GTK(winname, title);
#elif defined (HAVE_QT)
    return setWindowTitle_QT(winname, title);
#elif defined (HAVE_COCOA)
    return setWindowTitle_COCOA(winname, title);
#elif defined (HAVE_WAYLAND)
    return setWindowTitle_WAYLAND(winname, title);
#else
    CV_Error(Error::StsNotImplemented, "The function is not implemented. "
        "Rebuild the library with Windows, GTK+ 2.x or Cocoa support. "
        "If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script");
#endif
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

    {
        cv::AutoLock lock(cv::getWindowMutex());
        auto backend = getCurrentUIBackend();
        if (backend)
        {
            return backend->waitKeyEx(delay);
        }
    }

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
        use_legacy = utils::getConfigurationParameterBool("OPENCV_LEGACY_WAITKEY");
    }
    if (use_legacy > 0)
        return code;
#endif
    return (code != -1) ? (code & 0xff) : -1;
}

/*
 * process until queue is empty but don't wait.
 */
int cv::pollKey()
{
    CV_TRACE_FUNCTION();

    {
        cv::AutoLock lock(cv::getWindowMutex());
        auto backend = getCurrentUIBackend();
        if (backend)
        {
            return backend->pollKey();
        }
    }

#if defined(HAVE_WIN32UI)
    return pollKey_W32();
#else
    // fallback. please implement a proper polling function
    return cvWaitKey(1);
#endif
}

int cv::createTrackbar(const String& trackbarName, const String& winName,
                   int* value, int count, TrackbarCallback callback,
                   void* userdata)
{
    CV_TRACE_FUNCTION();

    CV_LOG_IF_WARNING(NULL, value, "UI/Trackbar(" << trackbarName << "@" << winName << "): Using 'value' pointer is unsafe and deprecated. Use NULL as value pointer. "
            "To fetch trackbar value setup callback.");

    {
        cv::AutoLock lock(cv::getWindowMutex());
        auto window = findWindow_(winName);
        if (window)
        {
            if (value)
            {
                auto cb = std::make_shared<TrackbarCallbackWithData>(value, callback, userdata);
                auto trackbar = window->createTrackbar(trackbarName, count, TrackbarCallbackWithData::onChangeCallback, cb.get());
                if (!trackbar)
                {
                    CV_LOG_ERROR(NULL, "OpenCV/UI: Can't create trackbar: '" << trackbarName << "'@'" << winName << "'");
                    return 0;
                }
                cb->trackbar_ = trackbar;
                getTrackbarCallbacksWithData().emplace_back(cb);
                getWindowsMap().emplace(trackbar->getID(), trackbar);
                trackbar->setPos(*value);
                return 1;
            }
            else
            {
                auto trackbar = window->createTrackbar(trackbarName, count, callback, userdata);
                if (!trackbar)
                {
                    CV_LOG_ERROR(NULL, "OpenCV/UI: Can't create trackbar: '" << trackbarName << "'@'" << winName << "'");
                    return 0;
                }
                getWindowsMap().emplace(trackbar->getID(), trackbar);
                return 1;
            }
        }
    }

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        CV_LOG_WARNING(NULL, "Can't find window with name: '" << winName << "'. Do nothing");
        CV_NOT_FOUND_DEPRECATION;
    }
    else
    {
        CV_LOG_WARNING(NULL, "No UI backends available. Use OPENCV_LOG_LEVEL=DEBUG for investigation");
    }
    return 0;
#else
    return cvCreateTrackbar2(trackbarName.c_str(), winName.c_str(),
                             value, count, callback, userdata);
#endif
}

void cv::setTrackbarPos( const String& trackbarName, const String& winName, int value )
{
    CV_TRACE_FUNCTION();

    {
        cv::AutoLock lock(cv::getWindowMutex());
        auto window = findWindow_(winName);
        if (window)
        {
            auto trackbar = window->findTrackbar(trackbarName);
            CV_Assert(trackbar);
            return trackbar->setPos(value);
        }
    }

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        CV_LOG_WARNING(NULL, "Can't find window with name: '" << winName << "'. Do nothing");
        CV_NOT_FOUND_DEPRECATION;
    }
    else
    {
        CV_LOG_WARNING(NULL, "No UI backends available. Use OPENCV_LOG_LEVEL=DEBUG for investigation");
    }
    return;
#else
    cvSetTrackbarPos(trackbarName.c_str(), winName.c_str(), value );
#endif
}

void cv::setTrackbarMax(const String& trackbarName, const String& winName, int maxval)
{
    CV_TRACE_FUNCTION();

    {
        cv::AutoLock lock(cv::getWindowMutex());
        auto window = findWindow_(winName);
        if (window)
        {
            auto trackbar = window->findTrackbar(trackbarName);
            CV_Assert(trackbar);
            Range old_range = trackbar->getRange();
            Range range(std::min(old_range.start, maxval), maxval);
            return trackbar->setRange(range);
        }
    }

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        CV_LOG_WARNING(NULL, "Can't find window with name: '" << winName << "'. Do nothing");
        CV_NOT_FOUND_DEPRECATION;
    }
    else
    {
        CV_LOG_WARNING(NULL, "No UI backends available. Use OPENCV_LOG_LEVEL=DEBUG for investigation");
    }
    return;
#else
    cvSetTrackbarMax(trackbarName.c_str(), winName.c_str(), maxval);
#endif
}

void cv::setTrackbarMin(const String& trackbarName, const String& winName, int minval)
{
    CV_TRACE_FUNCTION();

    {
        cv::AutoLock lock(cv::getWindowMutex());
        auto window = findWindow_(winName);
        if (window)
        {
            auto trackbar = window->findTrackbar(trackbarName);
            CV_Assert(trackbar);
            Range old_range = trackbar->getRange();
            Range range(minval, std::max(minval, old_range.end));
            return trackbar->setRange(range);
        }
    }

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        CV_LOG_WARNING(NULL, "Can't find window with name: '" << winName << "'. Do nothing");
        CV_NOT_FOUND_DEPRECATION;
    }
    else
    {
        CV_LOG_WARNING(NULL, "No UI backends available. Use OPENCV_LOG_LEVEL=DEBUG for investigation");
    }
    return;
#else
    cvSetTrackbarMin(trackbarName.c_str(), winName.c_str(), minval);
#endif
}

int cv::getTrackbarPos( const String& trackbarName, const String& winName )
{
    CV_TRACE_FUNCTION();

    {
        cv::AutoLock lock(cv::getWindowMutex());
        auto window = findWindow_(winName);
        if (window)
        {
            auto trackbar = window->findTrackbar(trackbarName);
            CV_Assert(trackbar);
            return trackbar->getPos();
        }
    }

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        CV_LOG_WARNING(NULL, "Can't find window with name: '" << winName << "'. Do nothing");
        CV_NOT_FOUND_DEPRECATION;
    }
    else
    {
        CV_LOG_WARNING(NULL, "No UI backends available. Use OPENCV_LOG_LEVEL=DEBUG for investigation");
    }
    return -1;
#else
    return cvGetTrackbarPos(trackbarName.c_str(), winName.c_str());
#endif
}

void cv::setMouseCallback( const String& windowName, MouseCallback onMouse, void* param)
{
    CV_TRACE_FUNCTION();

    {
        cv::AutoLock lock(cv::getWindowMutex());
        auto window = findWindow_(windowName);
        if (window)
        {
            return window->setMouseCallback(onMouse, param);
        }
    }

#if defined(OPENCV_HIGHGUI_WITHOUT_BUILTIN_BACKEND) && defined(ENABLE_PLUGINS)
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        CV_LOG_WARNING(NULL, "Can't find window with name: '" << windowName << "'. Do nothing");
        CV_NOT_FOUND_DEPRECATION;
    }
    else
    {
        CV_LOG_WARNING(NULL, "No UI backends available. Use OPENCV_LOG_LEVEL=DEBUG for investigation");
    }
    return;
#else
    cvSetMouseCallback(windowName.c_str(), onMouse, param);
#endif
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
    CV_Assert(size.width>0 && size.height>0);
    {
        cv::AutoLock lock(cv::getWindowMutex());
        cleanupClosedWindows_();
        auto& windowsMap = getWindowsMap();
        auto i = windowsMap.find(winname);
        if (i != windowsMap.end())
        {
            auto ui_base = i->second;
            if (ui_base)
            {
                auto window = std::dynamic_pointer_cast<UIWindow>(ui_base);
                if (!window)
                {
                    CV_LOG_ERROR(NULL, "OpenCV/UI: invalid window name: '" << winname << "'");
                }
                return window->imshow(_img);
            }
        }
        auto backend = getCurrentUIBackend();
        if (backend)
        {
            auto window = backend->createWindow(winname, WINDOW_AUTOSIZE);
            if (!window)
            {
                CV_LOG_ERROR(NULL, "OpenCV/UI: Can't create window: '" << winname << "'");
                return;
            }
            windowsMap.emplace(winname, window);
            return window->imshow(_img);
        }
    }

#ifndef HAVE_OPENGL
    {
        Mat img = _img.getMat();
        CvMat c_img = cvMat(img);
        cvShowImage(winname.c_str(), &c_img);
    }
#else
    const double useGl = getWindowProperty(winname, WND_PROP_OPENGL);

    if (useGl <= 0)
    {
        Mat img = _img.getMat();
        CvMat c_img = cvMat(img);
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
    CV_UNUSED(winname);
    CV_UNUSED(_tex);
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

const std::string cv::currentUIFramework()
{
    CV_TRACE_FUNCTION();

    // plugin and backend-compatible implementations
    auto backend = getCurrentUIBackend();
    if (backend)
    {
        return backend->getName();
    }

    // builtin backends
#if defined(HAVE_WIN32UI)
    CV_Assert(false); // backend-compatible
#elif defined (HAVE_GTK)
    CV_Assert(false); // backend-compatible
#elif defined (HAVE_QT)
    return std::string("QT");
#elif defined (HAVE_COCOA)
    return std::string("COCOA");
#elif defined (HAVE_WAYLAND)
    return std::string("WAYLAND");
#else
    return std::string();
#endif
}

// Without OpenGL

#ifndef HAVE_OPENGL

CV_IMPL void cvSetOpenGlDrawCallback(const char*, CvOpenGlDrawCallback, void*)
{
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
}

CV_IMPL void cvSetOpenGlContext(const char*)
{
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
}

CV_IMPL void cvUpdateWindow(const char*)
{
    CV_Error(cv::Error::OpenGlNotSupported, "The library is compiled without OpenGL support");
}

#endif // !HAVE_OPENGL

#if defined (HAVE_QT)

cv::QtFont cv::fontQt(const String& nameFont, int pointSize, Scalar color, int weight, int style, int spacing)
{
    CvFont f = cvFontQt(nameFont.c_str(), pointSize, cvScalar(color), weight, style, spacing);
    void* pf = &f; // to suppress strict-aliasing
    return *(cv::QtFont*)pf;
}

void cv::addText( const Mat& img, const String& text, Point org, const QtFont& font)
{
    CvMat _img = cvMat(img);
    cvAddText( &_img, text.c_str(), cvPoint(org), (CvFont*)&font);
}

void cv::addText( const Mat& img, const String& text, Point org, const String& nameFont,
        int pointSize, Scalar color, int weight, int style, int spacing)
{
    CvFont f = cvFontQt(nameFont.c_str(), pointSize, cvScalar(color), weight, style, spacing);
    CvMat _img = cvMat(img);
    cvAddText( &_img, text.c_str(), cvPoint(org), &f);
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
    CV_Error(cv::Error::StsNotImplemented, NO_QT_ERR_MSG);
}

void cv::addText( const Mat&, const String&, Point, const QtFont&)
{
    CV_Error(cv::Error::StsNotImplemented, NO_QT_ERR_MSG);
}

void cv::addText(const Mat&, const String&, Point, const String&, int, Scalar, int, int, int)
{
    CV_Error(cv::Error::StsNotImplemented, NO_QT_ERR_MSG);
}

void cv::displayStatusBar(const String&,  const String&, int)
{
    CV_Error(cv::Error::StsNotImplemented, NO_QT_ERR_MSG);
}

void cv::displayOverlay(const String&,  const String&, int )
{
    CV_Error(cv::Error::StsNotImplemented, NO_QT_ERR_MSG);
}

int cv::startLoop(int (*)(int argc, char *argv[]), int , char**)
{
    CV_Error(cv::Error::StsNotImplemented, NO_QT_ERR_MSG);
}

void cv::stopLoop()
{
    CV_Error(cv::Error::StsNotImplemented, NO_QT_ERR_MSG);
}

void cv::saveWindowParameters(const String&)
{
    CV_Error(cv::Error::StsNotImplemented, NO_QT_ERR_MSG);
}

void cv::loadWindowParameters(const String&)
{
    CV_Error(cv::Error::StsNotImplemented, NO_QT_ERR_MSG);
}

int cv::createButton(const String&, ButtonCallback, void*, int , bool )
{
    CV_Error(cv::Error::StsNotImplemented, NO_QT_ERR_MSG);
}

#endif

#if   defined (HAVE_WIN32UI)  // see window_w32.cpp
#elif defined (HAVE_GTK)      // see window_gtk.cpp
#elif defined (HAVE_COCOA)    // see window_cocoa.mm
#elif defined (HAVE_QT)       // see window_QT.cpp
#elif defined (HAVE_WAYLAND)  // see window_wayland.cpp
#elif defined (WINRT) && !defined (WINRT_8_0) // see window_winrt.cpp

#else

// No windowing system present at compile time ;-(
//
// We will build place holders that don't break the API but give an error
// at runtime. This way people can choose to replace an installed HighGUI
// version with a more capable one without a need to recompile dependent
// applications or libraries.

#define CV_NO_GUI_ERROR(funcname) \
    cv::error(cv::Error::StsError, \
    "The function is not implemented. " \
    "Rebuild the library with Windows, GTK+ 2.x or Cocoa support. "\
    "If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script", \
    funcname, __FILE__, __LINE__)


CV_IMPL int cvNamedWindow( const char*, int )
{
    CV_NO_GUI_ERROR("cvNamedWindow");
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
}

CV_IMPL int
cvCreateTrackbar2( const char* /*trackbar_name*/, const char* /*window_name*/,
                   int* /*val*/, int /*count*/, CvTrackbarCallback2 /*on_notify2*/,
                   void* /*userdata*/ )
{
    CV_NO_GUI_ERROR( "cvCreateTrackbar2" );
}

CV_IMPL void
cvSetMouseCallback( const char*, CvMouseCallback, void* )
{
    CV_NO_GUI_ERROR( "cvSetMouseCallback" );
}

CV_IMPL int cvGetTrackbarPos( const char*, const char* )
{
    CV_NO_GUI_ERROR( "cvGetTrackbarPos" );
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
}

CV_IMPL const char* cvGetWindowName( void* )
{
    CV_NO_GUI_ERROR( "cvGetWindowName" );
}

CV_IMPL int cvWaitKey( int )
{
    CV_NO_GUI_ERROR( "cvWaitKey" );
}

CV_IMPL int cvInitSystem( int , char** )
{

    CV_NO_GUI_ERROR( "cvInitSystem" );
}

CV_IMPL int cvStartWindowThread()
{

    CV_NO_GUI_ERROR( "cvStartWindowThread" );
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
    CV_UNUSED(argv);
    CV_NO_GUI_ERROR("cvStartLoop");
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
}

#endif

/* End of file. */
