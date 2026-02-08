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

#if defined(HAVE_GTK4)

#include <gtk/gtk.h>
#include <gdk/gdkkeysyms.h>
#include <thread>
#include <chrono>
#include <map>
#include <mutex>

namespace cv {
namespace highgui_backend {

//Global State

static std::mutex g_mutex;
static int g_last_key = -1;
static bool g_gtk_initialized = false;

static void ensureGtkInitialized()
{
    if (g_gtk_initialized)
        return;

    if (!gtk_is_initialized())
    {
        if (!gtk_init_check())
            CV_Error(Error::StsError, "GTK4 initialization failed");
    }

    g_gtk_initialized = true;
}

//Gtk4 Window

class GTK4Window : public UIWindow
{
public:
    GTK4Window(const std::string& name, int flags);
    ~GTK4Window() CV_OVERRIDE;

    const std::string& getID() const CV_OVERRIDE { return name_; }
    bool isActive() const CV_OVERRIDE { return window_ != NULL; }
    void destroy() CV_OVERRIDE;

    void imshow(InputArray image) CV_OVERRIDE;
    void setMouseCallback(MouseCallback, void*) CV_OVERRIDE {}

    bool setProperty(int prop_id, double value) CV_OVERRIDE;
    double getProperty(int prop_id) const CV_OVERRIDE;

    void resize(int width, int height) CV_OVERRIDE;
    void move(int, int) CV_OVERRIDE {}
    void setTitle(const std::string& title) CV_OVERRIDE;
    Rect getImageRect() const CV_OVERRIDE;

    std::shared_ptr<UITrackbar> createTrackbar(
        const std::string&, int, TrackbarCallback, void*) CV_OVERRIDE
    { return std::shared_ptr<UITrackbar>(); }

    std::shared_ptr<UITrackbar> findTrackbar(const std::string&) CV_OVERRIDE
    { return std::shared_ptr<UITrackbar>(); }

    void onKeyPressed(guint keyval);

private:
    std::string name_;
    int flags_;
    GtkWidget* window_;
    GtkWidget* picture_;

    static gboolean on_key_pressed_cb(GtkEventControllerKey*,
                                        guint keyval,
                                        guint,
                                        GdkModifierType,
                                        gpointer userdata)
    {
        static_cast<GTK4Window*>(userdata)->onKeyPressed(keyval);
        return TRUE;
    }
};

GTK4Window::GTK4Window(const std::string& name, int flags)
    : name_(name), flags_(flags), window_(NULL), picture_(NULL)
{
    ensureGtkInitialized();

    window_ = gtk_window_new();
    gtk_window_set_title(GTK_WINDOW(window_), name.c_str());
    gtk_window_set_default_size(GTK_WINDOW(window_), 640, 480);

    picture_ = gtk_picture_new();
    gtk_widget_set_hexpand(picture_, TRUE);
    gtk_widget_set_vexpand(picture_, TRUE);
    gtk_window_set_child(GTK_WINDOW(window_), picture_);

    GtkEventController* key_ctrl = gtk_event_controller_key_new();
    g_signal_connect(key_ctrl, "key-pressed",
                        G_CALLBACK(on_key_pressed_cb), this);
    gtk_widget_add_controller(window_, key_ctrl);

    gtk_window_present(GTK_WINDOW(window_));
}

GTK4Window::~GTK4Window()
{
    destroy();
}

void GTK4Window::destroy()
{
    if (window_)
    {
        gtk_window_destroy(GTK_WINDOW(window_));
        window_ = NULL;
    }
}

void GTK4Window::imshow(InputArray image)
{
    cv::Mat mat = image.getMat();
    if (mat.empty())
        return;

    cv::Mat rgb;
    if (mat.channels() == 3)
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    else if (mat.channels() == 4)
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGBA);
    else if (mat.channels() == 1)
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    else
        return;

    cv::Mat continuous = rgb.isContinuous() ? rgb : rgb.clone();

    GBytes* bytes = g_bytes_new(continuous.data,
                               continuous.total() * continuous.elemSize());

    GdkTexture* texture = gdk_memory_texture_new(
        continuous.cols,
        continuous.rows,
        (continuous.channels() == 3) ? GDK_MEMORY_R8G8B8
                                        : GDK_MEMORY_R8G8B8A8,
        bytes,
        continuous.step[0]);

    gtk_picture_set_paintable(GTK_PICTURE(picture_), GDK_PAINTABLE(texture));

    g_object_unref(texture);
    g_bytes_unref(bytes);
}

bool GTK4Window::setProperty(int prop_id, double value)
{
    if (prop_id == cv::WND_PROP_FULLSCREEN)
    {
        if ((int)value == cv::WINDOW_FULLSCREEN)
            gtk_window_fullscreen(GTK_WINDOW(window_));
        else
            gtk_window_unfullscreen(GTK_WINDOW(window_));
        return true;
    }
    return false;
}

double GTK4Window::getProperty(int prop_id) const
{
    if (prop_id == cv::WND_PROP_AUTOSIZE)
        return (flags_ & cv::WINDOW_AUTOSIZE) ? 1.0 : 0.0;
    return -1.0;
}

void GTK4Window::resize(int width, int height)
{
    gtk_window_set_default_size(GTK_WINDOW(window_), width, height);
}

void GTK4Window::setTitle(const std::string& title)
{
    gtk_window_set_title(GTK_WINDOW(window_), title.c_str());
}

Rect GTK4Window::getImageRect() const
{
    if (!picture_)
        return Rect();
    return Rect(0, 0,
                gtk_widget_get_width(picture_),
                gtk_widget_get_height(picture_));
}

void GTK4Window::onKeyPressed(guint keyval)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_key = (int)keyval;
}

//Gtk4 Backend UI

class GTK4BackendUI : public UIBackend
{
public:
    ~GTK4BackendUI() CV_OVERRIDE { destroyAllWindows(); }
    const std::string getName() const CV_OVERRIDE { return "GTK4"; }

    std::shared_ptr<UIWindow> createWindow(const std::string& winname, int flags) CV_OVERRIDE;
    void destroyAllWindows() CV_OVERRIDE;
    int waitKeyEx(int delay) CV_OVERRIDE;
    int pollKey() CV_OVERRIDE { return waitKeyEx(1); }

private:
    std::map<std::string, std::shared_ptr<GTK4Window>> windows_;
    std::mutex windows_mutex_;
};

std::shared_ptr<UIWindow> GTK4BackendUI::createWindow(const std::string& winname, int flags)
{
    std::lock_guard<std::mutex> lock(windows_mutex_);
    auto it = windows_.find(winname);
    if (it != windows_.end())
        return it->second;

    auto win = std::make_shared<GTK4Window>(winname, flags);
    windows_[winname] = win;
    return win;
}

void GTK4BackendUI::destroyAllWindows()
{
    std::lock_guard<std::mutex> lock(windows_mutex_);
    for (auto& p : windows_)
        p.second->destroy();
    windows_.clear();
}

int GTK4BackendUI::waitKeyEx(int delay)
{
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_last_key = -1;
    }

    auto start = std::chrono::steady_clock::now();
    while (true)
    {
        while (g_main_context_pending(NULL))
            g_main_context_iteration(NULL, FALSE);

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_last_key != -1)
                return g_last_key;
        }

        if (delay > 0)
        {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= delay)
                return -1;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

//Factory

std::shared_ptr<UIBackend> createUIBackendGTK4()
{
    return std::make_shared<GTK4BackendUI>();
}

} // namespace highgui_backend
} // namespace cv

//These below are stub implemetations without which the build wont run since gtk4 is backend compatible, i am looking to implement these in phase 2 or i could do whatever the maintainer asks for - @AdityaMishra3000

CV_IMPL int cvNamedWindow(const char* name, int flags)
{
    if (!name) return 0;
    cv::namedWindow(name, flags);
    return 1;
}

CV_IMPL void cvDestroyWindow(const char* name)
{
    if (!name) return;
    cv::destroyWindow(name);
}

CV_IMPL void cvDestroyAllWindows()
{
    cv::destroyAllWindows();
}

CV_IMPL void cvShowImage(const char* name, const CvArr* arr)
{
    if (!name || !arr) return;
    cv::Mat img = cv::cvarrToMat(arr);
    cv::imshow(name, img);
}

CV_IMPL void cvResizeWindow(const char* name, int width, int height)
{
    if (!name) return;
    cv::resizeWindow(name, width, height);
}

CV_IMPL void cvMoveWindow(const char* name, int x, int y)
{
    if (!name) return;
    cv::moveWindow(name, x, y);
}

CV_IMPL void cvSetMouseCallback(const char* name, CvMouseCallback on_mouse, void* param)
{
    if (!name) return;
    cv::setMouseCallback(name, on_mouse, param);
}

CV_IMPL int cvCreateTrackbar2(const char* trackbar_name, const char* window_name,int* value, int count, CvTrackbarCallback2 on_change,void* userdata)
{
    if (!trackbar_name || !window_name) return 0;
    return cv::createTrackbar(trackbar_name, window_name, value, count, on_change, userdata);
}

CV_IMPL int cvGetTrackbarPos(const char* trackbar_name, const char* window_name)
{
    if (!trackbar_name || !window_name) return -1;
    return cv::getTrackbarPos(trackbar_name, window_name);
}

CV_IMPL void cvSetTrackbarPos(const char* trackbar_name, const char* window_name, int pos)
{
    if (!trackbar_name || !window_name) return;
    cv::setTrackbarPos(trackbar_name, window_name, pos);
}

CV_IMPL void cvSetTrackbarMax(const char* trackbar_name, const char* window_name, int maxval)
{
    if (!trackbar_name || !window_name) return;
    cv::setTrackbarMax(trackbar_name, window_name, maxval);
}

CV_IMPL void cvSetTrackbarMin(const char* trackbar_name, const char* window_name, int minval)
{
    if (!trackbar_name || !window_name) return;
    cv::setTrackbarMin(trackbar_name, window_name, minval);
}

CV_IMPL int cvStartWindowThread()
{
    return 0;
}

#endif // HAVE_GTK4

/* End of file. */
