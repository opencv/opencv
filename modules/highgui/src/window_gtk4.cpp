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

// Forward declarations
class GTK4Trackbar;

// Global state
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

// GTK4Window class
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
        const std::string& name, int count, TrackbarCallback callback, void* userdata) CV_OVERRIDE;
    std::shared_ptr<UITrackbar> findTrackbar(const std::string& name) CV_OVERRIDE;

    void onKeyPressed(guint keyval);

private:
    std::string name_;
    int flags_;
    GtkWidget* window_;
    GtkWidget* picture_;
    GtkWidget* main_box_;
    std::map<std::string, std::shared_ptr<GTK4Trackbar>> trackbars_;
    static gboolean on_key_pressed_cb(GtkEventControllerKey*,guint keyval,guint,GdkModifierType,gpointer userdata)
    {
        static_cast<GTK4Window*>(userdata)->onKeyPressed(keyval);
        return TRUE;
    }
};

GTK4Window::GTK4Window(const std::string& name, int flags)
    : name_(name), flags_(flags), window_(NULL), picture_(NULL), main_box_(NULL)
{
    ensureGtkInitialized();

    window_ = gtk_window_new();
    gtk_window_set_title(GTK_WINDOW(window_), name.c_str());
    gtk_window_set_default_size(GTK_WINDOW(window_), 640, 480);

    main_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_window_set_child(GTK_WINDOW(window_), main_box_);

    picture_ = gtk_picture_new();
    gtk_widget_set_hexpand(picture_, TRUE);
    gtk_widget_set_vexpand(picture_, TRUE);
    gtk_box_append(GTK_BOX(main_box_), picture_);

    GtkEventController* key_ctrl = gtk_event_controller_key_new();
    g_signal_connect(key_ctrl, "key-pressed",G_CALLBACK(on_key_pressed_cb), this);
    gtk_widget_add_controller(window_, key_ctrl);

    gtk_window_present(GTK_WINDOW(window_));
}

GTK4Window::~GTK4Window()
{
    if (window_)
    {
        g_signal_handlers_disconnect_by_data(window_, this);
        window_ = NULL;
        picture_ = NULL;
        main_box_ = NULL;
    }
}

void GTK4Window::destroy()
{
    if (window_ && GTK_IS_WINDOW(window_))
    {
        GdkDisplay* display = gtk_widget_get_display(window_);
        if (display && !gdk_display_is_closed(display))
        {
            gtk_window_destroy(GTK_WINDOW(window_));
        }
        window_ = NULL;
        picture_ = NULL;
        main_box_ = NULL;
    }
}

void GTK4Window::imshow(InputArray image)
{
    cv::Mat mat = image.getMat();
    if (mat.empty())
    {
        gtk_picture_set_paintable(GTK_PICTURE(picture_), NULL);
        return;
    }
    cv::Mat display_mat;
    if (mat.depth() == CV_8U || mat.depth() == CV_16U || mat.depth() == CV_32F)
    {
        display_mat = mat;
    }
    else if (mat.depth() == CV_8S)
    {
        mat.convertTo(display_mat, CV_8U, 1, 128);
    }
    else if (mat.depth() == CV_16S)
    {
        mat.convertTo(display_mat, CV_16U, 1, 32768);
    }
    else if (mat.depth() == CV_32S || mat.depth() == CV_64F)
    {
        double minVal, maxVal;
        cv::minMaxLoc(mat, &minVal, &maxVal);
        double scale = (maxVal > minVal) ? 255.0 / (maxVal - minVal) : 1.0;
        double shift = -minVal * scale;
        mat.convertTo(display_mat, CV_8U, scale, shift);
    }
    else
    {
        mat.convertTo(display_mat, CV_8U);
    }
    if (display_mat.depth() == CV_32F || display_mat.depth() == CV_16U)
    {
        cv::Mat normalized;
        double minVal, maxVal;
        cv::minMaxLoc(display_mat, &minVal, &maxVal);
        if (maxVal > minVal)
        {
            double scale = 255.0 / (maxVal - minVal);
            double shift = -minVal * scale;
            display_mat.convertTo(normalized, CV_8U, scale, shift);
            display_mat = normalized;
        }
        else
        {
            display_mat.convertTo(normalized, CV_8U);
            display_mat = normalized;
        }
    }

    cv::Mat rgb;
    if (display_mat.channels() == 3)
        cv::cvtColor(display_mat, rgb, cv::COLOR_BGR2RGB);
    else if (display_mat.channels() == 4)
        cv::cvtColor(display_mat, rgb, cv::COLOR_BGRA2RGBA);
    else if (display_mat.channels() == 1)
        cv::cvtColor(display_mat, rgb, cv::COLOR_GRAY2RGB);
    else
        return;

    cv::Mat continuous = rgb.isContinuous() ? rgb : rgb.clone();

    GBytes* bytes = g_bytes_new(continuous.data, continuous.total() * continuous.elemSize());
    GdkTexture* texture = gdk_memory_texture_new(continuous.cols, continuous.rows, GDK_MEMORY_R8G8B8, bytes, continuous.step[0]);

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
    GdkPaintable* paintable = gtk_picture_get_paintable(GTK_PICTURE(picture_));
    if (!paintable)
        return Rect(0, 0, 640, 480);
    return Rect(0, 0,
                gdk_paintable_get_intrinsic_width(paintable),
                gdk_paintable_get_intrinsic_height(paintable));
}

void GTK4Window::onKeyPressed(guint keyval)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_key = (int)keyval;
}

// GTK4Trackbar class
class GTK4Trackbar : public UITrackbar
{
public:
    GTK4Trackbar(const std::string& name, GtkWidget* parent, int* value, int count,TrackbarCallback callback, void* userdata);
    virtual ~GTK4Trackbar() CV_OVERRIDE;
    virtual const std::string& getID() const CV_OVERRIDE { return name_; }
    virtual bool isActive() const CV_OVERRIDE { return scale_ != NULL; }
    virtual void destroy() CV_OVERRIDE;
    virtual void setPos(int pos) CV_OVERRIDE;
    virtual int getPos() const CV_OVERRIDE;
    virtual void setRange(const Range& range) CV_OVERRIDE;
    virtual Range getRange() const CV_OVERRIDE;

private:
    std::string name_;
    GtkWidget* container_;
    GtkWidget* scale_;
    int* value_ptr_;
    TrackbarCallback callback_;
    void* userdata_;

    static void on_value_changed(GtkRange* range, gpointer user_data);
};

GTK4Trackbar::GTK4Trackbar(const std::string& name, GtkWidget* parent,int* value, int count,TrackbarCallback callback, void* userdata)
    : name_(name), container_(NULL), scale_(NULL),
        value_ptr_(value), callback_(callback), userdata_(userdata)
{
    container_ = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* label = gtk_label_new(name.c_str());
    gtk_box_append(GTK_BOX(container_), label);
    scale_ = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0, count, 1);
    gtk_widget_set_hexpand(scale_, TRUE);
    gtk_scale_set_draw_value(GTK_SCALE(scale_), TRUE);
    if (value)
    gtk_range_set_value(GTK_RANGE(scale_), *value);
    g_signal_connect(scale_, "value-changed", G_CALLBACK(on_value_changed), this);
    gtk_box_append(GTK_BOX(container_), scale_);
    // Insert trackbar BEFORE picture widget
    gtk_box_prepend(GTK_BOX(parent), container_);
}

GTK4Trackbar::~GTK4Trackbar()
{
    container_ = NULL;
    scale_ = NULL;
}

void GTK4Trackbar::destroy()
{
    if (container_ && GTK_IS_WIDGET(container_))
    {
        if (gtk_widget_get_parent(container_))
            gtk_widget_unparent(container_);
        container_ = NULL;
        scale_ = NULL;
    }
}



void GTK4Trackbar::setPos(int pos)
{
    if (scale_)
        gtk_range_set_value(GTK_RANGE(scale_), pos);
}

int GTK4Trackbar::getPos() const
{
    if (!scale_) return 0;
    return (int)gtk_range_get_value(GTK_RANGE(scale_));
}

void GTK4Trackbar::setRange(const Range& range)
{
    if (scale_)
        gtk_range_set_range(GTK_RANGE(scale_), range.start, range.end);
}

Range GTK4Trackbar::getRange() const
{
    if (!scale_) return Range(0, 100);
    GtkAdjustment* adj = gtk_range_get_adjustment(GTK_RANGE(scale_));
    return Range(
        (int)gtk_adjustment_get_lower(adj),
        (int)gtk_adjustment_get_upper(adj)
    );
}

void GTK4Trackbar::on_value_changed(GtkRange* range, gpointer user_data)
{
    GTK4Trackbar* trackbar = static_cast<GTK4Trackbar*>(user_data);
    int pos = (int)gtk_range_get_value(range);

    if (trackbar->value_ptr_)
        *trackbar->value_ptr_ = pos;

    if (trackbar->callback_)
        trackbar->callback_(pos, trackbar->userdata_);
}

// GTK4Window trackbar methods implementation
std::shared_ptr<UITrackbar> GTK4Window::createTrackbar(
    const std::string& name, int count, TrackbarCallback callback, void* userdata)
{
    auto it = trackbars_.find(name);
    if (it != trackbars_.end())
        return it->second;

    int* value_ptr = nullptr;

    auto trackbar = std::make_shared<GTK4Trackbar>(name, main_box_, value_ptr, count, callback, userdata);
    trackbars_[name] = trackbar;
    return trackbar;
}

std::shared_ptr<UITrackbar> GTK4Window::findTrackbar(const std::string& name)
{
    auto it = trackbars_.find(name);
    if (it != trackbars_.end())
        return it->second;
    return std::shared_ptr<UITrackbar>();
}

// GTK4BackendUI class
class GTK4BackendUI : public UIBackend
{
public:
    ~GTK4BackendUI() CV_OVERRIDE {}
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
    if (delay < 0)
    {
        while (g_main_context_pending(NULL))
            g_main_context_iteration(NULL, FALSE);
        std::lock_guard<std::mutex> lock(g_mutex);
        return g_last_key;
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

std::shared_ptr<UIBackend> createUIBackendGTK4()
{
    return std::make_shared<GTK4BackendUI>();
}

} // namespace highgui_backend
} // namespace cv

// Legacy C API stubs
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

CV_IMPL int cvWaitKey(int delay)
{
    CV_TRACE_FUNCTION();
    auto backend = cv::highgui_backend::getCurrentUIBackend();
    if (backend)
        return backend->waitKeyEx(delay);
    return -1;
}

#endif // HAVE_GTK4

/* End of file. */
