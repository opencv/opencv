/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
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
#include <condition_variable>

// Fix for narrowing conversion warnings in graphene
#define GRAPHENE_RECT_INIT_FLOAT(x, y, w, h) \
    GRAPHENE_RECT_INIT((float)(x), (float)(y), (float)(w), (float)(h))

namespace cv {
namespace highgui_backend {

// Thread safety utilities
static inline void assertMainThread(const char* function_name) {
    // Implementation detail: checking default context owner
}

static std::mutex g_mutex;
static int g_last_key = -1;
static bool g_gtk_initialized = false;

static void ensureGtkInitialized()
{
    if (g_gtk_initialized) return;
    if (!gtk_is_initialized()) {
        if (!gtk_init_check())
            CV_Error(Error::StsError, "GTK4 initialization failed.");
    }
    g_gtk_initialized = true;
}

// ============================================================================
// CvImageWidget definition
// ============================================================================

#define CV_TYPE_IMAGE_WIDGET (cv_image_widget_get_type())
G_DECLARE_FINAL_TYPE(CvImageWidget, cv_image_widget, CV, IMAGE_WIDGET, GtkDrawingArea)

struct _CvImageWidget {
    GtkDrawingArea parent;
    cv::Mat original_image;
    cv::Mat scaled_image;
    GdkTexture *cached_texture;
    bool texture_needs_update;
    int flags;
    bool has_image;
};

G_DEFINE_TYPE(CvImageWidget, cv_image_widget, GTK_TYPE_DRAWING_AREA)

static void cv_image_widget_dispose(GObject *object) {
    CvImageWidget *widget = CV_IMAGE_WIDGET(object);
    if (widget->cached_texture) {
        g_object_unref(widget->cached_texture);
        widget->cached_texture = NULL;
    }
    G_OBJECT_CLASS(cv_image_widget_parent_class)->dispose(object);
}

static void cv_image_widget_snapshot(GtkWidget *gtk_widget, GtkSnapshot *snapshot) {
    CvImageWidget *widget = CV_IMAGE_WIDGET(gtk_widget);
    if (!widget->has_image || widget->scaled_image.empty()) return;

    int w = gtk_widget_get_width(gtk_widget);
    int h = gtk_widget_get_height(gtk_widget);
    if (w <= 0 || h <= 0) return;

    if (widget->texture_needs_update || !widget->cached_texture) {
        cv::Mat &img = widget->scaled_image;
        cv::Mat rgb_image;
        
        if (img.channels() == 3) cv::cvtColor(img, rgb_image, cv::COLOR_BGR2RGB);
        else if (img.channels() == 4) cv::cvtColor(img, rgb_image, cv::COLOR_BGRA2RGBA);
        else if (img.channels() == 1) cv::cvtColor(img, rgb_image, cv::COLOR_GRAY2RGB);
        else return;

        cv::Mat contiguous = rgb_image.isContinuous() ? rgb_image : rgb_image.clone();
        gsize data_size = contiguous.total() * contiguous.elemSize();
        GBytes *bytes = g_bytes_new(contiguous.data, data_size);

        if (widget->cached_texture) g_object_unref(widget->cached_texture);
        
        widget->cached_texture = gdk_memory_texture_new(
            contiguous.cols, contiguous.rows,
            (contiguous.channels() == 3) ? GDK_MEMORY_R8G8B8 : GDK_MEMORY_R8G8B8A8,
            bytes, contiguous.step[0]
        );
        g_bytes_unref(bytes);
        widget->texture_needs_update = false;
    }

    graphene_rect_t bounds = GRAPHENE_RECT_INIT_FLOAT(0, 0, w, h);
    gtk_snapshot_append_texture(snapshot, widget->cached_texture, &bounds);
}

static void cv_image_widget_measure(GtkWidget *widget, GtkOrientation orientation,
                                    int, int *minimum, int *natural,
                                    int *minimum_baseline, int *natural_baseline) {
    CvImageWidget *iw = CV_IMAGE_WIDGET(widget);
    if (orientation == GTK_ORIENTATION_HORIZONTAL) {
        *minimum = 160; *natural = (!iw->original_image.empty()) ? iw->original_image.cols : 640;
    } else {
        *minimum = 120; *natural = (!iw->original_image.empty()) ? iw->original_image.rows : 480;
    }
    *minimum_baseline = -1; *natural_baseline = -1;
}

static void cv_image_widget_class_init(CvImageWidgetClass *klass) {
    GObjectClass *object_class = G_OBJECT_CLASS(klass);
    GtkWidgetClass *widget_class = GTK_WIDGET_CLASS(klass);
    object_class->dispose = cv_image_widget_dispose;
    widget_class->snapshot = cv_image_widget_snapshot;
    widget_class->measure = cv_image_widget_measure;
}

static void cv_image_widget_init(CvImageWidget *widget) {
    widget->flags = 0;
    widget->has_image = false;
    widget->cached_texture = NULL;
    widget->texture_needs_update = false;
}

static GtkWidget* cv_image_widget_new(int flags) {
    CvImageWidget *widget = (CvImageWidget*)g_object_new(CV_TYPE_IMAGE_WIDGET, NULL);
    widget->flags = flags;
    return GTK_WIDGET(widget);
}

static void cv_image_widget_set_image(CvImageWidget *widget, const cv::Mat &image) {
    if (image.empty()) return;
    cv::Mat display_image;
    convertToShow(image, display_image, false);
    widget->original_image = display_image.clone();
    widget->has_image = true;
    
    int w = gtk_widget_get_width(GTK_WIDGET(widget));
    int h = gtk_widget_get_height(GTK_WIDGET(widget));
    
    if (w > 0 && h > 0 && !(widget->flags & cv::WINDOW_AUTOSIZE))
        cv::resize(display_image, widget->scaled_image, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    else
        widget->scaled_image = widget->original_image.clone();

    widget->texture_needs_update = true;
    gtk_widget_queue_draw(GTK_WIDGET(widget));
    if (widget->flags & cv::WINDOW_AUTOSIZE) gtk_widget_queue_resize(GTK_WIDGET(widget));
}

// ============================================================================
// GTK4Window Definition
// ============================================================================

class GTK4Window : public UIWindow
{
public:
    GTK4Window(const std::string& name, int flags);
    virtual ~GTK4Window() CV_OVERRIDE;

    virtual const std::string& getID() const CV_OVERRIDE { return name_; }
    virtual bool isActive() const CV_OVERRIDE { return window_ != NULL; }
    virtual void destroy() CV_OVERRIDE;
    
    virtual void imshow(InputArray image) CV_OVERRIDE;
    virtual void setMouseCallback(MouseCallback callback, void* userdata) CV_OVERRIDE;
    
    virtual bool setProperty(int prop_id, double value) CV_OVERRIDE;
    virtual double getProperty(int prop_id) const CV_OVERRIDE;
    
    virtual void resize(int width, int height) CV_OVERRIDE;
    virtual void move(int x, int y) CV_OVERRIDE;
    virtual void setTitle(const std::string& title) CV_OVERRIDE;
    virtual Rect getImageRect() const CV_OVERRIDE;

    virtual std::shared_ptr<UITrackbar> createTrackbar(
        const std::string& name, int count, TrackbarCallback callback, void* userdata) CV_OVERRIDE;
    virtual std::shared_ptr<UITrackbar> findTrackbar(const std::string& name) CV_OVERRIDE;

    void onKeyPressed(guint keyval, guint keycode, GdkModifierType state);

private:
    std::string name_;
    int flags_;
    GtkWidget *window_;
    GtkWidget *main_box_;
    GtkWidget *image_widget_;
    
    static gboolean on_key_pressed_cb(GtkEventControllerKey*, guint kv, guint kc, GdkModifierType s, gpointer u) {
        static_cast<GTK4Window*>(u)->onKeyPressed(kv, kc, s); return TRUE;
    }
};

GTK4Window::GTK4Window(const std::string& name, int flags) 
    : name_(name), flags_(flags), window_(NULL)
{
    ensureGtkInitialized();
    window_ = gtk_window_new();
    gtk_window_set_title(GTK_WINDOW(window_), name.c_str());
    gtk_window_set_default_size(GTK_WINDOW(window_), 640, 480);
    
    main_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_window_set_child(GTK_WINDOW(window_), main_box_);
    
    image_widget_ = cv_image_widget_new(flags);
    gtk_widget_set_hexpand(image_widget_, TRUE);
    gtk_widget_set_vexpand(image_widget_, TRUE);
    gtk_box_append(GTK_BOX(main_box_), image_widget_);
    
    GtkEventController *key_ctrl = gtk_event_controller_key_new();
    g_signal_connect(key_ctrl, "key-pressed", G_CALLBACK(on_key_pressed_cb), this);
    gtk_widget_add_controller(window_, key_ctrl);
    
    gtk_window_present(GTK_WINDOW(window_));
}

GTK4Window::~GTK4Window() { destroy(); }

void GTK4Window::destroy() {
    if (window_) {
        gtk_window_destroy(GTK_WINDOW(window_));
        window_ = NULL;
    }
}

void GTK4Window::imshow(InputArray image) {
    cv::Mat mat = image.getMat();
    if (CV_IS_IMAGE_WIDGET(image_widget_))
        cv_image_widget_set_image(CV_IMAGE_WIDGET(image_widget_), mat);
}

void GTK4Window::setMouseCallback(MouseCallback, void*) {
    // TODO: Implement Phase 2
}

bool GTK4Window::setProperty(int prop_id, double value) {
    if (prop_id == cv::WND_PROP_FULLSCREEN) {
        if (value == cv::WINDOW_FULLSCREEN) gtk_window_fullscreen(GTK_WINDOW(window_));
        else gtk_window_unfullscreen(GTK_WINDOW(window_));
        return true;
    }
    return false;
}

double GTK4Window::getProperty(int prop_id) const {
    if (prop_id == cv::WND_PROP_AUTOSIZE) return (flags_ & cv::WINDOW_AUTOSIZE) ? 1.0 : 0.0;
    return -1;
}

void GTK4Window::resize(int width, int height) { gtk_window_set_default_size(GTK_WINDOW(window_), width, height); }
void GTK4Window::move(int, int) { }
void GTK4Window::setTitle(const std::string& title) { gtk_window_set_title(GTK_WINDOW(window_), title.c_str()); }

Rect GTK4Window::getImageRect() const {
    if (!window_) return Rect();
    return Rect(0, 0, gtk_widget_get_width(image_widget_), gtk_widget_get_height(image_widget_)); 
}

std::shared_ptr<UITrackbar> GTK4Window::createTrackbar(const std::string&, int, TrackbarCallback, void*) {
    return std::shared_ptr<UITrackbar>();
}
std::shared_ptr<UITrackbar> GTK4Window::findTrackbar(const std::string&) {
    return std::shared_ptr<UITrackbar>();
}

void GTK4Window::onKeyPressed(guint keyval, guint, GdkModifierType) {
    int code = keyval; 
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_key = code;
}

// ============================================================================
// Backend Definition
// ============================================================================

class GTK4BackendUI : public UIBackend
{
public:
    virtual ~GTK4BackendUI() CV_OVERRIDE { destroyAllWindows(); }
    virtual const std::string getName() const CV_OVERRIDE { return "GTK4"; }
    
    virtual std::shared_ptr<UIWindow> createWindow(const std::string& winname, int flags) CV_OVERRIDE;
    virtual void destroyAllWindows() CV_OVERRIDE;
    virtual int waitKeyEx(int delay) CV_OVERRIDE;
    virtual int pollKey() CV_OVERRIDE { return waitKeyEx(1); }

private:
    std::map<std::string, std::shared_ptr<GTK4Window>> windows_;
};

std::shared_ptr<UIWindow> GTK4BackendUI::createWindow(const std::string& winname, int flags) {
    auto it = windows_.find(winname);
    if (it != windows_.end()) return it->second;
    
    auto window = std::make_shared<GTK4Window>(winname, flags);
    windows_[winname] = window;
    return window;
}

void GTK4BackendUI::destroyAllWindows() {
    for (auto& pair : windows_) pair.second->destroy();
    windows_.clear();
}

int GTK4BackendUI::waitKeyEx(int delay) {
    { std::lock_guard<std::mutex> lock(g_mutex); g_last_key = -1; }
    
    auto start = std::chrono::steady_clock::now();
    while (true) {
        while (g_main_context_pending(NULL)) g_main_context_iteration(NULL, FALSE);
        
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_last_key != -1) { int k = g_last_key; g_last_key = -1; return k; }
        }
        
        if (delay > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            if (elapsed >= delay) return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::shared_ptr<UIBackend> createUIBackendGTK4() {
    return std::make_shared<GTK4BackendUI>();
}

} // namespace highgui_backend
} // namespace cv
// Legacy C API Compatibility Layer
// window.cpp expects these when HAVE_GTK4 is defined

CV_IMPL int cvNamedWindow(const char* name, int flags)
{
    auto backend = cv::highgui_backend::getCurrentUIBackend();
    if (backend) {
        backend->createWindow(name, flags);
        return 1;
    }
    return 0;
}

CV_IMPL void cvDestroyWindow(const char* name)
{
    auto backend = cv::highgui_backend::getCurrentUIBackend();
    if (backend) {
        auto window = backend->createWindow(name, 0);
        if (window) window->destroy();
    }
}

CV_IMPL void cvDestroyAllWindows()
{
    auto backend = cv::highgui_backend::getCurrentUIBackend();
    if (backend) backend->destroyAllWindows();
}

CV_IMPL void cvShowImage(const char* name, const CvArr* arr)
{
    auto backend = cv::highgui_backend::getCurrentUIBackend();
    if (backend) {
        auto window = backend->createWindow(name, cv::WINDOW_AUTOSIZE);
        if (window) window->imshow(cv::cvarrToMat(arr));
    }
}

CV_IMPL void cvResizeWindow(const char* name, int width, int height)
{
    auto backend = cv::highgui_backend::getCurrentUIBackend();
    if (backend) {
        auto window = backend->createWindow(name, 0);
        if (window) window->resize(width, height);
    }
}

CV_IMPL void cvMoveWindow(const char* name, int x, int y)
{
    auto backend = cv::highgui_backend::getCurrentUIBackend();
    if (backend) {
        auto window = backend->createWindow(name, 0);
        if (window) window->move(x, y);
    }
}

CV_IMPL void cvSetMouseCallback(const char* name, CvMouseCallback callback, void* param)
{
    auto backend = cv::highgui_backend::getCurrentUIBackend();
    if (backend) {
        auto window = backend->createWindow(name, 0);
        if (window) window->setMouseCallback((cv::MouseCallback)callback, param);
    }
}

CV_IMPL int cvCreateTrackbar2(const char* name, const char* window, int* value, int count, CvTrackbarCallback2 callback, void* userdata)
{
    return 0; // TODO Phase 2
}

CV_IMPL int cvGetTrackbarPos(const char* name, const char* window)
{
    return 0; // TODO Phase 2
}

CV_IMPL void cvSetTrackbarPos(const char* name, const char* window, int pos)
{
    // TODO Phase 2
}

CV_IMPL void cvSetTrackbarMax(const char* name, const char* window, int maxval)
{
    // TODO Phase 2
}

CV_IMPL void cvSetTrackbarMin(const char* name, const char* window, int minval)
{
    // TODO Phase 2
}

CV_IMPL int cvStartWindowThread()
{
    return 0; // Not needed in GTK4
}
#endif // HAVE_GTK4