#include <gtk/gtk.h>
#include <iostream>
#include <chrono>

// Simulate a cv::Mat-like structure
struct FakeImage {
    unsigned char *data;
    int width;
    int height;
    int channels;
    
    FakeImage(int w, int h, int ch) : width(w), height(h), channels(ch) {
        // Use 4 channels to align with Cairo RGB24 (XRGB) requirements
        data = new unsigned char[w * h * ch];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = (y * w + x) * ch;
                data[idx + 0] = (x * 255) / w;      // B (Cairo expects BGRX)
                data[idx + 1] = (y * 255) / h;      // G
                data[idx + 2] = 128;                 // R
                data[idx + 3] = 255;                 // X (Padding)
            }
        }
    }
    
    ~FakeImage() {
        delete[] data;
    }
};

static void draw_cairo_method(GtkDrawingArea *area, cairo_t *cr, 
                              int width, int height, gpointer user_data)
{
    FakeImage *img = (FakeImage*)user_data;
    if (!img) return;

    auto start = std::chrono::high_resolution_clock::now();
    
    // Cairo RGB24 is 4 bytes per pixel
    int stride = img->width * 4;
    cairo_surface_t *surface = cairo_image_surface_create_for_data(
        img->data, CAIRO_FORMAT_RGB24, img->width, img->height, stride);
    
    double scale_x = (double)width / img->width;
    double scale_y = (double)height / img->height;
    cairo_scale(cr, scale_x, scale_y);
    
    cairo_set_source_surface(cr, surface, 0, 0);
    cairo_paint(cr);
    
    // FORCE CAIRO TO FINISH SO WE CAN ACTUALLY MEASURE IT
    cairo_surface_flush(surface); 
    
    cairo_surface_destroy(surface);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    static int frame_count = 0;
    if (++frame_count % 60 == 0) {
        std::cout << "Cairo: " << duration.count() << " µs/frame (~" 
                  << (1000000.0 / duration.count()) << " FPS max)" << std::endl;
    }
}

static void draw_texture_method(GtkWidget *widget, GtkSnapshot *snapshot, gpointer user_data)
{
    FakeImage *img = (FakeImage*)user_data;
    if (!img) return;

    int width = gtk_widget_get_width(widget);
    int height = gtk_widget_get_height(widget);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    gsize data_size = img->width * img->height * 4;
    guint8 *data_copy = (guint8*)g_memdup2(img->data, data_size);
    GBytes *bytes = g_bytes_new_take(data_copy, data_size);
    
    GdkTexture *texture = gdk_memory_texture_new(
        img->width, img->height, GDK_MEMORY_B8G8R8X8, bytes, img->width * 4);
    
    graphene_rect_t bounds = GRAPHENE_RECT_INIT(0.0f, 0.0f, (float)width, (float)height);
    gtk_snapshot_append_texture(snapshot, texture, &bounds);
    
    g_object_unref(texture);
    g_bytes_unref(bytes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    static int frame_count = 0;
    if (++frame_count % 60 == 0) {
        std::cout << "GdkTexture: " << duration.count() << " µs/frame (~" 
                  << (1000000.0 / duration.count()) << " FPS max)" << std::endl;
    }
}

// ... (TextureWidget boilerplate remains the same as your file) ...

#define TYPE_TEXTURE_WIDGET (texture_widget_get_type())
G_DECLARE_FINAL_TYPE(TextureWidget, texture_widget, TEXTURE, WIDGET, GtkWidget)
struct _TextureWidget { GtkWidget parent; FakeImage *image; };
G_DEFINE_TYPE(TextureWidget, texture_widget, GTK_TYPE_WIDGET)
static void texture_widget_snapshot(GtkWidget *widget, GtkSnapshot *snapshot) {
    TextureWidget *self = TEXTURE_WIDGET(widget);
    if (self->image) draw_texture_method(widget, snapshot, self->image);
}
static void texture_widget_class_init(TextureWidgetClass *klass) { GTK_WIDGET_CLASS(klass)->snapshot = texture_widget_snapshot; }
static void texture_widget_init(TextureWidget *self) {
    gtk_widget_set_hexpand(GTK_WIDGET(self), TRUE);
    gtk_widget_set_vexpand(GTK_WIDGET(self), TRUE);
}
static GtkWidget* texture_widget_new(FakeImage *img) {
    TextureWidget *widget = (TextureWidget*)g_object_new(TYPE_TEXTURE_WIDGET, NULL);
    widget->image = img;
    return GTK_WIDGET(widget);
}

static void activate(GtkApplication *app, gpointer user_data)
{
    // Important: Use 4 channels for 32-bit alignment
    FakeImage *img = new FakeImage(1920, 1080, 4);
    
    GtkWidget *window = gtk_window_new();
    gtk_window_set_application(GTK_WINDOW(window), app);
    gtk_window_set_default_size(GTK_WINDOW(window), 1280, 720);
    
    GtkWidget *notebook = gtk_notebook_new();
    gtk_window_set_child(GTK_WINDOW(window), notebook);
    
    GtkWidget *cairo_area = gtk_drawing_area_new();
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(cairo_area), draw_cairo_method, img, NULL);
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), cairo_area, gtk_label_new("Cairo"));
    
    GtkWidget *texture_widget = texture_widget_new(img);
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), texture_widget, gtk_label_new("GdkTexture"));
    
    g_timeout_add(16, [](gpointer data) -> gboolean {
        if (GTK_IS_WIDGET(data)) {
            gtk_widget_queue_draw(GTK_WIDGET(data));
            return G_SOURCE_CONTINUE;
        }
        return G_SOURCE_REMOVE;
    }, notebook);
    
    gtk_window_present(GTK_WINDOW(window));
}

int main(int argc, char **argv) {
    GtkApplication *app = gtk_application_new("org.opencv.test", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    return g_application_run(G_APPLICATION(app), argc, argv);
}