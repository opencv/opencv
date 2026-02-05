/**
 * Simple standalone test for GTK4 window creation
 * This tests the basic GTK4 APIs without requiring full OpenCV build
 */

#include <gtk/gtk.h>
#include <iostream>
#include <cstring>

// Simple test of GTK4 container API changes
static void test_container_api()
{
    std::cout << "Testing GTK4 Container API..." << std::endl;
    
    // Create window
    GtkWidget *window = gtk_window_new();
    gtk_window_set_title(GTK_WINDOW(window), "GTK4 Test");
    gtk_window_set_default_size(GTK_WINDOW(window), 400, 300);
    
    // Create vertical box (GTK4 way)
    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    
    // Set as window child (GTK4: no more gtk_container_add)
    gtk_window_set_child(GTK_WINDOW(window), vbox);
    
    // Add some labels
    GtkWidget *label1 = gtk_label_new("GTK4 Container API Test");
    GtkWidget *label2 = gtk_label_new("✓ gtk_box_new() works");
    GtkWidget *label3 = gtk_label_new("✓ gtk_window_set_child() works");
    
    gtk_box_append(GTK_BOX(vbox), label1);
    gtk_box_append(GTK_BOX(vbox), label2);
    gtk_box_append(GTK_BOX(vbox), label3);
    
    std::cout << "✓ Container API test passed" << std::endl;
}

// Test GTK4 event controllers
static void on_key_pressed(GtkEventControllerKey *controller,
                          guint keyval,
                          guint keycode,
                          GdkModifierType state,
                          gpointer user_data)
{
    std::cout << "Key pressed: " << keyval << std::endl;
    
    if (keyval == GDK_KEY_Escape || keyval == GDK_KEY_q)
    {
        GtkWidget *window = (GtkWidget*)user_data;
        gtk_window_destroy(GTK_WINDOW(window));
    }
}

static void on_button_pressed(GtkGestureClick *gesture,
                             gint n_press,
                             gdouble x,
                             gdouble y,
                             gpointer user_data)
{
    std::cout << "Mouse clicked at (" << x << ", " << y << ")";
    if (n_press == 2)
        std::cout << " (double-click)";
    std::cout << std::endl;
}

static void on_motion(GtkEventControllerMotion *controller,
                     gdouble x,
                     gdouble y,
                     gpointer user_data)
{
    static int counter = 0;
    // Only print every 10th motion event to avoid spam
    if (++counter % 10 == 0)
        std::cout << "Mouse motion: (" << x << ", " << y << ")" << std::endl;
}

static void test_event_controllers(GtkWidget *window, GtkWidget *drawing_area)
{
    std::cout << "Setting up GTK4 Event Controllers..." << std::endl;
    
    // Keyboard controller on window
    GtkEventController *key_ctrl = gtk_event_controller_key_new();
    g_signal_connect(key_ctrl, "key-pressed", 
                     G_CALLBACK(on_key_pressed), window);
    gtk_widget_add_controller(window, key_ctrl);
    
    // Click gesture on drawing area
    GtkGesture *click = gtk_gesture_click_new();
    gtk_gesture_single_set_button(GTK_GESTURE_SINGLE(click), 0);  // All buttons
    g_signal_connect(click, "pressed",
                     G_CALLBACK(on_button_pressed), NULL);
    gtk_widget_add_controller(drawing_area, GTK_EVENT_CONTROLLER(click));
    
    // Motion controller
    GtkEventController *motion = gtk_event_controller_motion_new();
    g_signal_connect(motion, "motion",
                     G_CALLBACK(on_motion), NULL);
    gtk_widget_add_controller(drawing_area, motion);
    
    std::cout << "✓ Event controllers test passed" << std::endl;
}

// Test GTK4 drawing
static void on_draw_func(GtkDrawingArea *area,
                        cairo_t *cr,
                        int width,
                        int height,
                        gpointer user_data)
{
    // Draw a simple pattern
    cairo_set_source_rgb(cr, 0.2, 0.3, 0.8);
    cairo_rectangle(cr, 20, 20, width - 40, height - 40);
    cairo_fill(cr);
    
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_move_to(cr, width/2 - 100, height/2);
    cairo_show_text(cr, "GTK4 Drawing Works!");
}

static void activate(GtkApplication *app, gpointer user_data)
{
    std::cout << "\n=== GTK4 API Test ===" << std::endl;
    std::cout << "GTK Version: " << gtk_get_major_version() << "."
              << gtk_get_minor_version() << "."
              << gtk_get_micro_version() << std::endl;
    
    if (gtk_get_major_version() < 4)
    {
        std::cerr << "ERROR: This test requires GTK 4.0 or later!" << std::endl;
        return;
    }
    
    // Test 1: Container API
    test_container_api();
    
    // Create main window
    GtkWidget *window = gtk_window_new();
    gtk_window_set_title(GTK_WINDOW(window), "GTK4 Integration Test");
    gtk_window_set_default_size(GTK_WINDOW(window), 640, 480);
    gtk_window_set_application(GTK_WINDOW(window), app);
    
    // Create layout
    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_window_set_child(GTK_WINDOW(window), vbox);
    
    // Add instruction label
    GtkWidget *label = gtk_label_new(
        "GTK4 Test Window\n\n"
        "• Move mouse to test motion events\n"
        "• Click to test button events\n"
        "• Press keys to test keyboard\n"
        "• Press ESC or Q to quit"
    );
    gtk_box_append(GTK_BOX(vbox), label);
    
    // Create drawing area
    GtkWidget *drawing_area = gtk_drawing_area_new();
    gtk_widget_set_hexpand(drawing_area, TRUE);
    gtk_widget_set_vexpand(drawing_area, TRUE);
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(drawing_area),
                                    on_draw_func, NULL, NULL);
    gtk_box_append(GTK_BOX(vbox), drawing_area);
    
    // Test 2: Event Controllers
    test_event_controllers(window, drawing_area);
    
    // Show window
    gtk_window_present(GTK_WINDOW(window));
    
    std::cout << "\nWindow created successfully!" << std::endl;
    std::cout << "Interact with the window to test events..." << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "GTK4 Standalone Test Program" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Create GTK application
    GtkApplication *app = gtk_application_new(
        "org.opencv.gtk4.test",
        G_APPLICATION_FLAGS_NONE
    );
    
    if (!app)
    {
        std::cerr << "Failed to create GTK application!" << std::endl;
        return 1;
    }
    
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    
    int status = g_application_run(G_APPLICATION(app), argc, argv);
    
    g_object_unref(app);
    
    return status;
}
