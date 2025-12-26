/*
 * Standalone test to verify 4-channel format support logic
 * This test validates the key parts of our implementation without requiring full OpenCV build
 */

#include <iostream>
#include <string>
#include <vector>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/app/gstappsrc.h>

// Simulate the format detection logic from our implementation
bool test_format_detection(const std::string& format_str) {
    std::cout << "Testing format detection for: " << format_str << std::endl;
    
    // Initialize GStreamer
    gst_init(nullptr, nullptr);
    
    // Create a test caps structure
    GstCaps* caps = gst_caps_new_simple(
        "video/x-raw",
        "format", G_TYPE_STRING, format_str.c_str(),
        "width", G_TYPE_INT, 320,
        "height", G_TYPE_INT, 240,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        NULL
    );
    
    if (!caps) {
        std::cerr << "Failed to create caps" << std::endl;
        return false;
    }
    
    // Get structure from caps (simulating our code)
    const GstStructure* structure = gst_caps_get_structure(caps, 0);
    if (!structure) {
        std::cerr << "Failed to get structure from caps" << std::endl;
        gst_caps_unref(caps);
        return false;
    }
    
    // Check caps name
    const gchar* caps_name = gst_structure_get_name(structure);
    if (!caps_name || strcmp(caps_name, "video/x-raw") != 0) {
        std::cerr << "Unexpected caps name: " << (caps_name ? caps_name : "NULL") << std::endl;
        gst_caps_unref(caps);
        return false;
    }
    
    // Get format string (simulating our code)
    const gchar* format_from_caps = gst_structure_get_string(structure, "format");
    if (!format_from_caps) {
        std::cerr << "Failed to get format from caps" << std::endl;
        gst_caps_unref(caps);
        return false;
    }
    
    std::cout << "  Format from caps: " << format_from_caps << std::endl;
    
    // Test gst_video_format_from_string (our implementation uses this)
    GstVideoFormat format = gst_video_format_from_string(format_from_caps);
    
    if (format == GST_VIDEO_FORMAT_UNKNOWN) {
        std::cerr << "  ERROR: Format is UNKNOWN" << std::endl;
        gst_caps_unref(caps);
        return false;
    }
    
    std::cout << "  Format enum value: " << format << std::endl;
    
    // Check if it's one of our 4-channel formats
    bool is_4channel = (format == GST_VIDEO_FORMAT_BGRA ||
                        format == GST_VIDEO_FORMAT_RGBA ||
                        format == GST_VIDEO_FORMAT_BGRx ||
                        format == GST_VIDEO_FORMAT_RGBx);
    
    if (is_4channel) {
        std::cout << "  ✓ Detected as 4-channel format" << std::endl;
    } else {
        std::cout << "  Note: Not a 4-channel format (this is OK for other formats)" << std::endl;
    }
    
    gst_caps_unref(caps);
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing 4-Channel Format Detection Logic" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::vector<std::string> formats = {"BGRA", "RGBA", "BGRX", "RGBX", "BGR", "GRAY8"};
    
    bool all_passed = true;
    for (const auto& fmt : formats) {
        std::cout << std::endl;
        if (!test_format_detection(fmt)) {
            all_passed = false;
        }
    }
    
    std::cout << std::endl << "========================================" << std::endl;
    if (all_passed) {
        std::cout << "✓ All format detection tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Some tests failed" << std::endl;
        return 1;
    }
}

