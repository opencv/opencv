#include "precomp.hpp"

#ifdef HAVE_GTK4

#include <gtk/gtk.h>
#include "opencv2/highgui.hpp"
#include "opencv2/core/types_c.h"

using namespace cv;

// --- 1. Create Window (Stub) ---
CV_IMPL int cvNamedWindow(const char* name, int flags) {
    // This is a stub to prove our build system works
    return 1;
}

// --- 2. Show Image (Stub) ---
CV_IMPL void cvShowImage(const char* name, const CvArr* arr) {
    // In the future, this will convert the image to GTK4 format
}

// --- 3. Wait Key (Stub) ---
CV_IMPL int cvWaitKey(int delay) {
    return -1;
}

// --- Required Empty Functions for Linker ---
CV_IMPL void cvDestroyWindow(const char* name) {}
CV_IMPL void cvDestroyAllWindows() {}
CV_IMPL void cvResizeWindow(const char* name, int width, int height) {}
CV_IMPL void cvMoveWindow(const char* name, int x, int y) {}
CV_IMPL int cvCreateTrackbar(const char* a, const char* b, int* c, int d, CvTrackbarCallback e) { return 0; }
CV_IMPL int cvGetTrackbarPos(const char* a, const char* b) { return 0; }
CV_IMPL void cvSetTrackbarPos(const char* a, const char* b, int c) {}
CV_IMPL void cvSetMouseCallback(const char* a, CvMouseCallback b, void* c) {}
CV_IMPL void* cvGetWindowHandle(const char* name) { return NULL; }
CV_IMPL const char* cvGetWindowName(void* window_handle) { return NULL; }

#endif // HAVE_GTK4
