// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

//
// Not a standalone header, part of va_intel.cpp
//

#include "opencv2/core/utils/plugin_loader.private.hpp"  // DynamicLib

namespace cv { namespace detail {

typedef VAStatus (*FN_vaDeriveImage)(VADisplay dpy, VASurfaceID surface, VAImage *image);
typedef VAStatus (*FN_vaDestroyImage)(VADisplay dpy, VAImageID image);
typedef VAStatus (*FN_vaMapBuffer)(VADisplay dpy, VABufferID buf_id, void **pbuf);
typedef VAStatus (*FN_vaSyncSurface)(VADisplay dpy, VASurfaceID render_target);
typedef VAStatus (*FN_vaUnmapBuffer)(VADisplay dpy, VABufferID buf_id);

static FN_vaDeriveImage fn_vaDeriveImage = NULL;
static FN_vaDestroyImage fn_vaDestroyImage = NULL;
static FN_vaMapBuffer fn_vaMapBuffer = NULL;
static FN_vaSyncSurface fn_vaSyncSurface = NULL;
static FN_vaUnmapBuffer fn_vaUnmapBuffer = NULL;

#define vaDeriveImage fn_vaDeriveImage
#define vaDestroyImage fn_vaDestroyImage
#define vaMapBuffer fn_vaMapBuffer
#define vaSyncSurface fn_vaSyncSurface
#define vaUnmapBuffer fn_vaUnmapBuffer


static std::shared_ptr<cv::plugin::impl::DynamicLib> loadLibVA()
{
    std::shared_ptr<cv::plugin::impl::DynamicLib> lib;
    const char* envPath = getenv("OPENCV_LIBVA_RUNTIME");
    if (envPath)
    {
        lib = std::make_shared<cv::plugin::impl::DynamicLib>(envPath);
        return lib;
    }
    static const char* const candidates[] = {
        "libva.so",
        "libva.so.2",
        "libva.so.1",
    };
    for (int i = 0; i < 3; ++i)
    {
        lib = std::make_shared<cv::plugin::impl::DynamicLib>(candidates[i]);
        if (lib->isLoaded())
            break;
    }
    return lib;
}
static void init_libva()
{
    static bool initialized = false;
    static auto library = loadLibVA();
    if (!initialized)
    {
        if (!library || !library->isLoaded())
        {
            library.reset();
            CV_Error(cv::Error::StsBadFunc, "OpenCV can't load VA library (libva)");
        }
        auto& lib = *library.get();
#define VA_LOAD_SYMBOL(name) fn_ ## name = reinterpret_cast<FN_ ## name>(lib.getSymbol(#name)); \
        if (!fn_ ## name) \
        { \
            library.reset(); \
            initialized = true; \
            CV_Error_(cv::Error::StsBadFunc, ("OpenCV can't load VA library (libva), missing symbol: %s", #name)); \
        }

        VA_LOAD_SYMBOL(vaDeriveImage);
        VA_LOAD_SYMBOL(vaDestroyImage);
        VA_LOAD_SYMBOL(vaMapBuffer);
        VA_LOAD_SYMBOL(vaSyncSurface);
        VA_LOAD_SYMBOL(vaUnmapBuffer);
        initialized = true;
    }
    if (!library)
        CV_Error(cv::Error::StsBadFunc, "OpenCV can't load/initialize VA library (libva)");
}

}}  // namespace
