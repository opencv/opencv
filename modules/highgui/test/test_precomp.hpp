#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#endif

#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#ifdef HAVE_CVCONFIG_H
# include "cvconfig.h"
#endif

#include "opencv2/ts/ts.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>

#if defined(HAVE_VIDEOINPUT)   || \
    defined(HAVE_TYZX)         || \
    defined(HAVE_VFW)          || \
    defined(HAVE_LIBV4L)       || \
    (defined(HAVE_CAMV4L) && defined(HAVE_CAMV4L2)) || \
    defined(HAVE_GSTREAMER)    || \
    defined(HAVE_DC1394_2)     || \
    defined(HAVE_DC1394)       || \
    defined(HAVE_CMU1394)      || \
    defined(HAVE_MIL)          || \
    defined(HAVE_QUICKTIME)    || \
    defined(HAVE_UNICAP)       || \
    defined(HAVE_PVAPI)        || \
    defined(HAVE_OPENNI)       || \
    defined(HAVE_XIMEA)        || \
    defined(HAVE_AVFOUNDATION) || \
    (0)
    //defined(HAVE_ANDROID_NATIVE_CAMERA) ||   - enable after #1193
#  define BUILD_WITH_CAMERA_SUPPORT 1
#else
#  define BUILD_WITH_CAMERA_SUPPORT 0
#endif

#if defined(HAVE_XINE)         || \
    defined(HAVE_GSTREAMER)    || \
    defined(HAVE_QUICKTIME)    || \
    defined(HAVE_AVFOUNDATION) || \
    /*defined(HAVE_OPENNI)     || too specialized */ \
    defined(HAVE_FFMPEG)       || \
    defined(WIN32) /* assume that we have ffmpeg */

#  define BUILD_WITH_VIDEO_INPUT_SUPPORT 1
#else
#  define BUILD_WITH_VIDEO_INPUT_SUPPORT 0
#endif

#if /*defined(HAVE_XINE)       || */\
    defined(HAVE_GSTREAMER)    || \
    defined(HAVE_QUICKTIME)    || \
    defined(HAVE_AVFOUNDATION) || \
    defined(HAVE_FFMPEG)       || \
    defined(WIN32) /* assume that we have ffmpeg */
#  define BUILD_WITH_VIDEO_OUTPUT_SUPPORT 1
#else
#  define BUILD_WITH_VIDEO_OUTPUT_SUPPORT 0
#endif

namespace cvtest
{

string fourccToString(int fourcc);

struct VideoFormat
{
    VideoFormat() { fourcc = -1; }
    VideoFormat(const string& _ext, int _fourcc) : ext(_ext), fourcc(_fourcc) {}
    bool empty() const { return ext.empty(); }

    string ext;
    int fourcc;
};

extern const VideoFormat g_specific_fmt_list[];

}

#endif
