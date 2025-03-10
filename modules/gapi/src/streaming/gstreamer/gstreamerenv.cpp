// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "gstreamerenv.hpp"
#include "gstreamerptr.hpp"

#ifdef HAVE_GSTREAMER
#include <gst/gst.h>
#endif // HAVE_GSTREAMER

namespace cv {
namespace gapi {
namespace wip {
namespace gst {

#ifdef HAVE_GSTREAMER

const GStreamerEnv& GStreamerEnv::init()
{
    static GStreamerEnv gInit;
    return gInit;
}

GStreamerEnv::GStreamerEnv()
{
    if (!gst_is_initialized())
    {
        GError* error = NULL;
        gst_init_check(NULL, NULL, &error);

        GStreamerPtr<GError> err(error);

        if (err)
        {
            cv::util::throw_error(
                std::runtime_error(std::string("GStreamer initializaton error! Details: ") +
                                   err->message));
        }
    }

    // FIXME: GStreamer libs which have same MAJOR and MINOR versions are API and ABI compatible.
    //        If GStreamer runtime MAJOR version differs from the version the application was
    //        compiled with, will it fail on the linkage stage? If so, the code below isn't needed.
    guint major, minor, micro, nano;
    gst_version(&major, &minor, &micro, &nano);
    if (GST_VERSION_MAJOR != major)
    {
        cv::util::throw_error(
            std::runtime_error(std::string("Incompatible GStreamer version: compiled with ") +
                               std::to_string(GST_VERSION_MAJOR) + '.' +
                               std::to_string(GST_VERSION_MINOR) + '.' +
                               std::to_string(GST_VERSION_MICRO) + '.' +
                               std::to_string(GST_VERSION_NANO) +
                               ", but runtime has " +
                               std::to_string(major) + '.' + std::to_string(minor) + '.' +
                               std::to_string(micro) + '.' + std::to_string(nano) + '.'));
    }
}

GStreamerEnv::~GStreamerEnv()
{
    gst_deinit();
}

#else // HAVE_GSTREAMER

const GStreamerEnv& GStreamerEnv::init()
{
    GAPI_Error("Built without GStreamer support!");
}

GStreamerEnv::GStreamerEnv()
{
    GAPI_Error("Built without GStreamer support!");
}

GStreamerEnv::~GStreamerEnv()
{
    // No need an assert here. The assert raise C4722 warning. Constructor have already got assert.
}

#endif // HAVE_GSTREAMER

} // namespace gst
} // namespace wip
} // namespace gapi
} // namespace cv
