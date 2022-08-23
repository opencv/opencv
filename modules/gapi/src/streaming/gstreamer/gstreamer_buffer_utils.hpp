// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMER_BUFFER_UTILS_HPP
#define OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMER_BUFFER_UTILS_HPP

#ifdef HAVE_GSTREAMER
#include <gst/gstbuffer.h>
#include <gst/video/video-frame.h>

namespace cv {
namespace gapi {
namespace wip {
namespace gstreamer_utils {

void mapBufferToFrame(GstBuffer& buffer, GstVideoInfo& info, GstVideoFrame& frame,
                      GstMapFlags map_flags);

} // namespace gstreamer_utils
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_GSTREAMER
#endif // OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMER_BUFFER_UTILS_HPP
