// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "gstreamer_buffer_utils.hpp"
#include "gstreamerptr.hpp"
#include <opencv2/gapi/own/assert.hpp>

#ifdef HAVE_GSTREAMER
namespace cv {
namespace gapi {
namespace wip {
namespace gstreamer_utils {

void mapBufferToFrame(GstBuffer& buffer, GstVideoInfo& info, GstVideoFrame& frame,
                      GstMapFlags mapFlags) {
    bool mapped = gst_video_frame_map(&frame, &info, &buffer, mapFlags);
    GAPI_Assert(mapped && "Failed to map GStreamer buffer to system memory as video-frame!");
}

} // namespace gstreamer_utils
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_GSTREAMER
