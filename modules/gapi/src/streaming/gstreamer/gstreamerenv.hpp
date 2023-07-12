// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERENV_HPP
#define OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERENV_HPP

namespace cv {
namespace gapi {
namespace wip {
namespace gst {

/*!
 * \brief The GStreamerEnv class
 * Initializes gstreamer once in the whole process
 *
 *
 * @note You need to build OpenCV with GStreamer support to use this class.
 */
class GStreamerEnv
{
public:
    static const GStreamerEnv& init();

private:
    GStreamerEnv();
    ~GStreamerEnv();
};

} // namespace gst
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERENV_HPP
