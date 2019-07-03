// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_CAP_HPP
#define OPENCV_GAPI_STREAMING_CAP_HPP

#include <string>

// FIXME: Think how it could be composed from the streaming API
// FIXME: Make it work only if videoio module is present
// FIXME: Think how to make it extensible from the outside
//  (support custom video sources?)

namespace cv {
namespace gapi {

// FIXME: namespace wip?
struct GVideoCapture
{
    // FIXME: extend to support camera
    std::string path;
};

} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_CAP_HPP
