// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 OpenCV & G-API team

#ifndef OPENCV_GAPI_GSTREAMING_TAG_HPP
#define OPENCV_GAPI_GSTREAMING_TAG_HPP

namespace cv {
namespace gapi {
namespace streaming {

/**
 * @brief Identifies a stream which output has been produced by the
 * GStreamingCompiled::pull() call.
 *
 * TBD
 */
struct tag {
    int id = -1;
    bool eos = true;
};

} // namespace streaming
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMING_TAG_HPP
