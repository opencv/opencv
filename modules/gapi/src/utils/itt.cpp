// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

// TODO: Consider using OpenCV's trace.hpp
#if defined(OPENCV_WITH_ITT)
#include <ittnotify.h>
#include <opencv2/gapi/own/exports.hpp>

namespace cv {
namespace gimpl {
    GAPI_EXPORTS __itt_domain* gapi_itt_domain = __itt_domain_create("GAPI Context");
} // namespace gimpl
}  // namespace cv
#endif // OPENCV_WITH_ITT
