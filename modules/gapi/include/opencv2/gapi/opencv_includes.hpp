// This file is part of OpenCV project.

// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OPENCV_INCLUDES_HPP
#define OPENCV_GAPI_OPENCV_INCLUDES_HPP

#if !defined(GAPI_STANDALONE)
#  include <opencv2/core/mat.hpp>
#  include <opencv2/core/cvdef.h>
#  include <opencv2/core/types.hpp>
#  include <opencv2/core/base.hpp>
#else   // Without OpenCV
#  include <opencv2/gapi/own/cvdefs.hpp>
#  include <opencv2/gapi/own/types.hpp>  // cv::gapi::own::Rect/Size/Point
#  include <opencv2/gapi/own/scalar.hpp> // cv::gapi::own::Scalar
#  include <opencv2/gapi/own/mat.hpp>
// replacement of cv's structures:
namespace cv {
    using Rect   = gapi::own::Rect;
    using Size   = gapi::own::Size;
    using Point  = gapi::own::Point;
    using Scalar = gapi::own::Scalar;
    using Mat    = gapi::own::Mat;
}  // namespace cv
#endif // !defined(GAPI_STANDALONE)

#endif // OPENCV_GAPI_OPENCV_INCLUDES_HPP
