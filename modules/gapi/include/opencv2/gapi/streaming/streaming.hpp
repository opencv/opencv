// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_STREAMING_API_HPP
#define OPENCV_GAPI_STREAMING_API_HPP

#include <opencv2/gapi/gkernel.hpp> // GKernelPackage
#include <opencv2/gapi/own/exports.hpp> // GAPI_EXPORTS

namespace cv {
namespace gapi {
namespace streaming {

GAPI_EXPORTS_W cv::gapi::GKernelPackage kernels();

} // namespace streaming
} // namespace gapi
} // namespace cv


#endif // OPENCV_GAPI_STREAMING_API_HPP
