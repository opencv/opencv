// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_PYTHON_BRIDGE_HPP
#define OPENCV_GAPI_PYTHON_BRIDGE_HPP

#include <opencv2/gapi/gkernel.hpp>

namespace cv {
namespace detail {

void allocateGraphOutputs(const cv::GTypesInfo &out_info,
                                cv::GRunArgs   &args,
                                cv::GRunArgsP  &outs);

} // namespace detail
} // namespace cv

#endif // OPENCV_GAPI_PYTHON_BRIDGE_HPP
