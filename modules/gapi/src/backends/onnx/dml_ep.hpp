// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#ifndef OPENCV_GAPI_DML_EP_HPP
#define OPENCV_GAPI_DML_EP_HPP

#include "opencv2/gapi/infer/onnx.hpp"
#ifdef HAVE_ONNX

#include <onnxruntime_cxx_api.h>

namespace cv {
namespace gimpl {
namespace onnx {
void addDMLExecutionProvider(Ort::SessionOptions *session_options,
                             const cv::gapi::onnx::ep::DirectML &dml_ep);
}}}

#endif  // HAVE_ONNX
#endif  // OPENCV_GAPI_DML_EP_HPP
