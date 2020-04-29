// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_INFER_IE_UTIL_HPP
#define OPENCV_GAPI_INFER_IE_UTIL_HPP

#ifdef HAVE_INF_ENGINE

// NOTE: This file is not included by default in infer/ie.hpp
// and won't be. infer/ie.hpp doesn't depend on IE headers itself.
// This file does -- so needs to be included separately by those who care.

#include "inference_engine.hpp"

#include <opencv2/core/cvdef.h>     // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

namespace cv {
namespace gapi {
namespace ie {
namespace util {

// NB: These functions are EXPORTed to make them accessible by the
// test suite only.
GAPI_EXPORTS std::vector<int> to_ocv(const InferenceEngine::SizeVector &dims);
GAPI_EXPORTS cv::Mat to_ocv(InferenceEngine::Blob::Ptr blob);
GAPI_EXPORTS InferenceEngine::Blob::Ptr to_ie(cv::Mat &blob);

}}}}

#endif //HAVE_INF_ENGINE

#endif // OPENCV_GAPI_INFER_IE_UTIL_HPP
