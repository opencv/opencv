// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#ifndef OPENCV_GAPI_INFER_OV_UTIL_HPP
#define OPENCV_GAPI_INFER_OV_UTIL_HPP

#if defined HAVE_INF_ENGINE && INF_ENGINE_RELEASE >= 2022010000

// NOTE: This file is not included by default in infer/ov.hpp
// and won't be. infer/ov.hpp doesn't depend on OV headers itself.
// This file does -- so needs to be included separately by those who care.

#include <openvino/openvino.hpp>

#include <opencv2/core/cvdef.h>     // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

namespace cv {
namespace gapi {
namespace ov {
namespace util {
// NB: These functions are EXPORTed to make them accessible by the
// test suite only.
GAPI_EXPORTS std::vector<int> to_ocv(const ::ov::Shape &shape);
GAPI_EXPORTS int to_ocv(const ::ov::element::Type &type);
GAPI_EXPORTS void to_ov(const cv::Mat &mat, ::ov::Tensor &tensor);
GAPI_EXPORTS void to_ocv(const ::ov::Tensor &tensor, cv::Mat &mat);
} // namespace util
namespace wrap {
GAPI_EXPORTS ::ov::Core getCore();
} // namespace wrap
} // namespace ov
} // namespace gapi
} // namespace cv

#endif // HAVE_INF_ENGINE && INF_ENGINE_RELEASE >= 2022010000

#endif // OPENCV_GAPI_INFER_OV_UTIL_HPP
