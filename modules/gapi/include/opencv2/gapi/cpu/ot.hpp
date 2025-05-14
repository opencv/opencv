// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_CPU_OT_API_HPP
#define OPENCV_GAPI_CPU_OT_API_HPP

#include <opencv2/core/cvdef.h>     // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

namespace cv {
namespace gapi {
/**
 * @brief This namespace contains G-API Operation Types for
 * VAS Object Tracking module functionality.
 */
namespace ot {
namespace cpu {
GAPI_EXPORTS_W GKernelPackage kernels();
} // namespace cpu
} // namespace ot
} // namespace gapi
} // namespace cv


#endif // OPENCV_GAPI_CPU_OT_API_HPP
