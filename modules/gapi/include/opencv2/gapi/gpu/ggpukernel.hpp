// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GGPUKERNEL_HPP
#define OPENCV_GAPI_GGPUKERNEL_HPP
/** @file
* @deprecated Use "opencv2/gapi/ocl/goclkernel.hpp" instead.
*/

#include "opencv2/gapi/ocl/goclkernel.hpp"
#define GAPI_GPU_KERNEL GAPI_OCL_KERNEL

namespace cv {
namespace gapi {
namespace core {
namespace gpu {
    using namespace ocl;
} // namespace gpu
} // namespace core
} // namespace gapi
} // namespace cv


#endif // OPENCV_GAPI_GGPUKERNEL_HPP
