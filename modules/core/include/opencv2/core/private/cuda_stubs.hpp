// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_PRIVATE_CUDA_STUBS_HPP
#define OPENCV_CORE_PRIVATE_CUDA_STUBS_HPP

#ifndef __OPENCV_BUILD
#  error this is a private header which should not be used from outside of the OpenCV library
#endif

#include "opencv2/core/cvdef.h"
#include "opencv2/core/base.hpp"

#ifndef HAVE_CUDA
static inline CV_NORETURN void throw_no_cuda() { CV_Error(cv::Error::GpuNotSupported, "The library is compiled without CUDA support"); }
#else
static inline CV_NORETURN void throw_no_cuda() { CV_Error(cv::Error::StsNotImplemented, "The called functionality is disabled for current build or platform"); }
#endif

#endif // OPENCV_CORE_PRIVATE_CUDA_STUBS_HPP
