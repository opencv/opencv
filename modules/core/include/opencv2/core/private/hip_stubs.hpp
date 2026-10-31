// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_PRIVATE_HIP_STUBS_HPP
#define OPENCV_CORE_PRIVATE_HIP_STUBS_HPP


#include "opencv2/core/cvdef.h"
#include "opencv2/core/base.hpp"

#ifndef HAVE_HIP
static inline CV_NORETURN void throw_no_hip() { CV_Error(cv::Error::GpuNotSupported, "The library is compiled without HIP support"); }
#else
static inline CV_NORETURN void throw_no_hip() { CV_Error(cv::Error::StsNotImplemented, "The called functionality is disabled for current build or platform"); }
#endif

#endif // OPENCV_CORE_PRIVATE_HIP_STUBS_HPP
