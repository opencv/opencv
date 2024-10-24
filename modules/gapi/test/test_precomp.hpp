// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


// FIXME: OpenCV header

#ifndef __OPENCV_GAPI_TEST_PRECOMP_HPP__
#define __OPENCV_GAPI_TEST_PRECOMP_HPP__

#include <cstdint>
#include <thread>
#include <vector>

#include <opencv2/ts.hpp>

#include <opencv2/core/utils/configuration.private.hpp>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/video.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/gpu/ggpukernel.hpp>
#include <opencv2/gapi/gpu/imgproc.hpp>
#include <opencv2/gapi/gpu/core.hpp>
#include <opencv2/gapi/gcompoundkernel.hpp>
#include <opencv2/gapi/operators.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/infer.hpp>

namespace cv {
static inline void countNonZero_is_forbidden_in_tests_use_norm_instead() {}
}
#define countNonZero() countNonZero_is_forbidden_in_tests_use_norm_instead()

#undef RAND_MAX
#define RAND_MAX RAND_MAX_is_banned_in_tests__use_cv_theRNG_instead

#endif // __OPENCV_GAPI_TEST_PRECOMP_HPP__
