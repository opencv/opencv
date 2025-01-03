// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024 Intel Corporation

#include "precomp.hpp"

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/core/utility.hpp>

cv::use_threaded_executor::use_threaded_executor()
    : num_threads(cv::getNumThreads()) {
}

cv::use_threaded_executor::use_threaded_executor(const uint32_t nthreads)
    : num_threads(nthreads) {
}
