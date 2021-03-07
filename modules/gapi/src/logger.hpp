// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef __OPENCV_GAPI_LOGGER_HPP__
#define __OPENCV_GAPI_LOGGER_HPP__

#if !defined(GAPI_STANDALONE)
#  include "opencv2/core/cvdef.h"
#  include "opencv2/core/utils/logger.hpp"
#  define GAPI_LOG_INFO(tag, ...)    CV_LOG_INFO(tag, __VA_ARGS__)
#  define GAPI_LOG_WARNING(tag, ...) CV_LOG_WARNING(tag, __VA_ARGS__)
#  define GAPI_LOG_DEBUG(tag, ...)    CV_LOG_DEBUG(tag, __VA_ARGS__)
#else
#  define GAPI_LOG_INFO(tag, ...)
#  define GAPI_LOG_WARNING(tag, ...)
#  define GAPI_LOG_DEBUG(tag, ...)
#endif //  !defined(GAPI_STANDALONE)


#endif // __OPENCV_GAPI_LOGGER_HPP__
