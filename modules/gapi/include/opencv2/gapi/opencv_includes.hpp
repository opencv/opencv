// This file is part of OpenCV project.

// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OPENCV_INCLUDES_HPP
#define OPENCV_GAPI_OPENCV_INCLUDES_HPP

#if !defined(GAPI_STANDALONE)
#  include <opencv2/core/mat.hpp>
#  include <opencv2/core/cvdef.h>
#  include <opencv2/core/types.hpp>
#  include <opencv2/core/base.hpp>
#else   // Without OpenCV
#  include <opencv2/gapi/own/cvdefs.hpp>
#endif // !defined(GAPI_STANDALONE)

#endif // OPENCV_GAPI_OPENCV_INCLUDES_HPP
