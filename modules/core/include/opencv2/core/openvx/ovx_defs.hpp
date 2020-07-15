// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

// OpenVX related definitions and declarations

#pragma once
#ifndef OPENCV_OVX_DEFS_HPP
#define OPENCV_OVX_DEFS_HPP

#include "cvconfig.h"

// utility macro for running OpenVX-based implementations
#ifdef HAVE_OPENVX

#define IVX_HIDE_INFO_WARNINGS
#define IVX_USE_OPENCV
#include "ivx.hpp"

namespace cv{
namespace ovx{
// Get common thread local OpenVX context
CV_EXPORTS_W ivx::Context& getOpenVXContext();

template <int kernel_id> inline bool skipSmallImages(int w, int h)     { return w*h < 3840 * 2160; }
}}

#define CV_OVX_RUN(condition, func, ...)          \
    if (cv::useOpenVX() && (condition) && func)   \
    {                                             \
        return __VA_ARGS__;                       \
    }

#else
    #define CV_OVX_RUN(condition, func, ...)
#endif // HAVE_OPENVX

// Throw an error in debug mode or try another implementation in release
#ifdef _DEBUG
#define VX_DbgThrow(s) CV_Error(cv::Error::StsInternal, (s))
#else
#define VX_DbgThrow(s) return false
#endif

#endif // OPENCV_OVX_DEFS_HPP
