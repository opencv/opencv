// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

// OpenVX related definitions and declarations

#pragma once
#ifndef OPENCV_OVX_HPP
#define OPENCV_OVX_HPP

#include "cvdef.h"

namespace cv
{
/// Check if use of OpenVX is possible
CV_EXPORTS_W bool haveOpenVX();

/// Check if use of OpenVX is enabled
CV_EXPORTS_W bool useOpenVX();

/// Enable/disable use of OpenVX
CV_EXPORTS_W void setUseOpenVX(bool flag);
} // namespace cv

#endif // OPENCV_OVX_HPP
