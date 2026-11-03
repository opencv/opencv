// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_VLM_PREPROCESS_COMMON_HPP
#define OPENCV_VLM_PREPROCESS_COMMON_HPP

#include "opencv2/core.hpp"

namespace cv { namespace vlm {

//! Converts BGR to three planar float channels, (pixel * rescale - mean) / std.
//! @p dst may wrap the destination tensor's memory, in which case it is written in place.
void toNormalizedPlanes(const Mat& imageBgr, float rescale, const Vec3f& mean, const Vec3f& std_,
                        Mat dst[3]);

}} // namespace cv::vlm

#endif // OPENCV_VLM_PREPROCESS_COMMON_HPP
