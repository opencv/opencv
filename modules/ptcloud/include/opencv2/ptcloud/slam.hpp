// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_PTCLOUD_SLAM_HPP
#define OPENCV_PTCLOUD_SLAM_HPP

/**
@defgroup slam SLAM and Visual Odometry

Monocular visual odometry pipeline. Entry point is @ref cv::slam::VisualOdometry.
Bootstraps an initial map from two-view geometry, then tracks subsequent frames
with PnP, growing the map at keyframe promotions.
*/

#include "opencv2/ptcloud/slam/types.hpp"
#include "opencv2/ptcloud/slam/map.hpp"
#include "opencv2/ptcloud/slam/odometry_params.hpp"
#include "opencv2/ptcloud/slam/visual_odometry.hpp"

#endif // OPENCV_PTCLOUD_SLAM_HPP
