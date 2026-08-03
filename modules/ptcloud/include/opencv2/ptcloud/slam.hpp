// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_PTCLOUD_SLAM_HPP
#define OPENCV_PTCLOUD_SLAM_HPP

/**
@defgroup slam SLAM and Visual Odometry

Monocular SLAM pipeline. Entry point is @ref cv::slam::VisualOdometry.

Bootstraps an initial map from two-view geometry, then tracks subsequent frames with PnP,
growing the map at keyframe promotions. Each keyframe additionally triggers pose-graph
refinement — pose-only and local bundle adjustment, plus appearance-based loop detection
and Sim(3) loop closure with essential-graph optimisation. A final global bundle
adjustment runs on @ref cv::slam::VisualOdometry::finalize.

The graph-optimisation stages require g2o (build with `-DWITH_G2O=ON`); without it the
module still builds and tracks, and every bundle-adjustment and loop-closure stage becomes
a no-op.

The pipeline is purely in-memory. Reading images from disk and exporting the resulting
trajectory and map is left to the caller — see `samples/slam/visual_odometry.cpp`.
*/

#include "opencv2/ptcloud/slam/types.hpp"
#include "opencv2/ptcloud/slam/map.hpp"
#include "opencv2/ptcloud/slam/odometry_params.hpp"
#include "opencv2/ptcloud/slam/visual_odometry.hpp"

#endif // OPENCV_PTCLOUD_SLAM_HPP
