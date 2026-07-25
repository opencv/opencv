// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_SLAM_POSE_OPTIMIZER_HPP
#define OPENCV_SLAM_POSE_OPTIMIZER_HPP

#include "../odometry/frame.hpp"

namespace cv {
namespace slam {

// Classify map point associations as inlier/outlier by reprojection distance.
// Returns inlier count. frame.poseCw is NOT modified.
int poseInlierCheck(Frame& frame, const Mat& K, double reprojThresh);

}} // namespace cv::slam

#endif // OPENCV_SLAM_POSE_OPTIMIZER_HPP
