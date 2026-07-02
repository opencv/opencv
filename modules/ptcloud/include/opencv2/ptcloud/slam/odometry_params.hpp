// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_SLAM_ODOMETRY_PARAMS_HPP
#define OPENCV_SLAM_ODOMETRY_PARAMS_HPP

#include "opencv2/core.hpp"

namespace cv {
namespace slam {

//! @addtogroup slam
//! @{

struct CV_EXPORTS_W_SIMPLE OdometryParams
{
    CV_WRAP OdometryParams() {}

    // Bootstrap
    CV_PROP_RW int minInitInliers = 40;
    CV_PROP_RW double minInitParallaxDeg = 1.5;
    CV_PROP_RW int minInitPoints = 50;
    CV_PROP_RW double hfRatioThresh = 0.45;
    CV_PROP_RW double minGrowthParallaxDeg = 0.1;
    CV_PROP_RW double essentialRansacThresh = 1.0;
    CV_PROP_RW double essentialRansacConfidence = 0.999;

    // Tracking (PnP)
    CV_PROP_RW double pnpReprojThresh = 4.0;
    CV_PROP_RW int pnpMinInliers = 6;
    CV_PROP_RW int pnpRansacIters = 500;
    CV_PROP_RW double pnpConfidence = 0.99;

    // Motion model
    CV_PROP_RW double motionModelRadius = 15.0;
    CV_PROP_RW double motionModelRadiusWide = 30.0;
    CV_PROP_RW int motionModelMinMatches = 20;
    CV_PROP_RW double descProjThresh = 1.0;

    // Optical flow fallback
    CV_PROP_RW int opticalFlowMinInliers = 10;

    // Keyframe promotion
    CV_PROP_RW int kfMinFrames = 1;
    CV_PROP_RW int kfMaxFrames = 30;
    CV_PROP_RW double kfInlierRatio = 0.70;
    CV_PROP_RW int kfMinInliers = 40;
    CV_PROP_RW double kfRotThreshDeg = 5.0;
    CV_PROP_RW double kfTransThresh = 0.5; 

    // Local map refinement
    CV_PROP_RW int localMapTopK = 10;
    CV_PROP_RW int localMapNeighborK = 5;
    CV_PROP_RW double localMapRadius = 7.0;
};

//! @}

}} // namespace cv::slam

#endif // OPENCV_SLAM_ODOMETRY_PARAMS_HPP
