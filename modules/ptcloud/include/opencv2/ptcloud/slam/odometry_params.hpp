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

/** @brief Tunable parameters for visual odometry: initialization, tracking, keyframe selection, and local-map refinement. */
struct CV_EXPORTS_W_SIMPLE OdometryParams
{
    CV_WRAP OdometryParams() {}

    // Bootstrap (two-view map initialization)
    CV_PROP_RW int minInitInliers = 40;             //!< Minimum match/inlier count at each bootstrap stage.
    CV_PROP_RW double minInitParallaxDeg = 1.5;     //!< Minimum parallax (deg) to trigger initialization.
    CV_PROP_RW int minInitPoints = 50;              //!< Minimum triangulated points to seed the map.
    CV_PROP_RW double hfRatioThresh = 0.45;         //!< Homography/fundamental score ratio above which homography is chosen.
    CV_PROP_RW double minGrowthParallaxDeg = 0.1;   //!< Minimum parallax (deg) to triangulate new points during map growth.
    CV_PROP_RW double essentialRansacThresh = 1.0;  //!< RANSAC reprojection threshold (px) for essential-matrix/homography estimation.
    CV_PROP_RW double essentialRansacConfidence = 0.999; //!< RANSAC confidence for essential-matrix estimation.

    // Tracking (PnP)
    CV_PROP_RW double pnpReprojThresh = 4.0;        //!< PnP RANSAC reprojection threshold (px).
    CV_PROP_RW int pnpMinInliers = 6;               //!< Minimum PnP inliers to accept a pose.
    CV_PROP_RW int pnpRansacIters = 500;            //!< Maximum PnP RANSAC iterations.
    CV_PROP_RW double pnpConfidence = 0.99;         //!< PnP RANSAC confidence.

    // Motion model (guided match search)
    CV_PROP_RW double motionModelRadius = 15.0;     //!< Guided-match search radius (px).
    CV_PROP_RW double motionModelRadiusWide = 30.0; //!< Wider fallback search radius (px) when the narrow search finds too few.
    CV_PROP_RW int motionModelMinMatches = 20;      //!< Matches below which the wider search runs.
    CV_PROP_RW double descProjThresh = 1.0;         //!< Descriptor-distance cutoff for a projected map-point match.

    // Optical flow fallback
    CV_PROP_RW int opticalFlowMinInliers = 10;      //!< Minimum correspondences for the optical-flow fallback.

    // Keyframe promotion
    CV_PROP_RW int kfMinFrames = 1;                 //!< Minimum frames since last keyframe before inserting one.
    CV_PROP_RW int kfMaxFrames = 30;                //!< Frames since last keyframe after which one is forced.
    CV_PROP_RW double kfInlierRatio = 0.70;         //!< Insert a keyframe when inliers drop below this fraction of the last keyframe's.
    CV_PROP_RW int kfMinInliers = 40;               //!< Absolute inlier floor: max(kfMinInliers, kfInlierRatio * lastKfInliers).
    CV_PROP_RW double kfRotThreshDeg = 5.0;         //!< Rotation (deg) from last keyframe that forces a new one.
    CV_PROP_RW double kfTransThresh = 0.5;          //!< Translation from last keyframe that forces a new one.

    // Local map refinement
    CV_PROP_RW int localMapTopK = 10;               //!< Top co-visible keyframes forming the local map.
    CV_PROP_RW int localMapNeighborK = 5;           //!< Covisibility neighbors expanded per local-map keyframe.
    CV_PROP_RW double localMapRadius = 7.0;         //!< Reprojection search radius (px) for local-map points.
};

//! @}

}} // namespace cv::slam

#endif // OPENCV_SLAM_ODOMETRY_PARAMS_HPP
