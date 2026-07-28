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
    CV_PROP_RW double minInitParallaxDeg = 3.0;     //!< Minimum parallax (deg) to trigger initialization.
    CV_PROP_RW int minInitPoints = 100;             //!< Minimum triangulated points to seed the map.
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
    CV_PROP_RW double kfInlierRatio = 0.75;         //!< Insert a keyframe when inliers drop below this fraction of the last keyframe's.
    CV_PROP_RW int kfMinInliers = 40;               //!< Absolute inlier floor: max(kfMinInliers, kfInlierRatio * lastKfInliers).
    CV_PROP_RW double kfRotThreshDeg = 5.0;         //!< Rotation (deg) from last keyframe that forces a new one.
    CV_PROP_RW double kfTransThresh = 0.5;          //!< Translation from last keyframe that forces a new one.

    // Local map refinement
    CV_PROP_RW int localMapTopK = 10;               //!< Top co-visible keyframes forming the local map.
    CV_PROP_RW int localMapNeighborK = 5;           //!< Covisibility neighbors expanded per local-map keyframe.
    CV_PROP_RW double localMapRadius = 7.0;         //!< Reprojection search radius (px) for local-map points.

    // Optimizers (g2o graph optimization; compile to no-ops when g2o is unavailable)
    CV_PROP_RW bool poseOptEnable = true;   //!< run g2o pose-only BA per frame
    CV_PROP_RW bool localBaEnable = true;   //!< run g2o local BA at each keyframe
    CV_PROP_RW bool globalBaEnable = true;  //!< run g2o global BA at end of run()
    CV_PROP_RW int globalBaIters = 10;      //!< global BA optimizer iterations
    CV_PROP_RW int globalBaMinObs = 2;      //!< min observations for a point in global BA

    // Loop detection
    CV_PROP_RW bool loopEnable = true;
    CV_PROP_RW int loopMinDbSize = 50;       //!< min keyframes in DB before searching
    CV_PROP_RW int loopVladK = 64;           //!< VLAD vocabulary size (k-means clusters)
    CV_PROP_RW int loopVladMaxTrain = 20000; //!< max descriptors used for k-means
    CV_PROP_RW int loopRecentGap = 30;       //!< keyframe IDs closer than this are excluded
    CV_PROP_RW float loopMinSimilarity = 0.0f; //!< VLAD cosine threshold
    CV_PROP_RW int loopTopK = 3;             //!< top-K candidates to geometrically verify
    CV_PROP_RW int loopMinRawMatches = 30;   //!< min LightGlue matches to attempt verify
    CV_PROP_RW int loopMinInliers = 40;      //!< min Essential-RANSAC inliers to accept
    CV_PROP_RW double loopMinInlierRatio = 0.4; //!< min inlier/raw ratio
    CV_PROP_RW int loopNConsistent = 3;      //!< consecutive detections required to act
    CV_PROP_RW int loopCloseCooldown = 20;   //!< keyframe gap before re-closing same region
    CV_PROP_RW int loopHashBits   = 256;     //!< binary LSH code length (bits, multiple of 8); 0 disables Hamming pre-filter
    CV_PROP_RW int loopCoarseTopk = 20;      //!< candidates kept after Hamming pre-filter before cosine rerank

    // Loop closure
    CV_PROP_RW bool loopCloseEnable = true;
    CV_PROP_RW int sim3RansacIters = 300;
    CV_PROP_RW int sim3MinInliers = 20;
    CV_PROP_RW double sim3MaxReprojErr2 = 9.21; //!< squared-pixel gate for Sim3 RANSAC
    CV_PROP_RW int essentialGraphIters = 20;
    CV_PROP_RW int essentialMinCovisWeight = 100; //!< min shared-MP count for covis edge
};

//! @}

}} // namespace cv::slam

#endif // OPENCV_SLAM_ODOMETRY_PARAMS_HPP
