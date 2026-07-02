// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_SLAM_TYPES_HPP
#define OPENCV_SLAM_TYPES_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"

#include <map>
#include <vector>

namespace cv {
namespace slam {

//! @addtogroup slam
//! @{

// Pipeline lifecycle state
enum OdometryState
{
    NOT_INITIALIZED = 0,
    INITIALIZING = 1,
    TRACKING = 2
};

struct MapPoint;
struct KeyFrame;

// 3D landmark, owned by Map
struct CV_EXPORTS MapPoint
{
    int id = -1;
    Point3d pos { 0, 0, 0 };
    Mat refDesc;

    std::map<KeyFrame*, size_t> observations; // kf -> keypoint index

    int visibleCount = 0;
    int foundCount = 0;
    bool bad = false; // soft-delete; check before use
};

// Keyframe: pose + keypoints + covisibility graph, owned by Map
struct CV_EXPORTS KeyFrame
{
    int id = -1;
    Matx44d poseCw = Matx44d::eye(); // world->camera

    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    std::vector<Point2f> undistKpts; // parallel to keypoints
    Size imageSize;

    std::vector<MapPoint*> mapPoints; // parallel to keypoints; null = unmatched

    std::map<KeyFrame*, int> covisibility;
    std::vector<std::pair<KeyFrame*, int>> orderedCovisibility; // sorted descending by count

    KeyFrame* parent = nullptr;
    Mat globalDesc;
};

//! @}

}} // namespace cv::slam

#endif // OPENCV_SLAM_TYPES_HPP
