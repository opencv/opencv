// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_SLAM_MAP_HPP
#define OPENCV_SLAM_MAP_HPP

#include "opencv2/ptcloud/slam/types.hpp"

#include <set>
#include <vector>

namespace cv {
namespace slam {

//! @addtogroup slam
//! @{

/** @brief Owns all persistent SLAM state (KeyFrames and MapPoints).
Pointers from addKeyframe / addMapPoint are valid until removeMapPoint / clear. */
class CV_EXPORTS Map
{
public:
    Map();
    ~Map();

    Map(const Map&) = delete;
    Map& operator=(const Map&) = delete;

    // Keyframes

    KeyFrame* addKeyframe(KeyFrame* kf); //!< takes ownership; assigns id if < 0
    KeyFrame* getKeyframe(int id) const;

    const std::set<KeyFrame*>& keyframes() const;
    int numKeyframes() const;

    // Map points

    MapPoint* addMapPoint(MapPoint* mp); //!< takes ownership; assigns id if < 0
    MapPoint* getMapPoint(int id) const;

    const std::set<MapPoint*>& mapPoints() const;
    int numMapPoints() const;

    void addObservation(KeyFrame* kf, size_t kpIdx, MapPoint* mp);
    void removeObservation(KeyFrame* kf, MapPoint* mp);
    void removeMapPoint(MapPoint* mp);

    // Reference / current keyframes

    void setRefKeyframe(KeyFrame* kf);
    KeyFrame* getRefKeyframe() const;

    void setCurrentKeyframe(KeyFrame* kf);
    KeyFrame* getCurrentKeyframe() const;

    // Trajectory

    void appendPose(const Matx44d& T_cw);
    const std::vector<Matx44d>& trajectory() const;

    // Lifecycle

    void clear();

private:
    struct Impl;
    Ptr<Impl> impl;
};

//! @}

}} // namespace cv::slam

#endif // OPENCV_SLAM_MAP_HPP
