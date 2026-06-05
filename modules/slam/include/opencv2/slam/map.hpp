// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_SLAM_MAP_HPP
#define OPENCV_SLAM_MAP_HPP

#include "types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace cv { namespace slam {

/** @brief Container for keyframes, map points, and the per-frame trajectory.

    Map owns all KeyFrame and MapPoint objects. External code holds only
    raw non-owning pointers. Ids are assigned automatically when the
    incoming id field is negative.

    @ingroup slam_odometry
*/
class CV_EXPORTS_W Map
{
public:
    CV_WRAP Map();
    ~Map();

    /** @brief Add a keyframe; assigns id if kf.id < 0. Returns the assigned id. */
    CV_WRAP int addKeyframe(KeyFrame& kf);

    /** @brief Return keyframe by id, or nullptr. */
    CV_WRAP KeyFrame* getKeyframe(int id);

    /** @brief All keyframes in insertion order. */
    CV_WRAP const std::vector<KeyFrame*>& keyframes() const;

    CV_WRAP int numKeyframes() const;

    /** @brief Add a map point; assigns id if mp.id < 0. Returns the assigned id. */
    CV_WRAP int addMapPoint(MapPoint& mp);

    /** @brief Return map point by id, or nullptr. */
    CV_WRAP MapPoint* getMapPoint(int id);

    /** @brief All live (non-bad) map points. */
    CV_WRAP std::vector<MapPoint*> mapPoints() const;

    CV_WRAP int numMapPoints() const;

    /** @brief Wire a bidirectional KF-MP observation. */
    CV_WRAP void addObservation(KeyFrame* kf, int kp_idx, MapPoint* mp);

    /** @brief Mark a map point bad and remove all cross-references. */
    CV_WRAP void removeMapPoint(int mp_id);

    /** @brief Append a world-to-camera pose to the trajectory. */
    CV_WRAP void appendPose(const Matx44d& T_cw);

    /** @brief All emitted world-to-camera poses in order. */
    CV_WRAP const std::vector<Matx44d>& trajectory() const;

    /** @brief Reset to empty state. */
    CV_WRAP void clear();

private:
    struct Impl;
    Ptr<Impl> impl_;
};

}} // namespace cv::slam

#endif // OPENCV_SLAM_MAP_HPP
