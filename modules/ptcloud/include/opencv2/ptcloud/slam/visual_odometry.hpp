// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_SLAM_VISUAL_ODOMETRY_HPP
#define OPENCV_SLAM_VISUAL_ODOMETRY_HPP

#include "opencv2/core.hpp"
#include "opencv2/features.hpp"

#include "opencv2/ptcloud/slam/types.hpp"
#include "opencv2/ptcloud/slam/map.hpp"
#include "opencv2/ptcloud/slam/odometry_params.hpp"

#include <vector>

namespace cv {
namespace slam {

//! @addtogroup slam
//! @{

/** @brief Monocular visual odometry pipeline.

State machine: NOT_INITIALIZED → INITIALIZING (H/F two-view bootstrap) → TRACKING
(per-frame PnP + local-map refinement). Tracking failure rewinds to INITIALIZING.

Purely in-memory: feed images with @ref processFrame and read back the trajectory
and map with @ref getTrajectory / @ref getMap. See samples/slam for driving this
from a directory of images and writing the result to disk (e.g. COLMAP format).
*/
class CV_EXPORTS_W VisualOdometry
{
public:
    virtual ~VisualOdometry();

    CV_WRAP static Ptr<VisualOdometry> create(
        const Ptr<Feature2D>& detector,
        const Ptr<DescriptorMatcher>& matcher,
        InputArray cameraMatrix,
        InputArray distCoeffs = noArray(),
        const OdometryParams& params = OdometryParams());

    /** @brief Feed one image. Returns true if a pose was emitted. */
    CV_WRAP virtual bool processFrame(InputArray image) = 0;

    /** @brief Reset to NOT_INITIALIZED, clearing map and trajectory. */
    CV_WRAP virtual void reset() = 0;

    CV_WRAP virtual OdometryState getState() const = 0;
    CV_WRAP virtual Matx44d getLastPose() const = 0;

    //! @note Not exposed to Python: Map holds raw pointers / non-convertible containers.
    virtual const Map& getMap() const = 0;

    //! @brief Number of keyframes in the map. Convenience for callers (e.g. Python) that
    //! can't use @ref getMap directly.
    CV_WRAP virtual int getNumKeyframes() const = 0;

    //! @brief Number of map points in the map. Convenience for callers (e.g. Python) that
    //! can't use @ref getMap directly.
    CV_WRAP virtual int getNumMapPoints() const = 0;

    CV_WRAP virtual const std::vector<Matx44d>& getTrajectory() const = 0;

    CV_WRAP virtual const OdometryParams& getParams() const = 0;
    CV_WRAP virtual void setParams(const OdometryParams& params) = 0;

protected:
    VisualOdometry();
};

//! @}

}} // namespace cv::slam

#endif // OPENCV_SLAM_VISUAL_ODOMETRY_HPP
