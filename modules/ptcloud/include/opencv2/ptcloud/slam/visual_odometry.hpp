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

The class is purely in-memory: it consumes frames through @ref processFrame and exposes
its results through getters. Reading an image sequence from disk and exporting the
result is the caller's responsibility — see `samples/slam/visual_odometry.cpp`.

Typical use:
@code
    Ptr<VisualOdometry> vo = VisualOdometry::create(detector, matcher, K);
    for (const String& path : imageFiles)
        vo->processFrame(imread(path));
    vo->finalize();                       // end-of-sequence global bundle adjustment
    writeColmapFiles(vo, K, imageSize, outputFolder);
@endcode
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

    /** @brief End-of-sequence refinement: runs global bundle adjustment over every
    keyframe pose and map point.

    Call once after the last @ref processFrame. Poses returned by @ref getMap and
    @ref getCorrectedTrajectory reflect the optimised graph only after this call.
    Returns true if the optimisation actually ran (it is skipped when global BA is
    disabled in @ref OdometryParams, when g2o is unavailable, or when the map is too
    small to optimise). */
    CV_WRAP virtual bool finalize() = 0;

    /** @brief Reset to NOT_INITIALIZED, clearing map and trajectory. */
    CV_WRAP virtual void reset() = 0;

    CV_WRAP virtual OdometryState getState() const = 0;
    CV_WRAP virtual Matx44d getLastPose() const = 0;

    //! @note Not exposed to Python: Map holds raw pointers / non-convertible containers.
    //! Use getNumKeyframes() / getNumMapPoints() for aggregate counts from Python.
    virtual const Map& getMap() const = 0;

    CV_WRAP virtual int getNumKeyframes() const = 0;
    CV_WRAP virtual int getNumMapPoints() const = 0;

    /** @brief Raw per-frame poses, appended at emission time.

    Never rewritten by loop closure or bundle adjustment, so this is the *uncorrected*
    trajectory. Use @ref getCorrectedTrajectory for poses that carry the optimisations. */
    CV_WRAP virtual const std::vector<Matx44d>& getTrajectory() const = 0;

    /** @brief Per-frame poses re-expressed on the current (optimised) keyframe graph.

    Each emitted frame is stored relative to its reference keyframe, so loop-closure and
    bundle-adjustment corrections applied to that keyframe propagate here. This is the
    trajectory to export or plot.

    Index-aligned with @ref getTrajectory and always the same length; a frame whose
    reference keyframe was culled falls back to its raw pose. */
    CV_WRAP virtual std::vector<Matx44d> getCorrectedTrajectory() const = 0;

    CV_WRAP virtual const OdometryParams& getParams() const = 0;
    CV_WRAP virtual void setParams(const OdometryParams& params) = 0;

protected:
    VisualOdometry();
};

//! @}

}} // namespace cv::slam

#endif // OPENCV_SLAM_VISUAL_ODOMETRY_HPP
