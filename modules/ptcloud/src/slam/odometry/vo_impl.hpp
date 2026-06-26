// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_SLAM_VO_IMPL_HPP
#define OPENCV_SLAM_VO_IMPL_HPP

#include "../precomp.hpp"
#include "frame.hpp"
#include "../optimizer/pose_optimizer.hpp"

#include <fstream>

namespace cv {
namespace slam {

/** @brief Concrete VisualOdometry implementation (pimpl target).

Stage logic is split across:
  - vo_bootstrap.cpp  : two-view H/F initialisation
  - vo_tracking.cpp   : per-frame localisation (motion model, fallback 1/2, local map)
  - vo_keyframe.cpp   : keyframe promotion decision + covisibility helpers
  - vo_map_growth.cpp : triangulation of new map points at promotion time
  - visual_odometry.cpp : factory, run(), processFrame(), IO writers
*/
class VisualOdometryImpl CV_FINAL : public VisualOdometry
{
public:
    VisualOdometryImpl(const Ptr<Feature2D>& detector,
                       const Ptr<DescriptorMatcher>& matcher,
                       const String& imagesFolder,
                       const String& outputFolder,
                       const Mat& cameraMatrix,
                       const Mat& distCoeffs,
                       const OdometryParams& params);

    // --- VisualOdometry interface -------------------------------------------

    bool run() CV_OVERRIDE;
    bool processFrame(InputArray image) CV_OVERRIDE;
    void reset() CV_OVERRIDE;

    OdometryState getState() const CV_OVERRIDE { return state; }
    Matx44d getLastPose() const CV_OVERRIDE { return lastPoseCw; }
    const Map& getMap() const CV_OVERRIDE { return map; }
    const std::vector<Matx44d>& getTrajectory() const CV_OVERRIDE { return map.trajectory(); }
    const OdometryParams& getParams() const CV_OVERRIDE { return params; }
    void setParams(const OdometryParams& p) CV_OVERRIDE { params = p; }

    const String& getImagesFolder() const CV_OVERRIDE { return imagesFolder; }
    const String& getOutputFolder() const CV_OVERRIDE { return outputFolder; }
    void setOutputFolder(const String& f) CV_OVERRIDE { outputFolder = f; }

    // --- Stage entry points -------------------------------------------------

    bool bootstrap(Frame& cur);
    bool track(Frame& cur);

    bool trackWithMotionModel(Frame& cur); // motion model
    bool trackWithReferenceKF(Frame& cur); // fallback 1
    bool trackWithOpticalFlow(Frame& cur); // fallback 2
    void trackLocalMap(Frame& cur);

    bool shouldPromoteKeyframe(int nInliers, const Matx44d& T_cw, String& reason) const;
    void promoteKeyframeAndGrowMap(Frame& cur);

    // --- Shared helpers (visual_odometry.cpp) --------------------------------

    void extractFeatures(InputArray image, Frame& out) const;

    void matchFrames(const std::vector<KeyPoint>& qKp, const Mat& qDesc, Size qSz,
                     const std::vector<KeyPoint>& tKp, const Mat& tDesc, Size tSz,
                     std::vector<DMatch>& matches) const;

    // --- IO helpers (visual_odometry.cpp) ------------------------------------

    void writeCameraIntrinsics(const String& path) const;
    void writeMapPoints(const String& path) const;
    void writeImagesTxt(const String& path) const;

    // --- Owned state ---------------------------------------------------------

    Ptr<Feature2D> detector;
    Ptr<DescriptorMatcher> matcher;
    Mat K;    // 3×3 CV_64F
    Mat dist; // distortion coefficients (may be empty)
    OdometryParams params;

    String imagesFolder;
    String outputFolder;

    OdometryState state = NOT_INITIALIZED;
    Matx44d lastPoseCw = Matx44d::eye();

    Frame refFrame;
    KeyFrame* lastKf = nullptr;
    int framesSinceKf = 0;
    int lastKfInliers = 0;

    Matx44d velocity = Matx44d::eye();
    bool hasVelocity = false;

    Frame prevFrame;
    bool hasPrevFrame = false;

    String lastEvent;
    std::vector<String> poseFilenames;

    Map map;
};

namespace detail {

double rotationAngleDeg(const Matx44d& A_cw, const Matx44d& B_cw);
double parallaxDeg(const Point3d& X_world, const Matx44d& A_cw, const Matx44d& B_cw);
Matx34d projectionFromPose(const Matx44d& T_cw);
Matx44d makePose(const Mat& R, const Mat& t);
Point3d cameraCenterWorld(const Matx44d& T_cw);
void updateCovisibility(KeyFrame* kf);

} // namespace detail

}} // namespace cv::slam

#endif // OPENCV_SLAM_VO_IMPL_HPP
