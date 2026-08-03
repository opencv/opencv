// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_SLAM_VO_IMPL_HPP
#define OPENCV_SLAM_VO_IMPL_HPP

#include "../precomp.hpp"
#include "frame.hpp"
#include "../optimizer/optimizer.hpp"

namespace cv {
namespace slam {

/** @brief Concrete VisualOdometry implementation (pimpl target).

Stage logic is split across:
  - vo_bootstrap.cpp  : two-view H/F initialisation
  - vo_tracking.cpp   : per-frame localisation (motion model, fallback 1/2, local map)
  - vo_keyframe.cpp   : keyframe promotion decision + covisibility helpers
  - vo_map_growth.cpp : triangulation of new map points at promotion time
  - visual_odometry.cpp : factory, processFrame(), finalize()
*/
class VisualOdometryImpl CV_FINAL : public VisualOdometry
{
public:
    VisualOdometryImpl(const Ptr<Feature2D>& _detector,
                       const Ptr<DescriptorMatcher>& _matcher,
                       const Mat& _cameraMatrix,
                       const Mat& _distCoeffs,
                       const OdometryParams& _params);

    // --- VisualOdometry interface -------------------------------------------

    bool processFrame(InputArray image) CV_OVERRIDE;
    bool finalize() CV_OVERRIDE;
    void reset() CV_OVERRIDE;

    OdometryState getState() const CV_OVERRIDE { return state; }
    Matx44d getLastPose() const CV_OVERRIDE { return lastPoseCw; }
    const Map& getMap() const CV_OVERRIDE { return map; }
    int getNumKeyframes() const CV_OVERRIDE { return map.numKeyframes(); }
    int getNumMapPoints() const CV_OVERRIDE { return map.numMapPoints(); }
    const std::vector<Matx44d>& getTrajectory() const CV_OVERRIDE { return map.trajectory(); }
    std::vector<Matx44d> getCorrectedTrajectory() const CV_OVERRIDE;
    const OdometryParams& getParams() const CV_OVERRIDE { return params; }
    void setParams(const OdometryParams& p) CV_OVERRIDE { params = p; }

    // --- Stage entry points -------------------------------------------------

    bool bootstrap(Frame& currentFrame);
    bool track(Frame& currentFrame);

    bool trackWithMotionModel(Frame& currentFrame); // motion model
    bool trackWithReferenceKF(Frame& currentFrame); // fallback 1
    bool trackWithOpticalFlow(Frame& currentFrame); // fallback 2
    void trackLocalMap(Frame& currentFrame);

    bool shouldPromoteKeyframe(int nInliers, const Matx44d& T_cw, String& reason) const;
    void promoteKeyframeAndGrowMap(Frame& currentFrame);

    // --- Loop detection / closure (loop/loop_detection.cpp, loop/loop_closing.cpp) ---

    void buildVocabulary();
    Mat  computeVlad(const Mat& descriptorsIn) const;
    int  geometricVerify(const KeyFrame* q, const KeyFrame* c,
                         const std::vector<DMatch>& matches) const;
    void detectLoop(KeyFrame* query);
    bool closeLoop(KeyFrame* Kc, KeyFrame* Km, const std::vector<DMatch>& matches);

    // --- Shared helpers (visual_odometry.cpp) --------------------------------

    void extractFeatures(InputArray image, Frame& out) const;

    void matchFrames(const std::vector<KeyPoint>& qKp, const Mat& qDesc, Size qSz,
                     const std::vector<KeyPoint>& tKp, const Mat& tDesc, Size tSz,
                     std::vector<DMatch>& matches) const;

    // --- Owned state ---------------------------------------------------------

    Ptr<Feature2D> detector;
    Ptr<DescriptorMatcher> matcher;
    Mat K;    // 3×3 CV_64F
    Mat dist; // distortion coefficients (may be empty)
    OdometryParams params;

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

    // per-frame record for corrected trajectory: relative pose to refKf at tracking time
    struct FrameRecord { Matx44d relPose; KeyFrame* refKf; };
    std::vector<FrameRecord> frameRecords;

    // Loop detection state
    Mat  vocab;
    bool vocabReady    = false;
    Mat  hashProj;             // (loopHashBits, K*D) random projection for binary Hamming pre-filter
    int  loopStreak    = 0;
    KeyFrame* loopLastCand   = nullptr;
    int  lastClosedKfId      = -1;

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
