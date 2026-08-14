// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_SLAM_OPTIMIZER_HPP
#define OPENCV_SLAM_OPTIMIZER_HPP

#include "../odometry/frame.hpp"
#include "../loop/sim3.hpp"

#include <map>
#include <set>

namespace cv {
namespace slam {

class Map; // forward declaration

// static optimisation routines — never instantiated, called as Optimizer::Xxx(...)
class CV_EXPORTS Optimizer   // exported so the accuracy tests can link them
{
public:
    Optimizer() = delete;

    // pose-only BA: refines frame.poseCw with map points fixed; falls back to reprojection check without g2o
    static int poseOptimization(Frame& frame, const Mat& K, double reprojThresh, bool enable);

    // local BA: jointly refines newKf + its top-10 covisible keyframes and all their map points
    static void localBundleAdjustment(KeyFrame* newKf, const Mat& K, bool enable,
                                      bool* stopFlag = nullptr);

    // diagnostics returned by globalBundleAdjustment
    struct GlobalBAStats
    {
        bool ran = false; // false if skipped (no g2o / too few KFs or points)
        int keyframes = 0;
        int points = 0;
        long observations = 0;
        int posesUpdated = 0;
        int culled = 0; // outlier observations removed after optimisation
        double chi2Before = 0.0;
        double chi2After = 0.0;
    };

    // global BA: optimises every keyframe pose and map point against all reprojection errors
    static void globalBundleAdjustment(Map& map, const Mat& K, int iterations,
                                       int minObservations, bool enable,
                                       bool* stopFlag = nullptr,
                                       GlobalBAStats* stats = nullptr);

    // essential graph optimisation: distributes loop-closure drift over the full trajectory, then corrects map points
    static bool optimizeEssentialGraph(
    Map& map, KeyFrame* loopKf, KeyFrame* curKf,
    const std::map<KeyFrame*, Sim3>& nonCorrectedScw,
    const std::map<KeyFrame*, Sim3>& correctedScw,
    const std::map<KeyFrame*, std::set<KeyFrame*>>& loopConnections,
    bool fixScale, int iterations, int minCovisWeight,
    double* outFinalChi2 = nullptr);

}; // class Optimizer
}} // namespace cv::slam

#endif // OPENCV_SLAM_OPTIMIZER_HPP
