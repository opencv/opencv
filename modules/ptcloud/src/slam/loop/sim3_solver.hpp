// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_SLAM_LOOP_SIM3_SOLVER_HPP
#define OPENCV_SLAM_LOOP_SIM3_SOLVER_HPP

#include "../precomp.hpp"
#include "sim3.hpp"

namespace cv {
namespace slam {

struct Sim3Result
{
    bool ok        = false;
    Sim3 Scm;
    int  nInliers  = 0;
    int  nPairs    = 0;
};

Sim3Result estimateSim3(const KeyFrame* Kc, const KeyFrame* Km,
                        const std::vector<DMatch>& matches,
                        const Mat& K, bool fixScale,
                        int ransacIters, int minInliers,
                        double maxReprojErr2);

}} // namespace cv::slam

#endif // OPENCV_SLAM_LOOP_SIM3_SOLVER_HPP
