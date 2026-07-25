// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "pose_optimizer.hpp"

namespace cv { namespace slam {

// pose is not optimized yet, only inlier classification is done.
// adding bundle adjustment based pose optimization in a later commit.

int poseInlierCheck(Frame& frame, const Mat& K, double reprojThresh)
{
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);
    const Matx44d& T = frame.poseCw;

    int nInliers = 0;
    for (size_t i = 0; i < frame.mapPoints.size(); ++i)
    {
        frame.outliers[i] = true;
        MapPoint* mp = frame.mapPoints[i];
        if (!mp || mp->bad) continue;

        const double Xc = T(0,0)*mp->pos.x + T(0,1)*mp->pos.y + T(0,2)*mp->pos.z + T(0,3);
        const double Yc = T(1,0)*mp->pos.x + T(1,1)*mp->pos.y + T(1,2)*mp->pos.z + T(1,3);
        const double Zc = T(2,0)*mp->pos.x + T(2,1)*mp->pos.y + T(2,2)*mp->pos.z + T(2,3);
        if (Zc <= 0.0) continue;

        const double u = fx * Xc / Zc + cx;
        const double v = fy * Yc / Zc + cy;
        const double dx = u - static_cast<double>(frame.undistKpts[i].x);
        const double dy = v - static_cast<double>(frame.undistKpts[i].y);

        if (std::sqrt(dx * dx + dy * dy) <= reprojThresh)
        {
            frame.outliers[i] = false;
            ++nInliers;
        }
    }
    return nInliers;
}

}} // namespace cv::slam
