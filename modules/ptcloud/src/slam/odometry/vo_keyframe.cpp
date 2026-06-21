// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "vo_impl.hpp"

namespace cv {
namespace slam {

namespace detail {

double rotationAngleDeg(const Matx44d& A, const Matx44d& B)
{
    Matx33d Ra(A(0,0),A(0,1),A(0,2), A(1,0),A(1,1),A(1,2), A(2,0),A(2,1),A(2,2));
    Matx33d Rb(B(0,0),B(0,1),B(0,2), B(1,0),B(1,1),B(1,2), B(2,0),B(2,1),B(2,2));
    Matx33d D = Rb * Ra.t();
    double tr = D(0,0) + D(1,1) + D(2,2);
    double c = std::max(-1.0, std::min(1.0, (tr - 1.0) * 0.5));
    return std::acos(c) * 180.0 / CV_PI;
}

Point3d cameraCenterWorld(const Matx44d& T_cw)
{
    Matx33d R(T_cw(0,0),T_cw(0,1),T_cw(0,2),
              T_cw(1,0),T_cw(1,1),T_cw(1,2),
              T_cw(2,0),T_cw(2,1),T_cw(2,2));
    Matx31d t(T_cw(0,3),T_cw(1,3),T_cw(2,3));
    Matx31d C = -R.t() * t;
    return Point3d(C(0),C(1),C(2));
}

double parallaxDeg(const Point3d& Xw, const Matx44d& A_cw, const Matx44d& B_cw)
{
    Point3d CA = cameraCenterWorld(A_cw), CB = cameraCenterWorld(B_cw);
    Point3d vA = Xw - CA, vB = Xw - CB;
    double nA = std::sqrt(vA.dot(vA)), nB = std::sqrt(vB.dot(vB));
    if (nA < 1e-9 || nB < 1e-9) return 0.0;
    double c = std::max(-1.0, std::min(1.0, vA.dot(vB) / (nA * nB)));
    return std::acos(c) * 180.0 / CV_PI;
}

Matx34d projectionFromPose(const Matx44d& T)
{
    return Matx34d(T(0,0),T(0,1),T(0,2),T(0,3),
                   T(1,0),T(1,1),T(1,2),T(1,3),
                   T(2,0),T(2,1),T(2,2),T(2,3));
}

Matx44d makePose(const Mat& R, const Mat& t)
{
    CV_Assert(R.rows == 3 && R.cols == 3 && t.total() == 3);
    Matx44d T = Matx44d::eye();
    Mat Rd, td;
    R.convertTo(Rd, CV_64F);
    t.reshape(1,3).convertTo(td, CV_64F);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j) T(i,j) = Rd.at<double>(i,j);
        T(i,3) = td.at<double>(i,0);
    }
    return T;
}

static void rebuildOrderedCovisibility(KeyFrame* target)
{
    target->orderedCovisibility.clear();
    for (auto& [other, cnt] : target->covisibility)
        target->orderedCovisibility.push_back({other, cnt});
    std::sort(target->orderedCovisibility.begin(),
              target->orderedCovisibility.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });
}

void updateCovisibility(KeyFrame* kf)
{
    if (!kf) return;

    kf->covisibility.clear();

    for (MapPoint* mp : kf->mapPoints)
    {
        if (!mp || mp->bad) continue;
        for (auto& [obsKf, kpIdx] : mp->observations)
        {
            if (obsKf == kf) continue;
            kf->covisibility[obsKf]++;
            obsKf->covisibility[kf]++;
        }
    }

    rebuildOrderedCovisibility(kf);
    for (auto& [nb, cnt] : kf->covisibility)
        rebuildOrderedCovisibility(nb);
}

} // namespace detail

bool VisualOdometryImpl::shouldPromoteKeyframe(int nInliers, const Matx44d& T_cw,
                                               String& reason) const
{
    if (framesSinceKf <= params.kfMinFrames) return false;

    if (framesSinceKf > params.kfMaxFrames)
    {
        reason = format("timeout(%d)", framesSinceKf);
        return true;
    }

    if (!lastKf) return false;

    // Motion-based triggers: rotation then translation (geometric, frame-count independent)
    double rot = detail::rotationAngleDeg(lastKf->poseCw, T_cw);
    if (rot > params.kfRotThreshDeg)
    {
        reason = format("rot=%.1fdeg", rot);
        return true;
    }

    // Tracking quality drop
    int minInliers = (lastKfInliers > 0)
        ? std::max(params.kfMinInliers, (int)(params.kfInlierRatio * lastKfInliers))
        : params.kfMinInliers;
    if (nInliers < minInliers)
    {
        reason = format("inliers=%d<%d", nInliers, minInliers);
        return true;
    }

    Point3d cCur  = detail::cameraCenterWorld(T_cw);
    Point3d cLast = detail::cameraCenterWorld(lastKf->poseCw);
    Point3d d     = cCur - cLast;
    double dist   = std::sqrt(d.dot(d));
    if (dist > params.kfTransThresh)
    {
        reason = format("trans=%.3f", dist);
        return true;
    }

    return false;
}

}} // namespace cv::slam
