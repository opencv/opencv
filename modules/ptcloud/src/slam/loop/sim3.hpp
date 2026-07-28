// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_SLAM_LOOP_SIM3_HPP
#define OPENCV_SLAM_LOOP_SIM3_HPP

#include "opencv2/core.hpp"

namespace cv {
namespace slam {

/** @brief 7-DoF similarity transform  (s·R, t)  acting as  p' = s·R·p + t.

Deliberately g2o-free so it can be used by the pure-cv correction code in
loop closing (Stage A / Stage C).  Only the essential-graph optimiser converts
this to/from g2o::Sim3 internally, behind HAVE_G2O.

Convention matches the module's poses: a camera pose is Sim3(R_cw, t_cw, s)
(world -> camera).  A relative S_cm maps a point in camera m into camera c:
  X_c = s·R·X_m + t. */
struct Sim3
{
    Matx33d R = Matx33d::eye();
    Vec3d   t = Vec3d::all(0.0);
    double  s = 1.0;
};

/** Composition: (a ∘ b) applied to p == a(b(p)).
    R = Ra·Rb,  t = sa·Ra·tb + ta,  s = sa·sb. */
inline Sim3 sim3Compose(const Sim3& a, const Sim3& b)
{
    Sim3 r;
    r.R = a.R * b.R;
    r.t = a.s * (a.R * b.t) + a.t;
    r.s = a.s * b.s;
    return r;
}

/** Inverse: s⁻¹, Rᵀ, −s⁻¹·Rᵀ·t. */
inline Sim3 sim3Inverse(const Sim3& S)
{
    Sim3 r;
    r.s = 1.0 / S.s;
    r.R = S.R.t();
    r.t = -r.s * (r.R * S.t);
    return r;
}

/** Apply the similarity to a point:  s·R·p + t. */
inline Vec3d sim3Map(const Sim3& S, const Vec3d& p)
{
    return S.s * (S.R * p) + S.t;
}

/** Build a Sim3 from a 4×4 world->camera pose (rotation+translation), scale s. */
inline Sim3 sim3FromPoseCW(const Matx44d& T, double s = 1.0)
{
    Sim3 r;
    r.s = s;
    r.R = Matx33d(T(0,0), T(0,1), T(0,2),
                  T(1,0), T(1,1), T(1,2),
                  T(2,0), T(2,1), T(2,2));
    r.t = Vec3d(T(0,3), T(1,3), T(2,3));
    return r;
}

/** Collapse a Sim3 to a 4×4 SE3 pose: rotation kept, translation de-scaled (t/s).
    Scale is folded into map points by Stage C so the stored pose stays a plain
    world->camera transform. */
inline Matx44d sim3ToPoseCW(const Sim3& S)
{
    Matx44d T = Matx44d::eye();
    const double invS = 1.0 / S.s;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j) T(i,j) = S.R(i,j);
        T(i,3) = S.t[i] * invS;
    }
    return T;
}

}} // namespace cv::slam

#endif // OPENCV_SLAM_LOOP_SIM3_HPP
