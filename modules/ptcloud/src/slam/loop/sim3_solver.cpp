// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Relative Sim(3) estimation between two keyframes: Horn's closed-form
// absolute orientation inside a RANSAC loop, recovering the 7-DoF
// transform (with scale) used by closeLoop().

#include "sim3_solver.hpp"

#include <algorithm>
#include <cmath>

namespace cv {
namespace slam {

namespace {

void computeCentroid(const Mat& P, Mat& Pr, Mat& C)
{
    reduce(P, C, 1, REDUCE_SUM);
    C = C / P.cols;
    Pr = P.clone();
    for (int i = 0; i < P.cols; ++i)
        Pr.col(i) = P.col(i) - C;
}

// Horn closed-form: given matched 3×N sets P1, P2 (CV_64F), find (s, R, t) with
// P1 ≈ s·R·P2 + t.
void computeSim3(const Mat& P1, const Mat& P2, bool fixScale,
                 Matx33d& Rout, Vec3d& tout, double& sout)
{
    Mat Pr1, Pr2, O1, O2;
    computeCentroid(P1, Pr1, O1);
    computeCentroid(P2, Pr2, O2);

    Mat M = Pr2 * Pr1.t();
    const double m00 = M.at<double>(0,0), m01 = M.at<double>(0,1), m02 = M.at<double>(0,2);
    const double m10 = M.at<double>(1,0), m11 = M.at<double>(1,1), m12 = M.at<double>(1,2);
    const double m20 = M.at<double>(2,0), m21 = M.at<double>(2,1), m22 = M.at<double>(2,2);

    const double N11 =  m00 + m11 + m22;
    const double N12 =  m12 - m21;
    const double N13 =  m20 - m02;
    const double N14 =  m01 - m10;
    const double N22 =  m00 - m11 - m22;
    const double N23 =  m01 + m10;
    const double N24 =  m20 + m02;
    const double N33 = -m00 + m11 - m22;
    const double N34 =  m12 + m21;
    const double N44 = -m00 - m11 + m22;
    Mat N = (Mat_<double>(4,4) <<
             N11, N12, N13, N14,
             N12, N22, N23, N24,
             N13, N23, N33, N34,
             N14, N24, N34, N44);

    Mat eval, evec;
    eigen(N, eval, evec);
    Mat vec(1, 3, CV_64F);
    evec.row(0).colRange(1, 4).copyTo(vec);
    const double w   = evec.at<double>(0, 0);
    const double nrm = norm(vec);

    Mat Rmat;
    if (nrm < 1e-12)
    {
        Rmat = Mat::eye(3, 3, CV_64F);
    }
    else
    {
        const double ang = std::atan2(nrm, w);
        vec = (2.0 * ang / nrm) * vec;
        Rodrigues(vec, Rmat);
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Rout(i,j) = Rmat.at<double>(i,j);

    Mat P3 = Rmat * Pr2;
    if (!fixScale)
    {
        const double nom = Pr1.dot(P3);
        Mat sq;
        pow(P3, 2.0, sq);
        const double den = sum(sq)[0];
        sout = (den > 1e-12) ? (nom / den) : 1.0;
    }
    else
    {
        sout = 1.0;
    }

    Mat tt = O1 - sout * (Rmat * O2);
    tout = Vec3d(tt.at<double>(0), tt.at<double>(1), tt.at<double>(2));
}

inline Matx33d rotationOf(const Matx44d& T) { return T.get_minor<3,3>(0, 0); }
inline Vec3d translationOf(const Matx44d& T) { return Vec3d(T(0,3), T(1,3), T(2,3)); }

} // anonymous namespace

Sim3Result estimateSim3(const KeyFrame* Kc, const KeyFrame* Km,
                        const std::vector<DMatch>& matches,
                        const Mat& K, bool fixScale,
                        int ransacIters, int minInliers,
                        double maxReprojErr2)
{
    Sim3Result res;
    if (!Kc || !Km || K.empty()) return res;

    const double fx = K.at<double>(0,0), fy = K.at<double>(1,1);
    const double cx = K.at<double>(0,2), cy = K.at<double>(1,2);

    const Matx33d Rc = rotationOf(Kc->poseCw); const Vec3d tc = translationOf(Kc->poseCw);
    const Matx33d Rm = rotationOf(Km->poseCw); const Vec3d tm = translationOf(Km->poseCw);

    // build 3D-3D correspondences in each camera frame
    std::vector<Vec3d>   X1, X2;
    std::vector<Point2f> obs1, obs2;
    X1.reserve(matches.size()); X2.reserve(matches.size());
    obs1.reserve(matches.size()); obs2.reserve(matches.size());

    for (const DMatch& m : matches)
    {
        const size_t qi = (size_t)m.queryIdx, ti = (size_t)m.trainIdx;
        if (qi >= Kc->mapPoints.size() || ti >= Km->mapPoints.size()) continue;
        if (qi >= Kc->undistKpts.size() || ti >= Km->undistKpts.size()) continue;
        MapPoint* mp1 = Kc->mapPoints[qi];
        MapPoint* mp2 = Km->mapPoints[ti];
        if (!mp1 || !mp2 || mp1->bad || mp2->bad) continue;

        const Vec3d x1 = Rc * Vec3d(mp1->pos.x, mp1->pos.y, mp1->pos.z) + tc;
        const Vec3d x2 = Rm * Vec3d(mp2->pos.x, mp2->pos.y, mp2->pos.z) + tm;
        if (x1[2] <= 0.0 || x2[2] <= 0.0) continue;

        X1.push_back(x1);  X2.push_back(x2);
        obs1.push_back(Kc->undistKpts[qi]);
        obs2.push_back(Km->undistKpts[ti]);
    }

    const int Np = (int)X1.size();
    res.nPairs = Np;
    if (Np < 3 || Np < minInliers) return res;

    auto countInliers = [&](const Matx33d& R, const Vec3d& t, double s,
                            std::vector<char>& mask) -> int
    {
        Sim3 S;  S.R = R; S.t = t; S.s = s;
        const Sim3 Si = sim3Inverse(S);
        int inl = 0;
        mask.assign(Np, 0);
        for (int k = 0; k < Np; ++k)
        {
            const Vec3d xc = sim3Map(S, X2[k]);
            if (xc[2] <= 0.0) continue;
            const double u1 = fx * xc[0] / xc[2] + cx, v1 = fy * xc[1] / xc[2] + cy;
            const double e1 = (u1 - obs1[k].x) * (u1 - obs1[k].x)
                            + (v1 - obs1[k].y) * (v1 - obs1[k].y);
            if (e1 >= maxReprojErr2) continue;

            const Vec3d xm = sim3Map(Si, X1[k]);
            if (xm[2] <= 0.0) continue;
            const double u2 = fx * xm[0] / xm[2] + cx, v2 = fy * xm[1] / xm[2] + cy;
            const double e2 = (u2 - obs2[k].x) * (u2 - obs2[k].x)
                            + (v2 - obs2[k].y) * (v2 - obs2[k].y);
            if (e2 >= maxReprojErr2) continue;

            mask[k] = 1;
            ++inl;
        }
        return inl;
    };

    RNG rng(0x5EED3D);
    Matx33d bestR = Matx33d::eye(); Vec3d bestT = Vec3d::all(0); double bestS = 1.0;
    int bestInliers = 0;
    std::vector<char> bestMask, mask;

    Mat P1(3, 3, CV_64F), P2(3, 3, CV_64F);
    const int iters = std::max(1, ransacIters);
    for (int it = 0; it < iters; ++it)
    {
        int idx[3];
        idx[0] = rng.uniform(0, Np);
        do { idx[1] = rng.uniform(0, Np); } while (idx[1] == idx[0]);
        do { idx[2] = rng.uniform(0, Np); } while (idx[2] == idx[0] || idx[2] == idx[1]);

        for (int c = 0; c < 3; ++c)
        {
            P1.at<double>(0,c) = X1[idx[c]][0]; P1.at<double>(1,c) = X1[idx[c]][1]; P1.at<double>(2,c) = X1[idx[c]][2];
            P2.at<double>(0,c) = X2[idx[c]][0]; P2.at<double>(1,c) = X2[idx[c]][1]; P2.at<double>(2,c) = X2[idx[c]][2];
        }

        Matx33d R; Vec3d t; double s;
        computeSim3(P1, P2, fixScale, R, t, s);
        if (!(s > 0.0) || !std::isfinite(s)) continue;

        const int inl = countInliers(R, t, s, mask);
        if (inl > bestInliers)
        {
            bestInliers = inl; bestR = R; bestT = t; bestS = s; bestMask = mask;
        }
    }

    if (bestInliers < minInliers) return res;

    // refit on all inliers for a tighter estimate
    Mat Q1(3, bestInliers, CV_64F), Q2(3, bestInliers, CV_64F);
    int col = 0;
    for (int k = 0; k < Np; ++k)
    {
        if (!bestMask[k]) continue;
        Q1.at<double>(0,col) = X1[k][0]; Q1.at<double>(1,col) = X1[k][1]; Q1.at<double>(2,col) = X1[k][2];
        Q2.at<double>(0,col) = X2[k][0]; Q2.at<double>(1,col) = X2[k][1]; Q2.at<double>(2,col) = X2[k][2];
        ++col;
    }
    {
        Matx33d R; Vec3d t; double s;
        computeSim3(Q1, Q2, fixScale, R, t, s);
        if (std::isfinite(s) && s > 0.0)
        {
            std::vector<char> refMask;
            const int refInliers = countInliers(R, t, s, refMask);
            if (refInliers >= bestInliers) { bestR = R; bestT = t; bestS = s; bestInliers = refInliers; }
        }
    }

    res.ok         = true;
    res.Scm.R      = bestR;
    res.Scm.t      = bestT;
    res.Scm.s      = bestS;
    res.nInliers   = bestInliers;
    return res;
}

}} // namespace cv::slam
