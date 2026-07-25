// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "vo_impl.hpp"

namespace cv {
namespace slam {

namespace {

inline Point2d projectThrough(const Mat& P, double X, double Y, double Z)
{
    double u = P.at<double>(0,0)*X + P.at<double>(0,1)*Y + P.at<double>(0,2)*Z + P.at<double>(0,3);
    double v = P.at<double>(1,0)*X + P.at<double>(1,1)*Y + P.at<double>(1,2)*Z + P.at<double>(1,3);
    double w = P.at<double>(2,0)*X + P.at<double>(2,1)*Y + P.at<double>(2,2)*Z + P.at<double>(2,3);
    if (std::abs(w) < 1e-12) return Point2d(0, 0);
    return Point2d(u / w, v / w);
}

inline double cameraDepth(const Matx44d& T_cw, double X, double Y, double Z)
{
    return T_cw(2,0)*X + T_cw(2,1)*Y + T_cw(2,2)*Z + T_cw(2,3);
}

} // anonymous namespace

void VisualOdometryImpl::promoteKeyframeAndGrowMap(Frame& cur)
{
    KeyFrame* newKf = new KeyFrame();
    newKf->poseCw = cur.poseCw;
    newKf->keypoints = cur.keypoints;
    newKf->descriptors = cur.descriptors.clone();
    newKf->undistKpts = cur.undistKpts;
    newKf->imageSize = cur.imageSize;
    newKf->mapPoints.assign(cur.keypoints.size(), nullptr);
    newKf->parent = lastKf;

    map.addKeyframe(newKf);

    for (size_t i = 0; i < cur.mapPoints.size(); ++i)
    {
        MapPoint* mp = cur.mapPoints[i];
        if (!mp || mp->bad || cur.outliers[i]) continue;
        map.addObservation(newKf, i, mp);
    }

    std::vector<DMatch> kfToCur;
    matchFrames(lastKf->keypoints, lastKf->descriptors, lastKf->imageSize,
                cur.keypoints, cur.descriptors, cur.imageSize, kfToCur);

    Mat Rt1(3, 4, CV_64F), Rt2(3, 4, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
        {
            Rt1.at<double>(i,j) = lastKf->poseCw(i,j);
            Rt2.at<double>(i,j) = cur.poseCw(i,j);
        }
    Mat P1 = K * Rt1;
    Mat P2 = K * Rt2;

    std::vector<Point2f> pts1, pts2;
    std::vector<int> triMatchIdx;
    for (size_t i = 0; i < kfToCur.size(); ++i)
    {
        const DMatch& m = kfToCur[i];
        if ((size_t)m.queryIdx >= lastKf->mapPoints.size()) continue;
        if (lastKf->mapPoints[m.queryIdx] != nullptr) continue;
        if ((size_t)m.trainIdx >= newKf->mapPoints.size()) continue;
        if (newKf->mapPoints[m.trainIdx] != nullptr) continue;
        pts1.push_back(lastKf->undistKpts[m.queryIdx]);
        pts2.push_back(cur.undistKpts[m.trainIdx]);
        triMatchIdx.push_back((int)i);
    }

    if (pts1.empty())
    {
        detail::updateCovisibility(newKf);
        map.setCurrentKeyframe(newKf);
        lastKf = newKf;
        lastKfInliers = 0;
        framesSinceKf = 0;
        return;
    }

    Mat pts4D;
    triangulatePoints(P1, P2, pts1, pts2, pts4D);

    int nNew = 0;
    for (int i = 0; i < pts4D.cols; ++i)
    {
        double w = pts4D.at<float>(3, i);
        if (std::abs(w) < 1e-9) continue;
        double X = pts4D.at<float>(0, i) / w;
        double Y = pts4D.at<float>(1, i) / w;
        double Z = pts4D.at<float>(2, i) / w;

        if (cameraDepth(lastKf->poseCw, X, Y, Z) <= 0) continue;
        if (cameraDepth(cur.poseCw, X, Y, Z) <= 0) continue;

        Point2d p1p = projectThrough(P1, X, Y, Z);
        Point2d p2p = projectThrough(P2, X, Y, Z);
        double e1 = std::hypot(p1p.x - pts1[i].x, p1p.y - pts1[i].y);
        double e2 = std::hypot(p2p.x - pts2[i].x, p2p.y - pts2[i].y);
        if (e1 > params.pnpReprojThresh || e2 > params.pnpReprojThresh) continue;

        Point3d Xw(X, Y, Z);
        if (detail::parallaxDeg(Xw, lastKf->poseCw, cur.poseCw)
                < params.minGrowthParallaxDeg) continue;

        MapPoint* mp = new MapPoint();
        mp->pos = Xw;
        const DMatch& dm = kfToCur[triMatchIdx[i]];
        mp->refDesc = cur.descriptors.row(dm.trainIdx).clone();

        map.addMapPoint(mp);
        map.addObservation(lastKf, (size_t)dm.queryIdx, mp);
        map.addObservation(newKf, (size_t)dm.trainIdx, mp);
        ++nNew;
    }

    detail::updateCovisibility(newKf);

    map.setCurrentKeyframe(newKf);
    lastKf = newKf;
    lastKfInliers = 0;
    framesSinceKf = 0;

    (void)nNew;
}

}} // namespace cv::slam
