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

// erase bad/low-quality map points; snapshot first since removeMapPoint mutates the live set.
void cullMapPoints(Map& map)
{
    const std::vector<MapPoint*> snapshot(map.mapPoints().begin(), map.mapPoints().end());
    for (MapPoint* mp : snapshot)
    {
        if (mp->bad ||
            (mp->visibleCount > 10 && mp->foundCount < 0.25 * mp->visibleCount))
            map.removeMapPoint(mp);
    }
}

void syncFrameMapPoints(Frame& frame, const KeyFrame* kf)
{
    const size_t n = std::min(frame.mapPoints.size(), kf->mapPoints.size());
    for (size_t i = 0; i < n; ++i)
        frame.mapPoints[i] = kf->mapPoints[i];
}

} // anonymous namespace

void VisualOdometryImpl::promoteKeyframeAndGrowMap(Frame& currentFrame)
{
    KeyFrame* newKf = new KeyFrame();
    newKf->poseCw = currentFrame.poseCw;
    newKf->keypoints = currentFrame.keypoints;
    newKf->descriptors = currentFrame.descriptors.clone();
    newKf->undistKpts = currentFrame.undistKpts;
    newKf->imageSize = currentFrame.imageSize;
    newKf->mapPoints.assign(currentFrame.keypoints.size(), nullptr);
    newKf->parent = lastKf;
    if (lastKf) lastKf->children.insert(newKf);

    map.addKeyframe(newKf);

    // register map points already tracked by this frame
    for (size_t i = 0; i < currentFrame.mapPoints.size(); ++i)
    {
        MapPoint* mp = currentFrame.mapPoints[i];
        if (!mp || mp->bad || currentFrame.outliers[i]) continue;
        map.addObservation(newKf, i, mp);
    }

    // match lastKf↔newKf and keep only pairs where neither keypoint has a map point yet
    std::vector<DMatch> kfToCur;
    matchFrames(lastKf->keypoints, lastKf->descriptors, lastKf->imageSize,
                currentFrame.keypoints, currentFrame.descriptors, currentFrame.imageSize, kfToCur);

    Mat Rt1(3, 4, CV_64F), Rt2(3, 4, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
        {
            Rt1.at<double>(i,j) = lastKf->poseCw(i,j);
            Rt2.at<double>(i,j) = currentFrame.poseCw(i,j);
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
        pts2.push_back(currentFrame.undistKpts[m.trainIdx]);
        triMatchIdx.push_back((int)i);
    }

    // no unmatched pairs left to triangulate
    if (pts1.empty())
    {
        detail::updateCovisibility(newKf);
        Optimizer::localBundleAdjustment(newKf, K, params.localBaEnable);
        detectLoop(newKf);
        cullMapPoints(map);
        syncFrameMapPoints(currentFrame, newKf);
        map.setCurrentKeyframe(newKf);
        lastKf = newKf;
        lastKfInliers = 0;
        framesSinceKf = 0;
        return;
    }

    // triangulate and add new map points, rejecting by depth / reprojection / parallax
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
        if (cameraDepth(currentFrame.poseCw, X, Y, Z) <= 0) continue;

        Point2d p1p = projectThrough(P1, X, Y, Z);
        Point2d p2p = projectThrough(P2, X, Y, Z);
        double e1 = std::hypot(p1p.x - pts1[i].x, p1p.y - pts1[i].y);
        double e2 = std::hypot(p2p.x - pts2[i].x, p2p.y - pts2[i].y);
        if (e1 > params.pnpReprojThresh || e2 > params.pnpReprojThresh) continue;

        Point3d Xw(X, Y, Z);
        if (detail::parallaxDeg(Xw, lastKf->poseCw, currentFrame.poseCw)
                < params.minGrowthParallaxDeg) continue;

        MapPoint* mp = new MapPoint();
        mp->pos = Xw;
        mp->refKf = newKf;
        const DMatch& dm = kfToCur[triMatchIdx[i]];
        mp->refDesc = currentFrame.descriptors.row(dm.trainIdx).clone();

        map.addMapPoint(mp);
        map.addObservation(lastKf, (size_t)dm.queryIdx, mp);
        map.addObservation(newKf, (size_t)dm.trainIdx, mp);
        ++nNew;
    }

    detail::updateCovisibility(newKf);

    Optimizer::localBundleAdjustment(newKf, K, params.localBaEnable);
    detectLoop(newKf);
    cullMapPoints(map);
    syncFrameMapPoints(currentFrame, newKf);

    map.setCurrentKeyframe(newKf);
    lastKf = newKf;
    lastKfInliers = 0;
    framesSinceKf = 0;

    (void)nNew;
}

}} // namespace cv::slam
