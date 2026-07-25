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

// Returns the homography candidate index with the most cheirality-consistent triangulations.
int bestHomographyCandidate(
    const std::vector<Mat>& Rs,
    const std::vector<Mat>& ts,
    const std::vector<Point2f>& pts1,
    const std::vector<Point2f>& pts2,
    const Mat& P1)
{
    int bestIdx = 0;
    int bestCount = -1;
    const int nSol = (int)Rs.size();

    for (int s = 0; s < nSol; ++s)
    {
        Mat Rt(3, 4, CV_64F);
        Rs[s].copyTo(Rt(Rect(0, 0, 3, 3)));
        ts[s].reshape(1, 3).copyTo(Rt(Rect(3, 0, 1, 3)));

        Mat Klocal = P1(Rect(0, 0, 3, 3)).clone();
        Mat P2_s = Klocal * Rt;

        Mat pts4D;
        triangulatePoints(P1, P2_s, pts1, pts2, pts4D);

        int count = 0;
        for (int i = 0; i < pts4D.cols; ++i)
        {
            float w = pts4D.at<float>(3, i);
            if (std::abs(w) < 1e-9f) continue;
            float X = pts4D.at<float>(0, i) / w;
            float Y = pts4D.at<float>(1, i) / w;
            float Z = pts4D.at<float>(2, i) / w;
            float Z2 = (float)(Rs[s].at<double>(2,0)*X + Rs[s].at<double>(2,1)*Y
                              + Rs[s].at<double>(2,2)*Z
                              + ts[s].reshape(1,3).at<double>(2,0));
            if (Z > 0 && Z2 > 0) ++count;
        }
        if (count > bestCount) { bestCount = count; bestIdx = s; }
    }
    return bestIdx;
}

} // anonymous namespace

bool VisualOdometryImpl::bootstrap(Frame& cur)
{
    std::vector<DMatch> matches;
    matchFrames(refFrame.keypoints, refFrame.descriptors, refFrame.imageSize,
                cur.keypoints, cur.descriptors, cur.imageSize, matches);

    if ((int)matches.size() < params.minInitInliers)
    {
        // slide ref forward so we don't keep trying against a frame that's too far
        refFrame = std::move(cur);
        return false;
    }

    std::vector<Point2f> refU, curU;
    refU.reserve(matches.size());
    curU.reserve(matches.size());
    for (const auto& m : matches)
    {
        refU.push_back(refFrame.undistKpts[m.queryIdx]);
        curU.push_back(cur.undistKpts[m.trainIdx]);
    }

    Mat maskE, maskH;
    Mat E = findEssentialMat(refU, curU, K, RANSAC,
                              params.essentialRansacConfidence,
                              params.essentialRansacThresh, 1000, maskE);
    Mat H = findHomography(refU, curU, RANSAC,
                           params.essentialRansacThresh, maskH);

    const int nE = (!E.empty() && !maskE.empty()) ? countNonZero(maskE) : 0;
    const int nH = (!H.empty() && !maskH.empty()) ? countNonZero(maskH) : 0;

    if (nE < params.minInitInliers && nH < params.minInitInliers)
        return false;

    const double RH = (double)nH / ((double)nH + (double)nE + 1e-9);

    Mat R, t, modelMask;
    Mat P1 = K * Mat::eye(3, 4, CV_64F);

    if (RH > params.hfRatioThresh && !H.empty())
    {
        std::vector<Mat> Rs, ts, normals;
        int nSol = decomposeHomographyMat(H, K, Rs, ts, normals);
        if (nSol <= 0) return false;

        std::vector<Point2f> p1In, p2In;
        for (size_t i = 0; i < matches.size(); ++i)
            if (!maskH.empty() && maskH.at<uchar>((int)i))
            { p1In.push_back(refU[i]); p2In.push_back(curU[i]); }
        if (p1In.empty()) return false;

        int best = bestHomographyCandidate(Rs, ts, p1In, p2In, P1);
        R = Rs[best].clone();
        t = ts[best].clone();
        modelMask = maskH;
    }
    else
    {
        if (E.empty() || nE < params.minInitInliers) return false;
        Mat recoverMask = maskE.clone();
        int nPose = recoverPose(E, refU, curU, K, R, t, recoverMask);
        if (nPose < params.minInitInliers) return false;
        modelMask = recoverMask;
    }

    std::vector<Point2f> refIn, curIn;
    std::vector<int> matchIn;
    for (size_t i = 0; i < matches.size(); ++i)
        if (!modelMask.empty() && modelMask.at<uchar>((int)i))
        {
            refIn.push_back(refU[i]);
            curIn.push_back(curU[i]);
            matchIn.push_back((int)i);
        }

    if ((int)refIn.size() < params.minInitInliers)
        return false;

    Mat Rt(3, 4, CV_64F);
    R.copyTo(Rt(Rect(0, 0, 3, 3)));
    t.reshape(1, 3).copyTo(Rt(Rect(3, 0, 1, 3)));
    Mat P2 = K * Rt;

    Mat pts4D;
    triangulatePoints(P1, P2, refIn, curIn, pts4D);

    Matx44d T_ref = Matx44d::eye();
    Matx44d T_cur = detail::makePose(R, t);

    std::vector<Point3d> goodPts;
    std::vector<int> goodMatch;
    int nValid = 0;
    int nPos = 0;

    for (int i = 0; i < pts4D.cols; ++i)
    {
        double w = pts4D.at<float>(3, i);
        if (std::abs(w) < 1e-9) continue;
        ++nValid;

        double X = pts4D.at<float>(0, i) / w;
        double Y = pts4D.at<float>(1, i) / w;
        double Z = pts4D.at<float>(2, i) / w;

        if (Z <= 0) continue;
        double Z2 = R.at<double>(2,0)*X + R.at<double>(2,1)*Y
                  + R.at<double>(2,2)*Z + t.reshape(1,3).at<double>(2,0);
        if (Z2 <= 0) continue;
        ++nPos;

        Point3d Xw(X, Y, Z);
        if (detail::parallaxDeg(Xw, T_ref, T_cur) < params.minInitParallaxDeg)
            continue;

        goodPts.push_back(Xw);
        goodMatch.push_back(matchIn[i]);
    }

    // reject degenerate decompositions (< 90% positive-depth ratio)
    if (nValid > 0 && (double)nPos / nValid < 0.9)
        return false;
    if ((int)goodPts.size() < params.minInitPoints)
        return false;

    // normalize scale: set median scene depth to 1.0
    std::vector<double> depths;
    depths.reserve(goodPts.size());
    for (const auto& p : goodPts) depths.push_back(p.z);
    std::nth_element(depths.begin(), depths.begin() + depths.size()/2, depths.end());
    double med = depths[depths.size()/2];
    if (med < 1e-9) return false;

    double scale = 1.0 / med;
    for (auto& p : goodPts) { p.x *= scale; p.y *= scale; p.z *= scale; }

    Mat tSc;
    t.reshape(1, 3).convertTo(tSc, CV_64F);
    tSc = tSc * scale;
    T_cur = detail::makePose(R, tSc);

    auto makeKF = [](const Frame& f) -> KeyFrame* {
        KeyFrame* kf = new KeyFrame();
        kf->poseCw = Matx44d::eye();
        kf->keypoints = f.keypoints;
        kf->descriptors = f.descriptors.clone();
        kf->undistKpts = f.undistKpts;
        kf->imageSize = f.imageSize;
        kf->mapPoints.assign(f.keypoints.size(), nullptr);
        return kf;
    };

    KeyFrame* kfRef = makeKF(refFrame);
    kfRef->poseCw = T_ref;

    KeyFrame* kfCur = makeKF(cur);
    kfCur->poseCw = T_cur;
    kfCur->parent = kfRef;

    map.addKeyframe(kfRef);
    map.addKeyframe(kfCur);

    for (size_t i = 0; i < goodPts.size(); ++i)
    {
        MapPoint* mp = new MapPoint();
        mp->pos = goodPts[i];
        const DMatch& m = matches[goodMatch[i]];
        mp->refDesc = refFrame.descriptors.row(m.queryIdx).clone();

        map.addMapPoint(mp);
        map.addObservation(kfRef, (size_t)m.queryIdx, mp);
        map.addObservation(kfCur, (size_t)m.trainIdx, mp);
    }

    detail::updateCovisibility(kfRef);

    map.setRefKeyframe(kfRef);
    map.setCurrentKeyframe(kfCur);

    lastKf = kfCur;
    framesSinceKf = 0;
    lastKfInliers = (int)goodPts.size();
    lastPoseCw = T_cur;
    state = TRACKING;
    hasVelocity = false;

    map.appendPose(T_ref);
    map.appendPose(T_cur);

    refFrame = Frame();
    return true;
}

}} // namespace cv::slam
