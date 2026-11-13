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

bool projectPoint(const Matx44d& T_cw, const Mat& K,
                  double Xw, double Yw, double Zw,
                  double& u, double& v)
{
    const double Xc = T_cw(0,0)*Xw + T_cw(0,1)*Yw + T_cw(0,2)*Zw + T_cw(0,3);
    const double Yc = T_cw(1,0)*Xw + T_cw(1,1)*Yw + T_cw(1,2)*Zw + T_cw(1,3);
    const double Zc = T_cw(2,0)*Xw + T_cw(2,1)*Yw + T_cw(2,2)*Zw + T_cw(2,3);
    if (Zc <= 0.0) return false;
    u = K.at<double>(0,0) * Xc / Zc + K.at<double>(0,2);
    v = K.at<double>(1,1) * Yc / Zc + K.at<double>(1,2);
    return true;
}

inline double descDist(const Mat& a, const Mat& b)
{
    if (a.empty() || b.empty()) return std::numeric_limits<double>::max();
    int normType = (a.type() == CV_8U) ? NORM_HAMMING : NORM_L2;
    return norm(a, b, normType);
}

void buildLocalMapPoints(const KeyFrame* kf, int topK,
                         std::set<MapPoint*>& localMps)
{
    if (!kf) return;
    for (MapPoint* mp : kf->mapPoints)
        if (mp && !mp->bad) localMps.insert(mp);

    int k = 0;
    for (const auto& [nbKf, cnt] : kf->orderedCovisibility)
    {
        if (k++ >= topK) break;
        for (MapPoint* mp : nbKf->mapPoints)
            if (mp && !mp->bad) localMps.insert(mp);
    }
}

int runPnP(const std::vector<Point3f>& obj, const std::vector<Point2f>& img,
           Frame& frame,
           const Mat& K, double reprojThresh, int maxIters, double confidence,
           int minInliers)
{
    if ((int)obj.size() < minInliers) return -1;

    Mat rvec, tvec, inlierIdx;
    bool ok = solvePnPRansac(obj, img, K, Mat(),
                              rvec, tvec, false,
                              maxIters, (float)reprojThresh, confidence,
                              inlierIdx, SOLVEPNP_AP3P);
    if (!ok || inlierIdx.rows < minInliers) return -1;

    std::vector<Point3f> objIn;
    std::vector<Point2f> imgIn;
    objIn.reserve(inlierIdx.rows); imgIn.reserve(inlierIdx.rows);
    for (int i = 0; i < inlierIdx.rows; ++i)
    {
        int idx = inlierIdx.at<int>(i);
        objIn.push_back(obj[idx]);
        imgIn.push_back(img[idx]);
    }
    solvePnPRefineLM(objIn, imgIn, K, Mat(), rvec, tvec);

    Mat R;
    Rodrigues(rvec, R);
    frame.poseCw = detail::makePose(R, tvec);
    return inlierIdx.rows;
}

} // anonymous namespace

// Motion model: constant-velocity pose prediction + projected map point search
bool VisualOdometryImpl::trackWithMotionModel(Frame& currentFrame)
{
    if (!lastKf) return false;

    currentFrame.poseCw = velocity * lastPoseCw;

    std::set<MapPoint*> localMps;
    buildLocalMapPoints(lastKf, params.localMapTopK, localMps);

    std::fill(currentFrame.mapPoints.begin(), currentFrame.mapPoints.end(), nullptr);
    std::fill(currentFrame.outliers.begin(), currentFrame.outliers.end(), false);

    auto doSearch = [&](float radius) -> int {
        int n = 0;
        for (MapPoint* mp : localMps)
        {
            double u, v;
            if (!projectPoint(currentFrame.poseCw, K, mp->pos.x, mp->pos.y, mp->pos.z, u, v))
                continue;
            if (u < 0 || u >= currentFrame.imageSize.width ||
                v < 0 || v >= currentFrame.imageSize.height) continue;

            mp->visibleCount++;

            auto cands = currentFrame.getKeypointsInRadius((float)u, (float)v, radius);
            double bestD = std::numeric_limits<double>::max();
            size_t bestI = std::numeric_limits<size_t>::max();
            for (size_t idx : cands)
            {
                if (currentFrame.mapPoints[idx]) continue;
                double d = descDist(mp->refDesc, currentFrame.descriptors.row((int)idx));
                if (d < bestD) { bestD = d; bestI = idx; }
            }
            if (bestI != std::numeric_limits<size_t>::max() &&
                bestD < params.descProjThresh)
            {
                currentFrame.mapPoints[bestI] = mp;
                mp->foundCount++;
                ++n;
            }
        }
        return n;
    };

    int n = doSearch((float)params.motionModelRadius);
    if (n < params.motionModelMinMatches)
    {
        std::fill(currentFrame.mapPoints.begin(), currentFrame.mapPoints.end(), nullptr);
        n = doSearch((float)params.motionModelRadiusWide);
    }

    if (n < params.pnpMinInliers) return false;

    std::vector<Point3f> obj; std::vector<Point2f> img;
    obj.reserve(n); img.reserve(n);
    for (size_t i = 0; i < currentFrame.mapPoints.size(); ++i)
    {
        if (!currentFrame.mapPoints[i]) continue;
        const MapPoint* mp = currentFrame.mapPoints[i];
        obj.push_back(Point3f((float)mp->pos.x,(float)mp->pos.y,(float)mp->pos.z));
        img.push_back(currentFrame.undistKpts[i]);
    }

    int nInliers = runPnP(obj, img, currentFrame, K,
                           params.pnpReprojThresh,
                           params.pnpRansacIters,
                           params.pnpConfidence,
                           params.pnpMinInliers);
    if (nInliers < 0) return false;

    int nOpt = Optimizer::poseOptimization(currentFrame, K, params.pnpReprojThresh, params.poseOptEnable);
    return nOpt >= params.pnpMinInliers;
}

// Fallback 1: descriptor match against the reference keyframe
bool VisualOdometryImpl::trackWithReferenceKF(Frame& currentFrame)
{
    if (!lastKf) return false;

    std::fill(currentFrame.mapPoints.begin(), currentFrame.mapPoints.end(), nullptr);
    std::fill(currentFrame.outliers.begin(), currentFrame.outliers.end(), false);

    std::vector<DMatch> matches;
    matchFrames(lastKf->keypoints, lastKf->descriptors, lastKf->imageSize,
                currentFrame.keypoints, currentFrame.descriptors, currentFrame.imageSize, matches);

    std::vector<Point3f> obj;
    std::vector<Point2f> img;
    std::vector<MapPoint*> corrMps;
    std::vector<int> corrKp;
    obj.reserve(matches.size()); img.reserve(matches.size());
    corrMps.reserve(matches.size()); corrKp.reserve(matches.size());

    for (const auto& m : matches)
    {
        if ((size_t)m.queryIdx >= lastKf->mapPoints.size()) continue;
        MapPoint* mp = lastKf->mapPoints[m.queryIdx];
        if (!mp || mp->bad) continue;
        obj.push_back(Point3f((float)mp->pos.x,(float)mp->pos.y,(float)mp->pos.z));
        img.push_back(currentFrame.undistKpts[m.trainIdx]);
        corrMps.push_back(mp);
        corrKp.push_back(m.trainIdx);
    }

    if ((int)obj.size() < params.pnpMinInliers)
    {
        lastEvent = format("refKF: 2d3d=%d < %d", (int)obj.size(), params.pnpMinInliers);
        return false;
    }

    int nInliers = runPnP(obj, img, currentFrame, K,
                           params.pnpReprojThresh, params.pnpRansacIters,
                           params.pnpConfidence, params.pnpMinInliers);
    if (nInliers < 0)
    {
        lastEvent = "refKF: PnP failed";
        return false;
    }

    for (size_t k = 0; k < corrMps.size(); ++k)
    {
        int kpIdx = corrKp[k];
        if ((size_t)kpIdx < currentFrame.mapPoints.size() && !currentFrame.mapPoints[kpIdx])
            currentFrame.mapPoints[kpIdx] = corrMps[k];
    }

    int nOpt = Optimizer::poseOptimization(currentFrame, K, params.pnpReprojThresh, params.poseOptEnable);
    return nOpt >= params.pnpMinInliers;
}

// Fallback 2: optical flow when descriptor match also fails
bool VisualOdometryImpl::trackWithOpticalFlow(Frame& currentFrame)
{
    if (!hasPrevFrame || prevFrame.image.empty()) return false;

    const Frame& prev = prevFrame;
    if (prev.keypoints.empty()) return false;

    std::vector<Point2f> prevPts;
    prevPts.reserve(prev.keypoints.size());
    for (const auto& kp : prev.keypoints) prevPts.push_back(kp.pt);

    std::vector<Point2f> curPts;
    std::vector<uchar> status;
    std::vector<float> err;

    calcOpticalFlowPyrLK(prev.image, currentFrame.image, prevPts, curPts,
                         status, err, Size(21, 21), 3,
                         TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01));

    std::vector<Point2f> curUndist;
    if (!dist.empty())
        undistortPoints(curPts, curUndist, K, dist, noArray(), K);
    else
        curUndist = curPts;

    std::vector<Point3f> obj; std::vector<Point2f> img;
    obj.reserve(prevPts.size()); img.reserve(prevPts.size());

    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (!status[i]) continue;
        if (i >= prev.mapPoints.size()) continue;
        MapPoint* mp = prev.mapPoints[i];
        if (!mp || mp->bad) continue;
        obj.push_back(Point3f((float)mp->pos.x,(float)mp->pos.y,(float)mp->pos.z));
        img.push_back(curUndist[i]);
    }

    if ((int)obj.size() < params.opticalFlowMinInliers)
    {
        lastEvent = format("optflow: corr=%d < %d", (int)obj.size(), params.opticalFlowMinInliers);
        return false;
    }

    int nInliers = runPnP(obj, img, currentFrame, K,
                            params.pnpReprojThresh, params.pnpRansacIters,
                            params.pnpConfidence, params.opticalFlowMinInliers);
    if (nInliers < 0)
    {
        lastEvent = "optflow: PnP failed";
        return false;
    }

    return true;
}

// Local map refinement: expand coverage + recheck inliers
void VisualOdometryImpl::trackLocalMap(Frame& currentFrame)
{
    if (!lastKf) return;

    std::set<MapPoint*> localMps;
    buildLocalMapPoints(lastKf, params.localMapTopK, localMps);

    int nbK = 0;
    for (const auto& [nbKf, cnt] : lastKf->orderedCovisibility)
    {
        if (nbK++ >= params.localMapTopK) break;
        int nb2K = 0;
        for (const auto& [nb2Kf, cnt2] : nbKf->orderedCovisibility)
        {
            if (nb2K++ >= params.localMapNeighborK) break;
            for (MapPoint* mp : nb2Kf->mapPoints)
                if (mp && !mp->bad) localMps.insert(mp);
        }
    }

    std::set<MapPoint*> alreadyMatched;
    for (MapPoint* mp : currentFrame.mapPoints)
        if (mp) alreadyMatched.insert(mp);

    bool anyNew = false;
    const float r = (float)params.localMapRadius;

    for (MapPoint* mp : localMps)
    {
        if (alreadyMatched.count(mp)) continue;

        double u, v;
        if (!projectPoint(currentFrame.poseCw, K, mp->pos.x, mp->pos.y, mp->pos.z, u, v))
            continue;
        if (u < 0 || u >= currentFrame.imageSize.width ||
            v < 0 || v >= currentFrame.imageSize.height) continue;

        mp->visibleCount++;

        auto cands = currentFrame.getKeypointsInRadius((float)u, (float)v, r);
        double bestD = std::numeric_limits<double>::max();
        size_t bestI = std::numeric_limits<size_t>::max();
        for (size_t idx : cands)
        {
            if (currentFrame.mapPoints[idx]) continue;
            double d = descDist(mp->refDesc, currentFrame.descriptors.row((int)idx));
            if (d < bestD) { bestD = d; bestI = idx; }
        }
        if (bestI != std::numeric_limits<size_t>::max() &&
            bestD < params.descProjThresh)
        {
            currentFrame.mapPoints[bestI] = mp;
            currentFrame.outliers[bestI] = false;
            mp->foundCount++;
            alreadyMatched.insert(mp);
            anyNew = true;
        }
    }

    if (anyNew)
        Optimizer::poseOptimization(currentFrame, K, params.pnpReprojThresh, params.poseOptEnable);
}

bool VisualOdometryImpl::track(Frame& currentFrame)
{
    if (!lastKf)
    {
        refFrame = currentFrame;
        state = INITIALIZING;
        return false;
    }

    bool ok = false;
    if (hasVelocity)
        ok = trackWithMotionModel(currentFrame);
    if (!ok)
        ok = trackWithReferenceKF(currentFrame);
    if (!ok)
        ok = trackWithOpticalFlow(currentFrame);

    if (!ok)
    {
        lastEvent = "track lost: all stages failed";
        refFrame = currentFrame;
        state = INITIALIZING;
        return false;
    }

    trackLocalMap(currentFrame);

    int nInliers = 0;
    for (size_t i = 0; i < currentFrame.mapPoints.size(); ++i)
        if (currentFrame.mapPoints[i] && !currentFrame.outliers[i]) ++nInliers;

    {
        Matx44d Tcw_last_inv = lastPoseCw.inv();
        velocity = currentFrame.poseCw * Tcw_last_inv;
        hasVelocity = true;
    }

    lastPoseCw = currentFrame.poseCw;
    map.appendPose(currentFrame.poseCw);
    frameRecords.push_back({ currentFrame.poseCw * lastKf->poseCw.inv(), lastKf });
    ++framesSinceKf;

    String kf_reason;
    if (shouldPromoteKeyframe(nInliers, currentFrame.poseCw, kf_reason))
    {
        int mpBefore = map.numMapPoints();
        promoteKeyframeAndGrowMap(currentFrame); // also runs local BA + detectLoop/closeLoop
        lastPoseCw = lastKf->poseCw;
        frameRecords.back() = { Matx44d::eye(), lastKf };

        String kfEv = format("keyframe: %s, +%d mp%s",
                             kf_reason.c_str(),
                             map.numMapPoints() - mpBefore,
#ifdef HAVE_G2O
                             params.localBaEnable ? " (LocalBA)" : "");
#else
                             "");
#endif
        lastEvent = lastEvent.empty() ? kfEv : (kfEv + " | " + lastEvent);
    }

    prevFrame = currentFrame;
    hasPrevFrame = true;

    return true;
}

}} // namespace cv::slam
