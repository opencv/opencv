// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../odometry/vo_impl.hpp"
#include "sim3.hpp"
#include "sim3_solver.hpp"

namespace cv {
namespace slam {

// Sim3 estimation → window correction → essential graph → map point fusion
bool VisualOdometryImpl::closeLoop(KeyFrame* Kc, KeyFrame* Km,
                                   const std::vector<DMatch>& matches)
{
    if (!params.loopCloseEnable || !Kc || !Km || Kc == Km) return false;

    Sim3Result sr = estimateSim3(Kc, Km, matches, K, /*fixScale*/false,
                                 params.sim3RansacIters,
                                 params.sim3MinInliers,
                                 params.sim3MaxReprojErr2);
    if (!sr.ok)
    {
        String ev = format("loop-close: rejected kf=%d->kf=%d (sim3 %d/%d pairs)",
                           Kc->id, Km->id, sr.nInliers, sr.nPairs);
        lastEvent = lastEvent.empty() ? ev : (lastEvent + " | " + ev);
        CV_LOG_INFO(NULL, "slam " << ev);
        return false;
    }

    const Sim3 ScwCorrected = sim3Compose(sr.Scm, sim3FromPoseCW(Km->poseCw, 1.0));

    // correction window: current keyframe + its covisible neighbours
    std::vector<KeyFrame*> window;
    window.push_back(Kc);
    for (const auto& [nb, cnt] : Kc->orderedCovisibility)
        if (nb && !nb->bad && nb != Km) window.push_back(nb);

    std::map<KeyFrame*, Sim3> correctedScw, nonCorrectedScw;
    const Matx44d Twc = Kc->poseCw.inv();
    correctedScw[Kc]    = ScwCorrected;
    nonCorrectedScw[Kc] = sim3FromPoseCW(Kc->poseCw, 1.0);
    for (KeyFrame* kfi : window)
    {
        if (kfi == Kc) continue;
        const Matx44d Tic = kfi->poseCw * Twc;
        const Sim3 Sic = sim3FromPoseCW(Tic, 1.0);
        correctedScw[kfi]    = sim3Compose(Sic, ScwCorrected);
        nonCorrectedScw[kfi] = sim3FromPoseCW(kfi->poseCw, 1.0);
    }

    // snapshot for rollback if optimisation diverges
    std::map<KeyFrame*, Matx44d>                          savedPose;
    std::map<MapPoint*, std::pair<Point3d, KeyFrame*>>    savedPoint;
    for (KeyFrame* kfi : window) savedPose[kfi] = kfi->poseCw;

    // shift window map points to the corrected side, then apply corrected poses
    std::set<MapPoint*> moved;
    for (KeyFrame* kfi : window)
    {
        const Sim3 corrSwi = sim3Inverse(correctedScw[kfi]);
        const Sim3 siw     = nonCorrectedScw[kfi];
        for (MapPoint* mp : kfi->mapPoints)
        {
            if (!mp || mp->bad || moved.count(mp)) continue;
            savedPoint[mp] = { mp->pos, mp->refKf };
            const Vec3d Pc = sim3Map(corrSwi,
                                     sim3Map(siw, Vec3d(mp->pos.x, mp->pos.y, mp->pos.z)));
            mp->pos    = Point3d(Pc[0], Pc[1], Pc[2]);
            mp->refKf  = kfi;
            moved.insert(mp);
        }
        kfi->poseCw = sim3ToPoseCW(correctedScw[kfi]);
    }

    // distribute drift over the full trajectory via essential graph optimisation
    std::map<KeyFrame*, std::set<KeyFrame*>> loopConnections;
    loopConnections[Kc].insert(Km);

    double chi2 = 0.0;
    const bool ok = Optimizer::OptimizeEssentialGraph(
        map, Km, Kc, nonCorrectedScw, correctedScw, loopConnections,
        /*fixScale*/false, params.essentialGraphIters,
        params.essentialMinCovisWeight, &chi2);

    if (!ok)
    {
        // roll back window
        for (auto& [kfi, pose] : savedPose) kfi->poseCw = pose;
        for (auto& [mp, st]    : savedPoint) { mp->pos = st.first; mp->refKf = st.second; }
        String ev = format("loop-close: rejected kf=%d->kf=%d (graph diverged)",
                           Kc->id, Km->id);
        lastEvent = lastEvent.empty() ? ev : (lastEvent + " | " + ev);
        CV_LOG_INFO(NULL, "slam " << ev);
        return false;
    }

    // fuse matched map points — keep the loop side, track fused set to avoid use-after-free
    int nFused = 0;
    std::set<MapPoint*> fused;
    for (const DMatch& m : matches)
    {
        const size_t qi = (size_t)m.queryIdx, ti = (size_t)m.trainIdx;
        if (ti >= Km->mapPoints.size() || qi >= Kc->mapPoints.size()) continue;
        MapPoint* loopMp = Km->mapPoints[ti];
        if (!loopMp || loopMp->bad || fused.count(loopMp)) continue;
        MapPoint* curMp = Kc->mapPoints[qi];
        if (curMp && fused.count(curMp)) continue; // stale slot, already deleted
        if (curMp && !curMp->bad && curMp != loopMp)
        {
            fused.insert(curMp);
            map.replaceMapPoint(curMp, loopMp);
            ++nFused;
        }
        else if (!curMp)
        {
            map.addObservation(Kc, qi, loopMp);
            ++nFused;
        }
    }

    // register loop edge and rebuild covisibility
    Kc->loopEdges.insert(Km);
    Km->loopEdges.insert(Kc);
    detail::updateCovisibility(Kc);
    detail::updateCovisibility(Km);

    // reset motion model to corrected pose
    lastPoseCw    = Kc->poseCw;
    velocity      = Matx44d::eye();
    hasVelocity   = false;

    // prevFrame may hold pointers into map points just fused or deleted
    hasPrevFrame  = false;

    lastClosedKfId = Kc->id;

    String ev = format("loop-close: CLOSED kf=%d->kf=%d (sim3 inliers=%d/%d s=%.3f, "
                       "window=%d, fused=%d, chi2=%.1f)",
                       Kc->id, Km->id, sr.nInliers, sr.nPairs, sr.Scm.s,
                       (int)window.size(), nFused, chi2);
    lastEvent = lastEvent.empty() ? ev : (lastEvent + " | " + ev);
    CV_LOG_INFO(NULL, "slam " << ev);
    return true;
}

}} // namespace cv::slam
