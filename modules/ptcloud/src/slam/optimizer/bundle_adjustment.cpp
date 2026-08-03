// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "optimizer.hpp"

// g2o bundle adjustment backends

#ifdef HAVE_G2O

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

// Sim(3) types for the loop-closure essential graph.
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#define OPENCV_SLAM_HAVE_G2O_SIM3 1

#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace cv { namespace slam {

// Binary reprojection edge for local/global BA: vertex 0 is the 3D world
// point, vertex 1 the camera pose (variable or fixed); both move.
class EdgeSE3ProjectXYZ :
    public g2o::BaseBinaryEdge<2, Eigen::Vector2d,
                               g2o::VertexSBAPointXYZ,
                               g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectXYZ(double fx, double fy, double cx, double cy)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
    bool read(std::istream&)        { return false; }
    bool write(std::ostream&) const { return false; }
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

    void computeError() CV_OVERRIDE
    {
        const auto* vp = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const auto* vT = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const Eigen::Vector3d Xc = vT->estimate().map(vp->estimate());
        _error = _measurement - Eigen::Vector2d(fx_ * Xc[0] / Xc[2] + cx_,
                                                 fy_ * Xc[1] / Xc[2] + cy_);
    }

    bool isDepthPositive() const
    {
        const auto* vp = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const auto* vT = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        return vT->estimate().map(vp->estimate())[2] > 0.0;
    }

    void linearizeOplus() CV_OVERRIDE
    {
        const auto* vp = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const auto* vT = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const Eigen::Vector3d Xc = vT->estimate().map(vp->estimate());
        const double x = Xc[0], y = Xc[1];
        const double invz = 1.0 / Xc[2], iz2 = invz * invz;
        const Eigen::Matrix3d R = vT->estimate().rotation().toRotationMatrix();

        for (int j = 0; j < 3; ++j)
        {
            _jacobianOplusXi(0, j) = -fx_ * invz * R(0,j) + fx_ * x * iz2 * R(2,j);
            _jacobianOplusXi(1, j) = -fy_ * invz * R(1,j) + fy_ * y * iz2 * R(2,j);
        }

        _jacobianOplusXj(0, 0) =  x * y * iz2 * fx_;
        _jacobianOplusXj(0, 1) = -(1.0 + x * x * iz2) * fx_;
        _jacobianOplusXj(0, 2) =  y * invz * fx_;
        _jacobianOplusXj(0, 3) = -invz * fx_;
        _jacobianOplusXj(0, 4) =  0.0;
        _jacobianOplusXj(0, 5) =  x * iz2 * fx_;

        _jacobianOplusXj(1, 0) =  (1.0 + y * y * iz2) * fy_;
        _jacobianOplusXj(1, 1) = -x * y * iz2 * fy_;
        _jacobianOplusXj(1, 2) = -x * invz * fy_;
        _jacobianOplusXj(1, 3) =  0.0;
        _jacobianOplusXj(1, 4) = -invz * fy_;
        _jacobianOplusXj(1, 5) =  y * iz2 * fy_;
    }

private:
    double fx_, fy_, cx_, cy_;
};

static void localBundleAdjustmentG2O(KeyFrame* newKf, const Mat& K, bool* stopFlag)
{
    if (!newKf) return;

    const double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);

    constexpr int    kMaxLocalKFs  = 10;
    constexpr double kChi2Thresh   = 5.991;
    constexpr double kHuberDelta   = 2.4495;
    constexpr int    kItersCoarse  = 5;
    constexpr int    kItersFine    = 10;

    // local KFs: newKf + top-K covisible
    std::vector<KeyFrame*> localKfs;
    std::set<KeyFrame*>    localKfSet;
    localKfs.push_back(newKf);
    localKfSet.insert(newKf);

    int k = 0;
    for (const auto& [nbKf, cnt] : newKf->orderedCovisibility)
    {
        if (k++ >= kMaxLocalKFs) break;
        if (!nbKf) continue;
        localKfs.push_back(nbKf);
        localKfSet.insert(nbKf);
    }
    if (localKfs.size() < 2) return;

    // local MPs: all seen by any local KF
    std::vector<MapPoint*> localMps;
    std::set<MapPoint*>    localMpSet;
    for (KeyFrame* kf : localKfs)
        for (MapPoint* mp : kf->mapPoints)
            if (mp && !mp->bad && !localMpSet.count(mp))
            { localMps.push_back(mp); localMpSet.insert(mp); }

    if (localMps.empty()) return;

    // fixed KFs: observe local MPs but lie outside the window
    std::set<KeyFrame*> fixedKfSet;
    for (MapPoint* mp : localMps)
        for (const auto& [obsKf, kpIdx] : mp->observations)
            if (!localKfSet.count(obsKf))
                fixedKfSet.insert(obsKf);

    // monocular gauge needs >= 2 anchors; promote oldest local KFs if short
    if (fixedKfSet.size() < 2)
    {
        std::vector<KeyFrame*> byAge = localKfs;
        std::sort(byAge.begin(), byAge.end(),
                  [](const KeyFrame* a, const KeyFrame* b){ return a->id < b->id; });
        for (KeyFrame* kf : byAge)
        {
            if (fixedKfSet.size() >= 2) break;
            if (kf == newKf) continue;
            localKfSet.erase(kf);
            fixedKfSet.insert(kf);
        }
    }

    int maxKFid = 0;
    for (KeyFrame* kf : localKfs)    maxKFid = std::max(maxKFid, kf->id);
    for (KeyFrame* kf : fixedKfSet)  maxKFid = std::max(maxKFid, kf->id);

    using Block   = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
    using LSolver = g2o::LinearSolverEigen<Block::PoseMatrixType>;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(
        new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<Block>(std::make_unique<LSolver>())));
    if (stopFlag) optimizer.setForceStopFlag(stopFlag);

    auto toSE3 = [](const KeyFrame* kf) -> g2o::SE3Quat {
        const Matx44d& T = kf->poseCw;
        Eigen::Matrix3d R;
        R << T(0,0), T(0,1), T(0,2),
             T(1,0), T(1,1), T(1,2),
             T(2,0), T(2,1), T(2,2);
        return g2o::SE3Quat(Eigen::Quaterniond(R).normalized(),
                            Eigen::Vector3d(T(0,3), T(1,3), T(2,3)));
    };

    for (KeyFrame* kf : localKfs)
    {
        if (!localKfSet.count(kf)) continue;
        auto* v = new g2o::VertexSE3Expmap();
        v->setId(kf->id);
        v->setEstimate(toSE3(kf));
        v->setFixed(false);
        optimizer.addVertex(v);
    }
    for (KeyFrame* kf : fixedKfSet)
    {
        auto* v = new g2o::VertexSE3Expmap();
        v->setId(kf->id);
        v->setEstimate(toSE3(kf));
        v->setFixed(true);
        optimizer.addVertex(v);
    }

    for (MapPoint* mp : localMps)
    {
        auto* v = new g2o::VertexSBAPointXYZ();
        v->setId(mp->id + maxKFid + 1);
        v->setEstimate(Eigen::Vector3d(mp->pos.x, mp->pos.y, mp->pos.z));
        v->setMarginalized(true);
        optimizer.addVertex(v);
    }

    struct EdgeRec { EdgeSE3ProjectXYZ* e; KeyFrame* kf; MapPoint* mp; size_t kpIdx; };
    std::vector<EdgeRec> recs;
    recs.reserve(localMps.size() * 4);

    for (MapPoint* mp : localMps)
    {
        const int ptVid = mp->id + maxKFid + 1;
        for (const auto& [obsKf, kpIdx] : mp->observations)
        {
            if (!localKfSet.count(obsKf) && !fixedKfSet.count(obsKf)) continue;
            if (kpIdx >= obsKf->undistKpts.size()) continue;

            auto* e = new EdgeSE3ProjectXYZ(fx, fy, cx, cy);
            e->setVertex(0, optimizer.vertex(ptVid));
            e->setVertex(1, optimizer.vertex(obsKf->id));
            e->setMeasurement(Eigen::Vector2d(obsKf->undistKpts[kpIdx].x,
                                              obsKf->undistKpts[kpIdx].y));
            e->setInformation(Eigen::Matrix2d::Identity());
            auto* rk = new g2o::RobustKernelHuber();
            rk->setDelta(kHuberDelta);
            e->setRobustKernel(rk);
            optimizer.addEdge(e);
            recs.push_back({e, obsKf, mp, kpIdx});
        }
    }

    if (recs.empty()) return;
    if (stopFlag && *stopFlag) return;

    // pass 1: coarse, Huber
    optimizer.initializeOptimization();
    optimizer.optimize(kItersCoarse);

    for (auto& r : recs)
    {
        const bool bad = (r.e->chi2() > kChi2Thresh) || !r.e->isDepthPositive();
        r.e->setLevel(bad ? 1 : 0);
        if (!bad) r.e->setRobustKernel(nullptr);
    }

    // pass 2: fine, L2, inliers only
    if (stopFlag && *stopFlag) return;
    {
        int nPass2 = 0;
        for (auto& r : recs)
            if (r.e->level() == 0) ++nPass2;
        if (nPass2 > 0)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(kItersFine);
        }
    }

    // write back KF poses (finite-guarded)
    for (KeyFrame* kf : localKfs)
    {
        if (!localKfSet.count(kf)) continue;
        auto* v = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(kf->id));
        if (!v) continue;
        const Eigen::Quaterniond q = v->estimate().rotation();
        const Eigen::Vector3d    t = v->estimate().translation();
        if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2])) continue;
        const Eigen::Matrix3d    R = q.toRotationMatrix();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) kf->poseCw(r, c) = R(r, c);
            kf->poseCw(r, 3) = t[r];
        }
        kf->poseCw(3, 0) = kf->poseCw(3, 1) = kf->poseCw(3, 2) = 0.0;
        kf->poseCw(3, 3) = 1.0;
    }

    // write back MP positions (finite-guarded)
    for (MapPoint* mp : localMps)
    {
        auto* v = static_cast<g2o::VertexSBAPointXYZ*>(
            optimizer.vertex(mp->id + maxKFid + 1));
        if (!v) continue;
        const auto est = v->estimate();
        if (std::isfinite(est[0]) && std::isfinite(est[1]) && std::isfinite(est[2]))
            mp->pos = cv::Point3d(est[0], est[1], est[2]);
    }

    // erase final outlier KF-MP links
    for (auto& r : recs)
    {
        if (r.e->level() == 0) continue;
        if (r.kpIdx < r.kf->mapPoints.size())
            r.kf->mapPoints[r.kpIdx] = nullptr;
        r.mp->observations.erase(r.kf);
        if (r.mp->observations.empty())
            r.mp->bad = true;
    }
}

static void globalBundleAdjustmentG2O(Map& map, const Mat& K, int iters,
                                      int minObs, bool* stopFlag,
                                      Optimizer::GlobalBAStats* stats)
{
    const double fx = K.at<double>(0,0), fy = K.at<double>(1,1);
    const double cx = K.at<double>(0,2), cy = K.at<double>(1,2);

    constexpr double kChi2Thresh = 5.991;
    constexpr double kHuberDelta = 2.4495;

    // collect non-bad keyframes; fix the lowest-id (gauge)
    std::vector<KeyFrame*> kfs;
    int minId = -1;
    for (KeyFrame* kf : map.keyframes())
        if (kf && !kf->bad)
        {
            kfs.push_back(kf);
            if (minId < 0 || kf->id < minId) minId = kf->id;
        }
    if (kfs.size() < 2) return;

    int maxKFid = 0;
    for (KeyFrame* kf : kfs) maxKFid = std::max(maxKFid, kf->id);

    // collect map points with enough observations
    std::vector<MapPoint*> mps;
    for (MapPoint* mp : map.mapPoints())
    {
        if (!mp || mp->bad) continue;
        int nobs = 0;
        for (const auto& [obsKf, kpIdx] : mp->observations)
            if (obsKf && !obsKf->bad) ++nobs;
        if (nobs >= std::max(2, minObs)) mps.push_back(mp);
    }
    if (mps.empty()) return;

    using Block   = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
    using LSolver = g2o::LinearSolverEigen<Block::PoseMatrixType>;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(
        new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<Block>(std::make_unique<LSolver>())));
    if (stopFlag) optimizer.setForceStopFlag(stopFlag);

    auto toSE3 = [](const KeyFrame* kf) -> g2o::SE3Quat {
        const Matx44d& T = kf->poseCw;
        Eigen::Matrix3d R;
        R << T(0,0), T(0,1), T(0,2),
             T(1,0), T(1,1), T(1,2),
             T(2,0), T(2,1), T(2,2);
        return g2o::SE3Quat(Eigen::Quaterniond(R).normalized(),
                            Eigen::Vector3d(T(0,3), T(1,3), T(2,3)));
    };

    for (KeyFrame* kf : kfs)
    {
        auto* v = new g2o::VertexSE3Expmap();
        v->setId(kf->id);
        v->setEstimate(toSE3(kf));
        v->setFixed(kf->id == minId);
        optimizer.addVertex(v);
    }

    // MP vertices + reprojection edges
    struct EdgeRec { EdgeSE3ProjectXYZ* e; KeyFrame* kf; MapPoint* mp; size_t kpIdx; };
    std::vector<EdgeRec> recs;
    recs.reserve(mps.size() * 3);
    size_t nEdges = 0;

    for (MapPoint* mp : mps)
    {
        const int pid = mp->id + maxKFid + 1;
        auto* vp = new g2o::VertexSBAPointXYZ();
        vp->setId(pid);
        vp->setEstimate(Eigen::Vector3d(mp->pos.x, mp->pos.y, mp->pos.z));
        vp->setMarginalized(true);
        optimizer.addVertex(vp);

        for (const auto& [obsKf, kpIdx] : mp->observations)
        {
            if (!obsKf || obsKf->bad) continue;
            if (!optimizer.vertex(obsKf->id)) continue;
            if (kpIdx >= obsKf->undistKpts.size()) continue;

            auto* e = new EdgeSE3ProjectXYZ(fx, fy, cx, cy);
            e->setVertex(0, optimizer.vertex(pid));
            e->setVertex(1, optimizer.vertex(obsKf->id));
            e->setMeasurement(Eigen::Vector2d(obsKf->undistKpts[kpIdx].x,
                                              obsKf->undistKpts[kpIdx].y));
            e->setInformation(Eigen::Matrix2d::Identity());
            auto* rk = new g2o::RobustKernelHuber();
            rk->setDelta(kHuberDelta);
            e->setRobustKernel(rk);
            optimizer.addEdge(e);
            recs.push_back({e, obsKf, mp, kpIdx});
            ++nEdges;
        }
    }
    if (recs.empty()) return;

    if (stats)
    {
        stats->keyframes    = (int)kfs.size();
        stats->points       = (int)mps.size();
        stats->observations = (long)nEdges;
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    const double chi2Before = optimizer.activeRobustChi2();
    optimizer.optimize(std::max(1, iters));
    optimizer.computeActiveErrors();
    const double chi2After = optimizer.activeRobustChi2();
    if (stats) { stats->chi2Before = chi2Before; stats->chi2After = chi2After; }

    CV_LOG_INFO(NULL, "slam global BA: " << kfs.size() << " keyframes, "
                      << mps.size() << " points, " << nEdges << " observations | "
                      << "reproj chi2 " << chi2Before << " -> " << chi2After);

    if (stopFlag && *stopFlag) return;

    // write back KF poses (finite-guarded)
    int nPoseWrites = 0;
    for (KeyFrame* kf : kfs)
    {
        if (kf->id == minId) continue;
        auto* v = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(kf->id));
        if (!v) continue;
        const g2o::SE3Quat est = v->estimate();
        const Eigen::Vector3d t = est.translation();
        if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2])) continue;
        const Eigen::Matrix3d R = est.rotation().toRotationMatrix();
        Matx44d T = Matx44d::eye();
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c) T(r, c) = R(r, c);
            T(r, 3) = t[r];
        }
        kf->poseCw = T;
        ++nPoseWrites;
    }

    // write back MP positions (finite-guarded)
    for (MapPoint* mp : mps)
    {
        auto* v = static_cast<g2o::VertexSBAPointXYZ*>(
            optimizer.vertex(mp->id + maxKFid + 1));
        if (!v) continue;
        const auto est = v->estimate();
        if (std::isfinite(est[0]) && std::isfinite(est[1]) && std::isfinite(est[2]))
            mp->pos = cv::Point3d(est[0], est[1], est[2]);
    }

    // cull surviving outlier observations
    int nCulled = 0;
    for (auto& r : recs)
    {
        if (r.e->chi2() <= kChi2Thresh && r.e->isDepthPositive()) continue;
        if (r.kpIdx < r.kf->mapPoints.size() && r.kf->mapPoints[r.kpIdx] == r.mp)
            r.kf->mapPoints[r.kpIdx] = nullptr;
        r.mp->observations.erase(r.kf);
        if (r.mp->observations.empty()) r.mp->bad = true;
        ++nCulled;
    }
    if (stats) { stats->posesUpdated = nPoseWrites; stats->culled = nCulled; stats->ran = true; }
    CV_LOG_INFO(NULL, "slam global BA: updated " << nPoseWrites
                      << " poses, culled " << nCulled << " outlier observations");
}

// Sim(3) essential-graph optimisation (loop closure); compiled only when
// g2o Sim(3) types are available.

#if OPENCV_SLAM_HAVE_G2O_SIM3

static g2o::Sim3 toG2oSim3(const Sim3& S)
{
    Eigen::Matrix3d R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R(i,j) = S.R(i,j);
    return g2o::Sim3(R, Eigen::Vector3d(S.t[0], S.t[1], S.t[2]), S.s);
}

static Sim3 fromG2oSim3(const g2o::Sim3& g)
{
    Sim3 S;
    const Eigen::Matrix3d R = g.rotation().toRotationMatrix();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) S.R(i,j) = R(i,j);
    const Eigen::Vector3d t = g.translation();
    S.t = Vec3d(t[0], t[1], t[2]);
    S.s = g.scale();
    return S;
}

static bool optimizeEssentialGraphG2O(
    Map& map, KeyFrame* loopKf, KeyFrame* curKf,
    const std::map<KeyFrame*, Sim3>& nonCorrectedScw,
    const std::map<KeyFrame*, Sim3>& correctedScw,
    const std::map<KeyFrame*, std::set<KeyFrame*>>& loopConnections,
    bool fixScale, int iterations, int minFeat, double* outFinalChi2)
{
    using Block   = g2o::BlockSolver<g2o::BlockSolverTraits<7,3>>;
    using LSolver = g2o::LinearSolverEigen<Block::PoseMatrixType>;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    auto* algo = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<Block>(std::make_unique<LSolver>()));
    algo->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(algo);

    const std::set<KeyFrame*>& kfs = map.keyframes();

    int minId = -1;
    for (KeyFrame* kf : kfs)
        if (kf && !kf->bad && (minId < 0 || kf->id < minId)) minId = kf->id;

    const Eigen::Matrix<double,7,7> I7 = Eigen::Matrix<double,7,7>::Identity();

    std::map<KeyFrame*, Sim3> vScw;
    std::map<KeyFrame*, Sim3> vCorrSwc;

    // Vertices
    for (KeyFrame* kf : kfs)
    {
        if (!kf || kf->bad) continue;
        Sim3 init;
        auto it = correctedScw.find(kf);
        init = (it != correctedScw.end()) ? it->second
                                          : sim3FromPoseCW(kf->poseCw, 1.0);
        vScw[kf] = init;

        auto* v = new g2o::VertexSim3Expmap();
        v->setEstimate(toG2oSim3(init));
        v->setFixed(kf->id == minId);
        v->setId(kf->id);
        v->setMarginalized(false);
        v->_fix_scale = fixScale;
        optimizer.addVertex(v);
    }

    auto poseOf = [&](KeyFrame* kf, const std::map<KeyFrame*, Sim3>& nc) -> Sim3 {
        auto it = nc.find(kf);
        return (it != nc.end()) ? it->second : vScw[kf];
    };
    auto addEdge = [&](int idi, int idj, const Sim3& Sji) {
        auto* e = new g2o::EdgeSim3();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(idi)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(idj)));
        e->setMeasurement(toG2oSim3(Sji));
        e->information() = I7;
        optimizer.addEdge(e);
    };

    std::set<std::pair<int,int>> inserted;
    auto edgeKey = [](int a, int b) { return std::make_pair(std::min(a,b), std::max(a,b)); };

    // Loop edges (from corrected poses)
    for (const auto& [kf, conns] : loopConnections)
    {
        if (!kf || kf->bad || !vScw.count(kf)) continue;
        const Sim3 Siw = vScw[kf];
        for (KeyFrame* kfj : conns)
        {
            if (!kfj || kfj->bad || !vScw.count(kfj)) continue;
            const bool isMain = (kf == curKf && kfj == loopKf);
            int w = 0;
            auto wit = kf->covisibility.find(kfj);
            if (wit != kf->covisibility.end()) w = wit->second;
            if (!isMain && w < minFeat) continue;
            addEdge(kf->id, kfj->id, sim3Compose(vScw[kfj], sim3Inverse(Siw)));
            inserted.insert(edgeKey(kf->id, kfj->id));
        }
    }
    // Guarantee the explicit cur<->loop edge
    if (curKf && loopKf && !curKf->bad && !loopKf->bad &&
        vScw.count(curKf) && vScw.count(loopKf) &&
        !inserted.count(edgeKey(curKf->id, loopKf->id)))
    {
        addEdge(curKf->id, loopKf->id,
                sim3Compose(vScw[loopKf], sim3Inverse(vScw[curKf])));
        inserted.insert(edgeKey(curKf->id, loopKf->id));
    }

    // Structural edges (spanning tree + covisibility, from non-corrected poses)
    for (KeyFrame* kf : kfs)
    {
        if (!kf || kf->bad || !vScw.count(kf)) continue;
        const Sim3 Swi = sim3Inverse(poseOf(kf, nonCorrectedScw));

        if (kf->parent && !kf->parent->bad && vScw.count(kf->parent))
            addEdge(kf->id, kf->parent->id,
                    sim3Compose(poseOf(kf->parent, nonCorrectedScw), Swi));

        for (KeyFrame* pLKF : kf->loopEdges)
        {
            if (!pLKF || pLKF->bad || !vScw.count(pLKF) || pLKF->id >= kf->id) continue;
            addEdge(kf->id, pLKF->id,
                    sim3Compose(poseOf(pLKF, nonCorrectedScw), Swi));
        }

        for (const auto& [kfn, w] : kf->orderedCovisibility)
        {
            if (w < minFeat) break;
            if (!kfn || kfn->bad || !vScw.count(kfn)) continue;
            if (kfn == kf->parent || kf->children.count(kfn) ||
                kf->loopEdges.count(kfn)) continue;
            if (kfn->id >= kf->id) continue;
            if (inserted.count(edgeKey(kf->id, kfn->id))) continue;
            addEdge(kf->id, kfn->id,
                    sim3Compose(poseOf(kfn, nonCorrectedScw), Swi));
        }
    }

    // Solve
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    const double err0 = optimizer.activeRobustChi2();
    optimizer.optimize(std::max(1, iterations));
    optimizer.computeActiveErrors();
    const double errEnd = optimizer.activeRobustChi2();
    if (outFinalChi2) *outFinalChi2 = errEnd;
    CV_LOG_INFO(NULL, "slam loop: essential graph chi2 " << err0 << " -> " << errEnd);

    if (!std::isfinite(errEnd)) return false;

    // Write back KF poses (Sim3 -> SE3, t/=s)
    for (KeyFrame* kf : kfs)
    {
        if (!kf || kf->bad || !vScw.count(kf)) continue;
        auto* v = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(kf->id));
        if (!v) continue;
        const Sim3 Siw = fromG2oSim3(v->estimate());
        if (!(Siw.s > 0.0) || !std::isfinite(Siw.s) ||
            !std::isfinite(Siw.t[0]) || !std::isfinite(Siw.t[1]) || !std::isfinite(Siw.t[2]))
            continue;
        vCorrSwc[kf] = sim3Inverse(Siw);
        kf->poseCw   = sim3ToPoseCW(Siw);
    }

    // Correct map points via their reference keyframe (Stage C)
    for (MapPoint* mp : map.mapPoints())
    {
        if (!mp || mp->bad) continue;
        KeyFrame* ref = mp->refKf;
        if (!ref || ref->bad) continue;
        auto itB = vScw.find(ref), itA = vCorrSwc.find(ref);
        if (itB == vScw.end() || itA == vCorrSwc.end()) continue;
        const Vec3d Pc = sim3Map(itA->second,
                                 sim3Map(itB->second, Vec3d(mp->pos.x, mp->pos.y, mp->pos.z)));
        if (std::isfinite(Pc[0]) && std::isfinite(Pc[1]) && std::isfinite(Pc[2]))
            mp->pos = Point3d(Pc[0], Pc[1], Pc[2]);
    }
    return true;
}
#endif // OPENCV_SLAM_HAVE_G2O_SIM3

}} // namespace cv::slam
#endif // HAVE_G2O


namespace cv { namespace slam {

void Optimizer::LocalBundleAdjustment(KeyFrame* newKf, const Mat& K, bool enable, bool* stopFlag)
{
    #ifdef HAVE_G2O
        if (enable)
            localBundleAdjustmentG2O(newKf, K, stopFlag);
    #else
        (void)newKf; (void)K; (void)enable; (void)stopFlag;
    #endif
}

bool Optimizer::OptimizeEssentialGraph(
    Map& map, KeyFrame* loopKf, KeyFrame* curKf,
    const std::map<KeyFrame*, Sim3>& nonCorrectedScw,
    const std::map<KeyFrame*, Sim3>& correctedScw,
    const std::map<KeyFrame*, std::set<KeyFrame*>>& loopConnections,
    bool fixScale, int iterations, int minCovisWeight, double* outFinalChi2)
{
#if defined(HAVE_G2O) && OPENCV_SLAM_HAVE_G2O_SIM3
    return optimizeEssentialGraphG2O(map, loopKf, curKf, nonCorrectedScw,
                                     correctedScw, loopConnections, fixScale,
                                     iterations, minCovisWeight, outFinalChi2);
#else
    (void)map; (void)loopKf; (void)curKf; (void)nonCorrectedScw;
    (void)correctedScw; (void)loopConnections; (void)fixScale;
    (void)iterations; (void)minCovisWeight;
    if (outFinalChi2) *outFinalChi2 = 0.0;
    return false;
#endif
}

void Optimizer::GlobalBundleAdjustment(Map& map, const Mat& K, int iterations,
                                       int minObservations, bool enable, bool* stopFlag,
                                       GlobalBAStats* stats)
{
    if (stats) *stats = GlobalBAStats{};
#ifdef HAVE_G2O
    if (enable)
        globalBundleAdjustmentG2O(map, K, iterations, minObservations, stopFlag, stats);
#else
    (void)map; (void)K; (void)iterations; (void)minObservations;
    (void)enable; (void)stopFlag;
#endif
}

}} // namespace cv::slam
