// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "optimizer.hpp"

namespace cv { namespace slam {

// reprojection-only inlier check; fallback for poseOptimization without g2o
static int poseOptimizationReproj(Frame& frame, const Mat& K, double reprojThresh)
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

        const double u  = fx * Xc / Zc + cx;
        const double v  = fy * Yc / Zc + cy;
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

// g2o pose-only BA

#ifdef HAVE_G2O

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace cv { namespace slam {

// real 6-DoF pose-only bundle adjustment
static int poseOptimizationG2O(Frame& frame, const Mat& K)
{
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    using Block   = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
    using LSolver = g2o::LinearSolverDense<Block::PoseMatrixType>;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(
        new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<Block>(std::make_unique<LSolver>())));

    const Matx44d& T = frame.poseCw;
    Eigen::Matrix3d R_eig;
    R_eig << T(0,0), T(0,1), T(0,2),
             T(1,0), T(1,1), T(1,2),
             T(2,0), T(2,1), T(2,2);
    const Eigen::Vector3d t_eig(T(0,3), T(1,3), T(2,3));

    auto* vPose = new g2o::VertexSE3Expmap();
    vPose->setId(0);
    vPose->setFixed(false);
    vPose->setEstimate(g2o::SE3Quat(Eigen::Quaterniond(R_eig).normalized(), t_eig));
    optimizer.addVertex(vPose);

    constexpr double kHuberDelta = 2.4495; // sqrt(5.991)
    constexpr double kChi2Thresh = 5.991;
    constexpr int    kIters      = 4;

    const int N = static_cast<int>(frame.mapPoints.size());
    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> edges(N, nullptr);

    for (int i = 0; i < N; ++i)
    {
        frame.outliers[i] = true;
        MapPoint* mp = frame.mapPoints[i];
        if (!mp || mp->bad) continue;

        auto* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
        e->Xw = Eigen::Vector3d(mp->pos.x, mp->pos.y, mp->pos.z);
        e->fx = fx; e->fy = fy; e->cx = cx; e->cy = cy;

        e->setId(i + 1);
        e->setVertex(0, vPose);
        e->setMeasurement(Eigen::Vector2d(frame.undistKpts[i].x,
                                          frame.undistKpts[i].y));
        e->setInformation(Eigen::Matrix2d::Identity());

        auto* rk = new g2o::RobustKernelHuber();
        rk->setDelta(kHuberDelta);
        e->setRobustKernel(rk);

        optimizer.addEdge(e);
        edges[i] = e;
    }

    // Pass 1: all edges, Huber
    optimizer.initializeOptimization(0);
    optimizer.optimize(kIters);

    for (int i = 0; i < N; ++i)
    {
        auto* e = edges[i];
        if (!e) continue;
        const bool good = (e->chi2() < kChi2Thresh) && e->isDepthPositive();
        if (good) { e->setLevel(0); e->setRobustKernel(nullptr); }
        else      { e->setLevel(1); }
    }

    // Pass 2: inliers only, L2
    {
        int nPass2 = 0;
        for (int i = 0; i < N; ++i)
            if (edges[i] && edges[i]->level() == 0) ++nPass2;
        if (nPass2 > 0)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(kIters);
        }
    }

    // Write refined pose back
    {
        const g2o::SE3Quat est = vPose->estimate();
        const Eigen::Quaterniond q = est.rotation();
        const Eigen::Vector3d    t = est.translation();
        const Eigen::Matrix3d    R = q.toRotationMatrix();
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c) frame.poseCw(r, c) = R(r, c);
            frame.poseCw(r, 3) = t[r];
        }
        frame.poseCw(3, 0) = frame.poseCw(3, 1) = frame.poseCw(3, 2) = 0.0;
        frame.poseCw(3, 3) = 1.0;
    }

    int nInliers = 0;
    for (int i = 0; i < N; ++i)
    {
        auto* e = edges[i];
        if (!e) continue;
        const bool good = (e->level() == 0)
                       && (e->chi2() < kChi2Thresh)
                       && e->isDepthPositive();
        frame.outliers[i] = !good;
        if (good) ++nInliers;
    }
    return nInliers;
}

}} // namespace cv::slam
#endif // HAVE_G2O

namespace cv { namespace slam {

int Optimizer::poseOptimization(Frame& frame, const Mat& K, double reprojThresh, bool enable)
{
#ifdef HAVE_G2O
    if (enable)
        return poseOptimizationG2O(frame, K);
#else
    (void)enable;
#endif
    return poseOptimizationReproj(frame, K, reprojThresh);
}

}} // namespace cv::slam
