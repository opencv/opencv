// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

// Tests for the vendored g2o subset in 3rdparty/g2o that the SLAM optimizers build
// on. Nothing from cv::slam is used here. Empty unless built with -DWITH_G2O=ON.

#ifdef HAVE_G2O

#include <g2o/core/block_solver.h>
#include <g2o/core/jacobian_workspace.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace opencv_test { namespace {

using g2o::SE3Quat;
using g2o::Sim3;
using g2o::Vector6;
using g2o::Vector7;

const double fx = 500.0, fy = 500.0, cx = 320.0, cy = 240.0;

using Block   = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
using LSolver = g2o::LinearSolverEigen<Block::PoseMatrixType>;

static void setLevenberg(g2o::SparseOptimizer& optimizer)
{
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(
        new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<Block>(std::make_unique<LSolver>())));
}

// g2o order: rotation first, then translation.
static Vector6 se3Vec(double wx, double wy, double wz, double ux, double uy, double uz)
{
    Vector6 v;
    v << wx, wy, wz, ux, uy, uz;
    return v;
}

static Eigen::Vector2d pinhole(const Eigen::Vector3d& Xc)
{
    return Eigen::Vector2d(fx * Xc[0] / Xc[2] + cx, fy * Xc[1] / Xc[2] + cy);
}

// Camera on an arc around the world origin, looking at it. Wide parallax keeps the
// bundle well conditioned (a forward-only baseline is near rank-deficient).
static SE3Quat lookAtPose(double angleDeg, double radius = 6.0)
{
    const double a = angleDeg * CV_PI / 180.0;
    const Eigen::Vector3d C(radius * std::sin(a), 0.0, -radius * std::cos(a));
    const Eigen::Vector3d zc = (-C).normalized();
    const Eigen::Vector3d xc = Eigen::Vector3d(0, 1, 0).cross(zc).normalized();
    const Eigen::Vector3d yc = zc.cross(xc);
    Eigen::Matrix3d R;
    R.row(0) = xc; R.row(1) = yc; R.row(2) = zc;
    return SE3Quat(Eigen::Quaterniond(R), Eigen::Vector3d(-R * C));
}

TEST(G2O_SE3Quat, exp_log_and_composition)
{
    const std::vector<Vector6> samples = {
        se3Vec(0, 0, 0, 0, 0, 0),                       // identity
        se3Vec(1e-9, -2e-9, 3e-9, 0.5, -0.25, 1.0),     // below the series cutoff
        se3Vec(0.2, -0.35, 0.1, 1.5, 0.5, -2.0),
        se3Vec(1.6, -1.9, 1.4, -0.3, 2.0, 0.7),         // theta ~ 2.8 rad, near pi
    };

    for (size_t i = 0; i < samples.size(); ++i)
    {
        SCOPED_TRACE(cv::format("sample %d", (int)i));
        const Vector6& v = samples[i];
        const SE3Quat T = SE3Quat::exp(v);

        const Vector6 v2 = T.log();
        EXPECT_LT((v2 - v).cwiseAbs().maxCoeff(), 1e-9);

        const Eigen::Matrix4d M = T.to_homogeneous_matrix();
        EXPECT_LT((M.row(3) - Eigen::Vector4d(0, 0, 0, 1).transpose()).cwiseAbs().maxCoeff(), 1e-15);

        // Rotation block == matrix exponential of skew(omega).
        const Eigen::Vector3d omega = v.head<3>();
        const double theta = omega.norm();
        Eigen::Matrix3d Rexp = Eigen::Matrix3d::Identity();
        if (theta > 1e-12)
            Rexp = Eigen::AngleAxisd(theta, omega / theta).toRotationMatrix();
        EXPECT_LT((M.topLeftCorner<3, 3>() - Rexp).cwiseAbs().maxCoeff(), 1e-12);

        EXPECT_LT((M.topLeftCorner<3, 3>() * M.topLeftCorner<3, 3>().transpose()
                   - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff(), 1e-12);
        EXPECT_NEAR((M.topLeftCorner<3, 3>().determinant()), 1.0, 1e-12);

        const Eigen::Vector3d p(0.7, -1.3, 4.2);
        const Eigen::Vector3d mapped = M.topLeftCorner<3, 3>() * p + M.topRightCorner<3, 1>();
        EXPECT_LT((T.map(p) - mapped).cwiseAbs().maxCoeff(), 1e-12);

        EXPECT_LT((T.inverse().to_homogeneous_matrix() - M.inverse()).cwiseAbs().maxCoeff(), 1e-12);

        const SE3Quat T2(Eigen::Quaterniond(M.topLeftCorner<3, 3>()),
                         Eigen::Vector3d(M.topRightCorner<3, 1>()));
        EXPECT_LT((T2.to_homogeneous_matrix() - M).cwiseAbs().maxCoeff(), 1e-12);
    }

    const SE3Quat A = SE3Quat::exp(samples[2]);
    const SE3Quat B = SE3Quat::exp(samples[3]);
    EXPECT_LT(((A * B).to_homogeneous_matrix()
               - A.to_homogeneous_matrix() * B.to_homogeneous_matrix()).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT(((A * A.inverse()).to_homogeneous_matrix()
               - Eigen::Matrix4d::Identity()).cwiseAbs().maxCoeff(), 1e-12);

    const Eigen::Vector3d p(-0.4, 0.9, 3.1);
    EXPECT_LT(((A * B).map(p) - A.map(B.map(p))).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(G2O_EdgeSE3ProjectXYZ, error_and_jacobians_match_numeric)
{
    g2o::SparseOptimizer optimizer;
    setLevenberg(optimizer);

    const SE3Quat pose = SE3Quat::exp(se3Vec(0.05, -0.12, 0.03, 0.4, -0.2, 0.1));
    const Eigen::Vector3d Xw(0.6, -0.4, 6.0);

    auto* vp = new g2o::VertexSBAPointXYZ();
    vp->setId(0);
    vp->setEstimate(Xw);
    optimizer.addVertex(vp);

    auto* vc = new g2o::VertexSE3Expmap();
    vc->setId(1);
    vc->setEstimate(pose);
    optimizer.addVertex(vc);

    // Offset from the true projection, so the jacobians are evaluated away from the minimum.
    const Eigen::Vector2d trueProj = pinhole(pose.map(Xw));
    const Eigen::Vector2d obs = trueProj + Eigen::Vector2d(1.7, -2.3);

    auto* e = new g2o::EdgeSE3ProjectXYZ();
    e->fx = fx; e->fy = fy; e->cx = cx; e->cy = cy;
    e->setVertex(0, vp);
    e->setVertex(1, vc);
    e->setMeasurement(obs);
    e->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(e);

    EXPECT_LT((e->cam_project(pose.map(Xw)) - trueProj).cwiseAbs().maxCoeff(), 1e-12);

    e->computeError();
    EXPECT_LT((e->error() - Eigen::Vector2d(1.7, -2.3)).cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_NEAR(e->chi2(), 1.7 * 1.7 + 2.3 * 2.3, 1e-9);
    EXPECT_TRUE(e->isDepthPositive());

    // The workspace overload is hidden by this type's linearizeOplus() override;
    // it maps _jacobianOplusXi/Xj onto real storage before evaluating.
    g2o::JacobianWorkspace jw;
    jw.updateSize(e);
    ASSERT_TRUE(jw.allocate());
    static_cast<g2o::OptimizableGraph::Edge*>(e)->linearizeOplus(jw);
    const Eigen::Matrix<double, 2, 3> Ji = e->jacobianOplusXi();
    const Eigen::Matrix<double, 2, 6> Jj = e->jacobianOplusXj();

    // Central differences through oplus, i.e. in the vertices' own parameterisation.
    // Templated on the concrete vertex type: taking the base pointer instead makes gcc
    // speculatively devirtualize push()/pop() to the wrong vertex and warn about it.
    const double h = 1e-7;
    auto errorAfterOplus = [&](auto* v, const std::vector<double>& d)
    {
        const auto backup = v->estimate();
        v->oplus(d.data());
        e->computeError();
        const Eigen::Vector2d err = e->error();
        v->setEstimate(backup);
        return err;
    };

    for (int k = 0; k < 3; ++k)
    {
        std::vector<double> dp(3, 0.0), dm(3, 0.0);
        dp[k] = h; dm[k] = -h;
        const Eigen::Vector2d num = (errorAfterOplus(vp, dp) - errorAfterOplus(vp, dm)) / (2 * h);
        for (int r = 0; r < 2; ++r)
            EXPECT_NEAR(Ji(r, k), num[r], 1e-4 * std::max(1.0, std::abs(num[r])))
                << "jacobianOplusXi(" << r << "," << k << ")";
    }

    for (int k = 0; k < 6; ++k)
    {
        std::vector<double> dp(6, 0.0), dm(6, 0.0);
        dp[k] = h; dm[k] = -h;
        const Eigen::Vector2d num = (errorAfterOplus(vc, dp) - errorAfterOplus(vc, dm)) / (2 * h);
        for (int r = 0; r < 2; ++r)
            EXPECT_NEAR(Jj(r, k), num[r], 1e-4 * std::max(1.0, std::abs(num[r])))
                << "jacobianOplusXj(" << r << "," << k << ")";
    }

    vp->setEstimate(Eigen::Vector3d(0.6, -0.4, -6.0));
    EXPECT_FALSE(e->isDepthPositive());
}

TEST(G2O_Optimizer, levenberg_recovers_perturbed_bundle)
{
    const int nPoses  = 3;
    const int nPoints = 20;

    RNG& rng = theRNG();

    std::vector<SE3Quat> gtPoses;
    for (int i = 0; i < nPoses; ++i)
        gtPoses.push_back(lookAtPose(-20.0 + 20.0 * i));
    std::vector<Eigen::Vector3d> gtPts(nPoints);
    for (int i = 0; i < nPoints; ++i)
        gtPts[i] = Eigen::Vector3d(rng.uniform(-1.0, 1.0),
                                   rng.uniform(-1.0, 1.0),
                                   rng.uniform(-1.0, 1.0));

    g2o::SparseOptimizer optimizer;
    setLevenberg(optimizer);

    // Two fixed poses remove the gauge freedom including scale, so ground truth is
    // the unique minimiser.
    for (int i = 0; i < nPoses; ++i)
    {
        auto* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        v->setFixed(i < 2);
        v->setEstimate(i < 2 ? gtPoses[i]
                             : SE3Quat::exp(se3Vec(0.01, 0.02, -0.015, 0.05, -0.03, 0.04)) * gtPoses[i]);
        optimizer.addVertex(v);
    }
    for (int i = 0; i < nPoints; ++i)
    {
        auto* v = new g2o::VertexSBAPointXYZ();
        v->setId(nPoses + i);
        v->setMarginalized(true);
        v->setEstimate(gtPts[i] + Eigen::Vector3d(rng.uniform(-0.05, 0.05),
                                                  rng.uniform(-0.05, 0.05),
                                                  rng.uniform(-0.05, 0.05)));
        optimizer.addVertex(v);
    }
    for (int i = 0; i < nPoints; ++i)
    {
        for (int c = 0; c < nPoses; ++c)
        {
            auto* e = new g2o::EdgeSE3ProjectXYZ();
            e->fx = fx; e->fy = fy; e->cx = cx; e->cy = cy;
            e->setVertex(0, optimizer.vertex(nPoses + i));
            e->setVertex(1, optimizer.vertex(c));
            e->setMeasurement(pinhole(gtPoses[c].map(gtPts[i])));   // noise-free
            e->setInformation(Eigen::Matrix2d::Identity());
            optimizer.addEdge(e);
        }
    }

    ASSERT_TRUE(optimizer.initializeOptimization());
    EXPECT_EQ(optimizer.activeEdges().size(), (size_t)(nPoses * nPoints));
    optimizer.computeActiveErrors();
    const double chi2Before = optimizer.activeChi2();
    EXPECT_GT(chi2Before, 1e3);

    optimizer.optimize(30);
    optimizer.computeActiveErrors();
    const double chi2After = optimizer.activeChi2();

    EXPECT_LT(chi2After, 1e-8);
    EXPECT_LT(chi2After, chi2Before * 1e-9);

    const auto* vFree = static_cast<const g2o::VertexSE3Expmap*>(optimizer.vertex(nPoses - 1));
    EXPECT_LT((vFree->estimate().to_homogeneous_matrix()
               - gtPoses[nPoses - 1].to_homogeneous_matrix()).cwiseAbs().maxCoeff(), 1e-5);

    double maxPtErr = 0.0;
    for (int i = 0; i < nPoints; ++i)
    {
        const auto* v = static_cast<const g2o::VertexSBAPointXYZ*>(optimizer.vertex(nPoses + i));
        maxPtErr = std::max(maxPtErr, (v->estimate() - gtPts[i]).cwiseAbs().maxCoeff());
    }
    EXPECT_LT(maxPtErr, 1e-4);

    for (int i = 0; i < 2; ++i)
    {
        const auto* v = static_cast<const g2o::VertexSE3Expmap*>(optimizer.vertex(i));
        EXPECT_DOUBLE_EQ((v->estimate().to_homogeneous_matrix()
                          - gtPoses[i].to_homogeneous_matrix()).cwiseAbs().maxCoeff(), 0.0);
    }
}

TEST(G2O_Sim3, algebra_and_pose_graph_closes_loop)
{
    const Eigen::Matrix3d R1 = Eigen::AngleAxisd(0.3, Eigen::Vector3d(0.2, -0.8, 0.5).normalized()).toRotationMatrix();
    const Eigen::Matrix3d R2 = Eigen::AngleAxisd(-0.7, Eigen::Vector3d(0.9, 0.1, -0.4).normalized()).toRotationMatrix();
    const Eigen::Vector3d t1(0.5, -1.2, 0.3), t2(-0.8, 0.4, 2.0);
    const double s1 = 1.7, s2 = 0.6;

    const Sim3 S1(Eigen::Quaterniond(R1), t1, s1);
    const Sim3 S2(Eigen::Quaterniond(R2), t2, s2);
    const Eigen::Vector3d p(0.9, -0.5, 3.0);

    EXPECT_LT((S1.map(p) - (s1 * (R1 * p) + t1)).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_DOUBLE_EQ(S1.scale(), s1);

    const Sim3 S12 = S1 * S2;
    EXPECT_NEAR(S12.scale(), s1 * s2, 1e-12);
    EXPECT_LT((S12.map(p) - S1.map(S2.map(p))).cwiseAbs().maxCoeff(), 1e-12);

    const Sim3 Sid = S1 * S1.inverse();
    EXPECT_NEAR(Sid.scale(), 1.0, 1e-12);
    EXPECT_LT((Sid.rotation().toRotationMatrix() - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT(Sid.translation().cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((S1.inverse().map(S1.map(p)) - p).cwiseAbs().maxCoeff(), 1e-12);

    Vector7 xi;
    xi << 0.2, -0.1, 0.35, 0.7, -1.1, 0.4, 0.25;   // omega, upsilon, sigma
    const Sim3 Sv(xi);
    EXPECT_NEAR(Sv.scale(), std::exp(xi[6]), 1e-12);
    EXPECT_LT((Sv.log() - xi).cwiseAbs().maxCoeff(), 1e-9);

    // Four consistent poses, three exact odometry edges, and a loop edge closing
    // 3 -> 0 with a 20% scale drift plus a small rotation error.
    const int nNodes = 4;
    std::vector<Sim3> gt;
    for (int i = 0; i < nNodes; ++i)
    {
        const Eigen::Matrix3d R =
            Eigen::AngleAxisd(0.4 * i, Eigen::Vector3d(0.1, 0.9, -0.3).normalized()).toRotationMatrix();
        gt.push_back(Sim3(Eigen::Quaterniond(R), Eigen::Vector3d(0.5 * i, 0.1 * i, -0.2 * i), 1.0));
    }
    const Sim3 drift(Eigen::Quaterniond(
                         Eigen::AngleAxisd(0.05, Eigen::Vector3d(0.3, 0.2, 0.9).normalized())),
                     Eigen::Vector3d(0.02, -0.01, 0.03), 1.2);

    struct Result { double chi2Before, chi2After, maxEdgeChi2, maxScaleDev; };
    auto runGraph = [&](bool fixScale) -> Result
    {
        g2o::SparseOptimizer optimizer;
        optimizer.setVerbose(false);
        using SBlock   = g2o::BlockSolver<g2o::BlockSolverTraits<7, 3>>;
        using SLSolver = g2o::LinearSolverEigen<SBlock::PoseMatrixType>;
        auto* algo = new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<SBlock>(std::make_unique<SLSolver>()));
        algo->setUserLambdaInit(1e-16);
        optimizer.setAlgorithm(algo);

        for (int i = 0; i < nNodes; ++i)
        {
            auto* v = new g2o::VertexSim3Expmap();
            v->setId(i);
            v->setEstimate(gt[i]);
            v->setFixed(i == 0);
            v->setMarginalized(false);
            v->_fix_scale = fixScale;
            optimizer.addVertex(v);
        }

        // The measurement for edge (i, j) is S_j * S_i^-1.
        auto addEdge = [&](int i, int j, const Sim3& meas)
        {
            auto* e = new g2o::EdgeSim3();
            e->setVertex(0, optimizer.vertex(i));
            e->setVertex(1, optimizer.vertex(j));
            e->setMeasurement(meas);
            e->information() = Eigen::Matrix<double, 7, 7>::Identity();
            optimizer.addEdge(e);
        };
        for (int i = 0; i + 1 < nNodes; ++i)
            addEdge(i, i + 1, gt[i + 1] * gt[i].inverse());
        addEdge(nNodes - 1, 0, drift * (gt[0] * gt[nNodes - 1].inverse()));

        EXPECT_TRUE(optimizer.initializeOptimization());
        optimizer.computeActiveErrors();
        Result r;
        r.chi2Before = optimizer.activeChi2();
        optimizer.optimize(30);
        optimizer.computeActiveErrors();
        r.chi2After = optimizer.activeChi2();

        r.maxEdgeChi2 = 0.0;
        for (auto* e : optimizer.activeEdges())
            r.maxEdgeChi2 = std::max(r.maxEdgeChi2, e->chi2());

        r.maxScaleDev = 0.0;
        for (int i = 0; i < nNodes; ++i)
        {
            const auto* v = static_cast<const g2o::VertexSim3Expmap*>(optimizer.vertex(i));
            r.maxScaleDev = std::max(r.maxScaleDev, std::abs(v->estimate().scale() - 1.0));
        }
        return r;
    };

    {
        SCOPED_TRACE("scale free");
        const Result r = runGraph(/*fixScale*/ false);
        EXPECT_GT(r.chi2Before, 1e-3);
        EXPECT_LT(r.chi2After, 0.5 * r.chi2Before);
        EXPECT_LT(r.maxEdgeChi2, 0.9 * r.chi2After);   // drift spread over the loop
        EXPECT_GT(r.maxScaleDev, 0.01);                // nodes absorb part of it
    }
    {
        SCOPED_TRACE("scale fixed");
        const Result r = runGraph(/*fixScale*/ true);
        EXPECT_LT(r.chi2After, r.chi2Before);
        EXPECT_DOUBLE_EQ(r.maxScaleDev, 0.0);
    }
}

}} // namespace opencv_test

#endif // HAVE_G2O
