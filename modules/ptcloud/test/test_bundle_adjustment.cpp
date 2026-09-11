// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "../src/slam/optimizer/optimizer.hpp"

namespace opencv_test { namespace {

// Synthetic scene with exact (noise-free) measurements, so the residual minimum is
// zero and ground truth can be asserted after perturbing the state. The g2o-free
// build keeps the poseOptimization fallback, hence the two unguarded tests.

using namespace cv::slam;

const int  nPoints    = 60;
const int  nKeyframes = 4;
const Size imageSize(640, 480);

// Cameras on an arc looking at the cloud: wide parallax keeps the bundle well
// conditioned (a forward-only baseline is near rank-deficient).
static const double camAnglesDeg[nKeyframes] = { -21.0, -7.0, 7.0, 21.0 };
static const double camRadius = 6.0;

static Mat cameraMatrix()
{
    Mat K = Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 500.0; K.at<double>(1, 1) = 500.0;
    K.at<double>(0, 2) = 320.0; K.at<double>(1, 2) = 240.0;
    return K;
}

static Vec3d gtCenter(int i)
{
    const double a = camAnglesDeg[i] * CV_PI / 180.0;
    return Vec3d(camRadius * std::sin(a), 0.0, -camRadius * std::cos(a));
}

// Look-at pose (world->camera) at gtCenter(i), aimed at the world origin.
static Matx44d gtPose(int i)
{
    const Vec3d C = gtCenter(i);
    Vec3d zc = -C;
    zc /= cv::norm(zc);
    Vec3d xc = Vec3d(0, 1, 0).cross(zc);
    xc /= cv::norm(xc);
    const Vec3d yc = zc.cross(xc);

    const Matx33d R(xc[0], xc[1], xc[2],
                    yc[0], yc[1], yc[2],
                    zc[0], zc[1], zc[2]);
    const Matx31d t = -R * Matx31d(C[0], C[1], C[2]);

    Matx44d T = Matx44d::eye();
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c) T(r, c) = R(r, c);
        T(r, 3) = t(r);
    }
    return T;
}

static Point2d project(const Matx44d& T, const Point3d& X)
{
    const double Xc = T(0,0)*X.x + T(0,1)*X.y + T(0,2)*X.z + T(0,3);
    const double Yc = T(1,0)*X.x + T(1,1)*X.y + T(1,2)*X.z + T(1,3);
    const double Zc = T(2,0)*X.x + T(2,1)*X.y + T(2,2)*X.z + T(2,3);
    return Point2d(500.0 * Xc / Zc + 320.0, 500.0 * Yc / Zc + 240.0);
}

static Point3d cameraCenter(const Matx44d& T)
{
    Matx33d R(T(0,0),T(0,1),T(0,2), T(1,0),T(1,1),T(1,2), T(2,0),T(2,1),T(2,2));
    const Matx31d C = -R.t() * Matx31d(T(0,3), T(1,3), T(2,3));
    return Point3d(C(0), C(1), C(2));
}

#ifdef HAVE_G2O
// Only the g2o-backed tests assert on pose accuracy; without g2o this would be
// an unused static function (-Wunused-function is an error on the CI builders).
static double rotationErrDeg(const Matx44d& A, const Matx44d& B)
{
    Matx33d Ra(A(0,0),A(0,1),A(0,2), A(1,0),A(1,1),A(1,2), A(2,0),A(2,1),A(2,2));
    Matx33d Rb(B(0,0),B(0,1),B(0,2), B(1,0),B(1,1),B(1,2), B(2,0),B(2,1),B(2,2));
    const Matx33d D = Ra * Rb.t();
    const double c = std::max(-1.0, std::min(1.0, (D(0,0) + D(1,1) + D(2,2) - 1.0) * 0.5));
    return std::acos(c) * 180.0 / CV_PI;
}
#endif // HAVE_G2O

// Rotates about the camera's own centre and then shifts that centre by @p dt.
// Rotating about the world origin instead would largely cancel in the image,
// since that is where the cloud sits.
static Matx44d perturbPose(const Matx44d& T, double rotDeg, const Vec3d& dt)
{
    Vec3d axis(0.3, -0.6, 0.74);
    axis /= cv::norm(axis);
    Matx33d dR;
    Rodrigues(Vec3d(axis * (rotDeg * CV_PI / 180.0)), dR);

    Matx33d R(T(0,0),T(0,1),T(0,2), T(1,0),T(1,1),T(1,2), T(2,0),T(2,1),T(2,2));
    const Matx33d Rn = dR * R;
    const Point3d C  = cameraCenter(T);
    const Matx31d tn = -Rn * Matx31d(C.x + dt[0], C.y + dt[1], C.z + dt[2]);

    Matx44d out = Matx44d::eye();
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c) out(r, c) = Rn(r, c);
        out(r, 3) = tn(r);
    }
    return out;
}

// Every landmark observed by every keyframe, at keypoint index == point index.
struct Scene
{
    Mat K = cameraMatrix();
    std::vector<Point3d>   gtPts;
    std::vector<KeyFrame*> kfs;   // owned by the map
    std::vector<MapPoint*> mps;   // owned by the map

    void build(Map& map)
    {
        RNG& rng = theRNG();
        // Wide angular extent and a 2:1 depth spread: that is what separates
        // rotation from translation when only the pose is being solved for.
        gtPts.resize(nPoints);
        for (int i = 0; i < nPoints; ++i)
            gtPts[i] = Point3d(rng.uniform(-2.0, 2.0),
                               rng.uniform(-1.4, 1.4),
                               rng.uniform(-2.0, 2.0));

        for (int f = 0; f < nKeyframes; ++f)
        {
            KeyFrame* kf = new KeyFrame();
            kf->poseCw    = gtPose(f);
            kf->imageSize = imageSize;
            kf->keypoints.reserve(nPoints);
            kf->undistKpts.reserve(nPoints);
            for (int i = 0; i < nPoints; ++i)
            {
                const Point2d p = project(gtPose(f), gtPts[i]);
                kf->keypoints.push_back(KeyPoint(Point2f((float)p.x, (float)p.y), 7.f));
                kf->undistKpts.push_back(Point2f((float)p.x, (float)p.y));
            }
            kf->mapPoints.assign(nPoints, nullptr);
            kfs.push_back(map.addKeyframe(kf));
        }

        for (int i = 0; i < nPoints; ++i)
        {
            MapPoint* mp = new MapPoint();
            mp->pos   = gtPts[i];
            mp->refKf = kfs[0];
            mps.push_back(map.addMapPoint(mp));
            for (int f = 0; f < nKeyframes; ++f)
                map.addObservation(kfs[f], (size_t)i, mps[i]);
        }

        for (int a = 0; a < nKeyframes; ++a)
        {
            for (int b = 0; b < nKeyframes; ++b)
            {
                if (a == b) continue;
                kfs[a]->covisibility[kfs[b]] = nPoints;
                kfs[a]->orderedCovisibility.push_back({ kfs[b], nPoints });
            }
            kfs[a]->parent = (a > 0) ? kfs[a - 1] : nullptr;
        }
    }

    double reprojRmse() const
    {
        double sum = 0.0;
        long   n   = 0;
        for (MapPoint* mp : mps)
        {
            if (mp->bad) continue;
            for (const auto& obs : mp->observations)
            {
                const Point2d p = project(obs.first->poseCw, mp->pos);
                const Point2f& m = obs.first->undistKpts[obs.second];
                sum += (p.x - m.x) * (p.x - m.x) + (p.y - m.y) * (p.y - m.y);
                ++n;
            }
        }
        return n ? std::sqrt(sum / (double)n) : 0.0;
    }

    long numObservations() const
    {
        long n = 0;
        for (MapPoint* mp : mps) n += (long)mp->observations.size();
        return n;
    }

    // Global BA fixes only keyframe 0, so a scaling about its centre stays free.
    double scaleToGt() const
    {
        const Point3d c0 = cameraCenter(kfs[0]->poseCw);
        const double est = cv::norm(cameraCenter(kfs[1]->poseCw) - c0);
        const double gt  = cv::norm(Point3d(gtCenter(1) - gtCenter(0)));
        return (est > 1e-12) ? gt / est : 1.0;
    }

    Point3d toGtFrame(const Point3d& p, double s) const
    {
        const Point3d c0 = cameraCenter(kfs[0]->poseCw);
        return c0 + s * (p - c0);
    }
};

static void makeFrame(const Scene& sc, int camIdx, Frame& frame)
{
    frame.imageSize = imageSize;
    frame.poseCw    = gtPose(camIdx);
    for (int i = 0; i < nPoints; ++i)
    {
        const Point2d p = project(gtPose(camIdx), sc.gtPts[i]);
        frame.keypoints.push_back(KeyPoint(Point2f((float)p.x, (float)p.y), 7.f));
        frame.undistKpts.push_back(Point2f((float)p.x, (float)p.y));
        frame.mapPoints.push_back(sc.mps[i]);
    }
    frame.outliers.assign(frame.mapPoints.size(), true);
}

#ifdef HAVE_G2O

TEST(SLAM_BundleAdjustment, pose_optimization_recovers_perturbed_pose)
{
    Map map;
    Scene sc;
    sc.build(map);

    Frame frame;
    makeFrame(sc, 1, frame);

    // Landmarks are fixed, so the gauge is determined and the pose must come back.
    const Matx44d gt = frame.poseCw;
    frame.poseCw = perturbPose(gt, 1.0, Vec3d(0.02, -0.015, 0.01));

    double errBefore = 0.0;
    for (int i = 0; i < nPoints; ++i)
    {
        const Point2d p = project(frame.poseCw, sc.mps[i]->pos);
        errBefore += cv::norm(p - Point2d(frame.undistKpts[i]));
    }
    errBefore /= nPoints;
    EXPECT_GT(errBefore, 5.0);

    const int nInliers = Optimizer::poseOptimization(frame, sc.K, 4.0, /*enable*/ true);

    EXPECT_GE(nInliers, (int)(0.9 * nPoints));
    EXPECT_LT(rotationErrDeg(frame.poseCw, gt), 0.05);
    EXPECT_LT(cv::norm(cameraCenter(frame.poseCw) - cameraCenter(gt)), 5e-3);

    double errAfter = 0.0;
    for (int i = 0; i < nPoints; ++i)
    {
        const Point2d p = project(frame.poseCw, sc.mps[i]->pos);
        errAfter += cv::norm(p - Point2d(frame.undistKpts[i]));
    }
    errAfter /= nPoints;
    EXPECT_LT(errAfter, 0.1);
    EXPECT_LT(errAfter, errBefore);

    int nFlagged = 0;
    for (size_t i = 0; i < frame.outliers.size(); ++i)
        if (frame.outliers[i]) ++nFlagged;
    EXPECT_EQ(nFlagged, nPoints - nInliers);
}

#endif // HAVE_G2O

// Runs with and without g2o.
TEST(SLAM_BundleAdjustment, pose_optimization_flags_outliers)
{
    Map map;
    Scene sc;
    sc.build(map);

    const int badIdx[]    = { 3, 17, 42 };   // measurement displaced by 30 px
    const int behindIdx[] = { 8, 23 };       // landmark behind the camera
    const int nullIdx[]   = { 11, 30 };      // unmatched keypoint
    const int badMpIdx    = 50;              // soft-deleted landmark

    std::vector<MapPoint*> behindMps;
    for (size_t k = 0; k < sizeof(behindIdx) / sizeof(behindIdx[0]); ++k)
    {
        MapPoint* mp = new MapPoint();
        mp->pos = Point3d(2.0 * gtCenter(1));   // past the camera centre, so Zc < 0
        behindMps.push_back(map.addMapPoint(mp));
    }
    sc.mps[badMpIdx]->bad = true;

    Frame frame;
    makeFrame(sc, 1, frame);
    for (int i : badIdx)  frame.undistKpts[i].x += 30.f;
    for (int i : nullIdx) frame.mapPoints[i] = nullptr;
    for (size_t k = 0; k < behindMps.size(); ++k)
        frame.mapPoints[behindIdx[k]] = behindMps[k];

    const int nExpectedInliers = nPoints - 3 - 2 - 2 - 1;

    auto checkClassification = [&](const Frame& f, int nInliers, const char* tag)
    {
        SCOPED_TRACE(tag);
        EXPECT_EQ(nInliers, nExpectedInliers);

        std::set<int> expectOutlier;
        for (int i : badIdx)    expectOutlier.insert(i);
        for (int i : behindIdx) expectOutlier.insert(i);
        for (int i : nullIdx)   expectOutlier.insert(i);
        expectOutlier.insert(badMpIdx);

        int nFlagged = 0;
        for (int i = 0; i < nPoints; ++i)
        {
            EXPECT_EQ(f.outliers[i], expectOutlier.count(i) > 0) << "index " << i;
            if (f.outliers[i]) ++nFlagged;
        }
        EXPECT_EQ(nFlagged, (int)expectOutlier.size());
    };

    {
        Frame f = frame;
        const int n = Optimizer::poseOptimization(f, sc.K, 4.0, /*enable*/ false);
        checkClassification(f, n, "reprojection fallback");
        EXPECT_DOUBLE_EQ(cv::norm(f.poseCw - gtPose(1)), 0.0);   // must not touch the pose
    }

#ifdef HAVE_G2O
    {
        Frame f = frame;
        const int n = Optimizer::poseOptimization(f, sc.K, 4.0, /*enable*/ true);
        checkClassification(f, n, "g2o pose-only BA");
        // Huber must keep the outliers from dragging the pose off truth.
        EXPECT_LT(rotationErrDeg(f.poseCw, gtPose(1)), 0.05);
        EXPECT_LT(cv::norm(cameraCenter(f.poseCw) - cameraCenter(gtPose(1))), 5e-3);
    }
#endif
}

#ifdef HAVE_G2O

TEST(SLAM_BundleAdjustment, global_ba_reduces_chi2_and_recovers_geometry)
{
    Map map;
    Scene sc;
    sc.build(map);

    // Keyframe 0 is the gauge, so perturb everything else.
    for (int f = 1; f < nKeyframes; ++f)
        sc.kfs[f]->poseCw = perturbPose(gtPose(f), 0.4 * f, Vec3d(0.01 * f, -0.008 * f, 0.006 * f));

    RNG& rng = theRNG();
    for (int i = 0; i < nPoints; ++i)
        sc.mps[i]->pos += Point3d(rng.uniform(-0.05, 0.05),
                                  rng.uniform(-0.05, 0.05),
                                  rng.uniform(-0.05, 0.05));

    const double rmseBefore = sc.reprojRmse();
    EXPECT_GT(rmseBefore, 1.0);

    Optimizer::GlobalBAStats stats;
    Optimizer::globalBundleAdjustment(map, sc.K, /*iterations*/ 20, /*minObservations*/ 2,
                                      /*enable*/ true, /*stopFlag*/ nullptr, &stats);

    ASSERT_TRUE(stats.ran);
    EXPECT_EQ(stats.keyframes, nKeyframes);
    EXPECT_EQ(stats.points, nPoints);
    EXPECT_EQ(stats.observations, (long)nKeyframes * nPoints);
    EXPECT_EQ(stats.posesUpdated, nKeyframes - 1);

    EXPECT_GT(stats.chi2Before, 1.0);
    EXPECT_LT(stats.chi2After, stats.chi2Before);
    EXPECT_LT(stats.chi2After, 1e-2);
    EXPECT_EQ(stats.culled, 0);
    EXPECT_EQ(sc.numObservations(), (long)nKeyframes * nPoints);
    EXPECT_LT(sc.reprojRmse(), 1e-2);

    EXPECT_DOUBLE_EQ(cv::norm(sc.kfs[0]->poseCw - gtPose(0)), 0.0);

    const double s = sc.scaleToGt();
    EXPECT_NEAR(s, 1.0, 0.1);
    for (int f = 1; f < nKeyframes; ++f)
    {
        SCOPED_TRACE(cv::format("keyframe %d", f));
        EXPECT_LT(rotationErrDeg(sc.kfs[f]->poseCw, gtPose(f)), 0.05);
        EXPECT_LT(cv::norm(sc.toGtFrame(cameraCenter(sc.kfs[f]->poseCw), s) - cameraCenter(gtPose(f))), 5e-3);
    }
    double maxPtErr = 0.0;
    for (int i = 0; i < nPoints; ++i)
        maxPtErr = std::max(maxPtErr, cv::norm(sc.toGtFrame(sc.mps[i]->pos, s) - sc.gtPts[i]));
    EXPECT_LT(maxPtErr, 1e-2);
}

TEST(SLAM_BundleAdjustment, local_ba_refines_window_and_fixes_anchors)
{
    Map map;
    Scene sc;
    sc.build(map);

    KeyFrame* newKf = sc.kfs[nKeyframes - 1];

    // With no covisible keyframe outside the window, local BA promotes the two
    // oldest to fixed anchors; leave those at ground truth and perturb the rest.
    sc.kfs[2]->poseCw = perturbPose(gtPose(2), 0.5, Vec3d(0.012, -0.009, 0.007));
    sc.kfs[3]->poseCw = perturbPose(gtPose(3), 0.8, Vec3d(0.018, -0.013, 0.011));

    RNG& rng = theRNG();
    for (int i = 0; i < nPoints; ++i)
        sc.mps[i]->pos += Point3d(rng.uniform(-0.04, 0.04),
                                  rng.uniform(-0.04, 0.04),
                                  rng.uniform(-0.04, 0.04));

    const double rmseBefore = sc.reprojRmse();
    EXPECT_GT(rmseBefore, 1.0);

    Optimizer::localBundleAdjustment(newKf, sc.K, /*enable*/ true);

    EXPECT_DOUBLE_EQ(cv::norm(sc.kfs[0]->poseCw - gtPose(0)), 0.0);
    EXPECT_DOUBLE_EQ(cv::norm(sc.kfs[1]->poseCw - gtPose(1)), 0.0);

    // The anchors pin the scale, so the window returns to absolute ground truth.
    for (int f = 2; f < nKeyframes; ++f)
    {
        SCOPED_TRACE(cv::format("keyframe %d", f));
        EXPECT_LT(rotationErrDeg(sc.kfs[f]->poseCw, gtPose(f)), 0.05);
        EXPECT_LT(cv::norm(cameraCenter(sc.kfs[f]->poseCw) - cameraCenter(gtPose(f))), 5e-3);
    }

    double maxPtErr = 0.0;
    for (int i = 0; i < nPoints; ++i)
        maxPtErr = std::max(maxPtErr, cv::norm(sc.mps[i]->pos - sc.gtPts[i]));
    EXPECT_LT(maxPtErr, 1e-2);

    EXPECT_LT(sc.reprojRmse(), 1e-2);
    EXPECT_LT(sc.reprojRmse(), rmseBefore);
    EXPECT_EQ(sc.numObservations(), (long)nKeyframes * nPoints);
}

#endif // HAVE_G2O

// Runs with and without g2o.
TEST(SLAM_BundleAdjustment, ba_disabled_and_guards_are_noops)
{
    Map map;
    Scene sc;
    sc.build(map);

    for (int f = 1; f < nKeyframes; ++f)
        sc.kfs[f]->poseCw = perturbPose(gtPose(f), 0.5, Vec3d(0.02, -0.01, 0.01));

    std::vector<Matx44d> poses;
    std::vector<Point3d> pts;
    auto snapshot = [&]()
    {
        poses.clear(); pts.clear();
        for (KeyFrame* kf : sc.kfs) poses.push_back(kf->poseCw);
        for (MapPoint* mp : sc.mps) pts.push_back(mp->pos);
    };
    auto expectUnchanged = [&](const char* tag)
    {
        SCOPED_TRACE(tag);
        for (int f = 0; f < nKeyframes; ++f)
            EXPECT_DOUBLE_EQ(cv::norm(sc.kfs[f]->poseCw - poses[f]), 0.0) << "keyframe " << f;
        for (int i = 0; i < nPoints; ++i)
            EXPECT_DOUBLE_EQ(cv::norm(sc.mps[i]->pos - pts[i]), 0.0) << "point " << i;
        EXPECT_EQ(sc.numObservations(), (long)nKeyframes * nPoints);
    };

    snapshot();

    Optimizer::localBundleAdjustment(sc.kfs[nKeyframes - 1], sc.K, /*enable*/ false);
    expectUnchanged("local BA disabled");

    Optimizer::GlobalBAStats stats;
    Optimizer::globalBundleAdjustment(map, sc.K, 10, 2, /*enable*/ false, nullptr, &stats);
    expectUnchanged("global BA disabled");
    EXPECT_FALSE(stats.ran);
    EXPECT_EQ(stats.keyframes, 0);
    EXPECT_EQ(stats.observations, 0);
    EXPECT_DOUBLE_EQ(stats.chi2Before, 0.0);
    EXPECT_DOUBLE_EQ(stats.chi2After, 0.0);

    // No point reaches minObservations.
    Optimizer::globalBundleAdjustment(map, sc.K, 10, /*minObservations*/ 99,
                                      /*enable*/ true, nullptr, &stats);
    expectUnchanged("minObservations too high");
    EXPECT_FALSE(stats.ran);

    // Stop requested before the write-back.
    bool stop = true;
    Optimizer::globalBundleAdjustment(map, sc.K, 10, 2, /*enable*/ true, &stop, &stats);
    expectUnchanged("stop flag set");
    EXPECT_FALSE(stats.ran);
#ifdef HAVE_G2O
    EXPECT_EQ(stats.observations, (long)nKeyframes * nPoints);   // assembled, then aborted
#endif

    // A single keyframe cannot constrain anything.
    Map tiny;
    KeyFrame* lone = new KeyFrame();
    lone->mapPoints.assign(1, nullptr);
    tiny.addKeyframe(lone);
    Optimizer::globalBundleAdjustment(tiny, sc.K, 10, 2, /*enable*/ true, nullptr, &stats);
    EXPECT_FALSE(stats.ran);
}

}} // namespace opencv_test
