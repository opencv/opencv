// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// The slam module has no committed image data, so - as in the geometry module's
// tests - the pipeline is exercised on a synthetic scene: a fixed 3D point cloud
// (generated with theRNG()) projected through a pin-hole camera that translates
// along +X. A stub Feature2D replays the projected keypoints frame by frame and
// a brute-force matcher recovers the ground-truth correspondences, so
// VisualOdometry runs end-to-end without a real detector or image files.

const int  descDim   = 8;
const int  cloudSize = 400;
const Size imageSize(640, 480);

// Camera centres along +X: a wide first baseline for good bootstrap parallax,
// then small uniform steps suited to per-frame tracking.
static const double camCenters[] = { 0.0, 0.8, 1.1, 1.4, 1.7, 2.0 };

static Matx33d cameraMatrix()
{
    return Matx33d(500, 0, 320,
                   0, 500, 240,
                   0,   0,   1);
}

// Each landmark gets a globally-unique, view-invariant descriptor: identical for
// the same 3D point in every frame (so the matcher pairs them at distance 0) and
// >= sqrt(descDim) apart for distinct points (well above descProjThresh, so
// the projection search during tracking never mismatches).
static Mat makeDescriptor(int id)
{
    return Mat(1, descDim, CV_32F, Scalar((double)(id + 1)));
}

// Stub Feature2D: replays pre-computed keypoints/descriptors, one frame per
// detectAndCompute() call (processFrame() invokes the detector exactly once).
class StubDetector CV_FINAL : public Feature2D
{
public:
    struct FrameFeatures { std::vector<KeyPoint> keypoints; Mat descriptors; };
    std::vector<FrameFeatures> frames;
    size_t next = 0;

    void detectAndCompute(InputArray, InputArray,
                          std::vector<KeyPoint>& keypoints,
                          OutputArray descriptors, bool) CV_OVERRIDE
    {
        CV_Assert(next < frames.size());
        keypoints = frames[next].keypoints;
        frames[next].descriptors.copyTo(descriptors);
        ++next;
    }
};

static std::vector<Point3f> makeCloud()
{
    RNG& rng = theRNG();   // seeded per-test by the ts framework -> reproducible
    std::vector<Point3f> pts(cloudSize);
    for (int i = 0; i < cloudSize; i++)
        pts[i] = Point3f(rng.uniform(-2.5f, 2.5f),
                         rng.uniform(-1.8f, 1.8f),
                         rng.uniform( 4.0f, 9.0f));   // varied depth -> non-planar
    return pts;
}

// Projects the cloud through the camera at centre (camX, 0, 0), keeping the
// points that land inside the image. Ground-truth pose is pure translation
// (R = I), so projectPoints takes a zero rotation vector.
static StubDetector::FrameFeatures renderFrame(const std::vector<Point3f>& cloud, double camX)
{
    Vec3d rvec(0, 0, 0), tvec(-camX, 0, 0);   // t = -R*C, with R = I, C = (camX,0,0)
    std::vector<Point2f> proj;
    projectPoints(cloud, rvec, tvec, cameraMatrix(), noArray(), proj);

    StubDetector::FrameFeatures ff;
    std::vector<int> ids;
    for (int i = 0; i < (int)cloud.size(); i++)
    {
        const Point2f& p = proj[i];
        if (p.x < 0 || p.x >= imageSize.width ||
            p.y < 0 || p.y >= imageSize.height) continue;
        ff.keypoints.push_back(KeyPoint(p, 7.f));
        ids.push_back(i);
    }
    ff.descriptors.create((int)ids.size(), descDim, CV_32F);
    for (int r = 0; r < (int)ids.size(); r++)
        makeDescriptor(ids[r]).copyTo(ff.descriptors.row(r));
    return ff;
}

// Builds a VisualOdometry fed by a stub detector pre-loaded with @p nFrames of
// the synthetic sequence and a ground-truth brute-force matcher.
static Ptr<slam::VisualOdometry> makeOdometry(int nFrames,
                                              const slam::OdometryParams& params = slam::OdometryParams())
{
    std::vector<Point3f> cloud = makeCloud();
    Ptr<StubDetector> detector = makePtr<StubDetector>();
    for (int f = 0; f < nFrames; f++)
        detector->frames.push_back(renderFrame(cloud, camCenters[f]));

    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_L2, true);
    return slam::VisualOdometry::create(detector, matcher, "", "",
                                        Mat(cameraMatrix()), noArray(), params);
}

static Mat blankImage() { return Mat::zeros(imageSize, CV_8UC1); }

// Rotation magnitude (deg) of @p T's rotation block relative to identity.
static double rotationFromIdentityDeg(const Matx44d& T)
{
    const double trace = T(0,0) + T(1,1) + T(2,2);
    const double c = std::max(-1.0, std::min(1.0, (trace - 1.0) * 0.5));
    return std::acos(c) * 180.0 / CV_PI;
}

// Camera centre in world coordinates: C = -R^T t.
static Point3d cameraCenter(const Matx44d& T)
{
    Matx33d R(T(0,0),T(0,1),T(0,2), T(1,0),T(1,1),T(1,2), T(2,0),T(2,1),T(2,2));
    Matx31d t(T(0,3), T(1,3), T(2,3));
    Matx31d C = -R.t() * t;
    return Point3d(C(0), C(1), C(2));
}

TEST(SLAM_VisualOdometry, create_validates_arguments)
{
    Ptr<Feature2D> detector = makePtr<StubDetector>();
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_L2, true);
    Mat K(cameraMatrix());

    EXPECT_THROW(slam::VisualOdometry::create(Ptr<Feature2D>(), matcher, "", "", K),
                 cv::Exception);                                  // null detector
    EXPECT_THROW(slam::VisualOdometry::create(detector, Ptr<DescriptorMatcher>(), "", "", K),
                 cv::Exception);                                  // null matcher
    EXPECT_THROW(slam::VisualOdometry::create(detector, matcher, "", "", Mat()),
                 cv::Exception);                                  // empty intrinsics
    EXPECT_THROW(slam::VisualOdometry::create(detector, matcher, "", "", Mat::eye(2, 2, CV_64F)),
                 cv::Exception);                                  // wrong-size intrinsics
}

TEST(SLAM_VisualOdometry, bootstrap_initializes_map)
{
    Ptr<slam::VisualOdometry> vo = makeOdometry(2);
    Mat image = blankImage();

    EXPECT_FALSE(vo->processFrame(image));            // frame 0 -> INITIALIZING
    EXPECT_EQ(vo->getState(), slam::INITIALIZING);

    EXPECT_TRUE(vo->processFrame(image));             // frame 1 -> TRACKING
    EXPECT_EQ(vo->getState(), slam::TRACKING);

    const slam::Map& map = vo->getMap();
    EXPECT_EQ(map.numKeyframes(), 2);
    EXPECT_GE(map.numMapPoints(), 100);               // OdometryParams::minInitPoints
    EXPECT_EQ(vo->getTrajectory().size(), 2u);

    // Reference keyframe is pinned to the world origin.
    const slam::KeyFrame* kfRef = map.getKeyframe(0);
    ASSERT_TRUE(kfRef != nullptr);
    EXPECT_LT(cv::norm(kfRef->poseCw - Matx44d::eye()), 1e-12);

    // Median scene depth is normalized to 1.
    std::vector<double> depths;
    depths.reserve(map.mapPoints().size());
    for (slam::MapPoint* mp : map.mapPoints())
        depths.push_back(mp->pos.z);
    ASSERT_FALSE(depths.empty());
    std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());
    EXPECT_NEAR(depths[depths.size() / 2], 1.0, 1e-2);

    // Second camera: ~no rotation; translation along +X up to scale.
    const slam::KeyFrame* kfCur = map.getKeyframe(1);
    ASSERT_TRUE(kfCur != nullptr);
    EXPECT_LT(rotationFromIdentityDeg(kfCur->poseCw), 2.0);
    Point3d C = cameraCenter(kfCur->poseCw);
    EXPECT_GT(C.x, 0.0);
    EXPECT_LT(std::abs(C.y), 0.2 * std::abs(C.x));
    EXPECT_LT(std::abs(C.z), 0.2 * std::abs(C.x));
}

TEST(SLAM_VisualOdometry, tracks_after_bootstrap)
{
    Ptr<slam::VisualOdometry> vo = makeOdometry(3);
    Mat image = blankImage();

    ASSERT_FALSE(vo->processFrame(image));
    ASSERT_TRUE(vo->processFrame(image));
    ASSERT_EQ(vo->getState(), slam::TRACKING);
    const Point3d cBootstrap = cameraCenter(vo->getLastPose());

    EXPECT_TRUE(vo->processFrame(image));             // frame 2 -> tracked by PnP
    EXPECT_EQ(vo->getState(), slam::TRACKING);
    EXPECT_EQ(vo->getTrajectory().size(), 3u);

    const Matx44d pose = vo->getLastPose();
    EXPECT_LT(rotationFromIdentityDeg(pose), 2.0);
    const Point3d C = cameraCenter(pose);
    EXPECT_GT(C.x, cBootstrap.x);                     // camera keeps advancing +X
    EXPECT_LT(std::abs(C.y), 0.2 * std::abs(C.x));
    EXPECT_LT(std::abs(C.z), 0.2 * std::abs(C.x));
}

TEST(SLAM_VisualOdometry, promotes_keyframes_during_tracking)
{
    slam::OdometryParams params;
    params.kfMaxFrames = 1;   // force a keyframe promotion early in tracking

    Ptr<slam::VisualOdometry> vo = makeOdometry(6, params);
    Mat image = blankImage();

    ASSERT_FALSE(vo->processFrame(image));            // -> INITIALIZING
    ASSERT_TRUE(vo->processFrame(image));             // -> TRACKING (2 keyframes)
    ASSERT_EQ(vo->getMap().numKeyframes(), 2);

    for (int f = 2; f < 6; f++)
        EXPECT_TRUE(vo->processFrame(image));

    EXPECT_EQ(vo->getState(), slam::TRACKING);
    EXPECT_GT(vo->getMap().numKeyframes(), 2);        // new keyframes were promoted

    const slam::KeyFrame* current = vo->getMap().getCurrentKeyframe();
    ASSERT_TRUE(current != nullptr);
    EXPECT_GT(current->id, 1);                        // current keyframe advanced
}

TEST(SLAM_VisualOdometry, reset_clears_state)
{
    Ptr<slam::VisualOdometry> vo = makeOdometry(2);
    Mat image = blankImage();

    vo->processFrame(image);
    vo->processFrame(image);
    ASSERT_EQ(vo->getState(), slam::TRACKING);
    ASSERT_GT(vo->getMap().numKeyframes(), 0);

    vo->reset();
    EXPECT_EQ(vo->getState(), slam::NOT_INITIALIZED);
    EXPECT_EQ(vo->getMap().numKeyframes(), 0);
    EXPECT_EQ(vo->getMap().numMapPoints(), 0);
    EXPECT_TRUE(vo->getTrajectory().empty());
    EXPECT_LT(cv::norm(vo->getLastPose() - Matx44d::eye()), 1e-12);
}

}} // namespace opencv_test
