// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// =============================================================================
// Map tests
// =============================================================================

TEST(Map, AddAndRetrieveKeyframe)
{
    cv::slam::Map map;

    cv::slam::KeyFrame kf;
    kf.pose_cw = cv::Matx44d::eye();

    int id = map.addKeyframe(kf);
    EXPECT_GE(id, 0);
    EXPECT_EQ(kf.id, id);

    cv::slam::KeyFrame* retrieved = map.getKeyframe(id);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->id, id);
    EXPECT_EQ(map.numKeyframes(), 1);
}

TEST(Map, AddAndRetrieveMapPoint)
{
    cv::slam::Map map;

    cv::slam::MapPoint mp;
    mp.pos = cv::Point3d(1.0, 2.0, 3.0);

    int id = map.addMapPoint(mp);
    EXPECT_GE(id, 0);
    EXPECT_EQ(mp.id, id);

    cv::slam::MapPoint* retrieved = map.getMapPoint(id);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_DOUBLE_EQ(retrieved->pos.x, 1.0);
    EXPECT_DOUBLE_EQ(retrieved->pos.y, 2.0);
    EXPECT_DOUBLE_EQ(retrieved->pos.z, 3.0);
    EXPECT_FALSE(retrieved->bad);
    EXPECT_EQ(map.numMapPoints(), 1);
}

TEST(Map, RemoveMapPointCleansObservations)
{
    cv::slam::Map map;

    // Add a keyframe with one keypoint slot.
    cv::slam::KeyFrame kf;
    kf.pose_cw  = cv::Matx44d::eye();
    kf.mappoints.assign(1, nullptr);
    kf.kpt_to_mp.assign(1, -1);
    int kf_id = map.addKeyframe(kf);

    // Add a map point.
    cv::slam::MapPoint mp;
    mp.pos = cv::Point3d(0, 0, 1);
    int mp_id = map.addMapPoint(mp);

    cv::slam::KeyFrame* kf_ptr = map.getKeyframe(kf_id);
    cv::slam::MapPoint* mp_ptr = map.getMapPoint(mp_id);
    ASSERT_NE(kf_ptr, nullptr);
    ASSERT_NE(mp_ptr, nullptr);

    // Wire the observation.
    map.addObservation(kf_ptr, 0, mp_ptr);
    EXPECT_EQ(mp_ptr->observations.size(), 1u);
    EXPECT_EQ(kf_ptr->mappoints[0], mp_ptr);

    // Remove the map point and verify cleanup.
    map.removeMapPoint(mp_id);
    EXPECT_TRUE(mp_ptr->bad);
    EXPECT_EQ(kf_ptr->mappoints[0], nullptr);
    EXPECT_EQ(kf_ptr->kpt_to_mp[0], -1);
    EXPECT_EQ(map.numMapPoints(), 0);
}

TEST(Map, TrajectoryAppendAndRetrieve)
{
    cv::slam::Map map;

    cv::Matx44d T1 = cv::Matx44d::eye();
    cv::Matx44d T2 = cv::Matx44d::eye();
    T2(0, 3) = 1.0;  // 1-unit translation

    map.appendPose(T1);
    map.appendPose(T2);

    const auto& traj = map.trajectory();
    ASSERT_EQ(traj.size(), 2u);
    EXPECT_DOUBLE_EQ(traj[1](0, 3), 1.0);
}

TEST(Map, ClearResetsState)
{
    cv::slam::Map map;

    cv::slam::KeyFrame kf;
    kf.pose_cw = cv::Matx44d::eye();
    map.addKeyframe(kf);

    cv::slam::MapPoint mp;
    mp.pos = cv::Point3d(1, 2, 3);
    map.addMapPoint(mp);

    map.appendPose(cv::Matx44d::eye());
    map.clear();

    EXPECT_EQ(map.numKeyframes(), 0);
    EXPECT_EQ(map.numMapPoints(), 0);
    EXPECT_TRUE(map.trajectory().empty());
}

// =============================================================================
// OdometryParams tests
// =============================================================================

TEST(OdometryParams, DefaultValues)
{
    cv::slam::OdometryParams p;
    EXPECT_EQ(p.min_init_inliers,           80);
    EXPECT_DOUBLE_EQ(p.min_init_parallax_deg, 3.0);
    EXPECT_EQ(p.min_init_points,            50);
    EXPECT_DOUBLE_EQ(p.hf_ratio_thresh,     0.45);
    EXPECT_DOUBLE_EQ(p.min_growth_parallax_deg, 1.0);
    EXPECT_DOUBLE_EQ(p.essential_ransac_thresh, 1.0);
    EXPECT_DOUBLE_EQ(p.essential_ransac_confidence, 0.999);
}

}} // namespace opencv_test
