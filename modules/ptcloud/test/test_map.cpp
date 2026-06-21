// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// Allocates a keyframe with @p n keypoint slots (parallel mapPoints[] zeroed),
// as the pipeline expects before any observation is wired.
static slam::KeyFrame* newKeyFrame(int n)
{
    slam::KeyFrame* kf = new slam::KeyFrame();
    kf->keypoints.resize(n);
    kf->mapPoints.assign(n, nullptr);
    return kf;
}

TEST(SLAM_Map, registers_and_clears)
{
    slam::Map map;

    // addKeyframe / addMapPoint assign sequential ids and the index round-trips.
    slam::KeyFrame* kf0 = map.addKeyframe(newKeyFrame(0));
    slam::KeyFrame* kf1 = map.addKeyframe(newKeyFrame(0));
    slam::MapPoint* mp0 = map.addMapPoint(new slam::MapPoint());
    EXPECT_EQ(kf0->id, 0);
    EXPECT_EQ(kf1->id, 1);
    EXPECT_EQ(mp0->id, 0);
    EXPECT_EQ(map.numKeyframes(), 2);
    EXPECT_EQ(map.numMapPoints(), 1);
    EXPECT_EQ(map.getKeyframe(1), kf1);
    EXPECT_EQ(map.getMapPoint(0), mp0);
    EXPECT_TRUE(map.getKeyframe(99) == nullptr);

    // An explicit id must advance the auto-id counter so it cannot collide.
    slam::KeyFrame* kfExplicit = new slam::KeyFrame();
    kfExplicit->id = 5;
    map.addKeyframe(kfExplicit);
    EXPECT_EQ(map.addKeyframe(newKeyFrame(0))->id, 6);

    // clear() drops all state and restarts the id counters.
    map.clear();
    EXPECT_EQ(map.numKeyframes(), 0);
    EXPECT_EQ(map.numMapPoints(), 0);
    EXPECT_EQ(map.addKeyframe(newKeyFrame(0))->id, 0);
}

TEST(SLAM_Map, wires_and_removes_observations)
{
    slam::Map map;
    slam::KeyFrame* kfA = map.addKeyframe(newKeyFrame(3));
    slam::KeyFrame* kfB = map.addKeyframe(newKeyFrame(3));
    slam::MapPoint* mp  = map.addMapPoint(new slam::MapPoint());

    // addObservation links both directions: keyframe slot <-> observation map.
    map.addObservation(kfA, 2, mp);
    map.addObservation(kfB, 0, mp);
    EXPECT_EQ(kfA->mapPoints[2], mp);
    EXPECT_EQ(mp->observations[kfA], 2u);
    EXPECT_EQ(mp->observations.size(), 2u);

    // Adding into an already-occupied slot is a no-op.
    slam::MapPoint* other = map.addMapPoint(new slam::MapPoint());
    map.addObservation(kfA, 2, other);
    EXPECT_EQ(kfA->mapPoints[2], mp);
    EXPECT_EQ(other->observations.count(kfA), 0u);

    // removeMapPoint unlinks every observing keyframe and drops the point.
    const int mpId = mp->id;
    map.removeMapPoint(mp);   // deletes mp; must not be dereferenced afterwards
    EXPECT_TRUE(map.getMapPoint(mpId) == nullptr);
    EXPECT_TRUE(kfA->mapPoints[2] == nullptr);
    EXPECT_TRUE(kfB->mapPoints[0] == nullptr);
}

}} // namespace opencv_test
