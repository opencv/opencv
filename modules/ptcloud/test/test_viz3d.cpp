// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/highgui.hpp>            // updateWindow / destroyWindow
#include "../src/viz3d/grid_ticks.hpp"    // GL-free grid spacing helper

namespace opencv_test { namespace {

using namespace cv;

// Regression for the grid tick_step infinite-loop hang (#1): the old code did
// logf(1.0) == 0 -> division -> inf -> the halving loop never terminated for
// distance*scale > 4. gridTickStep must terminate and stay finite/positive for
// every distance, including the region that used to hang. GL-free (runs in CI).
TEST(Viz3D, grid_tick_step)
{
    // Values span below/at/above the old hang threshold (dist_scale > 4).
    const float samples[] = { 0.03f, 0.9f, 3.0f, 4.0f, 5.0f, 13.4f, 40.0f, 400.0f, 4000.0f };
    for (float ds : samples)
    {
        float ts = viz3d::detail::gridTickStep(ds);
        ASSERT_TRUE(std::isfinite(ts)) << "ds=" << ds;
        ASSERT_GT(ts, 0.0f) << "ds=" << ds;
        float ratio = ds / ts;
        EXPECT_GE(ratio, 2.0f - 1e-3f) << "ds=" << ds << " ratio=" << ratio;
        EXPECT_LE(ratio, 4.0f + 1e-3f) << "ds=" << ds << " ratio=" << ratio;
    }
}

// viz3d is OpenGL/window based. Skip when no GL context can be created
// (e.g. headless CI without a display).
static bool viz3dAvailable()
{
    try
    {
        viz3d::showPoints("viz3d_gl_probe", "p", Mat::zeros(4, 6, CV_32F));
        destroyWindow("viz3d_gl_probe");
        return true;
    }
    catch (const cv::Exception&)
    {
        return false;
    }
}

// Smoke test for the public viz3d API. Builds a scene and forces redraws; passes
// if nothing throws. Exercises the paths touched by review fixes: #3 (showRGBD),
// #4 (degenerate up-vector trajectory), and the grid render path. The #1 hang
// itself is distance-triggered (no public camera-distance setter) and is covered
// by the GL-free Viz3D.grid_tick_step test above.
TEST(Viz3D, render_scene_smoke)
{
    if (!viz3dAvailable())
        throw cvtest::SkipTestException("viz3d/OpenGL not available (no GL context)");

    const String w = "viz3d_test";

    Mat pts(256, 6, CV_32F);
    randu(pts, 0.0f, 1.0f);
    EXPECT_NO_THROW(viz3d::showPoints(w, "pts", pts));
    EXPECT_NO_THROW(viz3d::setGridVisible(w, true));

    EXPECT_NO_THROW(viz3d::showBox(w, "box", Vec3f::all(1.0f), Vec3f(1, 0, 0)));
    EXPECT_NO_THROW(viz3d::showSphere(w, "sphere", 1.0f, Vec3f(0, 1, 0)));

    // forward = (0,+1,0) and (0,-1,0): the degenerate up-vector case guarded by #4.
    float traj[] = { 0,0,0, 0,1,0,   1,0,0, 0,-1,0 };
    EXPECT_NO_THROW(viz3d::showCameraTrajectory(w, "traj", Mat(2, 6, CV_32F, traj), 1.0f, 0.5f));

    Mat rgbd(16, 16, CV_32FC4, Scalar(120, 120, 120, 500));   // #3 showRGBD path
    EXPECT_NO_THROW(viz3d::showRGBD(w, "rgbd", rgbd, Matx33f(8, 0, 8, 0, 8, 8, 0, 0, 1), 0.1f));

    for (int i = 0; i < 4; ++i)
        EXPECT_NO_THROW(updateWindow(w));

    EXPECT_NO_THROW(viz3d::destroyObject(w, "pts"));
    destroyAllWindows();
}

}} // namespace
