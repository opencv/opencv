// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include <opencv2/highgui.hpp>            // updateWindow / destroyWindow
#include "../src/viz3d/grid_ticks.hpp"    // GL-free grid spacing helper
#include "../src/utils.hpp"               // GL-free 3DGS decode/sort helpers

#include <cstdio>
#include <fstream>

namespace opencv_test { namespace {

using namespace cv;

// Regression for the old gridTickStep infinite-loop hang: must stay finite/positive. GL-free.
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

// Identity rotation + unit scale must give the identity covariance. GL-free.
TEST(Splat, covariance_identity)
{
    Matx33f cov = splat::covariance(Vec3f(1.f, 1.f, 1.f), Vec4f(1.f, 0.f, 0.f, 0.f));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(cov(i, j), (i == j) ? 1.f : 0.f, 1e-5f) << "at " << i << "," << j;
}

// Axis-aligned scales must land on the diagonal as squares, and a non-unit
// quaternion must still be treated as a rotation (i.e. normalized).
TEST(Splat, covariance_scale_and_normalization)
{
    Matx33f cov = splat::covariance(Vec3f(2.f, 3.f, 4.f), Vec4f(1.f, 0.f, 0.f, 0.f));
    EXPECT_NEAR(cov(0, 0), 4.f, 1e-4f);
    EXPECT_NEAR(cov(1, 1), 9.f, 1e-4f);
    EXPECT_NEAR(cov(2, 2), 16.f, 1e-4f);
    EXPECT_NEAR(cov(0, 1), 0.f, 1e-4f);

    Matx33f scaled = splat::covariance(Vec3f(1.f, 1.f, 1.f), Vec4f(7.f, 0.f, 0.f, 0.f));
    for (int i = 0; i < 3; i++)
        EXPECT_NEAR(scaled(i, i), 1.f, 1e-5f) << "unnormalized quaternion changed the shape";
}

// A 90 degree rotation about Z must swap the X and Y variances.
TEST(Splat, covariance_rotation)
{
    const float s = std::sqrt(0.5f);
    Matx33f cov = splat::covariance(Vec3f(2.f, 3.f, 4.f), Vec4f(s, 0.f, 0.f, s));
    EXPECT_NEAR(cov(0, 0), 9.f, 1e-3f);
    EXPECT_NEAR(cov(1, 1), 4.f, 1e-3f);
    EXPECT_NEAR(cov(2, 2), 16.f, 1e-3f);
    EXPECT_NEAR(cov(0, 1), 0.f, 1e-3f);
}

// Covariance must stay symmetric for arbitrary inputs.
TEST(Splat, covariance_symmetric)
{
    RNG rng(0x5eed);
    for (int t = 0; t < 64; t++)
    {
        Vec3f scale(rng.uniform(0.01f, 3.f), rng.uniform(0.01f, 3.f), rng.uniform(0.01f, 3.f));
        Vec4f q(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f),
                rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f));
        if (cv::norm(q) < 1e-3)
            continue;
        Matx33f cov = splat::covariance(scale, q);
        EXPECT_NEAR(cov(0, 1), cov(1, 0), 1e-4f);
        EXPECT_NEAR(cov(0, 2), cov(2, 0), 1e-4f);
        EXPECT_NEAR(cov(1, 2), cov(2, 1), 1e-4f);
        EXPECT_GT(cov(0, 0), 0.f);
        EXPECT_GT(cov(1, 1), 0.f);
        EXPECT_GT(cov(2, 2), 0.f);
    }
}

TEST(Splat, decode_activations)
{
    Mat raw = Mat::zeros(1, splat::RAW_STRIDE, CV_32F);
    float* r = raw.ptr<float>(0);
    r[0] = 1.f; r[1] = 2.f; r[2] = 3.f;   // position
    r[10] = 1.f;                          // identity quaternion, zero scale, zero opacity, zero f_dc

    Mat splats;
    splat::decode(raw, splats);

    ASSERT_EQ(splats.rows, 1);
    ASSERT_EQ(splats.cols, (int)splat::STRIDE);

    const float* d = splats.ptr<float>(0);
    EXPECT_NEAR(d[splat::OFS_POS + 0], 1.f, 1e-6f);
    EXPECT_NEAR(d[splat::OFS_POS + 1], 2.f, 1e-6f);
    EXPECT_NEAR(d[splat::OFS_POS + 2], 3.f, 1e-6f);

    // exp(0) == 1 on every axis, so the covariance is the identity
    EXPECT_NEAR(d[splat::OFS_COV + 0], 1.f, 1e-5f);
    EXPECT_NEAR(d[splat::OFS_COV + 3], 1.f, 1e-5f);
    EXPECT_NEAR(d[splat::OFS_COV + 5], 1.f, 1e-5f);
    EXPECT_NEAR(d[splat::OFS_COV + 1], 0.f, 1e-5f);

    EXPECT_NEAR(d[splat::OFS_RGB + 0], 0.5f, 1e-5f);   // 0.5 + C0 * 0
    EXPECT_NEAR(d[splat::OFS_ALPHA], 0.5f, 1e-5f);     // sigmoid(0)
}

// Sorting must run strictly far to near, which is what alpha blending requires.
TEST(Splat, sort_far_to_near)
{
    Mat pos = Mat::zeros(4, 3, CV_32F);
    const float xs[] = { 1.f, 8.f, 4.f, 2.f };
    for (int i = 0; i < 4; i++)
        pos.ptr<float>(i)[0] = xs[i];

    std::vector<int> order;
    splat::sortByDepth(pos, Vec3f(0.f, 0.f, 0.f), order);

    ASSERT_EQ(order.size(), (size_t)4);
    EXPECT_EQ(order[0], 1);
    EXPECT_EQ(order[1], 2);
    EXPECT_EQ(order[2], 3);
    EXPECT_EQ(order[3], 0);

    for (size_t i = 1; i < order.size(); i++)
    {
        float prev = pos.ptr<float>(order[i - 1])[0];
        float cur = pos.ptr<float>(order[i])[0];
        EXPECT_GE(prev, cur);
    }
}

// Writes a minimal 3DGS PLY. Properties are deliberately out of order and an
// unrequested one is included, to prove the reader maps by name and skips extras.
static void writeSplatPly(const std::string& path)
{
    std::ofstream f(path);
    f << "ply\n"
      << "format ascii 1.0\n"
      << "element vertex 2\n"
      << "property float x\n"
      << "property float y\n"
      << "property float z\n"
      << "property float f_rest_0\n"
      << "property float opacity\n"
      << "property float scale_0\n"
      << "property float scale_1\n"
      << "property float scale_2\n"
      << "property float rot_0\n"
      << "property float rot_1\n"
      << "property float rot_2\n"
      << "property float rot_3\n"
      << "property float f_dc_0\n"
      << "property float f_dc_1\n"
      << "property float f_dc_2\n"
      << "end_header\n"
      << "1 2 3 99 0 0 0 0 1 0 0 0 0 0 0\n"
      << "4 5 6 99 0 0 0 0 1 0 0 0 0 0 0\n";
}

TEST(Splat, load_ply)
{
    std::string path = tempfile("splats.ply");
    writeSplatPly(path);

    Mat splats;
    loadGaussianSplats(path, splats);
    remove(path.c_str());

    ASSERT_FALSE(splats.empty());
    ASSERT_EQ(splats.rows, 2);
    ASSERT_EQ(splats.cols, (int)splat::STRIDE);

    EXPECT_NEAR(splats.ptr<float>(0)[0], 1.f, 1e-6f);
    EXPECT_NEAR(splats.ptr<float>(0)[2], 3.f, 1e-6f);
    EXPECT_NEAR(splats.ptr<float>(1)[0], 4.f, 1e-6f);
    EXPECT_NEAR(splats.ptr<float>(1)[2], 6.f, 1e-6f);

    for (int i = 0; i < 2; i++)
    {
        const float* d = splats.ptr<float>(i);
        EXPECT_NEAR(d[splat::OFS_ALPHA], 0.5f, 1e-5f);
        EXPECT_NEAR(d[splat::OFS_COV + 0], 1.f, 1e-5f);
        EXPECT_NEAR(d[splat::OFS_RGB + 0], 0.5f, 1e-5f);
    }
}

// A PLY without the 3DGS attributes must fail cleanly rather than return garbage.
TEST(Splat, load_ply_rejects_plain_cloud)
{
    std::string path = tempfile("plain.ply");
    {
        std::ofstream f(path);
        f << "ply\nformat ascii 1.0\nelement vertex 1\n"
          << "property float x\nproperty float y\nproperty float z\n"
          << "end_header\n0 0 0\n";
    }

    Mat splats;
    loadGaussianSplats(path, splats);
    remove(path.c_str());

    EXPECT_TRUE(splats.empty());
}

static void writeSplatFile(const std::string& path, int n)
{
    std::ofstream f(path, std::ios::binary);
    for (int i = 0; i < n; i++)
    {
        const float pos[3] = { 1.f + 3 * i, 2.f + 3 * i, 3.f + 3 * i };
        const float scale[3] = { 1.f, 1.f, 1.f };
        const uchar rgba[4] = { 128, 128, 128, 128 };
        const uchar rot[4] = { 255, 128, 128, 128 };
        f.write((const char*)pos, sizeof(pos));
        f.write((const char*)scale, sizeof(scale));
        f.write((const char*)rgba, sizeof(rgba));
        f.write((const char*)rot, sizeof(rot));
    }
}

TEST(Splat, load_splat)
{
    std::string path = tempfile("splats.splat");
    writeSplatFile(path, 2);

    Mat splats;
    loadGaussianSplats(path, splats);
    remove(path.c_str());

    ASSERT_FALSE(splats.empty());
    ASSERT_EQ(splats.rows, 2);
    ASSERT_EQ(splats.cols, (int)splat::STRIDE);

    EXPECT_NEAR(splats.ptr<float>(0)[0], 1.f, 1e-6f);
    EXPECT_NEAR(splats.ptr<float>(0)[2], 3.f, 1e-6f);
    EXPECT_NEAR(splats.ptr<float>(1)[0], 4.f, 1e-6f);
    EXPECT_NEAR(splats.ptr<float>(1)[2], 6.f, 1e-6f);

    for (int i = 0; i < 2; i++)
    {
        const float* d = splats.ptr<float>(i);
        EXPECT_NEAR(d[splat::OFS_ALPHA], 128.f / 255.f, 1e-5f);
        EXPECT_NEAR(d[splat::OFS_RGB + 0], 128.f / 255.f, 1e-5f);
        EXPECT_NEAR(d[splat::OFS_COV + 0], 1.f, 1e-5f);
        EXPECT_NEAR(d[splat::OFS_COV + 3], 1.f, 1e-5f);
        EXPECT_NEAR(d[splat::OFS_COV + 5], 1.f, 1e-5f);
        EXPECT_NEAR(d[splat::OFS_COV + 1], 0.f, 1e-5f);
    }
}

// A record is 32 bytes, so anything else is truncated or not a splat file at all.
TEST(Splat, load_splat_rejects_partial_record)
{
    std::string path = tempfile("partial.splat");
    writeSplatFile(path, 1);
    {
        std::ofstream f(path, std::ios::binary | std::ios::app);
        const uchar tail[8] = { 0 };
        f.write((const char*)tail, sizeof(tail));
    }

    Mat splats;
    loadGaussianSplats(path, splats);
    remove(path.c_str());

    EXPECT_TRUE(splats.empty());
}

// viz3d needs a GL context; skip when none is available (headless CI).
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

// Smoke test for the public viz3d API: build a scene, force redraws, pass if nothing throws.
TEST(Viz3D, render_scene_smoke)
{
    if (!viz3dAvailable())
        throw cvtest::SkipTestException("viz3d/OpenGL not available (no GL context)");

    const String w = "viz3d_test";

    Mat pts(256, 6, CV_32F);
    randu(pts, 0.0f, 1.0f);
    EXPECT_NO_THROW(viz3d::showPoints(w, "pts", pts));
    EXPECT_NO_THROW(viz3d::setGridVisible(w, true));
    EXPECT_NO_THROW(viz3d::setSky(w, Vec3f(0.2f, 0.5f, 0.8f)));   // exercises the sky-color render path

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

// Exercises the splat path: TextureBuffer, instanced draw, blend/depthMask state,
// and that splats compose with opaque objects in the same window.
TEST(Viz3D, render_splats_smoke)
{
    if (!viz3dAvailable())
        throw cvtest::SkipTestException("viz3d/OpenGL not available (no GL context)");

    const String w = "viz3d_splat_test";

    Mat raw(64, splat::RAW_STRIDE, CV_32F, Scalar::all(0.0));
    RNG rng(0x5B1A7C);
    for (int i = 0; i < raw.rows; i++)
    {
        float* r = raw.ptr<float>(i);
        r[0] = rng.uniform(-2.f, 2.f);
        r[1] = rng.uniform(-2.f, 2.f);
        r[2] = rng.uniform(-2.f, 2.f);
        r[6] = 1.0f;                                  // opacity -> sigmoid
        r[7] = r[8] = r[9] = -2.0f;                   // small scales
        r[10] = 1.0f;                                 // identity quaternion
    }

    Mat splats;
    splat::decode(raw, splats);

    EXPECT_NO_THROW(viz3d::showSplats(w, "splats", splats));

    // An opaque object in the same window: splats must be drawn after it.
    Mat pts(16, 6, CV_32F);
    randu(pts, 0.0f, 1.0f);
    EXPECT_NO_THROW(viz3d::showPoints(w, "pts", pts));

    for (int i = 0; i < 4; ++i)
        EXPECT_NO_THROW(updateWindow(w));

    // Changing the model matrix must invalidate the depth order, since the sort
    // runs in object space.
    EXPECT_NO_THROW(viz3d::setObjectPosition(w, "splats", Vec3f(1.0f, 0.0f, 0.0f)));
    EXPECT_NO_THROW(updateWindow(w));

    EXPECT_NO_THROW(viz3d::destroyObject(w, "splats"));
    destroyAllWindows();
}

// A single centered splat must not throw and must survive repeated redraws.
TEST(Viz3D, render_single_splat)
{
    if (!viz3dAvailable())
        throw cvtest::SkipTestException("viz3d/OpenGL not available (no GL context)");

    const String w = "viz3d_single_splat";

    Mat raw = Mat::zeros(1, splat::RAW_STRIDE, CV_32F);
    raw.ptr<float>(0)[6] = 2.0f;
    raw.ptr<float>(0)[10] = 1.0f;

    Mat splats;
    splat::decode(raw, splats);

    EXPECT_NO_THROW(viz3d::showSplats(w, "one", splats));
    for (int i = 0; i < 4; ++i)
        EXPECT_NO_THROW(updateWindow(w));

    destroyAllWindows();
}

}} // namespace
