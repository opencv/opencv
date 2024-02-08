// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <vector>
#include <cstdio>

#include "test_precomp.hpp"
#include "opencv2/ts.hpp"

namespace opencv_test { namespace {

TEST(PointCloud, LoadObj)
{
    std::vector<cv::Point3f> points_gold = {
        {-5.93915f, -0.13257f, 2.55837f},
        {-5.93915f, 1.86743f, 2.55837f},
        {-5.93915f, -0.13257f, -1.16339f},
        {-5.93915f, 1.86743f, -1.16339f},
        {0.399941f, -0.13257f, 2.55837f},
        {0.399941f, 1.86743f, 2.55837f},
        {0.399941f, -0.13257f, -1.16339f},
        {0.399941f, 1.86743f, -1.16339f}};

    std::vector<cv::Point3f> normals_gold = {
        {-1.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, -1.0000f},
        {1.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 1.0000f},
        {0.0000f, -1.0000f, 0.0000f},
        {0.0000f, 1.0000f, 0.0000f}};

    std::vector<cv::Point3_<uchar>> rgb_gold = {
            {19, 144, 149},
            {219, 28, 216},
            {218, 157, 101},
            {11, 161, 78},
            {248, 183, 214},
            {63, 196, 6},
            {165, 190, 153},
            {89, 203, 11}};

    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;
    std::vector<cv::Point3_<uchar>> rgb;

    auto folder = cvtest::TS::ptr()->get_data_path();
    cv::loadPointCloud(folder + "pointcloudio/orig.obj", points, normals, rgb);

    EXPECT_EQ(points_gold, points);
    EXPECT_EQ(rgb_gold, rgb);
    EXPECT_EQ(normals_gold, normals);
}

TEST(PointCloud, LoadObjNoNormals)
{
    std::vector<cv::Point3f> points_gold = {
        {-5.93915f, -0.13257f, 2.55837f},
        {-5.93915f, 1.86743f, 2.55837f},
        {-5.93915f, -0.13257f, -1.16339f},
        {-5.93915f, 1.86743f, -1.16339f},
        {0.399941f, -0.13257f, 2.55837f},
        {0.399941f, 1.86743f, 2.55837f},
        {0.399941f, -0.13257f, -1.16339f},
        {0.399941f, 1.86743f, -1.16339f}};

    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    auto folder = cvtest::TS::ptr()->get_data_path();
    cv::loadPointCloud(folder + "pointcloudio/orig_no_norms.obj", points, normals);

    EXPECT_EQ(points_gold, points);
    EXPECT_TRUE(normals.empty());
}

TEST(PointCloud, SaveObj)
{
    std::vector<cv::Point3f> points_gold;
    std::vector<cv::Point3f> normals_gold;
    std::vector<cv::Point3_<uchar>> rgb_gold;

    auto folder = cvtest::TS::ptr()->get_data_path();
    auto new_path = tempfile("new.obj");

    cv::loadPointCloud(folder + "pointcloudio/orig.obj", points_gold, normals_gold, rgb_gold);
    cv::savePointCloud(new_path, points_gold, normals_gold, rgb_gold);

    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;
    std::vector<cv::Point3_<uchar>> rgb;

    cv::loadPointCloud(new_path, points, normals, rgb);

    EXPECT_EQ(normals, normals_gold);
    EXPECT_EQ(points, points_gold);
    EXPECT_EQ(rgb, rgb_gold);
    std::remove(new_path.c_str());
}

TEST(PointCloud, LoadSavePly)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;
    std::vector<cv::Point3_<uchar>> rgb;

    auto folder = cvtest::TS::ptr()->get_data_path();
    std::string new_path = tempfile("new.ply");

    cv::loadPointCloud(folder + "pointcloudio/orig.ply", points, normals, rgb);
    cv::savePointCloud(new_path, points, normals, rgb);

    std::vector<cv::Point3f> points_gold;
    std::vector<cv::Point3f> normals_gold;
    std::vector<cv::Point3_<uchar>> rgb_gold;

    cv::loadPointCloud(new_path, points_gold, normals_gold, rgb_gold);

    EXPECT_EQ(normals_gold, normals);
    EXPECT_EQ(points_gold, points);
    EXPECT_EQ(rgb_gold, rgb);
    std::remove(new_path.c_str());
}

TEST(PointCloud, LoadSaveMeshObj)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;
    std::vector<std::vector<int32_t>> indices;

    auto folder = cvtest::TS::ptr()->get_data_path();
    std::string new_path = tempfile("new_mesh.obj");

    cv::loadMesh(folder + "pointcloudio/orig.obj", points, normals, indices);
    cv::saveMesh(new_path, points, normals, indices);

    std::vector<cv::Point3f> points_gold;
    std::vector<cv::Point3f> normals_gold;
    std::vector<std::vector<int32_t>> indices_gold;

    cv::loadMesh(new_path, points_gold, normals_gold, indices_gold);

    EXPECT_EQ(normals_gold, normals);
    EXPECT_EQ(points_gold, points);
    EXPECT_EQ(indices_gold, indices);
    EXPECT_TRUE(!indices.empty());
    std::remove(new_path.c_str());
}

typedef std::string PlyTestParamsType;
typedef testing::TestWithParam<PlyTestParamsType> PlyTest;

TEST_P(PlyTest, LoadSaveMesh)
{
    std::string fname = GetParam();

    std::vector<cv::Point3f> points_gold, normals_gold, colors_gold;
    std::vector<std::vector<int32_t>> indices_gold;

    auto folder = cvtest::TS::ptr()->get_data_path();
    std::string new_path = tempfile("new_mesh.ply");

    cv::loadMesh(folder + fname, points_gold, normals_gold, colors_gold, indices_gold);
    EXPECT_FALSE(points_gold.empty());
    EXPECT_FALSE(indices_gold.empty());

    cv::saveMesh(new_path, points_gold, normals_gold, colors_gold, indices_gold);

    std::vector<cv::Point3f> points, normals, colors;
    std::vector<std::vector<int32_t>> indices;
    cv::loadMesh(new_path, points, normals, colors, indices);

    if (!normals.empty())
    {
        EXPECT_LE(cv::norm(normals_gold, normals, NORM_INF), 0);
    }
    EXPECT_LE(cv::norm(points_gold, points, NORM_INF), 0);
    EXPECT_LE(cv::norm(colors_gold, colors, NORM_INF), 0);
    EXPECT_EQ(indices_gold, indices);
    std::remove(new_path.c_str());
}


INSTANTIATE_TEST_CASE_P(PointCloud, PlyTest,
    ::testing::Values("pointcloudio/orig.ply", "pointcloudio/orig_ascii_fidx.ply", "pointcloudio/orig_bin_fidx.ply",
                      "pointcloudio/orig_ascii_vidx.ply", "pointcloudio/orig_bin.ply", "viz/dragon.ply"));

TEST(PointCloud, NonexistentFile)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    auto folder = cvtest::TS::ptr()->get_data_path();
    cv::loadPointCloud(folder + "pointcloudio/fake.obj", points, normals);
    EXPECT_TRUE(points.empty());
    EXPECT_TRUE(normals.empty());
}

TEST(PointCloud, LoadBadExtension)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    auto folder = cvtest::TS::ptr()->get_data_path();
    cv::loadPointCloud(folder + "pointcloudio/fake.fake", points, normals);
    EXPECT_TRUE(points.empty());
    EXPECT_TRUE(normals.empty());
}

TEST(PointCloud, SaveBadExtension)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    auto folder = cvtest::TS::ptr()->get_data_path();
    cv::savePointCloud(folder + "pointcloudio/fake.fake", points, normals);
}

}} /* namespace opencv_test */
