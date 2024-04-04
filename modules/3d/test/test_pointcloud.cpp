// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <vector>
#include <cstdio>

#include "test_precomp.hpp"
#include "opencv2/ts.hpp"

namespace opencv_test { namespace {

TEST(PointCloud, LoadPointCloudObj)
{
    std::vector<cv::Point3f> points_gold = {
        {-5.93915f, -0.13257f,  2.55837f},
        {-5.93915f,  1.86743f,  2.55837f},
        {-5.93915f, -0.13257f, -1.16339f},
        {-5.93915f,  1.86743f, -1.16339f},
        {0.399941f, -0.13257f,  2.55837f},
        {0.399941f,  1.86743f,  2.55837f},
        {0.399941f, -0.13257f, -1.16339f},
        {0.399941f,  1.86743f, -1.16339f}
    };

    std::vector<cv::Point3f> normals_gold = {
        {-1.0000f,  0.0000f,  0.0000f},
        { 0.0000f,  0.0000f, -1.0000f},
        { 1.0000f,  0.0000f,  0.0000f},
        { 0.0000f,  0.0000f,  1.0000f},
        { 0.0000f, -1.0000f,  0.0000f},
        { 0.0000f,  1.0000f,  0.0000f}
    };

    std::vector<cv::Point3f> rgb_gold = {
        {0.0756f, 0.5651f, 0.5829f},
        {0.8596f, 0.1105f, 0.8455f},
        {0.8534f, 0.6143f, 0.3950f},
        {0.0438f, 0.6308f, 0.3065f},
        {0.9716f, 0.7170f, 0.8378f},
        {0.2472f, 0.7701f, 0.0234f},
        {0.6472f, 0.7467f, 0.5981f},
        {0.3502f, 0.7954f, 0.0443f}
    };

    std::vector<cv::Point3f> points, normals, rgb;

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

    std::vector<cv::Point3f> points, normals;

    auto folder = cvtest::TS::ptr()->get_data_path();
    cv::loadPointCloud(folder + "pointcloudio/orig_no_norms.obj", points, normals);

    EXPECT_EQ(points_gold, points);
    EXPECT_TRUE(normals.empty());
}

TEST(PointCloud, SaveObj)
{
    std::vector<cv::Point3f> points_gold, normals_gold, rgb_gold;

    auto folder = cvtest::TS::ptr()->get_data_path();
    auto new_path = tempfile("new.obj");

    cv::loadPointCloud(folder + "pointcloudio/orig.obj", points_gold, normals_gold, rgb_gold);
    cv::savePointCloud(new_path, points_gold, normals_gold, rgb_gold);

    std::vector<cv::Point3f> points, normals, rgb;

    cv::loadPointCloud(new_path, points, normals, rgb);

    EXPECT_EQ(normals, normals_gold);
    EXPECT_EQ(points, points_gold);
    EXPECT_EQ(rgb, rgb_gold);
    std::remove(new_path.c_str());
}

TEST(PointCloud, LoadSavePly)
{
    std::vector<cv::Point3f> points, normals, rgb;

    auto folder = cvtest::TS::ptr()->get_data_path();
    std::string new_path = tempfile("new.ply");

    cv::loadPointCloud(folder + "pointcloudio/orig.ply", points, normals, rgb);
    cv::savePointCloud(new_path, points, normals, rgb);

    std::vector<cv::Point3f> points_gold, normals_gold, rgb_gold;

    cv::loadPointCloud(new_path, points_gold, normals_gold, rgb_gold);

    EXPECT_EQ(normals_gold, normals);
    EXPECT_EQ(points_gold, points);
    EXPECT_EQ(rgb_gold, rgb);
    std::remove(new_path.c_str());
}

TEST(PointCloud, LoadSaveMeshObj)
{
    std::vector<cv::Point3f> points, normals, colors;
    std::vector<cv::Point2f> texCoords;
    std::vector<std::vector<int32_t>> indices;

    auto folder = cvtest::TS::ptr()->get_data_path();
    std::string new_path = tempfile("new_mesh.obj");

    cv::loadMesh(folder + "pointcloudio/orig.obj", points, indices, normals, colors, texCoords);
    EXPECT_FALSE(points.empty());
    EXPECT_FALSE(indices.empty());
    EXPECT_FALSE(normals.empty());
    EXPECT_FALSE(colors.empty());
    EXPECT_FALSE(texCoords.empty());
    cv::saveMesh(new_path, points, indices, normals, colors, texCoords);

    std::vector<cv::Point3f> points_gold, normals_gold, colors_gold;
    std::vector<cv::Point2f> texCoords_gold;
    std::vector<std::vector<int32_t>> indices_gold;

    cv::loadMesh(new_path, points_gold, indices_gold, normals_gold, colors_gold, texCoords_gold);
    EXPECT_FALSE(points_gold.empty());
    EXPECT_FALSE(indices_gold.empty());
    EXPECT_FALSE(normals_gold.empty());
    EXPECT_FALSE(colors_gold.empty());
    EXPECT_FALSE(texCoords_gold.empty());

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
    std::vector<cv::Vec3i> indices_gold;

    auto folder = cvtest::TS::ptr()->get_data_path();
    std::string new_path = tempfile("new_mesh.ply");

    cv::loadMesh(folder + fname, points_gold, indices_gold, normals_gold, colors_gold);
    EXPECT_FALSE(points_gold.empty());
    EXPECT_FALSE(indices_gold.empty());

    cv::saveMesh(new_path, points_gold, indices_gold, normals_gold, colors_gold);

    std::vector<cv::Point3f> points, normals, colors;
    std::vector<cv::Vec3i> indices;
    cv::loadMesh(new_path, points, indices, normals, colors);

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
