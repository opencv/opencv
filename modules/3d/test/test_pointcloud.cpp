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

    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    auto folder = cvtest::TS::ptr()->get_data_path();
    cv::loadPointCloud(folder + "pointcloudio/orig.obj", points, normals);

    EXPECT_EQ(points_gold, points);
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

    auto folder = cvtest::TS::ptr()->get_data_path();
    auto new_path = tempfile("new.obj");

    cv::loadPointCloud(folder + "pointcloudio/orig.obj", points_gold, normals_gold);
    cv::savePointCloud(new_path, points_gold, normals_gold);

    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    cv::loadPointCloud(new_path, points, normals);

    EXPECT_EQ(normals, normals_gold);
    EXPECT_EQ(points, points_gold);
    std::remove(new_path.c_str());
}

TEST(PointCloud, LoadSavePly)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    auto folder = cvtest::TS::ptr()->get_data_path();
    std::string new_path = tempfile("new.ply");

    cv::loadPointCloud(folder + "pointcloudio/orig.ply", points, normals);
    cv::savePointCloud(new_path, points, normals);

    std::vector<cv::Point3f> points_gold;
    std::vector<cv::Point3f> normals_gold;

    cv::loadPointCloud(new_path, points_gold, normals_gold);

    EXPECT_EQ(normals_gold, normals);
    EXPECT_EQ(points_gold, points);
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

TEST(PointCloud, LoadSaveMeshPly)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;
    std::vector<std::vector<int32_t>> indices;

    auto folder = cvtest::TS::ptr()->get_data_path();
    std::string new_path = tempfile("new_mesh.ply");

    // we don't support meshes in PLY format right now but it should exit silently
    cv::loadMesh(folder + "pointcloudio/orig.ply", points, normals, indices);
    EXPECT_TRUE(points.empty());
    EXPECT_TRUE(normals.empty());
    EXPECT_TRUE(indices.empty());

    cv::saveMesh(new_path, points, normals, indices);
    EXPECT_TRUE(points.empty());
    EXPECT_TRUE(normals.empty());
    EXPECT_TRUE(indices.empty());

    std::remove(new_path.c_str());
}

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
