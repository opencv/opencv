#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>

#include "test_precomp.hpp"
#include "opencv2/ts.hpp"

namespace opencv_test { namespace {

TEST(obj_test, point_cloud_load_tests)
{
    std::vector<cv::Point3f> points_1 = {
        {-5.93915f, -0.13257f, 2.55837f},
        {-5.93915f, 1.86743f, 2.55837f},
        {-5.93915f, -0.13257f, -1.16339f},
        {-5.93915f, 1.86743f, -1.16339f},
        {0.399941f, -0.13257f, 2.55837f},
        {0.399941f, 1.86743f, 2.55837f},
        {0.399941f, -0.13257f, -1.16339f},
        {0.399941f, 1.86743f, -1.16339f}};
    std::vector<cv::Point3f> normals_1 = {
        {-1.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, -1.0000f},
        {1.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 1.0000f},
        {0.0000f, -1.0000f, 0.0000f},
        {0.0000f, 1.0000f, 0.0000f}};

    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    std::string folder = string(cvtest::TS::ptr()->get_data_path()) + "pointcloudio/";
    std::string original_path = folder + "orig.obj";

    cv::loadPointCloud(original_path, points, normals);

    EXPECT_TRUE(points_1==points);
    EXPECT_TRUE(normals_1==normals);
}

TEST(obj_test, point_cloud_load_no_norms_tests)
{
    std::vector<cv::Point3f> points_1 = {
        {-5.93915f, -0.13257f, 2.55837f},
        {-5.93915f, 1.86743f, 2.55837f},
        {-5.93915f, -0.13257f, -1.16339f},
        {-5.93915f, 1.86743f, -1.16339f},
        {0.399941f, -0.13257f, 2.55837f},
        {0.399941f, 1.86743f, 2.55837f},
        {0.399941f, -0.13257f, -1.16339f},
        {0.399941f, 1.86743f, -1.16339f}};
    std::vector<cv::Point3f> normals_1 = {};

    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    std::string folder = string(cvtest::TS::ptr()->get_data_path()) + "pointcloudio/";
    std::string original_path = folder + "orig_no_norms.obj";

    cv::loadPointCloud(original_path, points, normals);

    EXPECT_TRUE(points_1==points);
    EXPECT_TRUE(normals_1==normals);
}

TEST(obj_test, point_cloud_save_tests)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    std::string folder = std::string(cvtest::TS::ptr()->get_data_path()) + "pointcloudio/";
    std::string original_path = folder + "orig.obj";
    std::string new_path = tempfile("new.obj");

    cv::loadPointCloud(original_path, points, normals);

    cv::savePointCloud(new_path, points, normals);

    std::vector<cv::Point3f> points_1;
    std::vector<cv::Point3f> normals_1;

    cv::loadPointCloud(new_path, points_1, normals_1);

    EXPECT_TRUE(normals_1==normals);
    EXPECT_TRUE(points_1==points);
    std::remove(new_path.c_str());
}

TEST(ply_test, point_cloud_load_tests)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    std::string folder = std::string(cvtest::TS::ptr()->get_data_path()) + "pointcloudio/";
    std::string original_path = folder + "orig.ply";
    std::string new_path = tempfile("new.ply");

    cv::loadPointCloud(original_path, points, normals);

    cv::savePointCloud(new_path, points, normals);

    std::vector<cv::Point3f> points_1;
    std::vector<cv::Point3f> normals_1;

    cv::loadPointCloud(new_path, points_1, normals_1);

    EXPECT_TRUE(normals_1==normals);
    EXPECT_TRUE(points_1==points);
    std::remove(new_path.c_str());
}

TEST(fake_file_test, point_cloud_load_tests)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    std::string folder = std::string(cvtest::TS::ptr()->get_data_path()) + "pointcloudio/";
    std::string original_path = folder + "fake.obj";

    EXPECT_THROW(cv::loadPointCloud(original_path, points, normals), cv::Exception);
}

TEST(fake_extention_test, point_cloud_load_tests)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    std::string folder = std::string(cvtest::TS::ptr()->get_data_path()) + "pointcloudio/";
    std::string original_path = folder + "fake.fake";

    EXPECT_THROW(cv::loadPointCloud(original_path, points, normals), cv::Exception);
}

TEST(fake_extention_test, point_cloud_save_tests)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Point3f> normals;

    std::string folder = std::string(cvtest::TS::ptr()->get_data_path()) + "pointcloudio/";
    std::string original_path = folder + "fake.fake";

    EXPECT_THROW(cv::savePointCloud(original_path, points, normals), cv::Exception);
}

}
}
