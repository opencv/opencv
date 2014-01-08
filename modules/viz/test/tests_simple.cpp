/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and / or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "test_precomp.hpp"

using namespace cv;
using namespace cv::viz;

TEST(Viz, DISABLED_show_cloud_bluberry)
{
    Mat dragon_cloud = readCloud(get_dragon_ply_file_path());

    Affine3d pose = Affine3d().rotate(Vec3d(0, 0.8, 0));

    Viz3d viz("show_cloud_bluberry");
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("dragon", WCloud(dragon_cloud, Color::bluberry()), pose);
    viz.spin();
}

TEST(Viz, DISABLED_show_cloud_random_color)
{
    Mat dragon_cloud = readCloud(get_dragon_ply_file_path());

    Mat colors(dragon_cloud.size(), CV_8UC3);
    theRNG().fill(colors, RNG::UNIFORM, 0, 255);

    Affine3d pose = Affine3d().rotate(Vec3d(0, 0.8, 0));

    Viz3d viz("show_cloud_random_color");
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("dragon", WCloud(dragon_cloud, colors), pose);
    viz.spin();
}

TEST(Viz, DISABLED_show_cloud_masked)
{
    Mat dragon_cloud = readCloud(get_dragon_ply_file_path());

    Vec3f qnan = Vec3f::all(std::numeric_limits<float>::quiet_NaN());
    for(size_t i = 0; i < dragon_cloud.total(); ++i)
        if (i % 15 != 0)
            dragon_cloud.at<Vec3f>(i) = qnan;

    Affine3d pose = Affine3d().rotate(Vec3d(0, 0.8, 0));

    Viz3d viz("show_cloud_masked");
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("dragon", WCloud(dragon_cloud), pose);
    viz.spin();
}

TEST(Viz, DISABLED_show_cloud_collection)
{
    Mat cloud = readCloud(get_dragon_ply_file_path());

    WCloudCollection ccol;
    ccol.addCloud(cloud, Color::white(), Affine3d().translate(Vec3d(0, 0, 0)).rotate(Vec3d(1.57, 0, 0)));
    ccol.addCloud(cloud, Color::blue(),  Affine3d().translate(Vec3d(1, 0, 0)));
    ccol.addCloud(cloud, Color::red(),   Affine3d().translate(Vec3d(2, 0, 0)));

    Viz3d viz("show_cloud_collection");
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("ccol", ccol);
    viz.spin();
}

TEST(Viz, DISABLED_show_mesh)
{
    Mesh3d mesh = Mesh3d::load(get_dragon_ply_file_path());

    Affine3d pose = Affine3d().rotate(Vec3d(0, 0.8, 0));

    Viz3d viz("show_mesh");
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("mesh", WMesh(mesh), pose);
    viz.spin();
}

TEST(Viz, DISABLED_show_mesh_random_colors)
{
    Mesh3d mesh = Mesh3d::load(get_dragon_ply_file_path());
    theRNG().fill(mesh.colors, RNG::UNIFORM, 0, 255);

    Affine3d pose = Affine3d().rotate(Vec3d(0, 0.8, 0));

    Viz3d viz("show_mesh_random_color");
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("mesh", WMesh(mesh), pose);
    viz.setRenderingProperty("mesh", SHADING, SHADING_PHONG);
    viz.spin();
}

TEST(Viz, DISABLED_show_polyline)
{
    Mat polyline(1, 32, CV_64FC3);
    for(size_t i = 0; i < polyline.total(); ++i)
        polyline.at<Vec3d>(i) = Vec3d(i/16.0, cos(i * CV_PI/6), sin(i * CV_PI/6));

    Viz3d viz("show_polyline");
    viz.showWidget("polyline", WPolyLine(Mat(polyline), Color::apricot()));
    viz.showWidget("coosys", WCoordinateSystem());
    viz.spin();
}

TEST(Viz, DISABLED_show_sampled_normals)
{
    Mesh3d mesh = Mesh3d::load(get_dragon_ply_file_path());
    computeNormals(mesh, mesh.normals);

    Affine3d pose = Affine3d().rotate(Vec3d(0, 0.8, 0));

    Viz3d viz("show_sampled_normals");
    viz.showWidget("mesh", WMesh(mesh), pose);
    viz.showWidget("normals", WCloudNormals(mesh.cloud, mesh.normals, 30, 0.1f, Color::green()), pose);
    viz.setRenderingProperty("normals", LINE_WIDTH, 2.0);
    viz.spin();
}

TEST(Viz, DISABLED_show_trajectories)
{
    std::vector<Affine3d> path = generate_test_trajectory<double>(), sub0, sub1, sub2, sub3, sub4, sub5;

    Mat(path).rowRange(0, path.size()/10+1).copyTo(sub0);
    Mat(path).rowRange(path.size()/10, path.size()/5+1).copyTo(sub1);
    Mat(path).rowRange(path.size()/5, 11*path.size()/12).copyTo(sub2);
    Mat(path).rowRange(11*path.size()/12, path.size()).copyTo(sub3);
    Mat(path).rowRange(3*path.size()/4, 33*path.size()/40).copyTo(sub4);
    Mat(path).rowRange(33*path.size()/40, 9*path.size()/10).copyTo(sub5);
    Matx33d K(1024.0, 0.0, 320.0, 0.0, 1024.0, 240.0, 0.0, 0.0, 1.0);

    Viz3d viz("show_trajectories");
    viz.showWidget("coos", WCoordinateSystem());
    viz.showWidget("sub0", WTrajectorySpheres(sub0, 0.25, 0.07));
    viz.showWidget("sub1", WTrajectory(sub1, WTrajectory::PATH, 0.2, Color::brown()));
    viz.showWidget("sub2", WTrajectory(sub2, WTrajectory::FRAMES, 0.2));
    viz.showWidget("sub3", WTrajectory(sub3, WTrajectory::BOTH, 0.2, Color::green()));
    viz.showWidget("sub4", WTrajectoryFrustums(sub4, K, 0.3));
    viz.showWidget("sub5", WTrajectoryFrustums(sub5, Vec2d(0.78, 0.78), 0.15));

    int i = 0;
    while(!viz.wasStopped())
    {
        double a = --i % 360;
        Vec3d pose(sin(a * CV_PI/180), 0.7, cos(a * CV_PI/180));
        viz.setViewerPose(makeCameraPose(pose * 7.5, Vec3d(0.0, 0.5, 0.0), Vec3d(0.0, 0.1, 0.0)));
        viz.spinOnce(20, true);
    }
    //viz.spin();
}

TEST(Viz, DISABLED_show_trajectory_reposition)
{
    std::vector<Affine3f> path = generate_test_trajectory<float>();

    Viz3d viz("show_trajectory_reposition_to_origin");
    viz.showWidget("coos", WCoordinateSystem());
    viz.showWidget("sub3", WTrajectory(Mat(path).rowRange(0, path.size()/3), WTrajectory::BOTH, 0.2, Color::brown()), path.front().inv());
    viz.spin();
}


TEST(Viz, DISABLED_spin_twice_____________________________TODO_UI_BUG)
{
    Mesh3d mesh = Mesh3d::load(get_dragon_ply_file_path());

    Viz3d viz("spin_twice");
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("mesh", WMesh(mesh));
    viz.spin();
    viz.spin();
}
