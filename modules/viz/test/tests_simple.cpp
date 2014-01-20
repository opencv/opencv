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

TEST(Viz, show_cloud_bluberry)
{
    Mat dragon_cloud = readCloud(get_dragon_ply_file_path());

    Affine3d pose = Affine3d().rotate(Vec3d(0, 0.8, 0));

    Viz3d viz("show_cloud_bluberry");
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("dragon", WCloud(dragon_cloud, Color::bluberry()), pose);

    viz.showWidget("text2d", WText("Bluberry cloud", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_cloud_random_color)
{
    Mat dragon_cloud = readCloud(get_dragon_ply_file_path());

    Mat colors(dragon_cloud.size(), CV_8UC3);
    theRNG().fill(colors, RNG::UNIFORM, 0, 255);

    Affine3d pose = Affine3d().rotate(Vec3d(0, 0.8, 0));

    Viz3d viz("show_cloud_random_color");
    viz.setBackgroundMeshLab();
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("dragon", WCloud(dragon_cloud, colors), pose);
    viz.showWidget("text2d", WText("Random color cloud", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_cloud_masked)
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
    viz.showWidget("text2d", WText("Nan masked cloud", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_cloud_collection)
{
    Mat cloud = readCloud(get_dragon_ply_file_path());

    WCloudCollection ccol;
    ccol.addCloud(cloud, Color::white(), Affine3d().translate(Vec3d(0, 0, 0)).rotate(Vec3d(CV_PI/2, 0, 0)));
    ccol.addCloud(cloud, Color::blue(),  Affine3d().translate(Vec3d(1, 0, 0)));
    ccol.addCloud(cloud, Color::red(),   Affine3d().translate(Vec3d(2, 0, 0)));

    Viz3d viz("show_cloud_collection");
    viz.setBackgroundColor(Color::mlab());
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("ccol", ccol);
    viz.showWidget("text2d", WText("Cloud collection", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_painted_clouds)
{
    Mat cloud = readCloud(get_dragon_ply_file_path());

    Viz3d viz("show_painted_clouds");
    viz.setBackgroundMeshLab();
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("cloud1", WPaintedCloud(cloud), Affine3d(Vec3d(0.0, -CV_PI/2, 0.0), Vec3d(-1.5, 0.0, 0.0)));
    viz.showWidget("cloud2", WPaintedCloud(cloud, Vec3d(0.0, -0.75, -1.0), Vec3d(0.0, 0.75, 0.0)), Affine3d(Vec3d(0.0, CV_PI/2, 0.0), Vec3d(1.5, 0.0, 0.0)));
    viz.showWidget("cloud3", WPaintedCloud(cloud, Vec3d(0.0, 0.0, -1.0), Vec3d(0.0, 0.0, 1.0), Color::blue(), Color::red()));
    viz.showWidget("arrow", WArrow(Vec3d(0.0, 1.0, -1.0), Vec3d(0.0, 1.0, 1.0), 0.009, Color::raspberry()));
    viz.showWidget("text2d", WText("Painted clouds", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_mesh)
{
    Mesh mesh = Mesh::load(get_dragon_ply_file_path());

    Affine3d pose = Affine3d().rotate(Vec3d(0, 0.8, 0));

    Viz3d viz("show_mesh");
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("mesh", WMesh(mesh), pose);
    viz.showWidget("text2d", WText("Just mesh", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_mesh_random_colors)
{
    Mesh mesh = Mesh::load(get_dragon_ply_file_path());
    theRNG().fill(mesh.colors, RNG::UNIFORM, 0, 255);

    Affine3d pose = Affine3d().rotate(Vec3d(0, 0.8, 0));

    Viz3d viz("show_mesh_random_color");
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("mesh", WMesh(mesh), pose);
    viz.setRenderingProperty("mesh", SHADING, SHADING_PHONG);
    viz.showWidget("text2d", WText("Random color mesh", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_textured_mesh)
{
    Mat lena = imread(Path::combine(cvtest::TS::ptr()->get_data_path(), "lena.png"));

    std::vector<Vec3d> points;
    std::vector<Vec2d> tcoords;
    std::vector<int> polygons;
    for(size_t i = 0; i < 64; ++i)
    {
        double angle = CV_PI/2 * i/64.0;
        points.push_back(Vec3d(0.00, cos(angle), sin(angle))*0.75);
        points.push_back(Vec3d(1.57, cos(angle), sin(angle))*0.75);
        tcoords.push_back(Vec2d(0.0, i/64.0));
        tcoords.push_back(Vec2d(1.0, i/64.0));
    }

    for(size_t i = 0; i < points.size()/2-1; ++i)
    {
        int polys[] = {3, 2*i, 2*i+1, 2*i+2, 3, 2*i+1, 2*i+2, 2*i+3};
        polygons.insert(polygons.end(), polys, polys + sizeof(polys)/sizeof(polys[0]));
    }

    cv::viz::Mesh mesh;
    mesh.cloud = Mat(points, true).reshape(3, 1);
    mesh.tcoords = Mat(tcoords, true).reshape(2, 1);
    mesh.polygons = Mat(polygons, true).reshape(1, 1);
    mesh.texture = lena;

    Viz3d viz("show_textured_mesh");
    viz.setBackgroundMeshLab();
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("mesh", WMesh(mesh));
    viz.setRenderingProperty("mesh", SHADING, SHADING_PHONG);
    viz.showWidget("text2d", WText("Textured mesh", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_polyline)
{
    Mat polyline(1, 32, CV_64FC3);
    for(size_t i = 0; i < polyline.total(); ++i)
        polyline.at<Vec3d>(i) = Vec3d(i/16.0, cos(i * CV_PI/6), sin(i * CV_PI/6));

    Viz3d viz("show_polyline");
    viz.showWidget("polyline", WPolyLine(Mat(polyline), Color::apricot()));
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("text2d", WText("Polyline", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_sampled_normals)
{
    Mesh mesh = Mesh::load(get_dragon_ply_file_path());
    computeNormals(mesh, mesh.normals);

    Affine3d pose = Affine3d().rotate(Vec3d(0, 0.8, 0));

    Viz3d viz("show_sampled_normals");
    viz.showWidget("mesh", WMesh(mesh), pose);
    viz.showWidget("normals", WCloudNormals(mesh.cloud, mesh.normals, 30, 0.1f, Color::green()), pose);
    viz.setRenderingProperty("normals", LINE_WIDTH, 2.0);
    viz.showWidget("text2d", WText("Cloud or mesh normals", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_trajectories)
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
    viz.showWidget("sub4", WTrajectoryFrustums(sub4, K, 0.3, Color::yellow()));
    viz.showWidget("sub5", WTrajectoryFrustums(sub5, Vec2d(0.78, 0.78), 0.15));
    viz.showWidget("text2d", WText("Different kinds of supported trajectories", Point(20, 20), 20, Color::green()));

    int i = 0;
    while(!viz.wasStopped())
    {
        double a = --i % 360;
        Vec3d pose(sin(a * CV_PI/180), 0.7, cos(a * CV_PI/180));
        viz.setViewerPose(makeCameraPose(pose * 7.5, Vec3d(0.0, 0.5, 0.0), Vec3d(0.0, 0.1, 0.0)));
        viz.spinOnce(20, true);
    }
    viz.resetCamera();
    viz.spin();
}

TEST(Viz, show_trajectory_reposition)
{
    std::vector<Affine3f> path = generate_test_trajectory<float>();

    Viz3d viz("show_trajectory_reposition_to_origin");
    viz.showWidget("coos", WCoordinateSystem());
    viz.showWidget("sub3", WTrajectory(Mat(path).rowRange(0, path.size()/3), WTrajectory::BOTH, 0.2, Color::brown()), path.front().inv());
    viz.showWidget("text2d", WText("Trajectory resposition to origin", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_camera_positions)
{
    Matx33d K(1024.0, 0.0, 320.0, 0.0, 1024.0, 240.0, 0.0, 0.0, 1.0);
    Mat lena = imread(Path::combine(cvtest::TS::ptr()->get_data_path(), "lena.png"));
    Mat gray = make_gray(lena);

    Affine3d poses[2];
    for(int i = 0; i < 2; ++i)
    {
        Vec3d pose = 5 * Vec3d(sin(3.14 + 2.7 + i*60 * CV_PI/180), 0.4 - i*0.3, cos(3.14 + 2.7 + i*60 * CV_PI/180));
        poses[i] = makeCameraPose(pose, Vec3d(0.0, 0.0, 0.0), Vec3d(0.0, -0.1, 0.0));
    }

    Viz3d viz("show_camera_positions");
    viz.showWidget("sphe", WSphere(Point3d(0,0,0), 1.0, 10, Color::orange_red()));
    viz.showWidget("coos", WCoordinateSystem(1.5));
    viz.showWidget("pos1", WCameraPosition(0.75), poses[0]);
    viz.showWidget("pos2", WCameraPosition(Vec2d(0.78, 0.78), lena, 2.2, Color::green()), poses[0]);
    viz.showWidget("pos3", WCameraPosition(0.75), poses[1]);
    viz.showWidget("pos4", WCameraPosition(K, gray, 3, Color::indigo()), poses[1]);
    viz.showWidget("text2d", WText("Camera positions with images", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_overlay_image)
{
    Mat lena = imread(Path::combine(cvtest::TS::ptr()->get_data_path(), "lena.png"));
    Mat gray = make_gray(lena);

    Size2d half_lsize = Size2d(lena.size()) * 0.5;

    Viz3d viz("show_overlay_image");
    viz.setBackgroundMeshLab();
    Size vsz = viz.getWindowSize();

    viz.showWidget("coos", WCoordinateSystem());
    viz.showWidget("cube", WCube());
    viz.showWidget("img1", WImageOverlay(lena, Rect(Point(10, 10), half_lsize)));
    viz.showWidget("img2", WImageOverlay(gray, Rect(Point(vsz.width-10-lena.cols/2, 10), half_lsize)));
    viz.showWidget("img3", WImageOverlay(gray, Rect(Point(10, vsz.height-10-lena.rows/2), half_lsize)));
    viz.showWidget("img5", WImageOverlay(lena, Rect(Point(vsz.width-10-lena.cols/2, vsz.height-10-lena.rows/2), half_lsize)));
    viz.showWidget("text2d", WText("Overlay images", Point(20, 20), 20, Color::green()));

    int i = 0;
    while(!viz.wasStopped())
    {
        double a = ++i % 360;
        Vec3d pose(sin(a * CV_PI/180), 0.7, cos(a * CV_PI/180));
        viz.setViewerPose(makeCameraPose(pose * 3, Vec3d(0.0, 0.5, 0.0), Vec3d(0.0, 0.1, 0.0)));
        viz.getWidget("img1").cast<WImageOverlay>().setImage(lena * pow(sin(i*10*CV_PI/180) * 0.5 + 0.5, 1.0));
        viz.spinOnce(1, true);
    }
    viz.showWidget("text2d", WText("Overlay images (stopped)", Point(20, 20), 20, Color::green()));
    viz.spin();
}


TEST(Viz, show_image_method)
{
    Mat lena = imread(Path::combine(cvtest::TS::ptr()->get_data_path(), "lena.png"));

    Viz3d viz("show_image_method");
    viz.showImage(lena);
    viz.spinOnce(1500, true);
    viz.showImage(lena, lena.size());
    viz.spinOnce(1500, true);

    cv::viz::imshow("show_image_method", make_gray(lena)).spin();
}

TEST(Viz, show_image_3d)
{
    Mat lena = imread(Path::combine(cvtest::TS::ptr()->get_data_path(), "lena.png"));
    Mat gray = make_gray(lena);

    Viz3d viz("show_image_3d");
    viz.setBackgroundMeshLab();
    viz.showWidget("coos", WCoordinateSystem());
    viz.showWidget("cube", WCube());
    viz.showWidget("arr0", WArrow(Vec3d(0.5, 0.0, 0.0), Vec3d(1.5, 0.0, 0.0), 0.009, Color::raspberry()));
    viz.showWidget("img0", WImage3D(lena, Size2d(1.0, 1.0)), Affine3d(Vec3d(0.0, CV_PI/2, 0.0), Vec3d(.5, 0.0, 0.0)));
    viz.showWidget("arr1", WArrow(Vec3d(-0.5, -0.5, 0.0), Vec3d(0.2, 0.2, 0.0), 0.009, Color::raspberry()));
    viz.showWidget("img1", WImage3D(gray, Size2d(1.0, 1.0), Vec3d(-0.5, -0.5, 0.0), Vec3d(1.0, 1.0, 0.0), Vec3d(0.0, 1.0, 0.0)));

    viz.showWidget("arr3", WArrow(Vec3d::all(-0.5), Vec3d::all(0.5), 0.009, Color::raspberry()));

    viz.showWidget("text2d", WText("Images in 3D", Point(20, 20), 20, Color::green()));

    int i = 0;
    while(!viz.wasStopped())
    {
        viz.getWidget("img0").cast<WImage3D>().setImage(lena * pow(sin(i++*7.5*CV_PI/180) * 0.5 + 0.5, 1.0));
        viz.spinOnce(1, true);
    }
    viz.showWidget("text2d", WText("Images in 3D (stopped)", Point(20, 20), 20, Color::green()));
    viz.spin();
}

TEST(Viz, show_simple_widgets)
{
    Viz3d viz("show_simple_widgets");
    viz.setBackgroundMeshLab();

    viz.showWidget("coos", WCoordinateSystem());
    viz.showWidget("cube", WCube());
    viz.showWidget("cub0", WCube(Vec3d::all(-1.0), Vec3d::all(-0.5), false, Color::indigo()));
    viz.showWidget("arro", WArrow(Vec3d::all(-0.5), Vec3d::all(0.5), 0.009, Color::raspberry()));
    viz.showWidget("cir1", WCircle(0.5, 0.01, Color::bluberry()));
    viz.showWidget("cir2", WCircle(0.5, Point3d(0.5, 0.0, 0.0), Vec3d(1.0, 0.0, 0.0), 0.01, Color::apricot()));

    viz.showWidget("cyl0", WCylinder(Vec3d(-0.5, 0.5, -0.5), Vec3d(0.5, 0.5, -0.5), 0.125, 30, Color::brown()));
    viz.showWidget("con0", WCone(0.25, 0.125, 6, Color::azure()));
    viz.showWidget("con1", WCone(0.125, Point3d(0.5, -0.5, 0.5), Point3d(0.5, -1.0, 0.5), 6, Color::turquoise()));

    viz.showWidget("text2d", WText("Different simple widgets", Point(20, 20), 20, Color::green()));
    viz.showWidget("text3d", WText3D("Simple 3D text", Point3d( 0.5,  0.5, 0.5), 0.125, false, Color::green()));

    viz.showWidget("plane1", WPlane(Size2d(0.25, 0.75)));
    viz.showWidget("plane2", WPlane(Vec3d(0.5, -0.5, -0.5), Vec3d(0.0, 1.0, 1.0), Vec3d(1.0, 1.0, 0.0), Size2d(1.0, 0.5), Color::gold()));

    viz.showWidget("grid1", WGrid(Vec2i(7,7), Vec2d::all(0.75), Color::gray()), Affine3d().translate(Vec3d(0.0, 0.0, -1.0)));

    viz.spin();
    viz.getWidget("text2d").cast<WText>().setText("Different simple widgets (updated)");
    viz.getWidget("text3d").cast<WText3D>().setText("Updated text 3D");
    viz.spin();
}

TEST(Viz, show_follower)
{
    Viz3d viz("show_follower");

    viz.showWidget("coos", WCoordinateSystem());
    viz.showWidget("cube", WCube());
    viz.showWidget("t3d_2", WText3D("Simple 3D follower", Point3d(-0.5, -0.5, 0.5), 0.125, true,  Color::green()));
    viz.showWidget("text2d", WText("Follower: text always facing camera", Point(20, 20), 20, Color::green()));
    viz.setBackgroundMeshLab();
    viz.spin();
    viz.getWidget("t3d_2").cast<WText3D>().setText("Updated follower 3D");
    viz.spin();
}
