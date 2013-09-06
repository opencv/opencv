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
#include <opencv2/viz.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <string>

#include <opencv2/viz.hpp>
using namespace cv;

cv::Mat cvcloud_load()
{
    cv::Mat cloud(1, 20000, CV_32FC3);
        std::ifstream ifs("/Users/nerei/cloud_dragon.ply");

    std::string str;
    for(size_t i = 0; i < 11; ++i)
        std::getline(ifs, str);

    cv::Point3f* data = cloud.ptr<cv::Point3f>();
    for(size_t i = 0; i < 20000; ++i)
        ifs >> data[i].x >> data[i].y >> data[i].z;

    return cloud;
}

TEST(Viz_viz3d, accuracy)
{
    cv::Mat cloud = cvcloud_load();
    cv::Mat colors(cloud.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    cv::Mat normals(cloud.size(), cloud.type(), cv::Scalar(0, 10, 0));
    //cv::viz::Mesh3d::Ptr mesh = cv::viz::Mesh3d::mesh_load("/Users/nerei/horse.ply");

    const Vec4d data[] = { Vec4d(0.0, 0.0, 0.0, 0.0), Vec4d(1.0, 1.0, 1.0, 1.0), cv::Vec4d(0.0, 2.0, 0.0, 0.0), cv::Vec4d(3.0, 4.0, 1.0, 1.0) };
    cv::Mat points(1, sizeof(data)/sizeof(data[0]), CV_64FC4, (void*)data);
    points = points.reshape(4, 2);

    cv::viz::Viz3d viz("abc");
    viz.setBackgroundColor();
    
    Vec3f angle = Vec3f::all(0);
    Vec3f pos = Vec3f::all(0);

    //viz.addPolygonMesh(*mesh, "pq");

    viz::Color color = viz::Color::black();

    viz::LineWidget lw(Point3f(0, 0, 0), Point3f(4.f, 4.f,4.f), viz::Color::green());
    viz::PlaneWidget pw(Vec4f(0.0,1.0,2.0,3.0));
    viz::PlaneWidget pw2(Vec4f(0.0,1.0,2.0,3.0), 2.0, viz::Color::red());
    viz::PlaneWidget pw3(Vec4f(0.0,1.0,2.0,3.0), 3.0, viz::Color::blue());
    viz::SphereWidget sw(Point3f(0, 0, 0), 0.2);
    viz::ArrowWidget aw(Point3f(0, 0, 0), Point3f(1, 1, 1), 0.01, viz::Color::red());
    viz::CircleWidget cw(Point3f(0, 0, 0), 0.5, 0.01, viz::Color::green());
    viz::CylinderWidget cyw(Point3f(0, 0, 0), Point3f(-1, -1, -1), 0.5, 30, viz::Color::green());
    viz::CubeWidget cuw(Point3f(-2, -2, -2), Point3f(-1, -1, -1));
    viz::CoordinateSystemWidget csw;
    viz::TextWidget tw("TEST", Point(100, 100), 20);
    viz::CloudWidget pcw(cloud, colors);
    viz::CloudWidget pcw2(cloud, viz::Color::magenta());
    
//     viz.showWidget("line", lw);
    viz.showWidget("plane", pw);
    viz.showWidget("plane2", pw2);
    viz.showWidget("plane3", pw3);
//     viz.showWidget("sphere", sw);
//     viz.showWidget("arrow", aw);
//     viz.showWidget("circle", cw);
//     viz.showWidget("cylinder", cyw);
//     viz.showWidget("cube", cuw);
    viz.showWidget("coordinateSystem", csw);
//     viz.showWidget("coordinateSystem2", viz::CoordinateSystemWidget(2.0), Affine3f().translate(Vec3f(2, 0, 0)));
//     viz.showWidget("text",tw);
//     viz.showWidget("pcw",pcw);
//     viz.showWidget("pcw2",pcw2);
    
//     viz::LineWidget lw2 = lw;
//     v.showPointCloud("cld",cloud, colors);

//     v.addPointCloudNormals(cloud, normals, 100, 0.02, "n");
    //viz::CloudNormalsWidget cnw(cloud, normals);
     //v.showWidget("n", cnw);

    
//     lw = v.getWidget("n").cast<viz::LineWidget>();
//     pw = v.getWidget("n").cast<viz::PlaneWidget>();
    
    
    viz::PolyLineWidget plw(points, viz::Color::green());
//     viz.showWidget("polyline", plw);
//     lw = v.getWidget("polyline").cast<viz::LineWidget>();
    
    viz::Mesh3d mesh = cv::viz::Mesh3d::loadMesh("/Users/nerei/horse.ply");
    
    viz::MeshWidget mw(mesh);
//     viz.showWidget("mesh", mw);
    
    Mat img = imread("opencv.png");
//     resize(img, img, Size(50,50));
//     viz.showWidget("img", viz::ImageOverlayWidget(img, Point2i(50,50)));
    
    Matx33f K(657, 0, 320, 
              0, 657, 240, 
              0, 0, 1);
    
    //viz::CameraPositionWidget cpw(Vec3f(0.5, 0.5, 3.0), Vec3f(0.0,0.0,0.0), Vec3f(0.0,-1.0,0.0), 0.5);
    viz::CameraPositionWidget cpw2(0.5);
    viz::CameraPositionWidget frustum(K, 2.0, viz::Color::green());
//     viz::CameraPositionWidget frustum2(K, 4.0, viz::Color::red());
    viz::CameraPositionWidget frustum2(K, 4.0, viz::Color::red());
    viz::CameraPositionWidget frustum3(Vec2f(CV_PI, CV_PI/2), 4.0);
    viz::Text3DWidget t3w1("Camera1", Point3f(0.4, 0.6, 3.0), 0.1);
    viz::Text3DWidget t3w2("Camera2", Point3f(0,0,0), 0.1);
    
//     viz.showWidget("CameraPositionWidget", cpw);
//     viz.showWidget("CameraPositionWidget2", cpw2, Affine3f(0.524, 0, 0, Vec3f(-1.0, 0.5, 0.5)));
//     viz.showWidget("camera_label", t3w1);
//     viz.showWidget("camera_label2", t3w2, Affine3f(0.524, 0, 0, Vec3f(-1.0, 0.5, 0.5)));
//     viz.showWidget("frustrum", frustum, Affine3f(0.524, 0, 0, Vec3f(-1.0, 0.5, 0.5)));
//     viz.showWidget("frustrum2", frustum2, Affine3f(0.524, 0, 0, Vec3f(-1.0, 0.5, 0.5)));
//     viz.showWidget("frustum3", frustum3, Affine3f(0.524, 0, 0, Vec3f(-1.0, 0.5, 0.5)));
    
    std::vector<Affine3f> trajectory;
    
    trajectory.push_back(Affine3f().translate(Vec3f(0.5,0.5,0.5)));
    trajectory.push_back(Affine3f().translate(Vec3f(1.0,0.0,0.0)));
    trajectory.push_back(Affine3f().translate(Vec3f(2.0,0.5,0.0)));
    trajectory.push_back(Affine3f(0.5, 0.0, 0.0, Vec3f(1.0,0.0,1.0)));
//     
    //viz.showWidget("trajectory1", viz::TrajectoryWidget(trajectory, viz::Color(0,255,255), true, 0.5));
    viz.showWidget("trajectory2", viz::TrajectoryWidget(trajectory, K, 1.0, viz::Color(255,0,255)));
    
    
    
//     viz.showWidget("trajectory1", viz::TrajectoryWidget(trajectory/*, viz::Color::yellow()*/));
    
//     viz.showWidget("CameraPositionWidget2", cpw2);
//     viz.showWidget("CameraPositionWidget3", cpw3);
    
    viz.spin();

    //viz::GridWidget gw(viz::Vec2i(100,100), viz::Vec2d(1,1));
    //v.showWidget("grid", gw);
//     lw = viz.getWidget("grid").cast<cv::viz::LineWidget>();
    
    //viz::Text3DWidget t3w("OpenCV", cv::Point3f(0.0, 2.0, 0.0), 1.0, viz::Color(255,255,0));
    //v.showWidget("txt3d", t3w);

//     float grid_x_angle = 0.0;
    
    
    while(!viz.wasStopped())
    {
        // Creating new point cloud with id cloud1
        cv::Affine3f cloudPosition(angle, pos);
        cv::Affine3f cloudPosition2(angle, pos + Vec3f(0.2f, 0.2f, 0.2f));

        lw.setColor(color);
//         lw.setLineWidth(pos_x * 10);
        
        //plw.setColor(viz::Color(col_blue, col_green, col_red));
        
//         sw.setPose(cloudPosition);
//         pw.setPose(cloudPosition);
        aw.setPose(cloudPosition);
        cw.setPose(cloudPosition);
        cyw.setPose(cloudPosition);
        
        frustum.setPose(cloudPosition);
//         lw.setPose(cloudPosition);
//         cpw.updatePose(Affine3f(0.1,0.0,0.0, cv::Vec3f(0.0,0.0,0.0)));
//         cpw.setPose(cloudPosition);
//         cnw.setPose(cloudPosition);
//         v.showWidget("pcw",pcw, cloudPosition);
//         v.showWidget("pcw2",pcw2, cloudPosition2);
//         v.showWidget("plane", pw, cloudPosition);
        
//         v.setWidgetPose("n",cloudPosition);
//         v.setWidgetPose("pcw2", cloudPosition);
        //cnw.setColor(viz::Color(col_blue, col_green, col_red));
        //pcw2.setColor(viz::Color(col_blue, col_green, col_red));
        
        //gw.updatePose(viz::Affine3f(0.0, 0.1, 0.0, cv::Vec3f(0.0,0.0,0.0)));
        
        angle[0] += 0.1f;
        angle[1] -= 0.1f;
        angle[2] += 0.1f;
        pos[0] = std::sin(angle[0]);
        pos[1] = std::sin(angle[1]);
        pos[2] = std::sin(angle[2]);

        color[0] = int(angle[0] * 10) % 256;
        color[1] = int(angle[0] * 20) % 256;
        color[2] = int(angle[0] * 30) % 256;

        viz.spinOnce(1, true);
    }
   

// 
// 
//     viz::ModelCoefficients mc;
//     mc.values.resize(4);
//     mc.values[0] = mc.values[1] = mc.values[2] = mc.values[3] = 1;
//     v.addPlane(mc);
// 
// 
//     viz::Mesh3d::Ptr mesh = viz::mesh_load("horse.ply");
//     v.addPolygonMesh(*mesh, "pq");
// 
//     v.spinOnce(1000, true);
// 
//     v.removeCoordinateSystem();
// 
//     for(int i = 0; i < mesh->cloud.cols; ++i)
//         mesh->cloud.ptr<cv::Point3f>()[i] += cv::Point3f(1, 1, 1);
// 
//     v.updatePolygonMesh(*mesh, "pq");
// 
// 
//     for(int i = 0; i < mesh->cloud.cols; ++i)
//         mesh->cloud.ptr<cv::Point3f>()[i] -= cv::Point3f(2, 2, 2);
//     v.addPolylineFromPolygonMesh(*mesh);
// 
// 
//     v.addText("===Abd sadfljsadlk", 100, 100, cv::Scalar(255, 0, 0), 15);
//     for(int i = 0; i < cloud.cols; ++i)
// 	cloud.ptr<cv::Point3f>()[i].x *=2;
// 
//     colors.setTo(cv::Scalar(255, 0, 0));
// 
//     v.addSphere(cv::Point3f(0, 0, 0), 0.3, viz::Color::blue());
// 
//     cv::Mat cvpoly(1, 5, CV_32FC3);
//     cv::Point3f* pdata = cvpoly.ptr<cv::Point3f>();
//     pdata[0] = cv::Point3f(0, 0, 0);
//     pdata[1] = cv::Point3f(0, 1, 1);
//     pdata[2] = cv::Point3f(3, 1, 2);
//     pdata[3] = cv::Point3f(0, 2, 4);
//     pdata[4] = cv::Point3f(7, 2, 3);
//     v.addPolygon(cvpoly, viz::Color::white());
// 
//     // Updating cloud1
//     v.showPointCloud("cloud1", cloud, colors);
//     v.spin();
}

