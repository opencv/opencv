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

#include <fstream>
#include <string>

#include <opencv2/viz/types.hpp>
#include <opencv2/viz/mesh_load.hpp>


cv::Mat cvcloud_load()
{
    cv::Mat cloud(1, 20000, CV_32FC3);
        std::ifstream ifs("d:/cloud_dragon.ply");

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
    temp_viz::Viz3d v("abc");
    //v.spin();

    v.setBackgroundColor();

    v.addCoordinateSystem(1.0, cv::Affine3f::Identity());

    cv::Mat cloud = cvcloud_load();

    cv::Mat colors(cloud.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    
    float angle_x = 0.0f;
    float angle_y = 0.0f;
    float angle_z = 0.0f;
    float pos_x = 0.0f;
    float pos_y = 0.0f;
    float pos_z = 0.0f;
    temp_viz::Mesh3d::Ptr mesh = temp_viz::mesh_load("d:/horse.ply");
    v.addPolygonMesh(*mesh, "pq");

    int col_blue = 0;
    int col_green = 0;
    int col_red = 0;

    while(!v.wasStopped())
    {
        // Creating new point cloud with id cloud1
        cv::Affine3f cloudPosition(angle_x, angle_y, angle_z, cv::Vec3f(pos_x, pos_y, pos_z));
        v.showPointCloud("cloud1", cloud, temp_viz::Color(col_blue, col_green, col_red), cloudPosition);

        angle_x += 0.1f;
        angle_y -= 0.1f;
        angle_z += 0.1f;
        pos_x = std::sin(angle_x);
        pos_y = std::sin(angle_x);
        pos_z = std::sin(angle_x);
        col_blue = int(angle_x * 10) % 256;
        col_green = int(angle_x * 20) % 256;
        col_red = int(angle_x * 30) % 256;

        v.spinOnce(1, true);
    }
   
//     cv::Mat normals(cloud.size(), CV_32FC3, cv::Scalar(0, 10, 0));
// 
//     v.addPointCloudNormals(cloud, normals, 100, 0.02, "n");
// 
// 
//     temp_viz::ModelCoefficients mc;
//     mc.values.resize(4);
//     mc.values[0] = mc.values[1] = mc.values[2] = mc.values[3] = 1;
//     v.addPlane(mc);
// 
// 
//     temp_viz::Mesh3d::Ptr mesh = temp_viz::mesh_load("horse.ply");
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
//     v.addSphere(cv::Point3f(0, 0, 0), 0.3, temp_viz::Color::blue());
// 
//     cv::Mat cvpoly(1, 5, CV_32FC3);
//     cv::Point3f* pdata = cvpoly.ptr<cv::Point3f>();
//     pdata[0] = cv::Point3f(0, 0, 0);
//     pdata[1] = cv::Point3f(0, 1, 1);
//     pdata[2] = cv::Point3f(3, 1, 2);
//     pdata[3] = cv::Point3f(0, 2, 4);
//     pdata[4] = cv::Point3f(7, 2, 3);
//     v.addPolygon(cvpoly, temp_viz::Color::white());
// 
//     // Updating cloud1
//     v.showPointCloud("cloud1", cloud, colors);
//     v.spin();
}

