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

TEST(Viz_viz3d, develop)
{
    cv::Mat cloud = cv::viz::readCloud(get_dragon_ply_file_path());

    //cv::viz::Mesh3d mesh = cv::viz::Mesh3d::load(get_dragon_ply_file_path());

    //theRNG().fill(mesh.colors, RNG::UNIFORM, 0, 255);

    cv::viz::Viz3d viz("abc");
    viz.setBackgroundColor(cv::viz::Color::mlab());
    viz.showWidget("coo", cv::viz::WCoordinateSystem(0.1));


//    double c = cos(CV_PI/6);
//    std::vector<Vec3d> pts;
//    pts.push_back(Vec3d(0, 0.0, -1.0));
//    pts.push_back(Vec3d(1,   c, -0.5));
//    pts.push_back(Vec3d(2,   c,  0.5));
//    pts.push_back(Vec3d(3, 0.0,  1.0));
//    pts.push_back(Vec3d(4,  -c,  0.5));
//    pts.push_back(Vec3d(5,  -c, -0.5));

//    viz.showWidget("pl", cv::viz::WPolyLine(Mat(pts), cv::viz::Color::green()));

    //viz.showWidget("pl", cv::viz::WPolyLine(cloud.colRange(0, 100), cv::viz::Color::green()));
    //viz.spin();

    //cv::Mat colors(cloud.size(), CV_8UC3, cv::Scalar(0, 255, 0));

    //viz.showWidget("h", cv::viz::Widget::fromPlyFile("d:/horse-red.ply"));
    //viz.showWidget("a", cv::viz::WArrow(cv::Point3f(0,0,0), cv::Point3f(1,1,1)));

    std::vector<cv::Affine3d> gt, es;
    cv::viz::readTrajectory(gt, "d:/Datasets/trajs/gt%05d.xml");
    //cv::viz::readTrajectory(es, "d:/Datasets/trajs/es%05d.xml");
    gt.resize(20);

    Affine3d inv = gt[0].inv();
    for(size_t i = 0; i < gt.size(); ++i)
        gt[i] = inv * gt[i];

    //viz.showWidget("gt", viz::WTrajectory(gt, viz::WTrajectory::PATH, 1.f, viz::Color::blue()), gt[0].inv());
    viz.showWidget("gt", viz::WTrajectory(gt, viz::WTrajectory::BOTH, 0.01f, viz::Color::blue()));

    //viz.showWidget("tr", viz::WTrajectory(es, viz::WTrajectory::PATH, 1.f, viz::Color::red()), gt[0].inv());

    //theRNG().fill(colors, cv::RNG::UNIFORM, 0, 255);
    //viz.showWidget("c", cv::viz::WCloud(cloud, colors));
    //viz.showWidget("c", cv::viz::WCloud(cloud, cv::viz::Color::bluberry()));

    //viz.showWidget("l", cv::viz::WLine(Point3f(0,0,0), Point3f(1,1,1)));
    //viz.showWidget("s", cv::viz::WSphere(Point3f(0,0,0), 1));
    //viz.showWidget("d", cv::viz::WCircle(Point3f(0,0,0), 1));
    viz.spin();
}
