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
    std::cout << std::string(cvtest::TS::ptr()->get_data_path()) + "dragon.ply" << std::endl;

    cv::Mat cloud = cv::viz::readCloud(String(cvtest::TS::ptr()->get_data_path()) + "dragon.ply");


//    for(size_t i = 0; i < cloud.total(); ++i)
//    {
//        if (i % 15 == 0)
//            continue;
//        const static float qnan = std::numeric_limits<float>::quiet_NaN();
//        cloud.at<Vec3f>(i) = Vec3f(qnan, qnan, qnan);
//    }

    cv::viz::Viz3d viz("abc");
    viz.showWidget("coo", cv::viz::WCoordinateSystem());

    cv::Mat colors(cloud.size(), CV_8UC3, cv::Scalar(0, 255, 0));

    //viz.showWidget("h", cv::viz::Widget::fromPlyFile("d:/horse-red.ply"));
    //viz.showWidget("a", cv::viz::WArrow(cv::Point3f(0,0,0), cv::Point3f(1,1,1)));

    cv::RNG rng;
    rng.fill(colors, cv::RNG::UNIFORM, 0, 255);
    viz.showWidget("c", cv::viz::WCloud(cloud, colors));
    //viz.showWidget("c", cv::viz::WCloud(cloud, cv::viz::Color::bluberry()));

    //viz.showWidget("l", cv::viz::WLine(Point3f(0,0,0), Point3f(1,1,1)));
    //viz.showWidget("s", cv::viz::WSphere(Point3f(0,0,0), 1));
    //viz.showWidget("d", cv::viz::WCircle(Point3f(0,0,0), 1));
    viz.spin();
}
