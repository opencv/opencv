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

    Viz3d viz("show_cloud_bluberry");
    viz.setBackgroundColor();
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("dragon", WCloud(dragon_cloud, Color::bluberry()));
    viz.spin();
}

TEST(Viz, DISABLED_show_cloud_random_color)
{
    Mat dragon_cloud = readCloud(get_dragon_ply_file_path());

    Mat colors(dragon_cloud.size(), CV_8UC3);
    theRNG().fill(colors, RNG::UNIFORM, 0, 255);

    Viz3d viz("show_cloud_random_color");
    viz.setBackgroundColor();
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("dragon", WCloud(dragon_cloud, colors));
    viz.spin();
}

TEST(Viz, DISABLED_show_cloud_masked)
{
    Mat dragon_cloud = readCloud(get_dragon_ply_file_path());

    Vec3f qnan = Vec3f::all(std::numeric_limits<float>::quiet_NaN());
    for(size_t i = 0; i < dragon_cloud.total(); ++i)
        if (i % 15 != 0)
            dragon_cloud.at<Vec3f>(i) = qnan;

    Viz3d viz("show_cloud_masked");
    viz.setBackgroundColor();
    viz.showWidget("coosys", WCoordinateSystem());
    viz.showWidget("dragon", WCloud(dragon_cloud));
    viz.spin();
}
