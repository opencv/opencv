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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
//     and/or other materials provided with the distribution.
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
// Authors:
//  * Ozan Tonkal, ozantonkal@gmail.com
//  * Anatoly Baksheev, Itseez Inc.  myname.mysurname <> mycompany.com
//
//M*/

#ifndef OPENCV_VIZ_TEST_COMMON_HPP
#define OPENCV_VIZ_TEST_COMMON_HPP

#include <opencv2/viz/vizcore.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <limits>

namespace cv
{
    struct Path
    {
        static String combine(const String& item1, const String& item2);
        static String combine(const String& item1, const String& item2, const String& item3);
        static String change_extension(const String& file, const String& ext);
    };

    inline cv::String get_dragon_ply_file_path()
    {
        return Path::combine(cvtest::TS::ptr()->get_data_path(), "dragon.ply");
    }

    template<typename _Tp>
    inline std::vector< Affine3<_Tp> > generate_test_trajectory()
    {
        std::vector< Affine3<_Tp> > result;

        for (int i = 0, j = 0; i <= 270; i += 3, j += 10)
        {
            double x = 2 * cos(i * 3 * CV_PI/180.0) * (1.0 + 0.5 * cos(1.2 + i * 1.2 * CV_PI/180.0));
            double y = 0.25 + i/270.0 + sin(j * CV_PI/180.0) * 0.2 * sin(0.6 + j * 1.5 * CV_PI/180.0);
            double z = 2 * sin(i * 3 * CV_PI/180.0) * (1.0 + 0.5 * cos(1.2 + i * CV_PI/180.0));
            result.push_back(viz::makeCameraPose(Vec3d(x, y, z), Vec3d::all(0.0), Vec3d(0.0, 1.0, 0.0)));
        }
        return result;
    }

    inline Mat make_gray(const Mat& image)
    {
        Mat chs[3]; split(image, chs);
        return 0.114 * chs[0] + 0.58 * chs[1] + 0.3 * chs[2];
    }
}

#endif
