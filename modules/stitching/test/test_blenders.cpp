/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
using namespace std;

TEST(MultiBandBlender, CanBlendTwoImages)
{    
    Mat image1 = imread(string(cvtest::TS::ptr()->get_data_path()) + "cv/shared/baboon.jpg");
    Mat image2 = imread(string(cvtest::TS::ptr()->get_data_path()) + "cv/shared/lena.jpg");
    ASSERT_EQ(image1.rows, image2.rows); ASSERT_EQ(image1.cols, image2.cols);

    Mat image1s, image2s;
    image1.convertTo(image1s, CV_16S);
    image2.convertTo(image2s, CV_16S);

    Mat mask1(image1s.size(), CV_8U);
    mask1(Rect(0, 0, mask1.cols/2, mask1.rows)).setTo(255);
    mask1(Rect(mask1.cols/2, 0, mask1.cols - mask1.cols/2, mask1.rows)).setTo(0);

    Mat mask2(image2s.size(), CV_8U);
    mask2(Rect(0, 0, mask2.cols/2, mask2.rows)).setTo(0);
    mask2(Rect(mask2.cols/2, 0, mask2.cols - mask2.cols/2, mask2.rows)).setTo(255);

    detail::MultiBandBlender blender(false, 5);

    blender.prepare(Rect(0, 0, max(image1s.cols, image2s.cols), max(image1s.rows, image2s.rows)));
    blender.feed(image1s, mask1, Point(0,0));
    blender.feed(image2s, mask2, Point(0,0));

    Mat result_s, result_mask;
    blender.blend(result_s, result_mask);
    Mat result; result_s.convertTo(result, CV_8U);

    Mat expected = imread(string(cvtest::TS::ptr()->get_data_path()) + "stitching/baboon_lena.png");
    double rmsErr = norm(expected, result, NORM_L2) / sqrt(double(expected.size().area()));
    ASSERT_LT(rmsErr, 1e-3);
}
