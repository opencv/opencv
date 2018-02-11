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
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Photo_Decolor, regression)
{
        string folder = string(cvtest::TS::ptr()->get_data_path()) + "decolor/";
        string original_path = folder + "color_image_1.png";

        Mat original = imread(original_path, IMREAD_COLOR);

        ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
        ASSERT_EQ(3, original.channels()) << "Load color input image " << original_path;

        Mat grayscale, color_boost;
        decolor(original, grayscale, color_boost);

        Mat reference_grayscale = imread(folder + "grayscale_reference.png", 0 /* == grayscale image*/);
        double gray_psnr = cvtest::PSNR(reference_grayscale, grayscale);
        EXPECT_GT(gray_psnr, 60.0);

        Mat reference_boost = imread(folder + "boost_reference.png");
        double boost_psnr = cvtest::PSNR(reference_boost, color_boost);
        EXPECT_GT(boost_psnr, 60.0);
}

}} // namespace
