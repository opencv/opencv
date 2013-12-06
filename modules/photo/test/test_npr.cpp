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
#include "opencv2/photo.hpp"
#include <string>

using namespace cv;
using namespace std;


TEST(Photo_NPR_EdgePreserveSmoothing_RecursiveFilter, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "npr/";
    string original_path = folder + "test1.png";

    Mat source = imread(original_path, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load input image " << original_path;

    Mat result;
    edgePreservingFilter(source,result,1);

    imwrite(folder + "smoothened_RF.png", result);

}

TEST(Photo_NPR_EdgePreserveSmoothing_NormConvFilter, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "npr/";
    string original_path = folder + "test1.png";

    Mat source = imread(original_path, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load input image " << original_path;

    Mat result;
    edgePreservingFilter(source,result,2);

    imwrite(folder + "smoothened_NCF.png", result);

}

TEST(Photo_NPR_DetailEnhance, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "npr/";
    string original_path = folder + "test1.png";

    Mat source = imread(original_path, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load input image " << original_path;

    Mat result;
    detailEnhance(source,result);

    imwrite(folder + "detail_enhanced.png", result);

}

TEST(Photo_NPR_PencilSketch, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "npr/";
    string original_path = folder + "test1.png";

    Mat source = imread(original_path, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load input image " << original_path;

    Mat result,result1;
    pencilSketch(source,result,result1, 10, 0.1f, 0.03f);

    imwrite(folder + "pencil_sketch.png", result);
    imwrite(folder + "color_pencil_sketch.png", result1);

}

TEST(Photo_NPR_Stylization, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "npr/";
    string original_path = folder + "test1.png";

    Mat source = imread(original_path, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load input image " << original_path;

    Mat result;
    stylization(source,result);

    imwrite(folder + "stylized.png", result);

}
