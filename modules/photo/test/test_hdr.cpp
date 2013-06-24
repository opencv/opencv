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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

TEST(Photo_MakeHdr, regression)
{
	string folder = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";
	
	vector<string>file_names(3);
	file_names[0] = folder + "grand_canal_1_45.jpg";
	file_names[1] = folder + "grand_canal_1_180.jpg";
	file_names[2] = folder + "grand_canal_1_750.jpg";
	vector<Mat>images(3);
	for(int i = 0; i < 3; i++) {
		images[i] = imread(file_names[i]);
		ASSERT_FALSE(images[i].empty()) << "Could not load input image " << file_names[i];
	}
	
	string expected_path = folder + "grand_canal_rle.hdr";
	Mat expected = imread(expected_path, -1);
	ASSERT_FALSE(expected.empty()) << "Could not load input image " << expected_path;

	vector<float>times(3);
	times[0] = 1.0f/45.0f;
	times[1] = 1.0f/180.0f;
	times[2] = 1.0f/750.0f;
	
	Mat result;
	makeHDR(images, times, result);
	double min = 0.0, max = 1.0;
	minMaxLoc(abs(result - expected), &min, &max);
	ASSERT_TRUE(max < 0.01);
}