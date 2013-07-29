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
#include <fstream>

using namespace cv;
using namespace std;

void loadImage(string path, Mat &img) 
{
	img = imread(path, -1);
	ASSERT_FALSE(img.empty()) << "Could not load input image " << path;
}

void checkEqual(Mat img0, Mat img1, double threshold)
{
	double max = 1.0;
	minMaxLoc(abs(img0 - img1), NULL, &max);
	ASSERT_FALSE(max > threshold);
}

TEST(Photo_HdrFusion, regression)
{
	string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";
	string fuse_path = test_path + "fusion/";
	
	vector<float> times;
	vector<Mat> images;

	ifstream list_file(fuse_path + "list.txt");
	ASSERT_TRUE(list_file.is_open());
	string name; 
	float val;
	while(list_file >> name >> val) {
		Mat img = imread(fuse_path + name);
		ASSERT_FALSE(img.empty()) << "Could not load input image " << fuse_path + name;
		images.push_back(img);
		times.push_back(1 / val);
	}
	list_file.close();

	Mat response, expected(256, 3, CV_32F);
	ifstream resp_file(test_path + "response.csv");
	for(int i = 0; i < 256; i++) {
		for(int channel = 0; channel < 3; channel++) {
			resp_file >> expected.at<float>(i, channel);
			resp_file.ignore(1);
		}
	}
	resp_file.close();

	estimateResponse(images, times, response);
	checkEqual(expected, response, 0.001);

	Mat result;
	loadImage(test_path + "no_calibration.hdr", expected);
	makeHDR(images, times, result);
	checkEqual(expected, result, 0.01);

	loadImage(test_path + "rle.hdr", expected);
	makeHDR(images, times, result, response);
	checkEqual(expected, result, 0.01);

	loadImage(test_path + "exp_fusion.png", expected);
	exposureFusion(images, result);
	result.convertTo(result, CV_8UC3, 255);
	checkEqual(expected, result, 0);
}

TEST(Photo_Tonemap, regression)
{
	string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/tonemap/";

	Mat img;
	loadImage(test_path + "../rle.hdr", img);
	ifstream list_file(test_path + "list.txt");
	ASSERT_TRUE(list_file.is_open());

	string name; 
	while(list_file >> name) {
		Mat expected = imread(test_path + name + ".png");
		ASSERT_FALSE(img.empty()) << "Could not load input image " << test_path + name + ".png";
		Ptr<Tonemap> mapper = Tonemap::create(name);	
		ASSERT_FALSE(mapper.empty()) << "Could not find mapper " << name;
		Mat result;
		mapper->process(img, result);
		result.convertTo(result, CV_8UC3, 255);
		checkEqual(expected, result, 0);
	}
	list_file.close();
}

TEST(Photo_Align, regression)
{
	const int TESTS_COUNT = 100;
	string folder = string(cvtest::TS::ptr()->get_data_path()) + "shared/";
	
	string file_name = folder + "lena.png";
	Mat img = imread(file_name);
	ASSERT_FALSE(img.empty()) << "Could not load input image " << file_name;
	cvtColor(img, img, COLOR_RGB2GRAY);

	int max_bits = 5;
	int max_shift = 32;
	srand(static_cast<unsigned>(time(0)));
	int errors = 0;

	for(int i = 0; i < TESTS_COUNT; i++) {
		Point shift(rand() % max_shift, rand() % max_shift);
		Mat res;
		shiftMat(img, shift, res);
		Point calc = getExpShift(img, res, max_bits);
		errors += (calc != -shift);
	}
	ASSERT_TRUE(errors < 5);
}
