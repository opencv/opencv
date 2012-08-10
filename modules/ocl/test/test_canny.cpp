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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

#include "precomp.hpp"

#ifdef WIN32
#define FILTER_IMAGE "C:/Users/Public/Pictures/Sample Pictures/Penguins.jpg"
#else
#define FILTER_IMAGE "/Users/Test/Valve_original.PNG" // user need to specify a valid image path
#endif
#define SHOW_RESULT 0

////////////////////////////////////////////////////////
// Canny

IMPLEMENT_PARAM_CLASS(AppertureSize, int);
IMPLEMENT_PARAM_CLASS(L2gradient, bool);

PARAM_TEST_CASE(Canny, AppertureSize, L2gradient)
{
    int apperture_size;
    bool useL2gradient;

    cv::Mat edges_gold;
	std::vector<cv::ocl::Info> oclinfo;
    virtual void SetUp()
    {
        apperture_size = GET_PARAM(0);
        useL2gradient = GET_PARAM(1);
		int devnums = getDevice(oclinfo);
		CV_Assert(devnums > 0);
    }
};

TEST_P(Canny, Accuracy)
{
    cv::Mat img = readImage(FILTER_IMAGE, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    double low_thresh = 50.0;
    double high_thresh = 100.0;

	cv::resize(img, img, cv::Size(512, 384));
	cv::ocl::oclMat ocl_img = cv::ocl::oclMat(img);

	cv::ocl::oclMat edges;
	cv::ocl::Canny(ocl_img, edges, low_thresh, high_thresh, apperture_size, useL2gradient);

	char filename [100];
	sprintf(filename, "G:/Valve_edges_a%d_L2Grad%d.jpg", apperture_size, (int)useL2gradient);

	cv::Mat edges_gold;
	cv::Canny(img, edges_gold, low_thresh, high_thresh, apperture_size, useL2gradient);

#if SHOW_RESULT
	cv::Mat edges_x2, ocl_edges(edges);
	edges_x2.create(edges.rows, edges.cols * 2, edges.type());
	edges_x2.setTo(0);
	cv::add(edges_gold,cv::Mat(edges_x2,cv::Rect(0,0,edges_gold.cols,edges_gold.rows)), cv::Mat(edges_x2,cv::Rect(0,0,edges_gold.cols,edges_gold.rows)));
	cv::add(ocl_edges,cv::Mat(edges_x2,cv::Rect(edges_gold.cols,0,edges_gold.cols,edges_gold.rows)), cv::Mat(edges_x2,cv::Rect(edges_gold.cols,0,edges_gold.cols,edges_gold.rows)));
	cv::namedWindow("Canny result (left: cpu, right: ocl)");
    cv::imshow("Canny result (left: cpu, right: ocl)", edges_x2);
	cv::waitKey();
#endif //OUTPUT_RESULT
	EXPECT_MAT_SIMILAR(edges_gold, edges, 1e-2);
}

INSTANTIATE_TEST_CASE_P(ocl_ImgProc, Canny, testing::Combine(
    testing::Values(AppertureSize(3), AppertureSize(5)),
    testing::Values(L2gradient(false), L2gradient(true))));
