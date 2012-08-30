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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//		Wenju He, wenju@multicorewareinc.com
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

#include "precomp.hpp"
#include "opencv2/core/core.hpp"
using namespace std;
#ifdef HAVE_OPENCL


PARAM_TEST_CASE(HOG,cv::Size,int)
{
	cv::Size winSize;
	int type;
	//vector<cv::ocl::Info> info;
	virtual void SetUp()
	{
		winSize = GET_PARAM(0);
		type = GET_PARAM(1);
		//cv::ocl::getDevice(info);
	}
};

TEST_P(HOG, GetDescriptors)
{
    // Load image
    cv::Mat img_rgb = readImage("../../../samples/gpu/road.png");
    ASSERT_FALSE(img_rgb.empty());

    // Convert image
    cv::Mat img;
    switch (type)
    {
    case CV_8UC1:
        cv::cvtColor(img_rgb, img, CV_BGR2GRAY);
        break;
    case CV_8UC4:
    default:
        cv::cvtColor(img_rgb, img, CV_BGR2BGRA);
        break;
    }
    cv::ocl::oclMat d_img(img);

    // HOGs
    cv::ocl::HOGDescriptor ocl_hog;
    ocl_hog.gamma_correction = true;
    cv::HOGDescriptor hog;
    hog.gammaCorrection = true;

    // Compute descriptor
    cv::ocl::oclMat d_descriptors;
    ocl_hog.getDescriptors(d_img, ocl_hog.win_size, d_descriptors, ocl_hog.DESCR_FORMAT_COL_BY_COL);
    cv::Mat down_descriptors;
    d_descriptors.download(down_descriptors);
    down_descriptors = down_descriptors.reshape(0, down_descriptors.cols * down_descriptors.rows);

    hog.setSVMDetector(hog.getDefaultPeopleDetector());
    std::vector<float> descriptors;
    switch (type)
    {
    case CV_8UC1:
        hog.compute(img, descriptors, ocl_hog.win_size);
        break;
    case CV_8UC4:
    default:
        hog.compute(img_rgb, descriptors, ocl_hog.win_size);
        break;
    }
    cv::Mat cpu_descriptors(descriptors);

    EXPECT_MAT_SIMILAR(down_descriptors, cpu_descriptors, 1e-2);
}


TEST_P(HOG, Detect)
{
    // Load image
    cv::Mat img_rgb = readImage("../../../samples/gpu/road.png");
    ASSERT_FALSE(img_rgb.empty());

    // Convert image
    cv::Mat img;
    switch (type)
    {
    case CV_8UC1:
        cv::cvtColor(img_rgb, img, CV_BGR2GRAY);
        break;
    case CV_8UC4:
    default:
        cv::cvtColor(img_rgb, img, CV_BGR2BGRA);
        break;
    }
    cv::ocl::oclMat d_img(img);

    // HOGs
    if ((winSize != cv::Size(48, 96)) && (winSize != cv::Size(64, 128)))
        winSize = cv::Size(64, 128);
    cv::ocl::HOGDescriptor ocl_hog(winSize);
    ocl_hog.gamma_correction = true;

    cv::HOGDescriptor hog;
    hog.winSize = winSize;
    hog.gammaCorrection = true;

    if (winSize.width == 48 && winSize.height == 96)
    {
        // daimler's base
        ocl_hog.setSVMDetector(ocl_hog.getPeopleDetector48x96());
        hog.setSVMDetector(hog.getDaimlerPeopleDetector());
    }
    else if (winSize.width == 64 && winSize.height == 128)
    {
        ocl_hog.setSVMDetector(ocl_hog.getPeopleDetector64x128());
        hog.setSVMDetector(hog.getDefaultPeopleDetector());
    }
    else
    {
        ocl_hog.setSVMDetector(ocl_hog.getDefaultPeopleDetector());
        hog.setSVMDetector(hog.getDefaultPeopleDetector());
    }

    // OpenCL detection
    std::vector<cv::Point> d_v_locations;
    ocl_hog.detect(d_img, d_v_locations, 0);
    cv::Mat d_locations(d_v_locations);
    
    // CPU detection
    std::vector<cv::Point> v_locations;
    switch (type)
    {
    case CV_8UC1:
        hog.detect(img, v_locations, 0);
        break;
    case CV_8UC4:
    default:
        hog.detect(img_rgb, v_locations, 0);
        break;
    }
    cv::Mat locations(v_locations);

    char s[100]={0};
    EXPECT_MAT_NEAR(d_locations, locations, 0, s);
}


INSTANTIATE_TEST_CASE_P(OCL_ObjDetect, HOG, testing::Combine(
                        testing::Values(cv::Size(64, 128), cv::Size(48, 96)),
                        testing::Values(MatType(CV_8UC1), MatType(CV_8UC4))));


#endif //HAVE_OPENCL
