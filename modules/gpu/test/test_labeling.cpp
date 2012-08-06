/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                          License Agreement
//               For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * The name of the copyright holders may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
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
//M*/

#include "precomp.hpp"
#include <string>
#include <iostream>

struct Labeling : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();
        cv::gpu::setDevice(devInfo.deviceID());
    }

    cv::Mat loat_image()
    {
        return cv::imread(std::string( cvtest::TS::ptr()->get_data_path() ) + "labeling/label.png");
    }
};

TEST_P(Labeling, ConnectedComponents)
{
    cv::Mat image;
    cvtColor(loat_image(), image, CV_BGR2GRAY);

    cv::gpu::GpuMat mask;
    mask.create(image.rows, image.cols, CV_8UC1);

    cv::gpu::GpuMat components;
    components.create(image.rows, image.cols, CV_32SC1);

    cv::gpu::connectivityMask(cv::gpu::GpuMat(image), mask, cv::Scalar::all(0), cv::Scalar::all(2));

    cv::gpu::labelComponents(mask, components);

    // std::cout << cv::Mat(components) << std::endl;
    // cv::imshow("test", image);
    // cv::waitKey(0);

    // for(int i = 0; i + 32 < image.rows; i += 32)
    //     for(int j = 0; j + 32 < image.cols; j += 32)
    //         cv::rectangle(image, cv::Rect(j, i, 32, 32) , CV_RGB(255, 255, 255));

    cv::imshow("test", image);
    cv::waitKey(0);
    cv::imshow("test", cv::Mat(mask) * 10);
    cv::waitKey(0);
    cv::imshow("test", cv::Mat(components) * 2);
    cv::waitKey(0);
}

INSTANTIATE_TEST_CASE_P(ConnectedComponents, Labeling, ALL_DEVICES);