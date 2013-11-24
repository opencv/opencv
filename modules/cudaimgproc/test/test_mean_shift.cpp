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

#ifdef HAVE_CUDA

using namespace cvtest;

////////////////////////////////////////////////////////////////////////////////
// MeanShift

struct MeanShift : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    cv::Mat img;

    int spatialRad;
    int colorRad;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());

        img = readImageType("meanshift/cones.png", CV_8UC4);
        ASSERT_FALSE(img.empty());

        spatialRad = 30;
        colorRad = 30;
    }
};

CUDA_TEST_P(MeanShift, Filtering)
{
    cv::Mat img_template;
    if (supportFeature(devInfo, cv::cuda::FEATURE_SET_COMPUTE_20))
        img_template = readImage("meanshift/con_result.png");
    else
        img_template = readImage("meanshift/con_result_CC1X.png");
    ASSERT_FALSE(img_template.empty());

    cv::cuda::GpuMat d_dst;
    cv::cuda::meanShiftFiltering(loadMat(img), d_dst, spatialRad, colorRad);

    ASSERT_EQ(CV_8UC4, d_dst.type());

    cv::Mat dst(d_dst);

    cv::Mat result;
    cv::cvtColor(dst, result, cv::COLOR_BGRA2BGR);

    EXPECT_MAT_NEAR(img_template, result, 0.0);
}

CUDA_TEST_P(MeanShift, Proc)
{
    cv::FileStorage fs;
    if (supportFeature(devInfo, cv::cuda::FEATURE_SET_COMPUTE_20))
        fs.open(std::string(cvtest::TS::ptr()->get_data_path()) + "meanshift/spmap.yaml", cv::FileStorage::READ);
    else
        fs.open(std::string(cvtest::TS::ptr()->get_data_path()) + "meanshift/spmap_CC1X.yaml", cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    cv::Mat spmap_template;
    fs["spmap"] >> spmap_template;
    ASSERT_FALSE(spmap_template.empty());

    cv::cuda::GpuMat rmap_filtered;
    cv::cuda::meanShiftFiltering(loadMat(img), rmap_filtered, spatialRad, colorRad);

    cv::cuda::GpuMat rmap;
    cv::cuda::GpuMat spmap;
    cv::cuda::meanShiftProc(loadMat(img), rmap, spmap, spatialRad, colorRad);

    ASSERT_EQ(CV_8UC4, rmap.type());

    EXPECT_MAT_NEAR(rmap_filtered, rmap, 0.0);
    EXPECT_MAT_NEAR(spmap_template, spmap, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, MeanShift, ALL_DEVICES);

////////////////////////////////////////////////////////////////////////////////
// MeanShiftSegmentation

namespace
{
    IMPLEMENT_PARAM_CLASS(MinSize, int);
}

PARAM_TEST_CASE(MeanShiftSegmentation, cv::cuda::DeviceInfo, MinSize)
{
    cv::cuda::DeviceInfo devInfo;
    int minsize;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        minsize = GET_PARAM(1);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(MeanShiftSegmentation, Regression)
{
    cv::Mat img = readImageType("meanshift/cones.png", CV_8UC4);
    ASSERT_FALSE(img.empty());

    std::ostringstream path;
    path << "meanshift/cones_segmented_sp10_sr10_minsize" << minsize;
    if (supportFeature(devInfo, cv::cuda::FEATURE_SET_COMPUTE_20))
        path << ".png";
    else
        path << "_CC1X.png";
    cv::Mat dst_gold = readImage(path.str());
    ASSERT_FALSE(dst_gold.empty());

    cv::Mat dst;
    cv::cuda::meanShiftSegmentation(loadMat(img), dst, 10, 10, minsize);

    cv::Mat dst_rgb;
    cv::cvtColor(dst, dst_rgb, cv::COLOR_BGRA2BGR);

    EXPECT_MAT_SIMILAR(dst_gold, dst_rgb, 1e-3);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, MeanShiftSegmentation, testing::Combine(
    ALL_DEVICES,
    testing::Values(MinSize(0), MinSize(4), MinSize(20), MinSize(84), MinSize(340), MinSize(1364))));

#endif // HAVE_CUDA
