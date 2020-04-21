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

namespace opencv_test { namespace {

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate8U

CV_ENUM(TemplateMethod, cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED)
#define ALL_TEMPLATE_METHODS testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_SQDIFF_NORMED), TemplateMethod(cv::TM_CCORR), TemplateMethod(cv::TM_CCORR_NORMED), TemplateMethod(cv::TM_CCOEFF), TemplateMethod(cv::TM_CCOEFF_NORMED))

namespace
{
    IMPLEMENT_PARAM_CLASS(TemplateSize, cv::Size);
}

PARAM_TEST_CASE(MatchTemplate8U, cv::cuda::DeviceInfo, cv::Size, TemplateSize, Channels, TemplateMethod)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    cv::Size templ_size;
    int cn;
    int method;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        templ_size = GET_PARAM(2);
        cn = GET_PARAM(3);
        method = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(MatchTemplate8U, Accuracy)
{
    cv::Mat image = randomMat(size, CV_MAKETYPE(CV_8U, cn));
    cv::Mat templ = randomMat(templ_size, CV_MAKETYPE(CV_8U, cn));

    cv::Ptr<cv::cuda::TemplateMatching> alg = cv::cuda::createTemplateMatching(image.type(), method);

    cv::cuda::GpuMat dst;
    alg->match(loadMat(image), loadMat(templ), dst);

    cv::Mat dst_gold;
    cv::matchTemplate(image, templ, dst_gold, method);

    cv::Mat h_dst(dst);
    ASSERT_EQ(dst_gold.size(), h_dst.size());
    ASSERT_EQ(dst_gold.type(), h_dst.type());
    for (int y = 0; y < h_dst.rows; ++y)
    {
        for (int x = 0; x < h_dst.cols; ++x)
        {
            float gold_val = dst_gold.at<float>(y, x);
            float actual_val = dst_gold.at<float>(y, x);
            ASSERT_FLOAT_EQ(gold_val, actual_val) << y << ", " << x;
        }
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, MatchTemplate8U, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(TemplateSize(cv::Size(5, 5)), TemplateSize(cv::Size(16, 16)), TemplateSize(cv::Size(30, 30))),
    testing::Values(Channels(1), Channels(3), Channels(4)),
    ALL_TEMPLATE_METHODS));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate32F

PARAM_TEST_CASE(MatchTemplate32F, cv::cuda::DeviceInfo, cv::Size, TemplateSize, Channels, TemplateMethod)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    cv::Size templ_size;
    int cn;
    int method;

    int n, m, h, w;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        templ_size = GET_PARAM(2);
        cn = GET_PARAM(3);
        method = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(MatchTemplate32F, Regression)
{
    cv::Mat image = randomMat(size, CV_MAKETYPE(CV_32F, cn));
    cv::Mat templ = randomMat(templ_size, CV_MAKETYPE(CV_32F, cn));

    cv::Ptr<cv::cuda::TemplateMatching> alg = cv::cuda::createTemplateMatching(image.type(), method);

    cv::cuda::GpuMat dst;
    alg->match(loadMat(image), loadMat(templ), dst);

    cv::Mat dst_gold;
    cv::matchTemplate(image, templ, dst_gold, method);

    cv::Mat h_dst(dst);
    ASSERT_EQ(dst_gold.size(), h_dst.size());
    ASSERT_EQ(dst_gold.type(), h_dst.type());
    for (int y = 0; y < h_dst.rows; ++y)
    {
        for (int x = 0; x < h_dst.cols; ++x)
        {
            float gold_val = dst_gold.at<float>(y, x);
            float actual_val = dst_gold.at<float>(y, x);
            ASSERT_FLOAT_EQ(gold_val, actual_val) << y << ", " << x;
        }
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, MatchTemplate32F, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(TemplateSize(cv::Size(5, 5)), TemplateSize(cv::Size(16, 16)), TemplateSize(cv::Size(30, 30))),
    testing::Values(Channels(1), Channels(3), Channels(4)),
    testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_CCORR))));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplateBlackSource

PARAM_TEST_CASE(MatchTemplateBlackSource, cv::cuda::DeviceInfo, TemplateMethod)
{
    cv::cuda::DeviceInfo devInfo;
    int method;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        method = GET_PARAM(1);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(MatchTemplateBlackSource, Accuracy)
{
    cv::Mat image = readImage("matchtemplate/black.png");
    ASSERT_FALSE(image.empty());

    cv::Mat pattern = readImage("matchtemplate/cat.png");
    ASSERT_FALSE(pattern.empty());

    cv::Ptr<cv::cuda::TemplateMatching> alg = cv::cuda::createTemplateMatching(image.type(), method);

    cv::cuda::GpuMat d_dst;
    alg->match(loadMat(image), loadMat(pattern), d_dst);

    cv::Mat dst(d_dst);

    double maxValue;
    cv::Point maxLoc;
    cv::minMaxLoc(dst, NULL, &maxValue, NULL, &maxLoc);

    cv::Point maxLocGold = cv::Point(284, 12);

    ASSERT_EQ(maxLocGold, maxLoc);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, MatchTemplateBlackSource, testing::Combine(
    ALL_DEVICES,
    testing::Values(TemplateMethod(cv::TM_CCOEFF_NORMED), TemplateMethod(cv::TM_CCORR_NORMED))));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate_CCOEF_NORMED

PARAM_TEST_CASE(MatchTemplate_CCOEF_NORMED, cv::cuda::DeviceInfo, std::pair<std::string, std::string>)
{
    cv::cuda::DeviceInfo devInfo;
    std::string imageName;
    std::string patternName;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        imageName = GET_PARAM(1).first;
        patternName = GET_PARAM(1).second;

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(MatchTemplate_CCOEF_NORMED, Accuracy)
{
    cv::Mat image = readImage(imageName);
    ASSERT_FALSE(image.empty());

    cv::Mat pattern = readImage(patternName);
    ASSERT_FALSE(pattern.empty());

    cv::Ptr<cv::cuda::TemplateMatching> alg = cv::cuda::createTemplateMatching(image.type(), cv::TM_CCOEFF_NORMED);

    cv::cuda::GpuMat d_dst;
    alg->match(loadMat(image), loadMat(pattern), d_dst);

    cv::Mat dst(d_dst);

    cv::Point minLoc, maxLoc;
    double minVal, maxVal;
    cv::minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::Mat dstGold;
    cv::matchTemplate(image, pattern, dstGold, cv::TM_CCOEFF_NORMED);

    double minValGold, maxValGold;
    cv::Point minLocGold, maxLocGold;
    cv::minMaxLoc(dstGold, &minValGold, &maxValGold, &minLocGold, &maxLocGold);

    ASSERT_EQ(minLocGold, minLoc);
    ASSERT_EQ(maxLocGold, maxLoc);
    ASSERT_LE(maxVal, 1.0);
    ASSERT_GE(minVal, -1.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, MatchTemplate_CCOEF_NORMED, testing::Combine(
    ALL_DEVICES,
    testing::Values(std::make_pair(std::string("matchtemplate/source-0.png"), std::string("matchtemplate/target-0.png")))));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate_CanFindBigTemplate

struct MatchTemplate_CanFindBigTemplate : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(MatchTemplate_CanFindBigTemplate, SQDIFF_NORMED)
{
    cv::Mat scene = readImage("matchtemplate/scene.png");
    ASSERT_FALSE(scene.empty());

    cv::Mat templ = readImage("matchtemplate/template.png");
    ASSERT_FALSE(templ.empty());

    cv::Ptr<cv::cuda::TemplateMatching> alg = cv::cuda::createTemplateMatching(scene.type(), cv::TM_SQDIFF_NORMED);

    cv::cuda::GpuMat d_result;
    alg->match(loadMat(scene), loadMat(templ), d_result);

    cv::Mat result(d_result);

    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(result, &minVal, 0, &minLoc, 0);

    ASSERT_GE(minVal, 0);
    ASSERT_LT(minVal, 1e-3);
    ASSERT_EQ(344, minLoc.x);
    ASSERT_EQ(0, minLoc.y);
}

CUDA_TEST_P(MatchTemplate_CanFindBigTemplate, SQDIFF)
{
    cv::Mat scene = readImage("matchtemplate/scene.png");
    ASSERT_FALSE(scene.empty());

    cv::Mat templ = readImage("matchtemplate/template.png");
    ASSERT_FALSE(templ.empty());

    cv::Ptr<cv::cuda::TemplateMatching> alg = cv::cuda::createTemplateMatching(scene.type(), cv::TM_SQDIFF);

    cv::cuda::GpuMat d_result;
    alg->match(loadMat(scene), loadMat(templ), d_result);

    cv::Mat result(d_result);

    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(result, &minVal, 0, &minLoc, 0);

    ASSERT_GE(minVal, 0);
    ASSERT_EQ(344, minLoc.x);
    ASSERT_EQ(0, minLoc.y);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, MatchTemplate_CanFindBigTemplate, ALL_DEVICES);


}} // namespace
#endif // HAVE_CUDA
