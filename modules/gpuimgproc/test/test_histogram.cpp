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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HistEven

struct HistEven : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(HistEven, Accuracy)
{
    cv::Mat img = readImage("stereobm/aloe-L.png");
    ASSERT_FALSE(img.empty());

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    int hbins = 30;
    float hranges[] = {0.0f, 180.0f};

    std::vector<cv::Mat> srcs;
    cv::split(hsv, srcs);

    cv::gpu::GpuMat hist;
    cv::gpu::histEven(loadMat(srcs[0]), hist, hbins, (int)hranges[0], (int)hranges[1]);

    cv::MatND histnd;
    int histSize[] = {hbins};
    const float* ranges[] = {hranges};
    int channels[] = {0};
    cv::calcHist(&hsv, 1, channels, cv::Mat(), histnd, 1, histSize, ranges);

    cv::Mat hist_gold = histnd;
    hist_gold = hist_gold.t();
    hist_gold.convertTo(hist_gold, CV_32S);

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, HistEven, ALL_DEVICES);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// CalcHist

namespace
{
    void calcHistGold(const cv::Mat& src, cv::Mat& hist)
    {
        hist.create(1, 256, CV_32SC1);
        hist.setTo(cv::Scalar::all(0));

        int* hist_row = hist.ptr<int>();
        for (int y = 0; y < src.rows; ++y)
        {
            const uchar* src_row = src.ptr(y);

            for (int x = 0; x < src.cols; ++x)
                ++hist_row[src_row[x]];
        }
    }
}

PARAM_TEST_CASE(CalcHist, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(CalcHist, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    cv::gpu::GpuMat hist;
    cv::gpu::calcHist(loadMat(src), hist);

    cv::Mat hist_gold;
    calcHistGold(src, hist_gold);

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, CalcHist, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// EqualizeHist

PARAM_TEST_CASE(EqualizeHist, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(EqualizeHist, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    cv::gpu::GpuMat dst;
    cv::gpu::equalizeHist(loadMat(src), dst);

    cv::Mat dst_gold;
    cv::equalizeHist(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 3.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, EqualizeHist, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// CLAHE

namespace
{
    IMPLEMENT_PARAM_CLASS(ClipLimit, double)
}

PARAM_TEST_CASE(CLAHE, cv::gpu::DeviceInfo, cv::Size, ClipLimit)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    double clipLimit;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        clipLimit = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(CLAHE, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    cv::Ptr<cv::gpu::CLAHE> clahe = cv::gpu::createCLAHE(clipLimit);
    cv::gpu::GpuMat dst;
    clahe->apply(loadMat(src), dst);

    cv::Ptr<cv::CLAHE> clahe_gold = cv::createCLAHE(clipLimit);
    cv::Mat dst_gold;
    clahe_gold->apply(src, dst_gold);

    ASSERT_MAT_NEAR(dst_gold, dst, 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, CLAHE, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(0.0, 40.0)));

#endif // HAVE_CUDA
