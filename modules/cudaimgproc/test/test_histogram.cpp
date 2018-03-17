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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HistEven

PARAM_TEST_CASE(HistEven, cv::cuda::DeviceInfo, cv::Size)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(HistEven, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    int hbins = 30;
    float hranges[] = {50.0f, 200.0f};

    cv::cuda::GpuMat hist;
    cv::cuda::histEven(loadMat(src), hist, hbins, (int) hranges[0], (int) hranges[1]);

    cv::Mat hist_gold;

    int histSize[] = {hbins};
    const float* ranges[] = {hranges};
    int channels[] = {0};
    cv::calcHist(&src, 1, channels, cv::Mat(), hist_gold, 1, histSize, ranges);

    hist_gold = hist_gold.t();
    hist_gold.convertTo(hist_gold, CV_32S);

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, HistEven, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// CalcHist

PARAM_TEST_CASE(CalcHist, cv::cuda::DeviceInfo, cv::Size)
{
    cv::cuda::DeviceInfo devInfo;

    cv::Size size;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(CalcHist, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    cv::cuda::GpuMat hist;
    cv::cuda::calcHist(loadMat(src), hist);

    cv::Mat hist_gold;

    const int hbins = 256;
    const float hranges[] = {0.0f, 256.0f};
    const int histSize[] = {hbins};
    const float* ranges[] = {hranges};
    const int channels[] = {0};

    cv::calcHist(&src, 1, channels, cv::Mat(), hist_gold, 1, histSize, ranges);
    hist_gold = hist_gold.reshape(1, 1);
    hist_gold.convertTo(hist_gold, CV_32S);

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, CalcHist, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES));

PARAM_TEST_CASE(CalcHistWithMask, cv::cuda::DeviceInfo, cv::Size)
{
    cv::cuda::DeviceInfo devInfo;

    cv::Size size;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(CalcHistWithMask, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);
    cv::Mat mask = randomMat(size, CV_8UC1);
    cv::Mat(mask, cv::Rect(0, 0, size.width / 2, size.height / 2)).setTo(0);

    cv::cuda::GpuMat hist;
    cv::cuda::calcHist(loadMat(src), loadMat(mask), hist);

    cv::Mat hist_gold;

    const int hbins = 256;
    const float hranges[] = {0.0f, 256.0f};
    const int histSize[] = {hbins};
    const float* ranges[] = {hranges};
    const int channels[] = {0};

    cv::calcHist(&src, 1, channels, mask, hist_gold, 1, histSize, ranges);
    hist_gold = hist_gold.reshape(1, 1);
    hist_gold.convertTo(hist_gold, CV_32S);

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, CalcHistWithMask, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// EqualizeHist

PARAM_TEST_CASE(EqualizeHist, cv::cuda::DeviceInfo, cv::Size)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(EqualizeHist, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    cv::cuda::GpuMat dst;
    cv::cuda::equalizeHist(loadMat(src), dst);

    cv::Mat dst_gold;
    cv::equalizeHist(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 3.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, EqualizeHist, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// CLAHE

namespace
{
    IMPLEMENT_PARAM_CLASS(ClipLimit, double)
}

PARAM_TEST_CASE(CLAHE, cv::cuda::DeviceInfo, cv::Size, ClipLimit)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    double clipLimit;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        clipLimit = GET_PARAM(2);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(CLAHE, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    cv::Ptr<cv::cuda::CLAHE> clahe = cv::cuda::createCLAHE(clipLimit);
    cv::cuda::GpuMat dst;
    clahe->apply(loadMat(src), dst);

    cv::Ptr<cv::CLAHE> clahe_gold = cv::createCLAHE(clipLimit);
    cv::Mat dst_gold;
    clahe_gold->apply(src, dst_gold);

    ASSERT_MAT_NEAR(dst_gold, dst, 1.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, CLAHE, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(0.0, 40.0)));


}} // namespace
#endif // HAVE_CUDA
