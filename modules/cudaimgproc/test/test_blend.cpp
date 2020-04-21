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

////////////////////////////////////////////////////////////////////////////
// Blend

namespace
{
    template <typename T>
    void blendLinearGold(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& weights1, const cv::Mat& weights2, cv::Mat& result_gold)
    {
        result_gold.create(img1.size(), img1.type());

        int cn = img1.channels();

        for (int y = 0; y < img1.rows; ++y)
        {
            const float* weights1_row = weights1.ptr<float>(y);
            const float* weights2_row = weights2.ptr<float>(y);
            const T* img1_row = img1.ptr<T>(y);
            const T* img2_row = img2.ptr<T>(y);
            T* result_gold_row = result_gold.ptr<T>(y);

            for (int x = 0; x < img1.cols * cn; ++x)
            {
                float w1 = weights1_row[x / cn];
                float w2 = weights2_row[x / cn];
                result_gold_row[x] = static_cast<T>((img1_row[x] * w1 + img2_row[x] * w2) / (w1 + w2 + 1e-5f));
            }
        }
    }
}

PARAM_TEST_CASE(Blend, cv::cuda::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Blend, Accuracy)
{
    int depth = CV_MAT_DEPTH(type);

    cv::Mat img1 = randomMat(size, type, 0.0, depth == CV_8U ? 255.0 : 1.0);
    cv::Mat img2 = randomMat(size, type, 0.0, depth == CV_8U ? 255.0 : 1.0);
    cv::Mat weights1 = randomMat(size, CV_32F, 0, 1);
    cv::Mat weights2 = randomMat(size, CV_32F, 0, 1);

    cv::cuda::GpuMat result;
    cv::cuda::blendLinear(loadMat(img1, useRoi), loadMat(img2, useRoi), loadMat(weights1, useRoi), loadMat(weights2, useRoi), result);

    cv::Mat result_gold;
    if (depth == CV_8U)
        blendLinearGold<uchar>(img1, img2, weights1, weights2, result_gold);
    else
        blendLinearGold<float>(img1, img2, weights1, weights2, result_gold);

    EXPECT_MAT_NEAR(result_gold, result, CV_MAT_DEPTH(type) == CV_8U ? 1.0 : 1e-5);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, Blend, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4), MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    WHOLE_SUBMAT));


}} // namespace
#endif // HAVE_CUDA
