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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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

#include "test_precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

TEST(blendLinear, accuracy_on_8U)
{
    Size size(607, 1021);
    RNG rng(0);
    for (int cn = 1; cn <= 4; ++cn)
    {
        Mat img1 = cvtest::randomMat(rng, size, CV_MAKE_TYPE(CV_8U, cn), 0, 255, false);
        Mat img2 = cvtest::randomMat(rng, size, CV_MAKE_TYPE(CV_8U, cn), 0, 255, false);
        Mat weights1 = cvtest::randomMat(rng, size, CV_32F, 0, 1, false);
        Mat weights2 = cvtest::randomMat(rng, size, CV_32F, 0, 1, false);
        Mat result_gold(size, CV_MAKE_TYPE(CV_8U, cn));
        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width * cn; ++x)
            {
                float w1 = weights1.at<float>(y, x / cn);
                float w2 = weights2.at<float>(y, x / cn);
                result_gold.at<uchar>(y, x) = static_cast<uchar>(
                    (img1.at<uchar>(y, x) * w1 + img2.at<uchar>(y, x) * w2) / (w1 + w2 + 1e-5f));
            }
        GpuMat d_result;
        blendLinear(GpuMat(img1), GpuMat(img2), GpuMat(weights1), GpuMat(weights2), d_result);
        ASSERT_LE(cvtest::norm(result_gold, Mat(d_result), NORM_INF), 1) << ", cn=" << cn;
    }
}

TEST(blendLinear, accuracy_on_32F)
{
    Size size(607, 1021);
    RNG rng(0);
    for (int cn = 1; cn <= 4; ++cn)
    {
        Mat img1 = cvtest::randomMat(rng, size, CV_MAKE_TYPE(CV_32F, cn), 0, 1, false);
        Mat img2 = cvtest::randomMat(rng, size, CV_MAKE_TYPE(CV_32F, cn), 0, 1, false);
        Mat weights1 = cvtest::randomMat(rng, size, CV_32F, 0, 1, false);
        Mat weights2 = cvtest::randomMat(rng, size, CV_32F, 0, 1, false);
        Mat result_gold(size, CV_MAKE_TYPE(CV_32F, cn));
        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width * cn; ++x)
            {
                float w1 = weights1.at<float>(y, x / cn);
                float w2 = weights2.at<float>(y, x / cn);
                result_gold.at<float>(y, x) = 
                    (img1.at<float>(y, x) * w1 + img2.at<float>(y, x) * w2) / (w1 + w2 + 1e-5f);
            }
        GpuMat d_result;
        blendLinear(GpuMat(img1), GpuMat(img2), GpuMat(weights1), GpuMat(weights2), d_result);
        ASSERT_LE(cvtest::norm(result_gold, Mat(d_result), NORM_INF), 1e-3) << ", cn=" << cn;
    }
}