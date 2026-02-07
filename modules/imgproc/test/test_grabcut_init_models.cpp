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

#include <opencv2/imgproc.hpp>
#include <limits>

namespace opencv_test {

static cv::Mat makeSynth(int W, int H)
{
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::rectangle(img, cv::Rect(W/4, H/4, W/2, H/2), cv::Scalar(140, 140, 140), cv::FILLED);
    cv::line(img, cv::Point(0, H/2), cv::Point(W - 1, H/2), cv::Scalar(255, 255, 255), 1);
    cv::circle(img, cv::Point(W/2, H/2), 1, cv::Scalar(0, 0, 0), cv::FILLED);
    return img;
}

TEST(Imgproc_GrabCut, InitModeIgnoresUninitializedModels_Rect)
{
    cv::Mat image = makeSynth(300, 300);
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::Rect rect(50, 50, image.cols - 100, image.rows - 100);

    cv::Mat bgdModel, fgdModel;
    bgdModel.create(1, 65, CV_64F);
    fgdModel.create(1, 65, CV_64F);

    // Deterministic "bad" contents (simulates uninitialized / garbage memory)
    const double nan = std::numeric_limits<double>::quiet_NaN();
    bgdModel.setTo(cv::Scalar::all(nan));
    fgdModel.setTo(cv::Scalar::all(nan));

    EXPECT_NO_THROW(cv::grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv::GC_INIT_WITH_RECT));
}

TEST(Imgproc_GrabCut, InitModeIgnoresUninitializedModels_Mask)
{
    cv::Mat image = makeSynth(100, 100);
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    mask.at<uchar>(image.rows / 2, image.cols / 2) = cv::GC_FGD; // seed one FG pixel

    cv::Mat bgdModel, fgdModel;
    bgdModel.create(1, 65, CV_64F);
    fgdModel.create(1, 65, CV_64F);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    bgdModel.setTo(cv::Scalar::all(nan));
    fgdModel.setTo(cv::Scalar::all(nan));

    cv::Rect dummyRect; // ignored for INIT_WITH_MASK
    EXPECT_NO_THROW(cv::grabCut(image, mask, dummyRect, bgdModel, fgdModel, 1, cv::GC_INIT_WITH_MASK));
}

} // namespace opencv_test
