/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;
using namespace cvtest;

TEST(Integral, _8u)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<uint> dst = integral_(d_src);

    Mat dst_gold;
    cv::integral(src, dst_gold);

    dst_gold = dst_gold(Rect(1, 1, size.width, size.height));

    ASSERT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST(Integral, _32f)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_32FC1, 0, 1);

    GpuMat_<float> d_src(src);

    GpuMat_<float> dst = integral_(d_src);

    Mat dst_gold;
    cv::integral(src, dst_gold, CV_32F);

    dst_gold = dst_gold(Rect(1, 1, size.width, size.height));

    ASSERT_PRED_FORMAT2(cvtest::MatComparator(1e-5, 0), dst_gold, Mat(dst));
}

TEST(Integral, _8u_opt)
{
    const Size size(640, 480);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<uint> dst = integral_(d_src);

    Mat dst_gold;
    cv::integral(src, dst_gold);

    dst_gold = dst_gold(Rect(1, 1, size.width, size.height));

    ASSERT_MAT_NEAR(dst_gold, dst, 0.0);
}
