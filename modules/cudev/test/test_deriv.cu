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

TEST(Sobel, Accuracy)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);
    Texture<uchar> tex_src(d_src);

    GpuMat_<short> dx = sobelX_(cvt_<int>(tex_src));
    GpuMat_<short> dy = sobelY_(cvt_<int>(tex_src));

    Mat dx_gold, dy_gold;
    cv::Sobel(src, dx_gold, CV_16S, 1, 0, 3, 1, 0, BORDER_REPLICATE);
    cv::Sobel(src, dy_gold, CV_16S, 0, 1, 3, 1, 0, BORDER_REPLICATE);

    EXPECT_MAT_NEAR(dx_gold, dx, 0.0);
    EXPECT_MAT_NEAR(dy_gold, dy, 0.0);
}

TEST(Scharr, Accuracy)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);
    Texture<uchar> tex_src(d_src);

    GpuMat_<short> dx = scharrX_(cvt_<int>(tex_src));
    GpuMat_<short> dy = scharrY_(cvt_<int>(tex_src));

    Mat dx_gold, dy_gold;
    cv::Scharr(src, dx_gold, CV_16S, 1, 0, 1, 0, BORDER_REPLICATE);
    cv::Scharr(src, dy_gold, CV_16S, 0, 1, 1, 0, BORDER_REPLICATE);

    EXPECT_MAT_NEAR(dx_gold, dx, 0.0);
    EXPECT_MAT_NEAR(dy_gold, dy, 0.0);
}

TEST(Laplacian, Accuracy)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);
    Texture<uchar> tex_src(d_src);

    GpuMat_<short> dst1 = laplacian_<1>(cvt_<int>(tex_src));
    GpuMat_<short> dst3 = laplacian_<3>(cvt_<int>(tex_src));

    Mat dst1_gold, dst3_gold;
    cv::Laplacian(src, dst1_gold, CV_16S, 1, 1, 0, BORDER_REPLICATE);
    cv::Laplacian(src, dst3_gold, CV_16S, 3, 1, 0, BORDER_REPLICATE);

    EXPECT_MAT_NEAR(dst1_gold, dst1, 0.0);
    EXPECT_MAT_NEAR(dst3_gold, dst3, 0.0);
}
