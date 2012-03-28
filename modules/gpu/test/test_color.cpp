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

#include "precomp.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////////
// cvtColor

PARAM_TEST_CASE(CvtColor, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    cv::Mat img;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        img = randomMat(size, CV_MAKE_TYPE(depth, 3), 0.0, depth == CV_32F ? 1.0 : 255.0);
    }
};

TEST_P(CvtColor, BGR2RGB)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2RGB);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2RGB);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR2RGBA)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2RGBA);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2RGBA);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR2BGRA)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2BGRA);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2BGRA);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2RGB)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGRA2RGB);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2RGB);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGRA2BGR);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2BGR);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2RGBA)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGRA2RGBA);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2RGBA);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR2GRAY)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2GRAY);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2GRAY);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, RGB2GRAY)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2GRAY);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2GRAY);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, GRAY2BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_GRAY2BGR);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_GRAY2BGR);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, GRAY2BGRA)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_GRAY2BGRA, 4);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_GRAY2BGRA, 4);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2GRAY)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGRA2GRAY);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2GRAY);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, RGBA2GRAY)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGBA2GRAY);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGBA2GRAY);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2BGR565)
{
    if (depth != CV_8U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2BGR565);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2BGR565);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, RGB2BGR565)
{
    if (depth != CV_8U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2BGR565);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2BGR565);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5652BGR)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR565);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR5652BGR);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5652BGR);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5652RGB)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR565);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR5652RGB);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5652RGB);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2BGR565)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGRA2BGR565);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2BGR565);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, RGBA2BGR565)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGBA2BGR565);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGBA2BGR565);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5652BGRA)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR565);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR5652BGRA, 4);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5652BGRA, 4);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5652RGBA)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR565);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR5652RGBA, 4);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5652RGBA, 4);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, GRAY2BGR565)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_GRAY2BGR565);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_GRAY2BGR565);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5652GRAY)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR565);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR5652GRAY);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5652GRAY);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR2BGR555)
{
    if (depth != CV_8U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2BGR555);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2BGR555);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, RGB2BGR555)
{
    if (depth != CV_8U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2BGR555);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2BGR555);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5552BGR)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR555);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR5552BGR);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5552BGR);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5552RGB)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR555);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR5552RGB);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5552RGB);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2BGR555)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGRA2BGR555);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2BGR555);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, RGBA2BGR555)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGBA2BGR555);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGBA2BGR555);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5552BGRA)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR555);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR5552BGRA, 4);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5552BGRA, 4);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5552RGBA)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR555);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR5552RGBA, 4);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5552RGBA, 4);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, GRAY2BGR555)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_GRAY2BGR555);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_GRAY2BGR555);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5552GRAY)
{
    if (depth != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR555);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR5552GRAY);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5552GRAY);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR2XYZ)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2XYZ);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2XYZ);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, RGB2XYZ)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2XYZ);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2XYZ);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2XYZ4)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2XYZ, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2XYZ);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, 1e-5);
}

TEST_P(CvtColor, BGRA2XYZ4)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2XYZ, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2XYZ);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, 1e-5);
}

TEST_P(CvtColor, XYZ2BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2XYZ);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_XYZ2BGR);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_XYZ2BGR);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, XYZ2RGB)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2XYZ);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_XYZ2RGB);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_XYZ2RGB);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, XYZ42BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2XYZ);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_XYZ2BGR);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_XYZ2BGR);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, XYZ42BGRA)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2XYZ);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_XYZ2BGR, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_XYZ2BGR, 4);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2YCrCb)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2YCrCb);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2YCrCb);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, RGB2YCrCb)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2YCrCb);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2YCrCb);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2YCrCb4)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2YCrCb, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2YCrCb);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, 1e-5);
}

TEST_P(CvtColor, RGBA2YCrCb4)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2YCrCb, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2YCrCb);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, 1e-5);
}

TEST_P(CvtColor, YCrCb2BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YCrCb);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_YCrCb2BGR);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YCrCb2BGR);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YCrCb2RGB)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YCrCb);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_YCrCb2RGB);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YCrCb2RGB);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YCrCb42RGB)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YCrCb);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YCrCb2RGB);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_YCrCb2RGB);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YCrCb42RGBA)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YCrCb);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YCrCb2RGB, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_YCrCb2RGB, 4);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2HSV)
{
    if (depth == CV_16U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2HSV);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2HSV);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HSV)
{
    if (depth == CV_16U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2HSV);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HSV);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HSV4)
{
    if (depth == CV_16U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2HSV, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HSV);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGBA2HSV4)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2HSV, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HSV);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, BGR2HLS)
{
    if (depth == CV_16U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2HLS);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2HLS);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HLS)
{
    if (depth == CV_16U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2HLS);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HLS);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HLS4)
{
    if (depth == CV_16U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2HLS, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HLS);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGBA2HLS4)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2HLS, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HLS);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV2BGR)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HSV2BGR);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2BGR);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV2RGB)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HSV2RGB);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2RGB);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV42BGR)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2BGR);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HSV2BGR);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV42BGRA)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2BGR, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HSV2BGR, 4);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS2BGR)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HLS2BGR);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2BGR);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS2RGB)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HLS2RGB);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2RGB);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS42RGB)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2RGB);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HLS2RGB);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS42RGBA)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2RGB, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);


    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HLS2RGB, 4);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, BGR2HSV_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2HSV_FULL);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2HSV_FULL);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HSV_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2HSV_FULL);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HSV_FULL);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HSV4_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2HSV_FULL, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HSV_FULL);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGBA2HSV4_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2HSV_FULL, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HSV_FULL);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, BGR2HLS_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2HLS_FULL);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2HLS_FULL);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HLS_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2HLS_FULL);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HLS_FULL);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HLS4_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2HLS_FULL, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HLS_FULL);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGBA2HLS4_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2HLS_FULL, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HLS_FULL);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV2BGR_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV_FULL);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HSV2BGR_FULL);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2BGR_FULL);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV2RGB_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV_FULL);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HSV2RGB_FULL);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2RGB_FULL);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV42RGB_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV_FULL);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2RGB_FULL);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HSV2RGB_FULL);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV42RGBA_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV_FULL);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2RGB_FULL, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HSV2RGB_FULL, 4);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS2BGR_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS_FULL);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HLS2BGR_FULL);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2BGR_FULL);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS2RGB_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS_FULL);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HLS2RGB_FULL);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2RGB_FULL);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS42RGB_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS_FULL);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2RGB_FULL);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HLS2RGB_FULL);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS42RGBA_FULL)
{
    if (depth == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS_FULL);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2RGB_FULL, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_HLS2RGB_FULL, 4);

    EXPECT_MAT_NEAR(dst_gold, dst, depth == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, BGR2YUV)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2YUV);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2YUV);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, RGB2YUV)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2YUV);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2YUV);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YUV2BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YUV);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_YUV2BGR);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YUV2BGR);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YUV42BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YUV);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YUV2BGR);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_YUV2BGR);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YUV42BGRA)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YUV);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YUV2BGR, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), depth, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_YUV2BGR, 4);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YUV2RGB)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_RGB2YUV);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_YUV2RGB);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YUV2RGB);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2YUV4)
{
    cv::Mat src = img;

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_BGR2YUV, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2YUV);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, 1e-5);
}

TEST_P(CvtColor, RGBA2YUV4)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);

    cv::gpu::GpuMat dst;
    cv::gpu::cvtColor(loadMat(src, useRoi), dst, cv::COLOR_RGB2YUV, 4);

    ASSERT_EQ(4, dst.channels());

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2YUV);

    cv::Mat h_dst(dst);

    cv::Mat channels[4];
    cv::split(h_dst, channels);
    cv::merge(channels, 3, h_dst);

    EXPECT_MAT_NEAR(dst_gold, h_dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, CvtColor, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_32F)),
    WHOLE_SUBMAT));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// swapChannels

PARAM_TEST_CASE(SwapChannels, cv::gpu::DeviceInfo, cv::Size, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(SwapChannels, Accuracy)
{
    cv::Mat src = readImageType("stereobm/aloe-L.png", CV_8UC4);
    ASSERT_FALSE(src.empty());

    cv::gpu::GpuMat d_src = loadMat(src, useRoi);

    const int dstOrder[] = {2, 1, 0, 3};
    cv::gpu::swapChannels(d_src, dstOrder);

    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2RGBA);

    EXPECT_MAT_NEAR(dst_gold, d_src, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, SwapChannels, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));
