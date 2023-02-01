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

namespace cv {

enum {
    COLOR_BGR2BGR = COLOR_BGR2RGB,
    COLOR_BGR2LRGB = COLOR_BGR2RGB,
    COLOR_BGR2LBGR = COLOR_BGR2RGB
};

}

#define CVT_COLOR_TEST(src_space, dst_space, src_cn, dst_cn) \
    TEST(CvtColor, src_space ## _to_ ## dst_space) \
    { \
        const Size size = randomSize(100, 400); \
        Mat bgrb = randomMat(size, CV_8UC3); \
        Mat srcb; \
        cv::cvtColor(bgrb, srcb, COLOR_BGR ## 2 ## src_space, src_cn); \
        GpuMat_<SelectIf<src_cn == 1, uchar, uchar ## src_cn>::type> d_srcb(srcb); \
        GpuMat_<SelectIf<dst_cn == 1, uchar, uchar ## dst_cn>::type> dstb = src_space ## _to_ ## dst_space ## _(d_srcb); \
        Mat dstb_gold; \
        cv::cvtColor(srcb, dstb_gold, COLOR_ ## src_space ## 2 ## dst_space); \
        EXPECT_MAT_NEAR(dstb_gold, dstb, 2.0); \
        Mat bgrf = randomMat(size, CV_32FC3, 0, 1); \
        Mat srcf; \
        cv::cvtColor(bgrf, srcf, COLOR_BGR ## 2 ## src_space, src_cn); \
        GpuMat_<SelectIf<src_cn == 1, float, float ## src_cn>::type> d_srcf(srcf); \
        GpuMat_<SelectIf<dst_cn == 1, float, float ## dst_cn>::type> dstf = src_space ## _to_ ## dst_space ## _(d_srcf); \
        Mat dstf_gold; \
        cv::cvtColor(srcf, dstf_gold, COLOR_ ## src_space ## 2 ## dst_space); \
        EXPECT_MAT_NEAR(dstf_gold, dstf, 2.0); \
    }

// RGB <-> BGR

CVT_COLOR_TEST(BGR, RGB, 3, 3)
CVT_COLOR_TEST(BGR, BGRA, 3, 4)
CVT_COLOR_TEST(BGR, RGBA, 3, 4)
CVT_COLOR_TEST(BGRA, BGR, 4, 3)
CVT_COLOR_TEST(BGRA, RGB, 4, 3)
CVT_COLOR_TEST(BGRA, RGBA, 4, 4)

// RGB <-> Gray

CVT_COLOR_TEST(BGR, GRAY, 3, 1)
CVT_COLOR_TEST(RGB, GRAY, 3, 1)
CVT_COLOR_TEST(BGRA, GRAY, 4, 1)
CVT_COLOR_TEST(RGBA, GRAY, 4, 1)

CVT_COLOR_TEST(GRAY, BGR, 1, 3)
CVT_COLOR_TEST(GRAY, BGRA, 1, 4)

// RGB <-> YUV

CVT_COLOR_TEST(RGB, YUV, 3, 3)
CVT_COLOR_TEST(BGR, YUV, 3, 3)

CVT_COLOR_TEST(YUV, RGB, 3, 3)
CVT_COLOR_TEST(YUV, BGR, 3, 3)

// RGB <-> YCrCb

CVT_COLOR_TEST(RGB, YCrCb, 3, 3)
CVT_COLOR_TEST(BGR, YCrCb, 3, 3)

CVT_COLOR_TEST(YCrCb, RGB, 3, 3)
CVT_COLOR_TEST(YCrCb, BGR, 3, 3)

// RGB <-> XYZ

CVT_COLOR_TEST(RGB, XYZ, 3, 3)
CVT_COLOR_TEST(BGR, XYZ, 3, 3)

CVT_COLOR_TEST(XYZ, RGB, 3, 3)
CVT_COLOR_TEST(XYZ, BGR, 3, 3)

// RGB <-> HSV

CVT_COLOR_TEST(RGB, HSV, 3, 3)
CVT_COLOR_TEST(BGR, HSV, 3, 3)

CVT_COLOR_TEST(HSV, RGB, 3, 3)
CVT_COLOR_TEST(HSV, BGR, 3, 3)

CVT_COLOR_TEST(RGB, HSV_FULL, 3, 3)
CVT_COLOR_TEST(BGR, HSV_FULL, 3, 3)

CVT_COLOR_TEST(HSV, RGB_FULL, 3, 3)
CVT_COLOR_TEST(HSV, BGR_FULL, 3, 3)

// RGB <-> HLS

CVT_COLOR_TEST(RGB, HLS, 3, 3)
CVT_COLOR_TEST(BGR, HLS, 3, 3)

CVT_COLOR_TEST(HLS, RGB, 3, 3)
CVT_COLOR_TEST(HLS, BGR, 3, 3)

CVT_COLOR_TEST(RGB, HLS_FULL, 3, 3)
CVT_COLOR_TEST(BGR, HLS_FULL, 3, 3)

CVT_COLOR_TEST(HLS, RGB_FULL, 3, 3)
CVT_COLOR_TEST(HLS, BGR_FULL, 3, 3)

// RGB <-> Lab

CVT_COLOR_TEST(RGB, Lab, 3, 3)
CVT_COLOR_TEST(BGR, Lab, 3, 3)

CVT_COLOR_TEST(Lab, RGB, 3, 3)
CVT_COLOR_TEST(Lab, BGR, 3, 3)

CVT_COLOR_TEST(LRGB, Lab, 3, 3)
CVT_COLOR_TEST(LBGR, Lab, 3, 3)

CVT_COLOR_TEST(Lab, LRGB, 3, 3)
CVT_COLOR_TEST(Lab, LBGR, 3, 3)

// RGB <-> Luv

CVT_COLOR_TEST(RGB, Luv, 3, 3)
CVT_COLOR_TEST(BGR, Luv, 3, 3)

CVT_COLOR_TEST(Luv, RGB, 3, 3)
CVT_COLOR_TEST(Luv, BGR, 3, 3)

CVT_COLOR_TEST(LRGB, Luv, 3, 3)
CVT_COLOR_TEST(LBGR, Luv, 3, 3)

CVT_COLOR_TEST(Luv, LRGB, 3, 3)
CVT_COLOR_TEST(Luv, LBGR, 3, 3)
