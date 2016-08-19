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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

///////////////////////////////////////////////////////////////////////////////////////////////////////
// cvtColor

PARAM_TEST_CASE(CvtColor, MatDepth, bool)
{
    int depth;
    bool use_roi;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        use_roi = GET_PARAM(1);
    }

    virtual void generateTestData(int channelsIn, int channelsOut)
    {
        const int srcType = CV_MAKE_TYPE(depth, channelsIn);
        const int dstType = CV_MAKE_TYPE(depth, channelsOut);

        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, srcType, 2, 100);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, dstType, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near(double threshold)
    {
        OCL_EXPECT_MATS_NEAR(dst, threshold);
    }

    void performTest(int channelsIn, int channelsOut, int code, double threshold = 1e-3)
    {
        for (int j = 0; j < test_loop_times; j++)
        {
            generateTestData(channelsIn, channelsOut);

            OCL_OFF(cv::cvtColor(src_roi, dst_roi, code, channelsOut));
            OCL_ON(cv::cvtColor(usrc_roi, udst_roi, code, channelsOut));

            int h_limit = 256;
            switch (code)
            {
            case COLOR_RGB2HLS: case COLOR_BGR2HLS:
                h_limit = 180;
            case COLOR_RGB2HLS_FULL: case COLOR_BGR2HLS_FULL:
            {
                ASSERT_EQ(dst_roi.type(), udst_roi.type());
                ASSERT_EQ(dst_roi.size(), udst_roi.size());
                Mat gold, actual;
                dst_roi.convertTo(gold, CV_32FC3);
                udst_roi.getMat(ACCESS_READ).convertTo(actual, CV_32FC3);
                Mat absdiff1, absdiff2, absdiff3;
                cv::absdiff(gold, actual, absdiff1);
                cv::absdiff(gold, actual + h_limit, absdiff2);
                cv::absdiff(gold, actual - h_limit, absdiff3);
                Mat diff = cv::min(cv::min(absdiff1, absdiff2), absdiff3);
                EXPECT_LE(cvtest::norm(diff, NORM_INF), threshold);
                break;
            }
            default:
                Near(threshold);
            }
        }
    }
};

#define CVTCODE(name) COLOR_ ## name

// RGB[A] <-> BGR[A]

OCL_TEST_P(CvtColor, BGR2BGRA) { performTest(3, 4, CVTCODE(BGR2BGRA)); }
OCL_TEST_P(CvtColor, RGB2RGBA) { performTest(3, 4, CVTCODE(RGB2RGBA)); }
OCL_TEST_P(CvtColor, BGRA2BGR) { performTest(4, 3, CVTCODE(BGRA2BGR)); }
OCL_TEST_P(CvtColor, RGBA2RGB) { performTest(4, 3, CVTCODE(RGBA2RGB)); }
OCL_TEST_P(CvtColor, BGR2RGBA) { performTest(3, 4, CVTCODE(BGR2RGBA)); }
OCL_TEST_P(CvtColor, RGB2BGRA) { performTest(3, 4, CVTCODE(RGB2BGRA)); }
OCL_TEST_P(CvtColor, RGBA2BGR) { performTest(4, 3, CVTCODE(RGBA2BGR)); }
OCL_TEST_P(CvtColor, BGRA2RGB) { performTest(4, 3, CVTCODE(BGRA2RGB)); }
OCL_TEST_P(CvtColor, BGR2RGB) { performTest(3, 3, CVTCODE(BGR2RGB)); }
OCL_TEST_P(CvtColor, RGB2BGR) { performTest(3, 3, CVTCODE(RGB2BGR)); }
OCL_TEST_P(CvtColor, BGRA2RGBA) { performTest(4, 4, CVTCODE(BGRA2RGBA)); }
OCL_TEST_P(CvtColor, RGBA2BGRA) { performTest(4, 4, CVTCODE(RGBA2BGRA)); }

// RGB <-> Gray

OCL_TEST_P(CvtColor, RGB2GRAY) { performTest(3, 1, CVTCODE(RGB2GRAY)); }
OCL_TEST_P(CvtColor, GRAY2RGB) { performTest(1, 3, CVTCODE(GRAY2RGB)); }
OCL_TEST_P(CvtColor, BGR2GRAY) { performTest(3, 1, CVTCODE(BGR2GRAY)); }
OCL_TEST_P(CvtColor, GRAY2BGR) { performTest(1, 3, CVTCODE(GRAY2BGR)); }
OCL_TEST_P(CvtColor, RGBA2GRAY) { performTest(4, 1, CVTCODE(RGBA2GRAY)); }
OCL_TEST_P(CvtColor, GRAY2RGBA) { performTest(1, 4, CVTCODE(GRAY2RGBA)); }
OCL_TEST_P(CvtColor, BGRA2GRAY) { performTest(4, 1, CVTCODE(BGRA2GRAY), cv::ocl::Device::getDefault().isNVidia() ? 1 : 1e-3); }
OCL_TEST_P(CvtColor, GRAY2BGRA) { performTest(1, 4, CVTCODE(GRAY2BGRA)); }

// RGB <-> YUV

OCL_TEST_P(CvtColor, RGB2YUV) { performTest(3, 3, CVTCODE(RGB2YUV)); }
OCL_TEST_P(CvtColor, BGR2YUV) { performTest(3, 3, CVTCODE(BGR2YUV)); }
OCL_TEST_P(CvtColor, RGBA2YUV) { performTest(4, 3, CVTCODE(RGB2YUV)); }
OCL_TEST_P(CvtColor, BGRA2YUV) { performTest(4, 3, CVTCODE(BGR2YUV)); }
OCL_TEST_P(CvtColor, YUV2RGB) { performTest(3, 3, CVTCODE(YUV2RGB)); }
OCL_TEST_P(CvtColor, YUV2BGR) { performTest(3, 3, CVTCODE(YUV2BGR)); }
OCL_TEST_P(CvtColor, YUV2RGBA) { performTest(3, 4, CVTCODE(YUV2RGB)); }
OCL_TEST_P(CvtColor, YUV2BGRA) { performTest(3, 4, CVTCODE(YUV2BGR)); }

// RGB <-> YCrCb

OCL_TEST_P(CvtColor, RGB2YCrCb) { performTest(3, 3, CVTCODE(RGB2YCrCb)); }
OCL_TEST_P(CvtColor, BGR2YCrCb) { performTest(3, 3, CVTCODE(BGR2YCrCb)); }
OCL_TEST_P(CvtColor, RGBA2YCrCb) { performTest(4, 3, CVTCODE(RGB2YCrCb)); }
OCL_TEST_P(CvtColor, BGRA2YCrCb) { performTest(4, 3, CVTCODE(BGR2YCrCb)); }
OCL_TEST_P(CvtColor, YCrCb2RGB) { performTest(3, 3, CVTCODE(YCrCb2RGB)); }
OCL_TEST_P(CvtColor, YCrCb2BGR) { performTest(3, 3, CVTCODE(YCrCb2BGR)); }
OCL_TEST_P(CvtColor, YCrCb2RGBA) { performTest(3, 4, CVTCODE(YCrCb2RGB)); }
OCL_TEST_P(CvtColor, YCrCb2BGRA) { performTest(3, 4, CVTCODE(YCrCb2BGR)); }

// RGB <-> XYZ

#ifdef HAVE_IPP
#define IPP_EPS depth <= CV_32S ? 1 : 5e-5
#else
#define IPP_EPS 1e-3
#endif

OCL_TEST_P(CvtColor, RGB2XYZ) { performTest(3, 3, CVTCODE(RGB2XYZ), IPP_EPS); }
OCL_TEST_P(CvtColor, BGR2XYZ) { performTest(3, 3, CVTCODE(BGR2XYZ), IPP_EPS); }
OCL_TEST_P(CvtColor, RGBA2XYZ) { performTest(4, 3, CVTCODE(RGB2XYZ), IPP_EPS); }
OCL_TEST_P(CvtColor, BGRA2XYZ) { performTest(4, 3, CVTCODE(BGR2XYZ), IPP_EPS); }

OCL_TEST_P(CvtColor, XYZ2RGB) { performTest(3, 3, CVTCODE(XYZ2RGB), IPP_EPS); }
OCL_TEST_P(CvtColor, XYZ2BGR) { performTest(3, 3, CVTCODE(XYZ2BGR), IPP_EPS); }
OCL_TEST_P(CvtColor, XYZ2RGBA) { performTest(3, 4, CVTCODE(XYZ2RGB), IPP_EPS); }
OCL_TEST_P(CvtColor, XYZ2BGRA) { performTest(3, 4, CVTCODE(XYZ2BGR), IPP_EPS); }

#undef IPP_EPS

// RGB <-> HSV

#ifdef HAVE_IPP
#define IPP_EPS depth <= CV_32S ? 1 : 4e-5
#else
#define IPP_EPS 1e-3
#endif

typedef CvtColor CvtColor8u32f;

OCL_TEST_P(CvtColor8u32f, RGB2HSV) { performTest(3, 3, CVTCODE(RGB2HSV), IPP_EPS); }
OCL_TEST_P(CvtColor8u32f, BGR2HSV) { performTest(3, 3, CVTCODE(BGR2HSV), IPP_EPS); }
OCL_TEST_P(CvtColor8u32f, RGBA2HSV) { performTest(4, 3, CVTCODE(RGB2HSV), IPP_EPS); }
OCL_TEST_P(CvtColor8u32f, BGRA2HSV) { performTest(4, 3, CVTCODE(BGR2HSV), IPP_EPS); }

OCL_TEST_P(CvtColor8u32f, RGB2HSV_FULL) { performTest(3, 3, CVTCODE(RGB2HSV_FULL), IPP_EPS); }
OCL_TEST_P(CvtColor8u32f, BGR2HSV_FULL) { performTest(3, 3, CVTCODE(BGR2HSV_FULL), IPP_EPS); }
OCL_TEST_P(CvtColor8u32f, RGBA2HSV_FULL) { performTest(4, 3, CVTCODE(RGB2HSV_FULL), IPP_EPS); }
OCL_TEST_P(CvtColor8u32f, BGRA2HSV_FULL) { performTest(4, 3, CVTCODE(BGR2HSV_FULL), IPP_EPS); }

#undef IPP_EPS

OCL_TEST_P(CvtColor8u32f, HSV2RGB) { performTest(3, 3, CVTCODE(HSV2RGB), depth == CV_8U ? 1 : 4e-1); }
OCL_TEST_P(CvtColor8u32f, HSV2BGR) { performTest(3, 3, CVTCODE(HSV2BGR), depth == CV_8U ? 1 : 4e-1); }
OCL_TEST_P(CvtColor8u32f, HSV2RGBA) { performTest(3, 4, CVTCODE(HSV2RGB), depth == CV_8U ? 1 : 4e-1); }
OCL_TEST_P(CvtColor8u32f, HSV2BGRA) { performTest(3, 4, CVTCODE(HSV2BGR), depth == CV_8U ? 1 : 4e-1); }

OCL_TEST_P(CvtColor8u32f, HSV2RGB_FULL) { performTest(3, 3, CVTCODE(HSV2RGB_FULL), depth == CV_8U ? 1 : 4e-1); }
OCL_TEST_P(CvtColor8u32f, HSV2BGR_FULL) { performTest(3, 3, CVTCODE(HSV2BGR_FULL), depth == CV_8U ? 1 : 4e-1); }
OCL_TEST_P(CvtColor8u32f, HSV2RGBA_FULL) { performTest(3, 4, CVTCODE(HSV2BGR_FULL), depth == CV_8U ? 1 : 4e-1); }
OCL_TEST_P(CvtColor8u32f, HSV2BGRA_FULL) { performTest(3, 4, CVTCODE(HSV2BGR_FULL), depth == CV_8U ? 1 : 4e-1); }

// RGB <-> HLS

#ifdef HAVE_IPP
#define IPP_EPS depth == CV_8U ? 2 : 1e-3
#else
#define IPP_EPS depth == CV_8U ? 1 : 1e-3
#endif

OCL_TEST_P(CvtColor8u32f, RGB2HLS) { performTest(3, 3, CVTCODE(RGB2HLS), depth == CV_8U ? 1 : 1e-3); }
OCL_TEST_P(CvtColor8u32f, BGR2HLS) { performTest(3, 3, CVTCODE(BGR2HLS), depth == CV_8U ? 1 : 1e-3); }
OCL_TEST_P(CvtColor8u32f, RGBA2HLS) { performTest(4, 3, CVTCODE(RGB2HLS), depth == CV_8U ? 1 : 1e-3); }
OCL_TEST_P(CvtColor8u32f, BGRA2HLS) { performTest(4, 3, CVTCODE(BGR2HLS), depth == CV_8U ? 1 : 1e-3); }

OCL_TEST_P(CvtColor8u32f, RGB2HLS_FULL) { performTest(3, 3, CVTCODE(RGB2HLS_FULL), IPP_EPS); }
OCL_TEST_P(CvtColor8u32f, BGR2HLS_FULL) { performTest(3, 3, CVTCODE(BGR2HLS_FULL), IPP_EPS); }
OCL_TEST_P(CvtColor8u32f, RGBA2HLS_FULL) { performTest(4, 3, CVTCODE(RGB2HLS_FULL), IPP_EPS); }
OCL_TEST_P(CvtColor8u32f, BGRA2HLS_FULL) { performTest(4, 3, CVTCODE(BGR2HLS_FULL), IPP_EPS); }

OCL_TEST_P(CvtColor8u32f, HLS2RGB) { performTest(3, 3, CVTCODE(HLS2RGB), 1); }
OCL_TEST_P(CvtColor8u32f, HLS2BGR) { performTest(3, 3, CVTCODE(HLS2BGR), 1); }
OCL_TEST_P(CvtColor8u32f, HLS2RGBA) { performTest(3, 4, CVTCODE(HLS2RGB), 1); }
OCL_TEST_P(CvtColor8u32f, HLS2BGRA) { performTest(3, 4, CVTCODE(HLS2BGR), 1); }

OCL_TEST_P(CvtColor8u32f, HLS2RGB_FULL) { performTest(3, 3, CVTCODE(HLS2RGB_FULL), 1); }
OCL_TEST_P(CvtColor8u32f, HLS2BGR_FULL) { performTest(3, 3, CVTCODE(HLS2BGR_FULL), 1); }
OCL_TEST_P(CvtColor8u32f, HLS2RGBA_FULL) { performTest(3, 4, CVTCODE(HLS2RGB_FULL), 1); }
OCL_TEST_P(CvtColor8u32f, HLS2BGRA_FULL) { performTest(3, 4, CVTCODE(HLS2BGR_FULL), 1); }

#undef IPP_EPS

// RGB5x5 <-> RGB

typedef CvtColor CvtColor8u;

OCL_TEST_P(CvtColor8u, BGR5652BGR) { performTest(2, 3, CVTCODE(BGR5652BGR)); }
OCL_TEST_P(CvtColor8u, BGR5652RGB) { performTest(2, 3, CVTCODE(BGR5652RGB)); }
OCL_TEST_P(CvtColor8u, BGR5652BGRA) { performTest(2, 4, CVTCODE(BGR5652BGRA)); }
OCL_TEST_P(CvtColor8u, BGR5652RGBA) { performTest(2, 4, CVTCODE(BGR5652RGBA)); }

OCL_TEST_P(CvtColor8u, BGR5552BGR) { performTest(2, 3, CVTCODE(BGR5552BGR)); }
OCL_TEST_P(CvtColor8u, BGR5552RGB) { performTest(2, 3, CVTCODE(BGR5552RGB)); }
OCL_TEST_P(CvtColor8u, BGR5552BGRA) { performTest(2, 4, CVTCODE(BGR5552BGRA)); }
OCL_TEST_P(CvtColor8u, BGR5552RGBA) { performTest(2, 4, CVTCODE(BGR5552RGBA)); }

OCL_TEST_P(CvtColor8u, BGR2BGR565) { performTest(3, 2, CVTCODE(BGR2BGR565)); }
OCL_TEST_P(CvtColor8u, RGB2BGR565) { performTest(3, 2, CVTCODE(RGB2BGR565)); }
OCL_TEST_P(CvtColor8u, BGRA2BGR565) { performTest(4, 2, CVTCODE(BGRA2BGR565)); }
OCL_TEST_P(CvtColor8u, RGBA2BGR565) { performTest(4, 2, CVTCODE(RGBA2BGR565)); }

OCL_TEST_P(CvtColor8u, BGR2BGR555) { performTest(3, 2, CVTCODE(BGR2BGR555)); }
OCL_TEST_P(CvtColor8u, RGB2BGR555) { performTest(3, 2, CVTCODE(RGB2BGR555)); }
OCL_TEST_P(CvtColor8u, BGRA2BGR555) { performTest(4, 2, CVTCODE(BGRA2BGR555)); }
OCL_TEST_P(CvtColor8u, RGBA2BGR555) { performTest(4, 2, CVTCODE(RGBA2BGR555)); }

// RGB5x5 <-> Gray

OCL_TEST_P(CvtColor8u, BGR5652GRAY) { performTest(2, 1, CVTCODE(BGR5652GRAY)); }
OCL_TEST_P(CvtColor8u, BGR5552GRAY) { performTest(2, 1, CVTCODE(BGR5552GRAY)); }

OCL_TEST_P(CvtColor8u, GRAY2BGR565) { performTest(1, 2, CVTCODE(GRAY2BGR565)); }
OCL_TEST_P(CvtColor8u, GRAY2BGR555) { performTest(1, 2, CVTCODE(GRAY2BGR555)); }

// RGBA <-> mRGBA

#ifdef HAVE_IPP
#define IPP_EPS depth <= CV_32S ? 1 : 1e-3
#else
#define IPP_EPS 1e-3
#endif

OCL_TEST_P(CvtColor8u, RGBA2mRGBA) { performTest(4, 4, CVTCODE(RGBA2mRGBA), IPP_EPS); }
OCL_TEST_P(CvtColor8u, mRGBA2RGBA) { performTest(4, 4, CVTCODE(mRGBA2RGBA)); }

// RGB <-> Lab

OCL_TEST_P(CvtColor8u32f, BGR2Lab) { performTest(3, 3, CVTCODE(BGR2Lab)); }
OCL_TEST_P(CvtColor8u32f, RGB2Lab) { performTest(3, 3, CVTCODE(RGB2Lab)); }
OCL_TEST_P(CvtColor8u32f, LBGR2Lab) { performTest(3, 3, CVTCODE(LBGR2Lab), IPP_EPS); }
OCL_TEST_P(CvtColor8u32f, LRGB2Lab) { performTest(3, 3, CVTCODE(LRGB2Lab), IPP_EPS); }
OCL_TEST_P(CvtColor8u32f, BGRA2Lab) { performTest(4, 3, CVTCODE(BGR2Lab)); }
OCL_TEST_P(CvtColor8u32f, RGBA2Lab) { performTest(4, 3, CVTCODE(RGB2Lab)); }
OCL_TEST_P(CvtColor8u32f, LBGRA2Lab) { performTest(4, 3, CVTCODE(LBGR2Lab), IPP_EPS); }
OCL_TEST_P(CvtColor8u32f, LRGBA2Lab) { performTest(4, 3, CVTCODE(LRGB2Lab), IPP_EPS); }

#undef IPP_EPS

OCL_TEST_P(CvtColor8u32f, Lab2BGR) { performTest(3, 3, CVTCODE(Lab2BGR), depth == CV_8U ? 1 : 1e-5); }
OCL_TEST_P(CvtColor8u32f, Lab2RGB) { performTest(3, 3, CVTCODE(Lab2RGB), depth == CV_8U ? 1 : 1e-5); }
OCL_TEST_P(CvtColor8u32f, Lab2LBGR) { performTest(3, 3, CVTCODE(Lab2LBGR), depth == CV_8U ? 1 : 1e-5); }
OCL_TEST_P(CvtColor8u32f, Lab2LRGB) { performTest(3, 3, CVTCODE(Lab2LRGB), depth == CV_8U ? 1 : 1e-5); }
OCL_TEST_P(CvtColor8u32f, Lab2BGRA) { performTest(3, 4, CVTCODE(Lab2BGR), depth == CV_8U ? 1 : 1e-5); }
OCL_TEST_P(CvtColor8u32f, Lab2RGBA) { performTest(3, 4, CVTCODE(Lab2RGB), depth == CV_8U ? 1 : 1e-5); }
OCL_TEST_P(CvtColor8u32f, Lab2LBGRA) { performTest(3, 4, CVTCODE(Lab2LBGR), depth == CV_8U ? 1 : 1e-5); }
OCL_TEST_P(CvtColor8u32f, Lab2LRGBA) { performTest(3, 4, CVTCODE(Lab2LRGB), depth == CV_8U ? 1 : 1e-5); }

// RGB -> Luv

OCL_TEST_P(CvtColor8u32f, BGR2Luv) { performTest(3, 3, CVTCODE(BGR2Luv), depth == CV_8U ? 1 : 1.5e-2); }
OCL_TEST_P(CvtColor8u32f, RGB2Luv) { performTest(3, 3, CVTCODE(RGB2Luv), depth == CV_8U ? 1 : 1.5e-2); }
OCL_TEST_P(CvtColor8u32f, LBGR2Luv) { performTest(3, 3, CVTCODE(LBGR2Luv), depth == CV_8U ? 1 : 6e-3); }
OCL_TEST_P(CvtColor8u32f, LRGB2Luv) { performTest(3, 3, CVTCODE(LRGB2Luv), depth == CV_8U ? 1 : 6e-3); }
OCL_TEST_P(CvtColor8u32f, BGRA2Luv) { performTest(4, 3, CVTCODE(BGR2Luv), depth == CV_8U ? 1 : 2e-2); }
OCL_TEST_P(CvtColor8u32f, RGBA2Luv) { performTest(4, 3, CVTCODE(RGB2Luv), depth == CV_8U ? 1 : 2e-2); }
OCL_TEST_P(CvtColor8u32f, LBGRA2Luv) { performTest(4, 3, CVTCODE(LBGR2Luv), depth == CV_8U ? 1 : 6e-3); }
OCL_TEST_P(CvtColor8u32f, LRGBA2Luv) { performTest(4, 3, CVTCODE(LRGB2Luv), depth == CV_8U ? 1 : 6e-3); }

OCL_TEST_P(CvtColor8u32f, Luv2BGR) { performTest(3, 3, CVTCODE(Luv2BGR), depth == CV_8U ? 1 : 7e-5); }
OCL_TEST_P(CvtColor8u32f, Luv2RGB) { performTest(3, 3, CVTCODE(Luv2RGB), depth == CV_8U ? 1 : 7e-5); }
OCL_TEST_P(CvtColor8u32f, Luv2LBGR) { performTest(3, 3, CVTCODE(Luv2LBGR), depth == CV_8U ? 1 : 1e-5); }
OCL_TEST_P(CvtColor8u32f, Luv2LRGB) { performTest(3, 3, CVTCODE(Luv2LRGB), depth == CV_8U ? 1 : 1e-5); }
OCL_TEST_P(CvtColor8u32f, Luv2BGRA) { performTest(3, 4, CVTCODE(Luv2BGR), depth == CV_8U ? 1 : 7e-5); }
OCL_TEST_P(CvtColor8u32f, Luv2RGBA) { performTest(3, 4, CVTCODE(Luv2RGB), depth == CV_8U ? 1 : 7e-5); }
OCL_TEST_P(CvtColor8u32f, Luv2LBGRA) { performTest(3, 4, CVTCODE(Luv2LBGR), depth == CV_8U ? 1 : 1e-5); }
OCL_TEST_P(CvtColor8u32f, Luv2LRGBA) { performTest(3, 4, CVTCODE(Luv2LRGB), depth == CV_8U ? 1 : 1e-5); }

// YUV420 -> RGBA

struct CvtColor_YUV2RGB_420 :
        public CvtColor
{
    void generateTestData(int channelsIn, int channelsOut)
    {
        const int srcType = CV_MAKE_TYPE(depth, channelsIn);
        const int dstType = CV_MAKE_TYPE(depth, channelsOut);

        Size roiSize = randomSize(1, MAX_VALUE);
        roiSize.width *= 2;
        roiSize.height *= 3;
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, srcType, 2, 100);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, dstType, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }
};

OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2RGBA_NV12) { performTest(1, 4, CVTCODE(YUV2RGBA_NV12)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2BGRA_NV12) { performTest(1, 4, CVTCODE(YUV2BGRA_NV12)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2RGB_NV12) { performTest(1, 3, CVTCODE(YUV2RGB_NV12)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2BGR_NV12) { performTest(1, 3, CVTCODE(YUV2BGR_NV12)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2RGBA_NV21) { performTest(1, 4, CVTCODE(YUV2RGBA_NV21)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2BGRA_NV21) { performTest(1, 4, CVTCODE(YUV2BGRA_NV21)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2RGB_NV21) { performTest(1, 3, CVTCODE(YUV2RGB_NV21)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2BGR_NV21) { performTest(1, 3, CVTCODE(YUV2BGR_NV21)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2RGBA_YV12) { performTest(1, 4, CVTCODE(YUV2RGBA_YV12)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2BGRA_YV12) { performTest(1, 4, CVTCODE(YUV2BGRA_YV12)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2RGB_YV12) { performTest(1, 3, CVTCODE(YUV2RGB_YV12)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2BGR_YV12) { performTest(1, 3, CVTCODE(YUV2BGR_YV12)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2RGBA_IYUV) { performTest(1, 4, CVTCODE(YUV2RGBA_IYUV)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2BGRA_IYUV) { performTest(1, 4, CVTCODE(YUV2BGRA_IYUV)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2RGB_IYUV) { performTest(1, 3, CVTCODE(YUV2RGB_IYUV)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2BGR_IYUV) { performTest(1, 3, CVTCODE(YUV2BGR_IYUV)); }
OCL_TEST_P(CvtColor_YUV2RGB_420, YUV2GRAY_420) { performTest(1, 1, CVTCODE(YUV2GRAY_420)); }

// RGBA -> YUV420

struct CvtColor_RGB2YUV_420 :
        public CvtColor
{
    void generateTestData(int channelsIn, int channelsOut)
    {
        const int srcType = CV_MAKE_TYPE(depth, channelsIn);
        const int dstType = CV_MAKE_TYPE(depth, channelsOut);

        Size srcRoiSize = randomSize(1, MAX_VALUE);
        srcRoiSize.width *= 2;
        srcRoiSize.height *= 2;
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, srcRoiSize, srcBorder, srcType, 2, 100);

        Size dstRoiSize(srcRoiSize.width, srcRoiSize.height / 2 * 3);
        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, dstRoiSize, dstBorder, dstType, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }
};

OCL_TEST_P(CvtColor_RGB2YUV_420, RGBA2YUV_YV12) { performTest(4, 1, CVTCODE(RGBA2YUV_YV12), 1); }
OCL_TEST_P(CvtColor_RGB2YUV_420, BGRA2YUV_YV12) { performTest(4, 1, CVTCODE(BGRA2YUV_YV12), 1); }
OCL_TEST_P(CvtColor_RGB2YUV_420, RGB2YUV_YV12) { performTest(3, 1, CVTCODE(RGB2YUV_YV12), 1); }
OCL_TEST_P(CvtColor_RGB2YUV_420, BGR2YUV_YV12) { performTest(3, 1, CVTCODE(BGR2YUV_YV12), 1); }
OCL_TEST_P(CvtColor_RGB2YUV_420, RGBA2YUV_IYUV) { performTest(4, 1, CVTCODE(RGBA2YUV_IYUV), 1); }
OCL_TEST_P(CvtColor_RGB2YUV_420, BGRA2YUV_IYUV) { performTest(4, 1, CVTCODE(BGRA2YUV_IYUV), 1); }
OCL_TEST_P(CvtColor_RGB2YUV_420, RGB2YUV_IYUV) { performTest(3, 1, CVTCODE(RGB2YUV_IYUV), 1); }
OCL_TEST_P(CvtColor_RGB2YUV_420, BGR2YUV_IYUV) { performTest(3, 1, CVTCODE(BGR2YUV_IYUV), 1); }

// YUV422 -> RGBA

struct CvtColor_YUV2RGB_422 :
        public CvtColor
{
    void generateTestData(int channelsIn, int channelsOut)
    {
        const int srcType = CV_MAKE_TYPE(depth, channelsIn);
        const int dstType = CV_MAKE_TYPE(depth, channelsOut);

        Size roiSize = randomSize(1, MAX_VALUE);
        roiSize.width *= 2;

        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, srcType, 2, 100);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, dstType, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }
};

OCL_TEST_P(CvtColor_YUV2RGB_422, YUV2RGB_UYVY) { performTest(2, 3, CVTCODE(YUV2RGB_UYVY)); }
OCL_TEST_P(CvtColor_YUV2RGB_422, YUV2BGR_UYVY) { performTest(2, 3, CVTCODE(YUV2BGR_UYVY)); }
OCL_TEST_P(CvtColor_YUV2RGB_422, YUV2RGBA_UYVY) { performTest(2, 4, CVTCODE(YUV2RGBA_UYVY)); }
OCL_TEST_P(CvtColor_YUV2RGB_422, YUV2BGRA_UYVY) { performTest(2, 4, CVTCODE(YUV2BGRA_UYVY)); }
OCL_TEST_P(CvtColor_YUV2RGB_422, YUV2RGB_YUY2) { performTest(2, 3, CVTCODE(YUV2RGB_YUY2)); }
OCL_TEST_P(CvtColor_YUV2RGB_422, YUV2BGR_YUY2) { performTest(2, 3, CVTCODE(YUV2BGR_YUY2)); }
OCL_TEST_P(CvtColor_YUV2RGB_422, YUV2RGBA_YUY2) { performTest(2, 4, CVTCODE(YUV2RGBA_YUY2)); }
OCL_TEST_P(CvtColor_YUV2RGB_422, YUV2BGRA_YUY2) { performTest(2, 4, CVTCODE(YUV2BGRA_YUY2)); }
OCL_TEST_P(CvtColor_YUV2RGB_422, YUV2RGB_YVYU) { performTest(2, 3, CVTCODE(YUV2RGB_YVYU)); }
OCL_TEST_P(CvtColor_YUV2RGB_422, YUV2BGR_YVYU) { performTest(2, 3, CVTCODE(YUV2BGR_YVYU)); }
OCL_TEST_P(CvtColor_YUV2RGB_422, YUV2RGBA_YVYU) { performTest(2, 4, CVTCODE(YUV2RGBA_YVYU)); }
OCL_TEST_P(CvtColor_YUV2RGB_422, YUV2BGRA_YVYU) { performTest(2, 4, CVTCODE(YUV2BGRA_YVYU)); }


OCL_INSTANTIATE_TEST_CASE_P(ImgProc, CvtColor8u,
                            testing::Combine(testing::Values(MatDepth(CV_8U)), Bool()));

OCL_INSTANTIATE_TEST_CASE_P(ImgProc, CvtColor8u32f,
                            testing::Combine(testing::Values(MatDepth(CV_8U), MatDepth(CV_32F)), Bool()));

OCL_INSTANTIATE_TEST_CASE_P(ImgProc, CvtColor,
                            testing::Combine(
                                testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_32F)),
                                Bool()));

OCL_INSTANTIATE_TEST_CASE_P(ImgProc, CvtColor_YUV2RGB_420,
                            testing::Combine(
                                testing::Values(MatDepth(CV_8U)),
                                Bool()));

OCL_INSTANTIATE_TEST_CASE_P(ImgProc, CvtColor_RGB2YUV_420,
                            testing::Combine(
                                testing::Values(MatDepth(CV_8U)),
                                Bool()));

OCL_INSTANTIATE_TEST_CASE_P(ImgProc, CvtColor_YUV2RGB_422,
                            testing::Combine(
                                testing::Values(MatDepth(CV_8U)),
                                Bool()));

} } // namespace cvtest::ocl

#endif
