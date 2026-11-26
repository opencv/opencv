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

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test { namespace {

CV_ENUM(DIS_TestPresets, DISOpticalFlow::PRESET_ULTRAFAST, DISOpticalFlow::PRESET_FAST, DISOpticalFlow::PRESET_MEDIUM)

typedef ocl::TSTestWithParam<DIS_TestPresets> OCL_DenseOpticalFlow_DIS;

OCL_TEST_P(OCL_DenseOpticalFlow_DIS, Mat)
{
    int preset = (int)GetParam();
    Mat frame1, frame2, GT;

    frame1 = imread(TS::ptr()->get_data_path() + "optflow/RubberWhale1.png");
    frame2 = imread(TS::ptr()->get_data_path() + "optflow/RubberWhale2.png");

    CV_Assert(!frame1.empty() && !frame2.empty());

    cvtColor(frame1, frame1, COLOR_BGR2GRAY);
    cvtColor(frame2, frame2, COLOR_BGR2GRAY);

    {
        Mat flow;
        UMat ocl_flow;

        Ptr<DenseOpticalFlow> algo = DISOpticalFlow::create(preset);
        OCL_OFF(algo->calc(frame1, frame2, flow));
        OCL_ON(algo->calc(frame1, frame2, ocl_flow));
        ASSERT_EQ(flow.rows, ocl_flow.rows);
        ASSERT_EQ(flow.cols, ocl_flow.cols);

        EXPECT_MAT_SIMILAR(flow, ocl_flow, 6e-3);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(Video, OCL_DenseOpticalFlow_DIS,
                            DIS_TestPresets::all());

OCL_TEST(DenseOpticalFlow_DIS, small_frame_preserves_finest_scale_and_patch_size)
{
    Mat frame1(240, 320, CV_8UC1), frame2(240, 320, CV_8UC1);
    theRNG().fill(frame1, RNG::UNIFORM, 0, 256);
    GaussianBlur(frame1, frame1, Size(5, 5), 1.2);
    Mat warp(Matx23d(1.0, 0.0, 3.5, 0.0, 1.0, -2.0));
    warpAffine(frame1, frame2, warp, frame1.size(), INTER_LINEAR, BORDER_REFLECT);

    Mat small1 = frame1(Rect(80, 60, 64, 64)).clone();
    Mat small2 = frame2(Rect(80, 60, 64, 64)).clone();
    UMat uFrame1, uFrame2, uSmall1, uSmall2;
    frame1.copyTo(uFrame1);
    frame2.copyTo(uFrame2);
    small1.copyTo(uSmall1);
    small2.copyTo(uSmall2);

    // PRESET_ULTRAFAST (finest_scale = 2, patch_size = 8)
    Ptr<DISOpticalFlow> dis = DISOpticalFlow::create(DISOpticalFlow::PRESET_ULTRAFAST);
    Ptr<DISOpticalFlow> fresh = DISOpticalFlow::create(DISOpticalFlow::PRESET_ULTRAFAST);
    UMat uFlowSmall, uFlowReused, uFlowFresh;
    OCL_ON(dis->calc(uSmall1, uSmall2, uFlowSmall));
    EXPECT_EQ(2, dis->getFinestScale());
    EXPECT_EQ(8, dis->getPatchSize());

    OCL_ON(dis->calc(uFrame1, uFrame2, uFlowReused));
    OCL_ON(fresh->calc(uFrame1, uFrame2, uFlowFresh));
    EXPECT_EQ(0.0, cvtest::norm(uFlowReused, uFlowFresh, NORM_INF));

    // PRESET_FAST with setFinestScale(3) (patch_size = 8)
    Ptr<DISOpticalFlow> disScale3 = DISOpticalFlow::create(DISOpticalFlow::PRESET_FAST);
    disScale3->setFinestScale(3);
    Ptr<DISOpticalFlow> cpuScale3 = DISOpticalFlow::create(DISOpticalFlow::PRESET_FAST);
    cpuScale3->setFinestScale(3);
    UMat uSmall3, uLargeReused, uLargeFresh;
    Mat cpuSmall3;
    OCL_ON(disScale3->calc(uSmall1, uSmall2, uSmall3));
    OCL_OFF(cpuScale3->calc(small1, small2, cpuSmall3));
    EXPECT_EQ(3, disScale3->getFinestScale());
    EXPECT_EQ(8, disScale3->getPatchSize());
    EXPECT_LE(cvtest::norm(uSmall3.getMat(ACCESS_READ), cpuSmall3, NORM_INF), 1e-4);

    Ptr<DISOpticalFlow> freshScale3 = DISOpticalFlow::create(DISOpticalFlow::PRESET_FAST);
    freshScale3->setFinestScale(3);
    OCL_ON(disScale3->calc(uFrame1, uFrame2, uLargeReused));
    OCL_ON(freshScale3->calc(uFrame1, uFrame2, uLargeFresh));
    EXPECT_EQ(0.0, cvtest::norm(uLargeReused, uLargeFresh, NORM_INF));
}

OCL_TEST(DenseOpticalFlow_DIS, uses_initial_flow)
{
    Mat frame1(240, 320, CV_8UC1), frame2(240, 320, CV_8UC1);
    theRNG().fill(frame1, RNG::UNIFORM, 0, 256);
    GaussianBlur(frame1, frame1, Size(7, 7), 1.5);

    const double dx = 96.0, dy = 48.0;
    Mat warp(Matx23d(1.0, 0.0, dx, 0.0, 1.0, dy));
    warpAffine(frame1, frame2, warp, frame1.size(), INTER_LINEAR, BORDER_REPLICATE);

    Mat initFlow(frame1.size(), CV_32FC2, Scalar(0.8 * dx, 0.8 * dy));
    UMat uFrame1, uFrame2, uFlowNone, uFlowInit, uFlowAfter;
    frame1.copyTo(uFrame1);
    frame2.copyTo(uFrame2);
    initFlow.copyTo(uFlowInit);

    Ptr<DISOpticalFlow> dis = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
    OCL_ON(dis->calc(uFrame1, uFrame2, uFlowNone));
    OCL_ON(dis->calc(uFrame1, uFrame2, uFlowInit));

    Mat cpuFlowInit = initFlow.clone();
    OCL_OFF(DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM)->calc(frame1, frame2, cpuFlowInit));

    EXPECT_GT(cvtest::norm(uFlowInit, uFlowNone, NORM_INF), 20.0);
    EXPECT_LE(cvtest::norm(uFlowInit.getMat(ACCESS_READ), cpuFlowInit, NORM_INF), 1e-4);

    // Subsequent call without initial flow resumes the uninitialized OpenCL path
    OCL_ON(dis->calc(uFrame1, uFrame2, uFlowAfter));
    EXPECT_EQ(0.0, cvtest::norm(uFlowAfter, uFlowNone, NORM_INF));
}

}} // namespace

#endif // HAVE_OPENCL
