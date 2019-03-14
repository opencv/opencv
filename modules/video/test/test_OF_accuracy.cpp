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

namespace opencv_test { namespace {

static string getDataDir() { return TS::ptr()->get_data_path(); }

static string getRubberWhaleFrame1() { return getDataDir() + "optflow/RubberWhale1.png"; }

static string getRubberWhaleFrame2() { return getDataDir() + "optflow/RubberWhale2.png"; }

static string getRubberWhaleGroundTruth() { return getDataDir() + "optflow/RubberWhale.flo"; }

static bool isFlowCorrect(float u) { return !cvIsNaN(u) && (fabs(u) < 1e9); }

static float calcRMSE(Mat flow1, Mat flow2)
{
    float sum = 0;
    int counter = 0;
    const int rows = flow1.rows;
    const int cols = flow1.cols;

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            Vec2f flow1_at_point = flow1.at<Vec2f>(y, x);
            Vec2f flow2_at_point = flow2.at<Vec2f>(y, x);

            float u1 = flow1_at_point[0];
            float v1 = flow1_at_point[1];
            float u2 = flow2_at_point[0];
            float v2 = flow2_at_point[1];

            if (isFlowCorrect(u1) && isFlowCorrect(u2) && isFlowCorrect(v1) && isFlowCorrect(v2))
            {
                sum += (u1 - u2) * (u1 - u2) + (v1 - v2) * (v1 - v2);
                counter++;
            }
        }
    }
    return (float)sqrt(sum / (1e-9 + counter));
}

bool readRubberWhale(Mat &dst_frame_1, Mat &dst_frame_2, Mat &dst_GT)
{
    const string frame1_path = getRubberWhaleFrame1();
    const string frame2_path = getRubberWhaleFrame2();
    const string gt_flow_path = getRubberWhaleGroundTruth();

    dst_frame_1 = imread(frame1_path);
    dst_frame_2 = imread(frame2_path);
    dst_GT = readOpticalFlow(gt_flow_path);

    if (dst_frame_1.empty() || dst_frame_2.empty() || dst_GT.empty())
        return false;
    else
        return true;
}

TEST(DenseOpticalFlow_DIS, ReferenceAccuracy)
{
    Mat frame1, frame2, GT;
    ASSERT_TRUE(readRubberWhale(frame1, frame2, GT));
    int presets[] = {DISOpticalFlow::PRESET_ULTRAFAST, DISOpticalFlow::PRESET_FAST, DISOpticalFlow::PRESET_MEDIUM};
    float target_RMSE[] = {0.86f, 0.74f, 0.49f};
    cvtColor(frame1, frame1, COLOR_BGR2GRAY);
    cvtColor(frame2, frame2, COLOR_BGR2GRAY);

    Ptr<DenseOpticalFlow> algo;

    // iterate over presets:
    for (int i = 0; i < 3; i++)
    {
        Mat flow;
        algo = DISOpticalFlow::create(presets[i]);
        algo->calc(frame1, frame2, flow);
        ASSERT_EQ(GT.rows, flow.rows);
        ASSERT_EQ(GT.cols, flow.cols);
        EXPECT_LE(calcRMSE(GT, flow), target_RMSE[i]);
    }
}

TEST(DenseOpticalFlow_VariationalRefinement, ReferenceAccuracy)
{
    Mat frame1, frame2, GT;
    ASSERT_TRUE(readRubberWhale(frame1, frame2, GT));
    float target_RMSE = 0.86f;
    cvtColor(frame1, frame1, COLOR_BGR2GRAY);
    cvtColor(frame2, frame2, COLOR_BGR2GRAY);

    Ptr<VariationalRefinement> var_ref;
    var_ref = VariationalRefinement::create();
    var_ref->setAlpha(20.0f);
    var_ref->setDelta(5.0f);
    var_ref->setGamma(10.0f);
    var_ref->setSorIterations(25);
    var_ref->setFixedPointIterations(25);
    Mat flow(frame1.size(), CV_32FC2);
    flow.setTo(0.0f);
    var_ref->calc(frame1, frame2, flow);
    ASSERT_EQ(GT.rows, flow.rows);
    ASSERT_EQ(GT.cols, flow.cols);
    EXPECT_LE(calcRMSE(GT, flow), target_RMSE);
}

}} // namespace
