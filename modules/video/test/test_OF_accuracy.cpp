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

TEST(DenseOpticalFlow_DIS, InvalidImgSize_CoarsestLevelLessThanZero)
{
    cv::Ptr<cv::DISOpticalFlow> of = cv::DISOpticalFlow::create();
    const int mat_size = 10;

    cv::Mat x(mat_size, mat_size, CV_8UC1, 42);
    cv::Mat y(mat_size, mat_size, CV_8UC1, 42);
    cv::Mat flow;

    ASSERT_THROW(of->calc(x, y, flow), cv::Exception);
}

// make sure that autoSelectPatchSizeAndScales() works properly.
TEST(DenseOpticalFlow_DIS, InvalidImgSize_CoarsestLevelLessThanFinestLevel)
{
    cv::Ptr<cv::DISOpticalFlow> of = cv::DISOpticalFlow::create();
    const int mat_size = 80;

    cv::Mat x(mat_size, mat_size, CV_8UC1, 42);
    cv::Mat y(mat_size, mat_size, CV_8UC1, 42);
    cv::Mat flow;

    of->calc(x, y, flow);

    ASSERT_EQ(flow.rows, mat_size);
    ASSERT_EQ(flow.cols, mat_size);
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

TEST(DenseOpticalFlow_DIS, calc_after_collectGarbage)
{
    Mat frame1(240, 320, CV_8UC1), frame2(240, 320, CV_8UC1);
    theRNG().fill(frame1, RNG::UNIFORM, 0, 256);
    GaussianBlur(frame1, frame1, Size(5, 5), 1.0);
    Mat M = (Mat_<double>(2, 3) << 1, 0, 2.0, 0, 1, -1.0);
    warpAffine(frame1, frame2, M, frame1.size(), INTER_LINEAR, BORDER_REFLECT);

    Ptr<DISOpticalFlow> dis = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
    dis->collectGarbage();
    Mat flow1, flow2, reference;
    dis->calc(frame1, frame2, flow1);
    dis->collectGarbage();
    dis->calc(frame1, frame2, flow2);

    DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM)->calc(frame1, frame2, reference);
    EXPECT_EQ(0, cvtest::norm(flow1, reference, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(flow2, reference, NORM_INF));
}

TEST(DenseOpticalFlow_DIS, no_stale_initial_flow)
{
    Mat frame1(240, 320, CV_8UC1), frame2(240, 320, CV_8UC1);
    theRNG().fill(frame1, RNG::UNIFORM, 0, 256);
    GaussianBlur(frame1, frame1, Size(5, 5), 1.0);
    Mat M = (Mat_<double>(2, 3) << 1, 0, 2.0, 0, 1, -1.0);
    warpAffine(frame1, frame2, M, frame1.size(), INTER_LINEAR, BORDER_REFLECT);

    // a call with an initial flow, then one without: the second must not reuse the first one's initial flow
    Ptr<DISOpticalFlow> dis = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
    Mat initial(frame1.size(), CV_32FC2, Scalar(5, -3));
    dis->calc(frame1, frame2, initial);
    Mat flow, reference;
    dis->calc(frame1, frame2, flow);

    DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM)->calc(frame1, frame2, reference);
    EXPECT_EQ(0, cvtest::norm(flow, reference, NORM_INF));
}

TEST(DenseOpticalFlow_VariationalRefinement, calc_reuse_matches_fresh)
{
    Mat frame1(120, 160, CV_8UC1), frame2(120, 160, CV_8UC1);
    theRNG().fill(frame1, RNG::UNIFORM, 0, 256);
    GaussianBlur(frame1, frame1, Size(5, 5), 1.2);
    Mat M = (Mat_<double>(2, 3) << 1, 0, 1.5, 0, 1, -0.75);
    warpAffine(frame1, frame2, M, frame1.size(), INTER_LINEAR, BORDER_REFLECT);

    Ptr<VariationalRefinement> vr = VariationalRefinement::create();
    Mat flow1(frame1.size(), CV_32FC2, Scalar::all(0));
    Mat flow2(frame1.size(), CV_32FC2, Scalar::all(0));
    vr->calc(frame1, frame2, flow1);
    vr->calc(frame1, frame2, flow2);
    EXPECT_EQ(0, cvtest::norm(flow1, flow2, NORM_INF));
    vr->collectGarbage();
    Mat flow3(frame1.size(), CV_32FC2, Scalar::all(0));
    vr->calc(frame1, frame2, flow3);
    EXPECT_EQ(0, cvtest::norm(flow1, flow3, NORM_INF));
}

TEST(DenseOpticalFlow_DIS, ManualCoarsestScale)
{
    Mat prev = Mat::zeros(Size(320, 240), CV_8UC1);
    Mat next = Mat::zeros(Size(320, 240), CV_8UC1);
    randu(prev, 0, 255);
    randu(next, 0, 255);

    Ptr<DISOpticalFlow> dis = DISOpticalFlow::create(DISOpticalFlow::PRESET_FAST);

    EXPECT_EQ(dis->getCoarsestScale(), -1);

    Mat flow;
    dis->calc(prev, next, flow);
    EXPECT_FALSE(flow.empty());
    int manual_scale = 3;
    dis->setCoarsestScale(manual_scale);
    EXPECT_EQ(dis->getCoarsestScale(), manual_scale);

    dis->calc(prev, next, flow);
    EXPECT_FALSE(flow.empty());

    dis->setCoarsestScale(-1);
    EXPECT_EQ(dis->getCoarsestScale(), -1);
}

// See https://github.com/opencv/opencv/issues/20185
TEST(DenseOpticalFlow_DIS, regression_20185_patch_larger_than_border)
{
    Mat prev(240, 320, CV_8UC1), next(240, 320, CV_8UC1);
    theRNG().fill(prev, RNG::UNIFORM, 0, 256);
    theRNG().fill(next, RNG::UNIFORM, 0, 256);

    Ptr<DISOpticalFlow> dis = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
    dis->setPatchStride(10);

    Mat flow(prev.size(), CV_32FC2, Scalar(60.f, 60.f));
    ASSERT_NO_THROW(dis->calc(prev, next, flow));
    EXPECT_EQ(flow.size(), prev.size());

    dis->setPatchSize(25);
    flow.setTo(Scalar(60.f, 60.f));
    ASSERT_NO_THROW(dis->calc(prev, next, flow));
    EXPECT_EQ(flow.size(), prev.size());
}
TEST(Video_OpticalFlowFarneback, InitialFlow_Validation)
{
    cv::Mat frame1 = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::Mat frame2 = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::Mat flow;

    // Test 1: Empty flow with OPTFLOW_USE_INITIAL_FLOW flag
    EXPECT_ANY_THROW(cv::calcOpticalFlowFarneback(frame1, frame2, flow, 0.5, 3, 15, 3, 5, 1.2, cv::OPTFLOW_USE_INITIAL_FLOW));

    // Test 2: Mismatched size (image is 100x100, but flow is 50x50)
    cv::Mat wrong_size_flow = cv::Mat::zeros(50, 50, CV_32FC2);
    EXPECT_ANY_THROW(cv::calcOpticalFlowFarneback(frame1, frame2, wrong_size_flow, 0.5, 3, 15, 3, 5, 1.2, cv::OPTFLOW_USE_INITIAL_FLOW));

    // Test 3: Mismatched type (CV_8UC1 instead of CV_32FC2)
    cv::Mat wrong_type_flow = cv::Mat::zeros(100, 100, CV_8UC1);
    EXPECT_ANY_THROW(cv::calcOpticalFlowFarneback(frame1, frame2, wrong_type_flow, 0.5, 3, 15, 3, 5, 1.2, cv::OPTFLOW_USE_INITIAL_FLOW));
}
}} // namespace
