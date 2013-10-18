///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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
#include <iomanip>

#ifdef HAVE_OPENCL

using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
using namespace std;

//////////////////////////////////////////////////////
// GoodFeaturesToTrack
namespace
{
    IMPLEMENT_PARAM_CLASS(MinDistance, double)
}
PARAM_TEST_CASE(GoodFeaturesToTrack, MinDistance)
{
    double minDistance;

    virtual void SetUp()
    {
        minDistance = GET_PARAM(0);
    }
};

OCL_TEST_P(GoodFeaturesToTrack, Accuracy)
{
    cv::Mat frame = readImage("gpu/opticalflow/rubberwhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty());

    int maxCorners = 1000;
    double qualityLevel = 0.01;

    cv::ocl::GoodFeaturesToTrackDetector_OCL detector(maxCorners, qualityLevel, minDistance);

    cv::ocl::oclMat d_pts;
    detector(oclMat(frame), d_pts);

    ASSERT_FALSE(d_pts.empty());

    std::vector<cv::Point2f> pts(d_pts.cols);

    detector.downloadPoints(d_pts, pts);

    std::vector<cv::Point2f> pts_gold;
    cv::goodFeaturesToTrack(frame, pts_gold, maxCorners, qualityLevel, minDistance);

    ASSERT_EQ(pts_gold.size(), pts.size());

    size_t mistmatch = 0;
    for (size_t i = 0; i < pts.size(); ++i)
    {
        cv::Point2i a = pts_gold[i];
        cv::Point2i b = pts[i];

        bool eq = std::abs(a.x - b.x) < 1 && std::abs(a.y - b.y) < 1;

        if (!eq)
            ++mistmatch;
    }

    double bad_ratio = static_cast<double>(mistmatch) / pts.size();

    ASSERT_LE(bad_ratio, 0.01);
}

OCL_TEST_P(GoodFeaturesToTrack, EmptyCorners)
{
    int maxCorners = 1000;
    double qualityLevel = 0.01;

    cv::ocl::GoodFeaturesToTrackDetector_OCL detector(maxCorners, qualityLevel, minDistance);

    cv::ocl::oclMat src(100, 100, CV_8UC1, cv::Scalar::all(0));
    cv::ocl::oclMat corners(1, maxCorners, CV_32FC2);

    detector(src, corners);

    ASSERT_TRUE(corners.empty());
}

INSTANTIATE_TEST_CASE_P(OCL_Video, GoodFeaturesToTrack,
    testing::Values(MinDistance(0.0), MinDistance(3.0)));

//////////////////////////////////////////////////////////////////////////
PARAM_TEST_CASE(TVL1, bool)
{
    bool useRoi;

    virtual void SetUp()
    {
        useRoi = GET_PARAM(0);
    }

};

OCL_TEST_P(TVL1, DISABLED_Accuracy) // TODO implementations of TV1 in video module are different in 2.4 and master branches
{
    cv::Mat frame0 = readImage("gpu/opticalflow/rubberwhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("gpu/opticalflow/rubberwhale2.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    cv::ocl::OpticalFlowDual_TVL1_OCL d_alg;
    cv::Mat flowx = randomMat(frame0.size(), CV_32FC1, 0, 0, useRoi);
    cv::Mat flowy = randomMat(frame0.size(), CV_32FC1, 0, 0, useRoi);
    cv::ocl::oclMat d_flowx(flowx), d_flowy(flowy);
    d_alg(oclMat(frame0), oclMat(frame1), d_flowx, d_flowy);

    cv::Ptr<cv::DenseOpticalFlow> alg = cv::createOptFlow_DualTVL1();
    cv::Mat flow;
    alg->calc(frame0, frame1, flow);
    cv::Mat gold[2];
    cv::split(flow, gold);

    EXPECT_MAT_SIMILAR(gold[0], d_flowx, 3e-3);
    EXPECT_MAT_SIMILAR(gold[1], d_flowy, 3e-3);
}
INSTANTIATE_TEST_CASE_P(OCL_Video, TVL1, Values(false, true));


/////////////////////////////////////////////////////////////////////////////////////////////////
// PyrLKOpticalFlow

PARAM_TEST_CASE(Sparse, bool, bool)
{
    bool useGray;
    bool UseSmart;

    virtual void SetUp()
    {
        UseSmart = GET_PARAM(0);
        useGray = GET_PARAM(1);
    }
};

OCL_TEST_P(Sparse, Mat)
{
    cv::Mat frame0 = readImage("gpu/opticalflow/rubberwhale1.png", useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("gpu/opticalflow/rubberwhale2.png", useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame1.empty());

    cv::Mat gray_frame;
    if (useGray)
        gray_frame = frame0;
    else
        cv::cvtColor(frame0, gray_frame, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(gray_frame, pts, 1000, 0.01, 0.0);

    cv::ocl::oclMat d_pts;
    cv::Mat pts_mat(1, (int)pts.size(), CV_32FC2, (void *)&pts[0]);
    d_pts.upload(pts_mat);

    cv::ocl::PyrLKOpticalFlow pyrLK;

    cv::ocl::oclMat oclFrame0;
    cv::ocl::oclMat oclFrame1;
    cv::ocl::oclMat d_nextPts;
    cv::ocl::oclMat d_status;
    cv::ocl::oclMat d_err;

    oclFrame0 = frame0;
    oclFrame1 = frame1;

    pyrLK.sparse(oclFrame0, oclFrame1, d_pts, d_nextPts, d_status, &d_err);

    std::vector<cv::Point2f> nextPts(d_nextPts.cols);
    cv::Mat nextPts_mat(1, d_nextPts.cols, CV_32FC2, (void *)&nextPts[0]);
    d_nextPts.download(nextPts_mat);

    std::vector<unsigned char> status(d_status.cols);
    cv::Mat status_mat(1, d_status.cols, CV_8UC1, (void *)&status[0]);
    d_status.download(status_mat);

    std::vector<float> err(d_err.cols);
    cv::Mat err_mat(1, d_err.cols, CV_32FC1, (void*)&err[0]);
    d_err.download(err_mat);

    std::vector<cv::Point2f> nextPts_gold;
    std::vector<unsigned char> status_gold;
    std::vector<float> err_gold;
    cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts_gold, status_gold, err_gold);

    ASSERT_EQ(nextPts_gold.size(), nextPts.size());
    ASSERT_EQ(status_gold.size(), status.size());

    size_t mistmatch = 0;
    for (size_t i = 0; i < nextPts.size(); ++i)
    {
        if (status[i] != status_gold[i])
        {
            ++mistmatch;
            continue;
        }

        if (status[i])
        {
            cv::Point2i a = nextPts[i];
            cv::Point2i b = nextPts_gold[i];

            bool eq = std::abs(a.x - b.x) < 1 && std::abs(a.y - b.y) < 1;
            float errdiff = 0.0f;

            if (!eq || errdiff > 1e-1)
                ++mistmatch;
        }
    }

    double bad_ratio = static_cast<double>(mistmatch) / (nextPts.size());

    ASSERT_LE(bad_ratio, 0.02f);
}

INSTANTIATE_TEST_CASE_P(OCL_Video, Sparse, Combine(Bool(), Bool()));

//////////////////////////////////////////////////////
// FarnebackOpticalFlow

namespace
{
    IMPLEMENT_PARAM_CLASS(PyrScale, double)
        IMPLEMENT_PARAM_CLASS(PolyN, int)
        CV_FLAGS(FarnebackOptFlowFlags, 0, OPTFLOW_FARNEBACK_GAUSSIAN)
        IMPLEMENT_PARAM_CLASS(UseInitFlow, bool)
}

PARAM_TEST_CASE(Farneback, PyrScale, PolyN, FarnebackOptFlowFlags, UseInitFlow)
{
    double pyrScale;
    int polyN;
    int flags;
    bool useInitFlow;

    virtual void SetUp()
    {
        pyrScale = GET_PARAM(0);
        polyN = GET_PARAM(1);
        flags = GET_PARAM(2);
        useInitFlow = GET_PARAM(3);
    }
};

OCL_TEST_P(Farneback, Accuracy)
{
    cv::Mat frame0 = readImage("gpu/opticalflow/rubberwhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("gpu/opticalflow/rubberwhale2.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    double polySigma = polyN <= 5 ? 1.1 : 1.5;

    cv::ocl::FarnebackOpticalFlow farn;
    farn.pyrScale = pyrScale;
    farn.polyN = polyN;
    farn.polySigma = polySigma;
    farn.flags = flags;

    cv::ocl::oclMat d_flowx, d_flowy;
    farn(oclMat(frame0), oclMat(frame1), d_flowx, d_flowy);

    cv::Mat flow;
    if (useInitFlow)
    {
        cv::Mat flowxy[] = {cv::Mat(d_flowx), cv::Mat(d_flowy)};
        cv::merge(flowxy, 2, flow);

        farn.flags |= cv::OPTFLOW_USE_INITIAL_FLOW;
        farn(oclMat(frame0), oclMat(frame1), d_flowx, d_flowy);
    }

    cv::calcOpticalFlowFarneback(
        frame0, frame1, flow, farn.pyrScale, farn.numLevels, farn.winSize,
        farn.numIters, farn.polyN, farn.polySigma, farn.flags);

    std::vector<cv::Mat> flowxy;
    cv::split(flow, flowxy);

    EXPECT_MAT_SIMILAR(flowxy[0], d_flowx, 0.1);
    EXPECT_MAT_SIMILAR(flowxy[1], d_flowy, 0.1);
}

INSTANTIATE_TEST_CASE_P(OCL_Video, Farneback, testing::Combine(
    testing::Values(PyrScale(0.3), PyrScale(0.5), PyrScale(0.8)),
    testing::Values(PolyN(5), PolyN(7)),
    testing::Values(FarnebackOptFlowFlags(0), FarnebackOptFlowFlags(cv::OPTFLOW_FARNEBACK_GAUSSIAN)),
    testing::Values(UseInitFlow(false), UseInitFlow(true))));

#endif // HAVE_OPENCL
