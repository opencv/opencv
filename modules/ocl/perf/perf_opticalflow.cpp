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
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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
#include "perf_precomp.hpp"

///////////// PyrLKOpticalFlow ////////////////////////

using namespace perf;
using std::tr1::get;
using std::tr1::tuple;
using std::tr1::make_tuple;

CV_ENUM(LoadMode, IMREAD_GRAYSCALE, IMREAD_COLOR)

typedef tuple<int, tuple<string, string, LoadMode> > PyrLKOpticalFlowParamType;
typedef TestBaseWithParam<PyrLKOpticalFlowParamType> PyrLKOpticalFlowFixture;

PERF_TEST_P(PyrLKOpticalFlowFixture,
            PyrLKOpticalFlow,
            ::testing::Combine(
                ::testing::Values(1000, 2000, 4000),
                ::testing::Values(
                    make_tuple<string, string, LoadMode>
                    (
                        string("gpu/opticalflow/rubberwhale1.png"),
                        string("gpu/opticalflow/rubberwhale2.png"),
                        LoadMode(IMREAD_COLOR)
                        ),
                    make_tuple<string, string, LoadMode>
                    (
                        string("gpu/stereobm/aloe-L.png"),
                        string("gpu/stereobm/aloe-R.png"),
                        LoadMode(IMREAD_GRAYSCALE)
                        )
                    )
                )
            )
{
    PyrLKOpticalFlowParamType params = GetParam();
    tuple<string, string, LoadMode> fileParam = get<1>(params);
    const int pointsCount = get<0>(params);
    const int openMode = static_cast<int>(get<2>(fileParam));
    const string fileName0 = get<0>(fileParam), fileName1 = get<1>(fileParam);
    Mat frame0 = imread(getDataPath(fileName0), openMode);
    Mat frame1 = imread(getDataPath(fileName1), openMode);

    declare.in(frame0, frame1);

    ASSERT_FALSE(frame0.empty()) << "can't load " << fileName0;
    ASSERT_FALSE(frame1.empty()) << "can't load " << fileName1;

    Mat grayFrame;
    if (openMode == IMREAD_COLOR)
        cvtColor(frame0, grayFrame, COLOR_BGR2GRAY);
    else
        grayFrame = frame0;

    vector<Point2f> pts, nextPts;
    vector<unsigned char> status;
    vector<float> err;
    goodFeaturesToTrack(grayFrame, pts, pointsCount, 0.01, 0.0);
    Mat ptsMat(1, static_cast<int>(pts.size()), CV_32FC2, (void *)&pts[0]);

    if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE()
                cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, err);
    }
    else if (RUN_OCL_IMPL)
    {
        ocl::PyrLKOpticalFlow oclPyrLK;
        ocl::oclMat oclFrame0(frame0), oclFrame1(frame1);
        ocl::oclMat oclPts(ptsMat);
        ocl::oclMat oclNextPts, oclStatus, oclErr;

        OCL_TEST_CYCLE()
                oclPyrLK.sparse(oclFrame0, oclFrame1, oclPts, oclNextPts, oclStatus, &oclErr);
    }
    else
        OCL_PERF_ELSE

    int value = 0;
    SANITY_CHECK(value);
}

PERF_TEST(tvl1flowFixture, tvl1flow)
{
    Mat frame0 = imread(getDataPath("gpu/opticalflow/rubberwhale1.png"), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty()) << "can't load rubberwhale1.png";

    Mat frame1 = imread(getDataPath("gpu/opticalflow/rubberwhale2.png"), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty()) << "can't load rubberwhale2.png";

    const Size srcSize = frame0.size();
    const double eps = 1.2;
    Mat flow(srcSize, CV_32FC2), flow1(srcSize, CV_32FC1), flow2(srcSize, CV_32FC1);
    declare.in(frame0, frame1).out(flow1, flow2).time(159);

    if (RUN_PLAIN_IMPL)
    {
        Ptr<DenseOpticalFlow> alg = createOptFlow_DualTVL1();

        TEST_CYCLE() alg->calc(frame0, frame1, flow);

        alg->collectGarbage();
        Mat flows[2] = { flow1, flow2 };
        split(flow, flows);

        SANITY_CHECK(flow1, eps);
        SANITY_CHECK(flow2, eps);
    }
    else if (RUN_OCL_IMPL)
    {
        ocl::OpticalFlowDual_TVL1_OCL oclAlg;
        ocl::oclMat oclFrame0(frame0), oclFrame1(frame1), oclFlow1(srcSize, CV_32FC1),
                oclFlow2(srcSize, CV_32FC1);

        OCL_TEST_CYCLE() oclAlg(oclFrame0, oclFrame1, oclFlow1, oclFlow2);

        oclAlg.collectGarbage();

        oclFlow1.download(flow1);
        oclFlow2.download(flow2);

        SANITY_CHECK(flow1, eps);
        SANITY_CHECK(flow2, eps);
    }
    else
        OCL_PERF_ELSE
}

///////////// FarnebackOpticalFlow ////////////////////////

CV_ENUM(farneFlagType, 0, OPTFLOW_FARNEBACK_GAUSSIAN)

typedef tuple<tuple<int, double>, farneFlagType, bool> FarnebackOpticalFlowParams;
typedef TestBaseWithParam<FarnebackOpticalFlowParams> FarnebackOpticalFlowFixture;

PERF_TEST_P(FarnebackOpticalFlowFixture, FarnebackOpticalFlow,
            ::testing::Combine(
                ::testing::Values(make_tuple<int, double>(5, 1.1),
                                  make_tuple<int, double>(7, 1.5)),
                farneFlagType::all(),
                ::testing::Bool()))
{
    Mat frame0 = imread(getDataPath("gpu/opticalflow/rubberwhale1.png"), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty()) << "can't load rubberwhale1.png";

    Mat frame1 = imread(getDataPath("gpu/opticalflow/rubberwhale2.png"), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty()) << "can't load rubberwhale2.png";

    const Size srcSize = frame0.size();

    const FarnebackOpticalFlowParams params = GetParam();
    const tuple<int, double> polyParams = get<0>(params);
    const int polyN = get<0>(polyParams), flags = get<1>(params);
    const double polySigma = get<1>(polyParams), pyrScale = 0.5;
    const bool useInitFlow = get<2>(params);
    const double eps = 1.5;

    Mat flowx(srcSize, CV_32FC1), flowy(srcSize, CV_32FC1), flow(srcSize, CV_32FC2);
    declare.in(frame0, frame1).out(flowx, flowy);

    ocl::FarnebackOpticalFlow farn;
    farn.pyrScale = pyrScale;
    farn.polyN = polyN;
    farn.polySigma = polySigma;
    farn.flags = flags;

    if (RUN_PLAIN_IMPL)
    {
        if (useInitFlow)
        {
            calcOpticalFlowFarneback(
                        frame0, frame1, flow, farn.pyrScale, farn.numLevels, farn.winSize,
                        farn.numIters, farn.polyN, farn.polySigma, farn.flags);
            farn.flags |= OPTFLOW_USE_INITIAL_FLOW;
        }

        TEST_CYCLE()
                calcOpticalFlowFarneback(
                    frame0, frame1, flow, farn.pyrScale, farn.numLevels, farn.winSize,
                    farn.numIters, farn.polyN, farn.polySigma, farn.flags);

        Mat flowxy[2] = { flowx, flowy };
        split(flow, flowxy);

        SANITY_CHECK(flowx, eps);
        SANITY_CHECK(flowy, eps);
    }
    else if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclFrame0(frame0), oclFrame1(frame1),
                oclFlowx(srcSize, CV_32FC1), oclFlowy(srcSize, CV_32FC1);

        if (useInitFlow)
        {
            farn(oclFrame0, oclFrame1, oclFlowx, oclFlowy);
            farn.flags |= OPTFLOW_USE_INITIAL_FLOW;
        }

        OCL_TEST_CYCLE()
                farn(oclFrame0, oclFrame1, oclFlowx, oclFlowy);

        oclFlowx.download(flowx);
        oclFlowy.download(flowy);

        SANITY_CHECK(flowx, eps);
        SANITY_CHECK(flowy, eps);
    }
    else
        OCL_PERF_ELSE
}
