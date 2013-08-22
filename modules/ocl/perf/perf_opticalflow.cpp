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
//     and/or other oclMaterials provided with the distribution.
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

template <typename T>
static vector<T> & MatToVector(const ocl::oclMat & oclSrc, vector<T> & instance)
{
    Mat src;
    oclSrc.download(src);

    for (int i = 0; i < src.cols; ++i)
        instance.push_back(src.at<T>(0, i));

    return instance;
}

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
                        string("gpu/opticalflow/rubberwhale1.png"),
                        LoadMode(IMREAD_COLOR)
                        )
//                    , make_tuple<string, string, LoadMode>
//                    (
//                        string("gpu/stereobm/aloe-L.png"),
//                        string("gpu/stereobm/aloe-R.png"),
//                        LoadMode(IMREAD_GRAYSCALE)
//                        )
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
    const string impl = getSelectedImpl();

    ASSERT_FALSE(frame0.empty()) << "can't load " << fileName0;
    ASSERT_FALSE(frame1.empty()) << "can't load " << fileName1;

    Mat grayFrame;
    if (openMode == IMREAD_COLOR)
        cvtColor(frame0, grayFrame, COLOR_BGR2GRAY);
    else
        grayFrame = frame0;

    // initialization
    vector<Point2f> pts, nextPts;
    vector<unsigned char> status;
    vector<float> err;
    goodFeaturesToTrack(grayFrame, pts, pointsCount, 0.01, 0.0);

    // selecting implementation
    if (impl == "plain")
    {
        TEST_CYCLE()
                cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, err);

        SANITY_CHECK(nextPts);
        SANITY_CHECK(status);
        SANITY_CHECK(err);
    }
    else if (impl == "ocl")
    {
        ocl::PyrLKOpticalFlow oclPyrLK;
        ocl::oclMat oclFrame0(frame0), oclFrame1(frame1);
        ocl::oclMat oclPts(1, static_cast<int>(pts.size()), CV_32FC2, (void *)&pts[0]);
        ocl::oclMat oclNextPts, oclStatus, oclErr;

        TEST_CYCLE()
                oclPyrLK.sparse(oclFrame0, oclFrame1, oclPts, oclNextPts, oclStatus, &oclErr);

        MatToVector(oclNextPts, nextPts);
        MatToVector(oclStatus, status);
        MatToVector(oclErr, err);

        SANITY_CHECK(nextPts);
        SANITY_CHECK(status);
        SANITY_CHECK(err);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();

//    size_t mismatch = 0;
//    for (int i = 0; i < (int)nextPts.size(); ++i)
//    {
//        if(status[i] != ocl_status.at<unsigned char>(0, i))
//        {
//            mismatch++;
//            continue;
//        }
//        if(status[i])
//        {
//            Point2f gpu_rst = ocl_nextPts.at<Point2f>(0, i);
//            Point2f cpu_rst = nextPts[i];
//            if(fabs(gpu_rst.x - cpu_rst.x) >= 1. || fabs(gpu_rst.y - cpu_rst.y) >= 1.)
//                mismatch++;
//        }
//    }
//    double ratio = (double)mismatch / (double)nextPts.size();
//    if(ratio < .02)
//        TestSystem::instance().setAccurate(1, ratio);
//    else
//        TestSystem::instance().setAccurate(0, ratio);
}


PERFTEST(tvl1flow)
{
    cv::Mat frame0 = imread("rubberwhale1.png", cv::IMREAD_GRAYSCALE);
    assert(!frame0.empty());

    cv::Mat frame1 = imread("rubberwhale2.png", cv::IMREAD_GRAYSCALE);
    assert(!frame1.empty());

    cv::ocl::OpticalFlowDual_TVL1_OCL d_alg;
    cv::ocl::oclMat d_flowx(frame0.size(), CV_32FC1);
    cv::ocl::oclMat d_flowy(frame1.size(), CV_32FC1);

    cv::Ptr<cv::DenseOpticalFlow> alg = cv::createOptFlow_DualTVL1();
    cv::Mat flow;


    SUBTEST << frame0.cols << 'x' << frame0.rows << "; rubberwhale1.png; "<<frame1.cols<<'x'<<frame1.rows<<"; rubberwhale2.png";

    alg->calc(frame0, frame1, flow);

    CPU_ON;
    alg->calc(frame0, frame1, flow);
    CPU_OFF;

    cv::Mat gold[2];
    cv::split(flow, gold);

    cv::ocl::oclMat d0(frame0.size(), CV_32FC1);
    d0.upload(frame0);
    cv::ocl::oclMat d1(frame1.size(), CV_32FC1);
    d1.upload(frame1);

    WARMUP_ON;
    d_alg(d0, d1, d_flowx, d_flowy);
    WARMUP_OFF;
    /*
        double diff1 = 0.0, diff2 = 0.0;
        if(ExceptedMatSimilar(gold[0], cv::Mat(d_flowx), 3e-3, diff1) == 1
            &&ExceptedMatSimilar(gold[1], cv::Mat(d_flowy), 3e-3, diff2) == 1)
            TestSystem::instance().setAccurate(1);
        else
            TestSystem::instance().setAccurate(0);

        TestSystem::instance().setDiff(diff1);
        TestSystem::instance().setDiff(diff2);
    */


    GPU_ON;
    d_alg(d0, d1, d_flowx, d_flowy);
    d_alg.collectGarbage();
    GPU_OFF;


    cv::Mat flowx, flowy;

    GPU_FULL_ON;
    d0.upload(frame0);
    d1.upload(frame1);
    d_alg(d0, d1, d_flowx, d_flowy);
    d_alg.collectGarbage();
    d_flowx.download(flowx);
    d_flowy.download(flowy);
    GPU_FULL_OFF;

    TestSystem::instance().ExceptedMatSimilar(gold[0], flowx, 3e-3);
    TestSystem::instance().ExceptedMatSimilar(gold[1], flowy, 3e-3);
}

///////////// FarnebackOpticalFlow ////////////////////////
PERFTEST(FarnebackOpticalFlow)
{
    cv::Mat frame0 = imread("rubberwhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = imread("rubberwhale2.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    cv::ocl::oclMat d_frame0(frame0), d_frame1(frame1);

    int polyNs[2] = { 5, 7 };
    double polySigmas[2] = { 1.1, 1.5 };
    int farneFlags[2] = { 0, cv::OPTFLOW_FARNEBACK_GAUSSIAN };
    bool UseInitFlows[2] = { false, true };
    double pyrScale = 0.5;

    string farneFlagStrs[2] = { "BoxFilter", "GaussianBlur" };
    string useInitFlowStrs[2] = { "", "UseInitFlow" };

    for ( int i = 0; i < 2; ++i)
    {
        int polyN = polyNs[i];
        double polySigma = polySigmas[i];

        for ( int j = 0; j < 2; ++j)
        {
            int flags = farneFlags[j];

            for ( int k = 0; k < 2; ++k)
            {
                bool useInitFlow = UseInitFlows[k];
                SUBTEST << "polyN(" << polyN << "); " << farneFlagStrs[j] << "; " << useInitFlowStrs[k];

                cv::ocl::FarnebackOpticalFlow farn;
                farn.pyrScale = pyrScale;
                farn.polyN = polyN;
                farn.polySigma = polySigma;
                farn.flags = flags;

                cv::ocl::oclMat d_flowx, d_flowy;
                cv::Mat flow, flowBuf, flowxBuf, flowyBuf;

                WARMUP_ON;
                farn(d_frame0, d_frame1, d_flowx, d_flowy);

                if (useInitFlow)
                {
                    cv::Mat flowxy[] = {cv::Mat(d_flowx), cv::Mat(d_flowy)};
                    cv::merge(flowxy, 2, flow);
                    flow.copyTo(flowBuf);
                    flowxy[0].copyTo(flowxBuf);
                    flowxy[1].copyTo(flowyBuf);

                    farn.flags |= cv::OPTFLOW_USE_INITIAL_FLOW;
                    farn(d_frame0, d_frame1, d_flowx, d_flowy);
                }
                WARMUP_OFF;

                cv::calcOpticalFlowFarneback(
                    frame0, frame1, flow, farn.pyrScale, farn.numLevels, farn.winSize,
                    farn.numIters, farn.polyN, farn.polySigma, farn.flags);

                std::vector<cv::Mat> flowxy;
                cv::split(flow, flowxy);

                Mat md_flowx = cv::Mat(d_flowx);
                Mat md_flowy = cv::Mat(d_flowy);
                TestSystem::instance().ExceptedMatSimilar(flowxy[0], md_flowx, 0.1);
                TestSystem::instance().ExceptedMatSimilar(flowxy[1], md_flowy, 0.1);

                if (useInitFlow)
                {
                    cv::Mat flowx, flowy;
                    farn.flags = (flags | cv::OPTFLOW_USE_INITIAL_FLOW);

                    CPU_ON;
                    cv::calcOpticalFlowFarneback(
                        frame0, frame1, flowBuf, farn.pyrScale, farn.numLevels, farn.winSize,
                        farn.numIters, farn.polyN, farn.polySigma, farn.flags);
                    CPU_OFF;

                    GPU_ON;
                    farn(d_frame0, d_frame1, d_flowx, d_flowy);
                    GPU_OFF;

                    GPU_FULL_ON;
                    d_frame0.upload(frame0);
                    d_frame1.upload(frame1);
                    d_flowx.upload(flowxBuf);
                    d_flowy.upload(flowyBuf);
                    farn(d_frame0, d_frame1, d_flowx, d_flowy);
                    d_flowx.download(flowx);
                    d_flowy.download(flowy);
                    GPU_FULL_OFF;
                }
                else
                {
                    cv::Mat flow, flowx, flowy;
                    cv::ocl::oclMat d_flowx, d_flowy;

                    farn.flags = flags;

                    CPU_ON;
                    cv::calcOpticalFlowFarneback(
                        frame0, frame1, flow, farn.pyrScale, farn.numLevels, farn.winSize,
                        farn.numIters, farn.polyN, farn.polySigma, farn.flags);
                    CPU_OFF;

                    GPU_ON;
                    farn(d_frame0, d_frame1, d_flowx, d_flowy);
                    GPU_OFF;

                    GPU_FULL_ON;
                    d_frame0.upload(frame0);
                    d_frame1.upload(frame1);
                    farn(d_frame0, d_frame1, d_flowx, d_flowy);
                    d_flowx.download(flowx);
                    d_flowy.download(flowy);
                    GPU_FULL_OFF;
                }
            }
        }
    }
}
