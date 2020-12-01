// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_TESTS_INL_HPP
#define OPENCV_GAPI_VIDEO_TESTS_INL_HPP

#include "gapi_video_tests.hpp"
#include <opencv2/gapi/streaming/cap.hpp>
#include "opencv2/ts.hpp"

namespace opencv_test
{

TEST_P(BuildOptFlowPyramidTest, AccuracyTest)
{
    std::vector<Mat> outPyrOCV,          outPyrGAPI;
    int              outMaxLevelOCV = 0, outMaxLevelGAPI = 0;

    BuildOpticalFlowPyramidTestParams params { fileName, winSize, maxLevel,
                                              withDerivatives, pyrBorder, derivBorder,
                                              tryReuseInputImage, getCompileArgs() };

    BuildOpticalFlowPyramidTestOutput outOCV  { outPyrOCV,  outMaxLevelOCV };
    BuildOpticalFlowPyramidTestOutput outGAPI { outPyrGAPI, outMaxLevelGAPI };

    runOCVnGAPIBuildOptFlowPyramid(*this, params, outOCV, outGAPI);

    compareOutputPyramids(outOCV, outGAPI);
}

TEST_P(OptFlowLKTest, AccuracyTest)
{
    std::vector<cv::Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>       outStatusOCV, outStatusGAPI;
    std::vector<float>       outErrOCV,    outErrGAPI;

    OptFlowLKTestParams params { fileNamePattern, channels, pointsNum,
                                 winSize, criteria, getCompileArgs() };

    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    runOCVnGAPIOptFlowLK(*this, inPts, params, outOCV, outGAPI);

    compareOutputsOptFlow(outOCV, outGAPI);
}

TEST_P(OptFlowLKTestForPyr, AccuracyTest)
{
    std::vector<cv::Mat>     inPyr1, inPyr2;
    std::vector<cv::Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>       outStatusOCV, outStatusGAPI;
    std::vector<float>       outErrOCV,    outErrGAPI;

    OptFlowLKTestParams params { fileNamePattern, channels, pointsNum,
                                 winSize, criteria, getCompileArgs() };

    OptFlowLKTestInput<std::vector<cv::Mat>> in { inPyr1, inPyr2, inPts };
    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    runOCVnGAPIOptFlowLKForPyr(*this, in, params, withDeriv, outOCV, outGAPI);

    compareOutputsOptFlow(outOCV, outGAPI);
}

TEST_P(BuildPyr_CalcOptFlow_PipelineTest, AccuracyTest)
{
    std::vector<Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>   outStatusOCV, outStatusGAPI;
    std::vector<float>   outErrOCV,    outErrGAPI;

    BuildOpticalFlowPyramidTestParams params { fileNamePattern, winSize, maxLevel,
                                              withDerivatives, BORDER_DEFAULT, BORDER_DEFAULT,
                                              true, getCompileArgs() };

    auto customKernel  = gapi::kernels<GCPUMinScalar>();
    auto kernels       = gapi::combine(customKernel,
                                       params.compileArgs[0].get<gapi::GKernelPackage>());
    params.compileArgs = compile_args(kernels);

    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    runOCVnGAPIOptFlowPipeline(*this, params, outOCV, outGAPI, inPts);

    compareOutputsOptFlow(outOCV, outGAPI);
}

#ifdef HAVE_OPENCV_VIDEO
TEST_P(BackgroundSubtractorTest, AccuracyTest)
{
    initTestDataPath();

    cv::gapi::video::BackgroundSubtractorType opType;
    double thr = -1;
    std::tie(opType, thr) = typeAndThreshold;

    cv::gapi::video::BackgroundSubtractorParams bsp(opType, histLength, thr,
                                                    detectShadows, learningRate);

    // G-API graph declaration
    cv::GMat in;
    cv::GMat out = cv::gapi::BackgroundSubtractor(in, bsp);
    // Preserving 'in' in output to have possibility to compare with OpenCV reference
    cv::GComputation c(cv::GIn(in), cv::GOut(cv::gapi::copy(in), out));

    // G-API compilation of graph for streaming mode
    auto gapiBackSub = c.compileStreaming(getCompileArgs());

    // Testing G-API Background Substractor in streaming mode
    auto path = findDataFile("cv/video/768x576.avi");
    try
    {
        gapiBackSub.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(path));
    }
    catch (...)
    { throw SkipTestException("Video file can't be opened."); }

    cv::Ptr<cv::BackgroundSubtractor> pOCVBackSub;

    if (opType == cv::gapi::video::TYPE_BS_MOG2)
        pOCVBackSub = cv::createBackgroundSubtractorMOG2(histLength, thr,
                                                         detectShadows);
    else if (opType == cv::gapi::video::TYPE_BS_KNN)
        pOCVBackSub = cv::createBackgroundSubtractorKNN(histLength, thr,
                                                        detectShadows);

    // Allowing 1% difference of all pixels between G-API and reference OpenCV results
    testBackgroundSubtractorStreaming(gapiBackSub, pOCVBackSub, 1, 1, learningRate, testNumFrames);
}

TEST_P(KalmanFilterTest, AccuracyTest)
{
    GAPI_Assert(depth == CV_32F || depth == CV_64F);
    GAPI_Assert(dDim > 0 && mDim > 0 && cDim >= 0);

    cv::gapi::video::KalmanParams kp(dDim, mDim, cDim, depth);

    //measurements vector
    cv::Mat measure_vec(mDim, 1, depth);

    //control vector
    cv::Mat ctrl_vec;

    if (cDim > 0)
        ctrl_vec = Mat::zeros(cDim, 1, depth);
    else if (cDim == 0)
        ctrl_vec = Mat::zeros(dDim, 1, depth);

    cv::randu(measure_vec, Scalar::all(-1), Scalar::all(1));

    if (cDim > 0)
        cv::randu(ctrl_vec, Scalar::all(-1), Scalar::all(1));

    cv::Mat gapiKState(dDim, 1, depth);
    cv::Mat ocvKState(dDim, 1, depth);

    // G-API code
    cv::GMat m, ctrl;
    cv::GOpaque<bool> have_m;
    cv::GMat out = cv::gapi::KalmanFilter(m, have_m, ctrl, kp);
    cv::GComputation comp(cv::GIn(m, have_m, ctrl), cv::GOut(out));

    comp.apply(cv::gin(measure_vec, haveMeasure, ctrl_vec), cv::gout(gapiKState));

    cv::KalmanFilter ocvKalman(dDim, mDim, cDim, depth);

    if (cDim > 0)
        ocvKState = ocvKalman.predict(ctrl_vec);
    else if (cDim == 0)
        ocvKState = ocvKalman.predict();

    if (haveMeasure)
        ocvKState = ocvKalman.correct(measure_vec);

    // Comparison //////////////////////////////////////////////////////////////
    {
        double diff = 0;
        vector<int> idx;
        const double eps = 1.0;
        EXPECT_TRUE(cmpEps(gapiKState, ocvKState, &diff, eps, &idx, false) >= 0);
    }
}
#endif

} // opencv_test

#endif // OPENCV_GAPI_VIDEO_TESTS_INL_HPP
