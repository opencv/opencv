// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_TESTS_INL_HPP
#define OPENCV_GAPI_VIDEO_TESTS_INL_HPP

#include "gapi_video_tests.hpp"
#include <opencv2/gapi/streaming/cap.hpp>

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
    cv::gapi::video::KalmanParams kp;
    kp.depth = type;

    kp.statePre = Mat::zeros(dDim, 1, type);
    kp.errorCovPre = Mat::zeros(dDim, dDim, type);
    kp.statePost = Mat::zeros(dDim, 1, type);
    kp.errorCovPost = Mat::zeros(dDim, dDim, type);

    kp.transitionMatrix = Mat::ones(dDim, dDim, type)*2;
    kp.processNoiseCov = Mat::eye(dDim, dDim, type);
    kp.measurementMatrix = Mat::eye(mDim, dDim, type)*2;
    kp.measurementNoiseCov = Mat::eye(mDim, mDim, type);

    if (cDim > 0)
        kp.controlMatrix = Mat::eye(dDim, cDim, type);

    cv::randu(kp.statePre, Scalar::all(-1), Scalar::all(1));

    setIdentity(kp.measurementMatrix);
    setIdentity(kp.processNoiseCov, Scalar::all(1e-5));
    setIdentity(kp.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kp.errorCovPre, Scalar::all(1e-5));

    //measurement vector
    cv::Mat measure_vec(mDim, 1, type);
    cv::randu(measure_vec, Scalar::all(-1), Scalar::all(1));

    //control vector
    cv::Mat ctrl_vec = Mat::zeros(cDim > 0 ? cDim : 2, 1, type);

    // G-API Kalman's output state
    cv::Mat gapiKState(dDim, 1, type);
    // OCV Kalman's output state
    cv::Mat ocvKState(dDim, 1, type);

    // G-API graph initialization
    cv::GMat m, ctrl;
    cv::GOpaque<bool> have_m;
    cv::GMat out = cv::gapi::KalmanFilter(m, have_m, ctrl, kp);
    cv::GComputation comp(cv::GIn(m, have_m, ctrl), cv::GOut(out));

    // OpenCV reference KalmanFilter initialization
    cv::KalmanFilter ocvKalman(dDim, mDim, cDim, type);

    ocvKalman.statePre = kp.statePre;
    ocvKalman.errorCovPre = kp.errorCovPre;
    ocvKalman.statePost = kp.statePost;
    ocvKalman.errorCovPost = kp.errorCovPost;

    ocvKalman.transitionMatrix = kp.transitionMatrix;
    ocvKalman.controlMatrix = kp.controlMatrix;
    ocvKalman.measurementMatrix = kp.measurementMatrix;
    ocvKalman.measurementNoiseCov = kp.measurementNoiseCov;
    ocvKalman.processNoiseCov = kp.processNoiseCov;

    cv::RNG& rng = cv::theRNG();
    bool haveMeasure;

    for (int i = 0; i < numIter; i++)
    {
        haveMeasure = (rng(2u) == 1) ? true : false; // returns 0 or 1 - whether we have measurement at this iteration or not

        if (haveMeasure)
            cv::randu(measure_vec, Scalar::all(-1), Scalar::all(1));
        if (cDim > 0)
            cv::randu(ctrl_vec, Scalar::all(-1), Scalar::all(1));

        // G-API
        comp.apply(cv::gin(measure_vec, haveMeasure, ctrl_vec), cv::gout(gapiKState));

        // OpenCV
        ocvKState = cDim > 0 ? ocvKalman.predict(ctrl_vec) : ocvKalman.predict();
        if (haveMeasure)
            ocvKState = ocvKalman.correct(measure_vec);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        double diff = 0;
        vector<int> idx;
        EXPECT_TRUE(cmpEps(gapiKState, ocvKState, &diff, 1.0, &idx, false) >= 0);
    }
}

TEST_P(KalmanFilterNoControlTest, AccuracyTest)
{
    cv::gapi::video::KalmanParams kp;
    kp.depth = type;

    kp.statePre = Mat::zeros(dDim, 1, type);
    kp.errorCovPre = Mat::zeros(dDim, dDim, type);
    kp.statePost = Mat::zeros(dDim, 1, type);
    kp.errorCovPost = Mat::zeros(dDim, dDim, type);

    kp.transitionMatrix = Mat::ones(dDim, dDim, type) * 2;
    kp.processNoiseCov = Mat::eye(dDim, dDim, type);
    kp.measurementMatrix = Mat::eye(mDim, dDim, type) * 2;
    kp.measurementNoiseCov = Mat::eye(mDim, mDim, type);

    cv::randu(kp.statePre, Scalar::all(-1), Scalar::all(1));

    setIdentity(kp.processNoiseCov, Scalar::all(1e-5));
    setIdentity(kp.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kp.errorCovPost, Scalar::all(1));

    //measurement vector
    cv::Mat measure_vec(mDim, 1, type);
    cv::randu(measure_vec, Scalar::all(-1), Scalar::all(1));

    // G-API Kalman's output state
    cv::Mat gapiKState(dDim, 1, type);
    // OCV Kalman's output state
    cv::Mat ocvKState(dDim, 1, type);

    // G-API graph initialization
    cv::GMat m;
    cv::GOpaque<bool> have_m;
    cv::GMat out = cv::gapi::KalmanFilter(m, have_m, kp);
    cv::GComputation comp(cv::GIn(m, have_m), cv::GOut(out));

    // OpenCV reference KalmanFilter initialization
    cv::KalmanFilter ocvKalman(dDim, mDim, 0, type);

    ocvKalman.statePre = kp.statePre;
    ocvKalman.errorCovPre = kp.errorCovPre;
    ocvKalman.statePost = kp.statePost;
    ocvKalman.errorCovPost = kp.errorCovPost;

    ocvKalman.transitionMatrix = kp.transitionMatrix;
    ocvKalman.measurementMatrix = kp.measurementMatrix;
    ocvKalman.measurementNoiseCov = kp.measurementNoiseCov;
    ocvKalman.processNoiseCov = kp.processNoiseCov;

    cv::RNG& rng = cv::theRNG();
    bool haveMeasure;

    for (int i = 0; i < numIter; i++)
    {
        haveMeasure = (rng(2u) == 1) ? true : false; // returns 0 or 1 - whether we have measurement at this iteration or not

        if (haveMeasure)
            cv::randu(measure_vec, Scalar::all(-1), Scalar::all(1));

        // G-API
        comp.apply(cv::gin(measure_vec, haveMeasure), cv::gout(gapiKState));

        // OpenCV
        ocvKState = ocvKalman.predict();
        if (haveMeasure)
            ocvKState = ocvKalman.correct(measure_vec);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        double diff = 0;
        vector<int> idx;
        EXPECT_TRUE(cmpEps(gapiKState, ocvKState, &diff, 1.0, &idx, false) >= 0);
    }
}

static inline Point calcPoint(Point2f center, double R, double angle)
{
    return center + Point2f((float)cos(angle), (float)-sin(angle))*(float)R;
}

TEST_P(KalmanFilterSampleTest, AccuracyTest)
{
    // auxiliary variables
    cv::Mat img(500, 500, CV_8UC3);
    cv::Mat processNoise(2, 1, type);
#if 0
    char code = (char)-1;
#endif
    // Input mesurement
    cv::Mat measurement = Mat::zeros(1, 1, type);
    // Angle and it's delta(phi, delta_phi)
    cv::Mat state(2, 1, type);

    // G-API graph initialization
    cv::gapi::video::KalmanParams kp;
    kp.depth = type;
    kp.statePre = Mat::zeros(2, 1, type);
    kp.statePost = Mat::zeros(2, 1, type);
    kp.errorCovPre = Mat::zeros(2, 2, type);
    kp.errorCovPost = Mat::zeros(2, 2, type);

    if (type == CV_32F)
        kp.transitionMatrix = (Mat_<float>(2, 2) << 1, 1, 0, 1);
    else
        kp.transitionMatrix = (Mat_<double>(2, 2) << 1, 1, 0, 1);

    kp.processNoiseCov = Mat::zeros(2, 2, type);
    kp.measurementMatrix = Mat::zeros(1, 2, type);
    kp.measurementNoiseCov = Mat::zeros(1, 1, type);

    setIdentity(kp.measurementMatrix);
    setIdentity(kp.processNoiseCov, Scalar::all(1e-5));
    setIdentity(kp.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kp.errorCovPost, Scalar::all(1));

    randn(kp.statePost, Scalar::all(0), Scalar::all(0.1));

    cv::GMat m;
    cv::GOpaque<bool> have_mesure;
    cv::GMat out = cv::gapi::KalmanFilter(m, have_mesure, kp);
    cv::GComputation comp(cv::GIn(m, have_mesure), cv::GOut(out));

    randn(state, Scalar::all(0), Scalar::all(0.1));

    // Corrected state
    cv::Mat correction(2, 1, type);

    bool haveMeasure;
#if 0
    for (;;)
#else
    for (int i = 0; i < numIter; ++i)
#endif
    {
        Point2f center(img.cols*0.5f, img.rows*0.5f);
        float R = img.cols / 3.f;
        double stateAngle = (type == CV_32F) ? state.at<float>(0): state.at<double>(0);
#if 0
        Point statePt = calcPoint(center, R, stateAngle);
#endif
        haveMeasure = false;
        // Predicted state
        cv::Mat prediction(2, 1, type);
        comp.apply(cv::gin(measurement, haveMeasure), cv::gout(prediction));

        double predictAngle = (type == CV_32F) ? prediction.at<float>(0): prediction.at<double>(0);
#if 0
        Point predictPt = calcPoint(center, R, predictAngle);
#endif
        randn(measurement, Scalar::all(0), Scalar::all((type == CV_32F) ?
              kp.measurementNoiseCov.at<float>(0) : kp.measurementNoiseCov.at<double>(0)));

        // generate measurement
        measurement += kp.measurementMatrix*state;
#if 0
        double measAngle = (type == CV_32F) ? measurement.at<float>(0): measurement.at<double>(0);
        Point measPt = calcPoint(center, R, measAngle);

        // plot points
        #define drawCross( center, color, d )                                        \
                line( img, Point( center.x - d, center.y - d ),                          \
                             Point( center.x + d, center.y + d ), color, 1, LINE_AA, 0); \
                line( img, Point( center.x + d, center.y - d ),                          \
                             Point( center.x - d, center.y + d ), color, 1, LINE_AA, 0 )

        img = Scalar::all(0);
        drawCross(statePt, Scalar(255, 255, 255), 3);
        drawCross(measPt, Scalar(0, 0, 255), 3);
        drawCross(predictPt, Scalar(0, 255, 0), 3);
        line(img, statePt, measPt, Scalar(0, 0, 255), 3, LINE_AA, 0);
        line(img, statePt, predictPt, Scalar(0, 255, 255), 3, LINE_AA, 0);
#endif
        if (theRNG().uniform(0, 4) != 0)
        {
            haveMeasure = true;
            comp.apply(cv::gin(measurement, haveMeasure), cv::gout(correction));
        }

        randn(processNoise, Scalar(0), Scalar::all(sqrt(type == CV_32F ?
                                                   kp.processNoiseCov.at<float>(0, 0):
                                                   kp.processNoiseCov.at<double>(0, 0))));
        state = kp.transitionMatrix*state + processNoise;
#if 0
        imshow("Kalman", img);
        code = (char)waitKey(100);

        if (code > 0)
            break;
#endif
    }
}

#endif
} // opencv_test

#endif // OPENCV_GAPI_VIDEO_TESTS_INL_HPP
