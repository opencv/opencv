// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_PERF_TESTS_INL_HPP
#define OPENCV_GAPI_VIDEO_PERF_TESTS_INL_HPP

#include <iostream>

#include "gapi_video_perf_tests.hpp"

namespace opencv_test
{

  using namespace perf;

//------------------------------------------------------------------------------

PERF_TEST_P_(OptFlowLKPerfTest, TestPerformance)
{
    std::vector<cv::Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>       outStatusOCV, outStatusGAPI;
    std::vector<float>       outErrOCV,    outErrGAPI;

    OptFlowLKTestParams params;
    std::tie(params.fileNamePattern, params.channels,
             params.pointsNum, params.winSize, params.criteria,
             params.compileArgs) = GetParam();

    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    cv::GComputation c = runOCVnGAPIOptFlowLK(*this, inPts, params, outOCV, outGAPI);

    declare.in(in_mat1, in_mat2, inPts).out(outPtsGAPI, outStatusGAPI, outErrGAPI);

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1, in_mat2, inPts, std::vector<cv::Point2f>{ }),
                cv::gout(outPtsGAPI, outStatusGAPI, outErrGAPI));
    }

    // Comparison //////////////////////////////////////////////////////////////
    compareOutputsOptFlow(outOCV, outGAPI);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(OptFlowLKForPyrPerfTest, TestPerformance)
{
    std::vector<cv::Mat>     inPyr1, inPyr2;
    std::vector<cv::Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>       outStatusOCV, outStatusGAPI;
    std::vector<float>       outErrOCV,    outErrGAPI;

    bool withDeriv = false;
    OptFlowLKTestParams params;
    std::tie(params.fileNamePattern, params.channels,
             params.pointsNum, params.winSize, params.criteria,
             withDeriv, params.compileArgs) = GetParam();

    OptFlowLKTestInput<std::vector<cv::Mat>> in { inPyr1, inPyr2, inPts };
    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    cv::GComputation c = runOCVnGAPIOptFlowLKForPyr(*this, in, params, withDeriv, outOCV, outGAPI);

    declare.in(inPyr1, inPyr2, inPts).out(outPtsGAPI, outStatusGAPI, outErrGAPI);

    TEST_CYCLE()
    {
        c.apply(cv::gin(inPyr1, inPyr2, inPts, std::vector<cv::Point2f>{ }),
                cv::gout(outPtsGAPI, outStatusGAPI, outErrGAPI));
    }

    // Comparison //////////////////////////////////////////////////////////////
    compareOutputsOptFlow(outOCV, outGAPI);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

} // opencv_test

#endif // OPENCV_GAPI_VIDEO_PERF_TESTS_INL_HPP
