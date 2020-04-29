// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_TESTS_INL_HPP
#define OPENCV_GAPI_VIDEO_TESTS_INL_HPP

#include "gapi_video_tests.hpp"

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

} // opencv_test

#endif // OPENCV_GAPI_VIDEO_TESTS_INL_HPP
