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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
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
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

/////////////////////////////////////////////////////////////////////////////////////////////////
// FarnebackOpticalFlow
namespace
{
    IMPLEMENT_PARAM_CLASS(PyrScale, double)
    IMPLEMENT_PARAM_CLASS(PolyN, int)
    CV_FLAGS(FarnebackOptFlowFlags, 0, OPTFLOW_FARNEBACK_GAUSSIAN)
    IMPLEMENT_PARAM_CLASS(UseInitFlow, bool)
}

PARAM_TEST_CASE(FarnebackOpticalFlow, PyrScale, PolyN, FarnebackOptFlowFlags, UseInitFlow)
{
    int numLevels;
    int winSize;
    int numIters;
    double pyrScale;
    int polyN;
    int flags;
    bool useInitFlow;

    virtual void SetUp()
    {
        numLevels = 5;
        winSize = 13;
        numIters = 10;
        pyrScale = GET_PARAM(0);
        polyN = GET_PARAM(1);
        flags = GET_PARAM(2);
        useInitFlow = GET_PARAM(3);
    }
};

OCL_TEST_P(FarnebackOpticalFlow, Mat)
{
    cv::Mat frame0 = readImage("optflow/RubberWhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("optflow/RubberWhale2.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    double polySigma = polyN <= 5 ? 1.1 : 1.5;

    cv::Mat flow; cv::UMat uflow;
    if (useInitFlow)
    {
        OCL_ON(cv::calcOpticalFlowFarneback(frame0, frame1, uflow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags));
        uflow.copyTo(flow);
        flags |= cv::OPTFLOW_USE_INITIAL_FLOW;
    }
    OCL_OFF(cv::calcOpticalFlowFarneback(frame0, frame1, flow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags));
    OCL_ON(cv::calcOpticalFlowFarneback(frame0, frame1, uflow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags));

    EXPECT_MAT_SIMILAR(flow, uflow, 0.1)
}


OCL_INSTANTIATE_TEST_CASE_P(Video, FarnebackOpticalFlow,
                            Combine(
                                Values(PyrScale(0.3), PyrScale(0.5), PyrScale(0.8)),
                                Values(PolyN(5), PolyN(7)),
                                Values(FarnebackOptFlowFlags(0), FarnebackOptFlowFlags(cv::OPTFLOW_FARNEBACK_GAUSSIAN)),
                                Values(UseInitFlow(false), UseInitFlow(true))
                                )
                           );


} } // namespace cvtest::ocl

#endif // HAVE_OPENCL