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

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

/////////////////////////////////////////////////////////////////////////////////////////////////
// Optical_flow_tvl1
namespace
{
    IMPLEMENT_PARAM_CLASS(UseInitFlow, bool)
    IMPLEMENT_PARAM_CLASS(MedianFiltering, int)
    IMPLEMENT_PARAM_CLASS(ScaleStep, double)
}

PARAM_TEST_CASE(OpticalFlowTVL1, UseInitFlow, MedianFiltering, ScaleStep)
{
    bool useInitFlow;
    int medianFiltering;
    double scaleStep;
    virtual void SetUp()
    {
        useInitFlow = GET_PARAM(0);
        medianFiltering = GET_PARAM(1);
        scaleStep = GET_PARAM(2);
    }
};

OCL_TEST_P(OpticalFlowTVL1, Mat)
{
    cv::Mat frame0 = readImage("optflow/RubberWhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("optflow/RubberWhale2.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    cv::Mat flow; cv::UMat uflow;

    //create algorithm
    cv::Ptr<cv::DualTVL1OpticalFlow> alg = cv::createOptFlow_DualTVL1();

    //set parameters
    alg->setScaleStep(scaleStep);
    alg->setMedianFiltering(medianFiltering);

    //create initial flow as result of algorithm calculation
    if (useInitFlow)
    {
        OCL_ON(alg->calc(frame0, frame1, uflow));
        uflow.copyTo(flow);
    }

    //set flag to use initial flow as it is ready to use
    alg->setUseInitialFlow(useInitFlow);

    OCL_OFF(alg->calc(frame0, frame1, flow));
    OCL_ON(alg->calc(frame0, frame1, uflow));

    EXPECT_MAT_SIMILAR(flow, uflow, 1e-2);
}

OCL_INSTANTIATE_TEST_CASE_P(Video, OpticalFlowTVL1,
    Combine(
    Values(UseInitFlow(false), UseInitFlow(true)),
    Values(MedianFiltering(3), MedianFiltering(-1)),
    Values(ScaleStep(0.3),ScaleStep(0.5))
    )
    );

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
