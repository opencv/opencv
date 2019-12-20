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

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

///////////// FarnebackOpticalFlow ////////////////////////
CV_ENUM(farneFlagType, 0, OPTFLOW_FARNEBACK_GAUSSIAN)

typedef tuple< tuple<int, double>, farneFlagType, bool > FarnebackOpticalFlowParams;
typedef TestBaseWithParam<FarnebackOpticalFlowParams> FarnebackOpticalFlowFixture;

OCL_PERF_TEST_P(FarnebackOpticalFlowFixture, FarnebackOpticalFlow,
                ::testing::Combine(
                    ::testing::Values(
                                      make_tuple<int, double>(5, 1.1),
                                      make_tuple<int, double>(7, 1.5)
                                     ),
                    farneFlagType::all(),
                    ::testing::Bool()
                    )
                )
{
    Mat frame0 = imread(getDataPath("gpu/opticalflow/rubberwhale1.png"), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty()) << "can't load rubberwhale1.png";

    Mat frame1 = imread(getDataPath("gpu/opticalflow/rubberwhale2.png"), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty()) << "can't load rubberwhale2.png";

    const Size srcSize = frame0.size();

    const int numLevels = 5;
    const int winSize = 13;
    const int numIters = 10;

    const FarnebackOpticalFlowParams params = GetParam();
    const tuple<int, double> polyParams = get<0>(params);
    const int polyN = get<0>(polyParams);
    const double polySigma = get<1>(polyParams);
    const double pyrScale = 0.5;
    int flags = get<1>(params);
    const bool useInitFlow = get<2>(params);
    const double eps = 0.1;

    UMat uFrame0; frame0.copyTo(uFrame0);
    UMat uFrame1; frame1.copyTo(uFrame1);
    UMat uFlow(srcSize, CV_32FC2);
    declare.in(uFrame0, uFrame1, WARMUP_READ).out(uFlow, WARMUP_READ);
    if (useInitFlow)
    {
        cv::calcOpticalFlowFarneback(uFrame0, uFrame1, uFlow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);
        flags |= OPTFLOW_USE_INITIAL_FLOW;
    }

    OCL_TEST_CYCLE()
            cv::calcOpticalFlowFarneback(uFrame0, uFrame1, uFlow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);


    SANITY_CHECK(uFlow, eps, ERROR_RELATIVE);
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL