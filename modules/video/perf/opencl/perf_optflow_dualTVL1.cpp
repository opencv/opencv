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

using std::tr1::make_tuple;

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

///////////// OpticalFlow Dual TVL1 ////////////////////////
typedef tuple< tuple<int, double>, bool> OpticalFlowDualTVL1Params;
typedef TestBaseWithParam<OpticalFlowDualTVL1Params> OpticalFlowDualTVL1Fixture;

OCL_PERF_TEST_P(OpticalFlowDualTVL1Fixture, OpticalFlowDualTVL1,
            ::testing::Combine(
                        ::testing::Values(make_tuple<int, double>(-1, 0.3),
                                          make_tuple<int, double>(3, 0.5)),
                        ::testing::Bool()
                                )
            )
    {
        Mat frame0 = imread(getDataPath("cv/optflow/RubberWhale1.png"), cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(frame0.empty()) << "can't load RubberWhale1.png";

        Mat frame1 = imread(getDataPath("cv/optflow/RubberWhale2.png"), cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(frame1.empty()) << "can't load RubberWhale2.png";

        const Size srcSize = frame0.size();

        const OpticalFlowDualTVL1Params params = GetParam();
        const tuple<int, double> filteringScale = get<0>(params);
            const int medianFiltering = get<0>(filteringScale);
            const double scaleStep = get<1>(filteringScale);
        const bool useInitFlow = get<1>(params);
        double eps = 0.9;

        UMat uFrame0; frame0.copyTo(uFrame0);
        UMat uFrame1; frame1.copyTo(uFrame1);
        UMat uFlow(srcSize, CV_32FC2);
        declare.in(uFrame0, uFrame1, WARMUP_READ).out(uFlow, WARMUP_READ);

        //create algorithm
        cv::Ptr<cv::DenseOpticalFlow> alg = cv::createOptFlow_DualTVL1();

        //set parameters
        alg->set("scaleStep", scaleStep);
        alg->setInt("medianFiltering", medianFiltering);

        if (useInitFlow)
        {
            //calculate initial flow as result of optical flow
            alg->calc(uFrame0, uFrame1, uFlow);
        }

        //set flag to use initial flow
        alg->setBool("useInitialFlow", useInitFlow);
        OCL_TEST_CYCLE()
            alg->calc(uFrame0, uFrame1, uFlow);

        SANITY_CHECK(uFlow, eps, ERROR_RELATIVE);
    }
}
} // namespace cvtest::ocl

#endif // HAVE_OPENCL