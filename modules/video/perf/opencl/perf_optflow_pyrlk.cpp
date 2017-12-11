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

typedef tuple< int > PyrLKOpticalFlowParams;
typedef TestBaseWithParam<PyrLKOpticalFlowParams> PyrLKOpticalFlowFixture;

OCL_PERF_TEST_P(PyrLKOpticalFlowFixture, PyrLKOpticalFlow,
                ::testing::Values(1000, 2000, 4000)
                )
{
    Mat frame0 = imread(getDataPath("gpu/opticalflow/rubberwhale1.png"), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty()) << "can't load rubberwhale1.png";

    Mat frame1 = imread(getDataPath("gpu/opticalflow/rubberwhale2.png"), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty()) << "can't load rubberwhale2.png";

    UMat uFrame0; frame0.copyTo(uFrame0);
    UMat uFrame1; frame1.copyTo(uFrame1);

    const Size winSize = Size(21, 21);
    const int maxLevel = 3;
    const TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    const int flags = 0;
    const float minEigThreshold = 1e-4f;
    const double eps = 1.0;

    const PyrLKOpticalFlowParams params = GetParam();
    const int pointsCount = get<0>(params);

    // SKIP unstable tests
#ifdef __linux__
    if (cvtest::skipUnstableTests && ocl::useOpenCL())
    {
         if (ocl::Device::getDefault().isIntel())
             throw ::perf::TestBase::PerfSkipTestException();
    }
#endif

    vector<Point2f> pts;
    goodFeaturesToTrack(frame0, pts, pointsCount, 0.01, 0.0);
    Mat ptsMat(1, static_cast<int>(pts.size()), CV_32FC2, (void *)&pts[0]);

    declare.in(uFrame0, uFrame1, WARMUP_READ);
    UMat uNextPts, uStatus, uErr;
    OCL_TEST_CYCLE()
        cv::calcOpticalFlowPyrLK(uFrame0, uFrame1, pts, uNextPts, uStatus, uErr, winSize, maxLevel, criteria, flags, minEigThreshold);

    SANITY_CHECK(uNextPts, eps);
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
