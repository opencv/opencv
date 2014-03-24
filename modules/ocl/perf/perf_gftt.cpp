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
//    Peng Xiao, pengxiao@outlook.com
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


#include "perf_precomp.hpp"

using namespace perf;
using std::tr1::tuple;
using std::tr1::get;

///////////// GoodFeaturesToTrack ////////////////////////

typedef tuple<String, double, bool> GoodFeaturesToTrackParams;
typedef TestBaseWithParam<GoodFeaturesToTrackParams> GoodFeaturesToTrackFixture;

OCL_PERF_TEST_P(GoodFeaturesToTrackFixture, GoodFeaturesToTrack,
                ::testing::Combine(OCL_PERF_ENUM(String("gpu/opticalflow/rubberwhale1.png")),
                                   OCL_PERF_ENUM(0.0, 3.0), testing::Bool()))
{
    const GoodFeaturesToTrackParams params = GetParam();
    const String fileName = get<0>(params);
    const double minDistance = get<1>(params), qualityLevel = 0.01;
    const bool harrisDetector = get<2>(params);
    const int maxCorners = 1000;

    Mat frame = imread(getDataPath(fileName), IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "no input image";

    vector<Point2f> pts_gold;
    declare.in(frame);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclFrame(frame), pts_oclmat;
        ocl::GoodFeaturesToTrackDetector_OCL detector(maxCorners, qualityLevel, minDistance, 3,
                                                      harrisDetector);

        OCL_TEST_CYCLE() detector(oclFrame, pts_oclmat);

        detector.downloadPoints(pts_oclmat, pts_gold);

        SANITY_CHECK(pts_gold);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::goodFeaturesToTrack(frame, pts_gold,
                                             maxCorners, qualityLevel, minDistance, noArray(), 3, harrisDetector);

        SANITY_CHECK(pts_gold);
    }
    else
        OCL_PERF_ELSE
}
