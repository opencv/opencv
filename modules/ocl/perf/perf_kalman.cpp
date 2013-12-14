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

#include "perf_precomp.hpp"

#ifdef HAVE_CLAMDBLAS

using namespace perf;
using namespace std;
using namespace cv::ocl;
using namespace cv;
using std::tr1::tuple;
using std::tr1::get;

///////////// Kalman Filter ////////////////////////

typedef tuple<int> KalmanFilterType;
typedef TestBaseWithParam<KalmanFilterType> KalmanFilterFixture;

PERF_TEST_P(KalmanFilterFixture, KalmanFilter,
    ::testing::Values(1000, 1500))
{
    KalmanFilterType params = GetParam();
    const int dim = get<0>(params);

    cv::Mat sample(dim, 1, CV_32FC1), dresult;
    randu(sample, -1, 1);

    cv::Mat statePre_;

    if (RUN_PLAIN_IMPL)
    {
        cv::KalmanFilter kalman;
        TEST_CYCLE()
        {
            kalman.init(dim, dim);
            kalman.correct(sample);
            kalman.predict();
        }
        statePre_ = kalman.statePre;
    }
    else if(RUN_OCL_IMPL)
    {
        cv::ocl::oclMat dsample(sample);
        cv::ocl::KalmanFilter kalman_ocl;
        OCL_TEST_CYCLE()
        {
            kalman_ocl.init(dim, dim);
            kalman_ocl.correct(dsample);
            kalman_ocl.predict();
        }
        kalman_ocl.statePre.download(statePre_);
    }
    else
        OCL_PERF_ELSE

    SANITY_CHECK(statePre_);
}

#endif // HAVE_CLAMDBLAS
