/*M///////////////////////////////////////////////////////////////////////////////////////
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//                        (3-clause BSD License)
//
// Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * Neither the names of the copyright holders nor the names of the contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "perf_precomp.hpp"
#include <algorithm>
#include <functional>

namespace opencv_test
{
using namespace perf;

CV_ENUM(Method, RANSAC, LMEDS)
typedef tuple<int, double, Method, size_t> TranslationParams;
typedef TestBaseWithParam<TranslationParams> EstimateTranslation2DPerf;
#define ESTIMATE_PARAMS Combine(Values(1000), Values(0.95), Method::all(), Values(10, 0))

static float rngIn(float from, float to) { return from + (to - from) * (float)theRNG(); }

static cv::Mat rngTranslationMat()
{
    double tx = rngIn(-2.f, 2.f);
    double ty = rngIn(-2.f, 2.f);
    double t[2*3] = { 1.0, 0.0, tx,
                      0.0, 1.0, ty };
    return cv::Mat(2, 3, CV_64F, t).clone();
}

PERF_TEST_P(EstimateTranslation2DPerf, EstimateTranslation2D, ESTIMATE_PARAMS)
{
    TranslationParams params = GetParam();
    const int n              = get<0>(params);
    const double confidence  = get<1>(params);
    const int method         = get<2>(params);
    const size_t refining    = get<3>(params);

    //fixed seed so the generated data are deterministic
    cv::theRNG().state = 0x12345678;
    // ground-truth pure translation
    cv::Mat T = rngTranslationMat();

    // LMEDS can't handle more than 50% outliers (by design)
    int m;
    if (method == LMEDS)
        m = 3*n/5;
    else
        m = 2*n/5;

    const float shift_outl   = 15.f;
    const float noise_level  = 20.f;

    cv::Mat fpts(1, n, CV_32FC2);
    cv::Mat tpts(1, n, CV_32FC2);

    randu(fpts, 0.f, 100.f);
    transform(fpts, tpts, T);

    // add outliers to the tail [m, n)
    cv::Mat outliers = tpts.colRange(m, n);
    outliers.reshape(1) += shift_outl;

    cv::Mat noise(outliers.size(), outliers.type());
    randu(noise, 0.f, noise_level);
    outliers += noise;

    cv::Vec2d T_est;
    std::vector<uchar> inliers(n);

    warmup(inliers, WARMUP_WRITE);
    warmup(fpts, WARMUP_READ);
    warmup(tpts, WARMUP_READ);

    TEST_CYCLE()
    {
        T_est = estimateTranslation2D(fpts, tpts, inliers, method,
                                      /*ransacReprojThreshold=*/3.0,
                                      /*maxIters=*/2000,
                                      /*confidence=*/confidence,
                                      /*refineIters=*/refining);
    }

    // Convert to Mat for SANITY_CHECK consistency
    cv::Mat T_est_mat = (cv::Mat_<double>(2,1) << T_est[0], T_est[1]);
    SANITY_CHECK(T_est_mat, 1e-6);
}

} // namespace opencv_test
