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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "perf_precomp.hpp"

using namespace std;
using namespace testing;
using namespace perf;

DEF_PARAM_TEST_1(Count, int);

//////////////////////////////////////////////////////////////////////
// ProjectPoints

PERF_TEST_P(Count, Calib3D_ProjectPoints,
            Values(5000, 10000, 20000))
{
    const int count = GetParam();

    cv::Mat src(1, count, CV_32FC3);
    declare.in(src, WARMUP_RNG);

    const cv::Mat rvec = cv::Mat::ones(1, 3, CV_32FC1);
    const cv::Mat tvec = cv::Mat::ones(1, 3, CV_32FC1);
    const cv::Mat camera_mat = cv::Mat::ones(3, 3, CV_32FC1);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::projectPoints(d_src, rvec, tvec, camera_mat, cv::Mat(), dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::projectPoints(src, rvec, tvec, camera_mat, cv::noArray(), dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// SolvePnPRansac

PERF_TEST_P(Count, Calib3D_SolvePnPRansac,
            Values(5000, 10000, 20000))
{
    declare.time(10.0);

    const int count = GetParam();

    cv::Mat object(1, count, CV_32FC3);
    declare.in(object, WARMUP_RNG);

    cv::Mat camera_mat(3, 3, CV_32FC1);
    cv::randu(camera_mat, 0.5, 1);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;

    const cv::Mat dist_coef(1, 8, CV_32F, cv::Scalar::all(0));

    cv::Mat rvec_gold(1, 3, CV_32FC1);
    cv::randu(rvec_gold, 0, 1);

    cv::Mat tvec_gold(1, 3, CV_32FC1);
    cv::randu(tvec_gold, 0, 1);

    std::vector<cv::Point2f> image_vec;
    cv::projectPoints(object, rvec_gold, tvec_gold, camera_mat, dist_coef, image_vec);

    const cv::Mat image(1, count, CV_32FC2, &image_vec[0]);

    cv::Mat rvec;
    cv::Mat tvec;

    if (PERF_RUN_CUDA())
    {
        TEST_CYCLE() cv::cuda::solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);

        CUDA_SANITY_CHECK(rvec, 1e-3);
        CUDA_SANITY_CHECK(tvec, 1e-3);
    }
    else
    {
        TEST_CYCLE() cv::solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);

        CPU_SANITY_CHECK(rvec, 1e-6);
        CPU_SANITY_CHECK(tvec, 1e-6);
    }
}
