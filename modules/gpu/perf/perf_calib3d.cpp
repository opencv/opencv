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

//////////////////////////////////////////////////////////////////////
// StereoBM

typedef std::tr1::tuple<string, string> pair_string;
DEF_PARAM_TEST_1(ImagePair, pair_string);

PERF_TEST_P(ImagePair, Calib3D_StereoBM,
            Values(pair_string("gpu/perf/aloe.png", "gpu/perf/aloeR.png")))
{
    declare.time(300.0);

    const cv::Mat imgLeft = readImage(GET_PARAM(0), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(imgLeft.empty());

    const cv::Mat imgRight = readImage(GET_PARAM(1), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(imgRight.empty());

    const int preset = 0;
    const int ndisp = 256;

    if (PERF_RUN_GPU())
    {
        cv::gpu::StereoBM_GPU d_bm(preset, ndisp);

        const cv::gpu::GpuMat d_imgLeft(imgLeft);
        const cv::gpu::GpuMat d_imgRight(imgRight);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() d_bm(d_imgLeft, d_imgRight, dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::StereoBM bm(preset, ndisp);

        cv::Mat dst;

        TEST_CYCLE() bm(imgLeft, imgRight, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// StereoBeliefPropagation

PERF_TEST_P(ImagePair, Calib3D_StereoBeliefPropagation,
            Values(pair_string("gpu/stereobp/aloe-L.png", "gpu/stereobp/aloe-R.png")))
{
    declare.time(300.0);

    const cv::Mat imgLeft = readImage(GET_PARAM(0));
    ASSERT_FALSE(imgLeft.empty());

    const cv::Mat imgRight = readImage(GET_PARAM(1));
    ASSERT_FALSE(imgRight.empty());

    const int ndisp = 64;

    if (PERF_RUN_GPU())
    {
        cv::gpu::StereoBeliefPropagation d_bp(ndisp);

        const cv::gpu::GpuMat d_imgLeft(imgLeft);
        const cv::gpu::GpuMat d_imgRight(imgRight);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() d_bp(d_imgLeft, d_imgRight, dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// StereoConstantSpaceBP

PERF_TEST_P(ImagePair, Calib3D_StereoConstantSpaceBP,
            Values(pair_string("gpu/stereobm/aloe-L.png", "gpu/stereobm/aloe-R.png")))
{
    declare.time(300.0);

    const cv::Mat imgLeft = readImage(GET_PARAM(0), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(imgLeft.empty());

    const cv::Mat imgRight = readImage(GET_PARAM(1), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(imgRight.empty());

    const int ndisp = 128;

    if (PERF_RUN_GPU())
    {
        cv::gpu::StereoConstantSpaceBP d_csbp(ndisp);

        const cv::gpu::GpuMat d_imgLeft(imgLeft);
        const cv::gpu::GpuMat d_imgRight(imgRight);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() d_csbp(d_imgLeft, d_imgRight, dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// DisparityBilateralFilter

PERF_TEST_P(ImagePair, Calib3D_DisparityBilateralFilter,
            Values(pair_string("gpu/stereobm/aloe-L.png", "gpu/stereobm/aloe-disp.png")))
{
    const cv::Mat img = readImage(GET_PARAM(0), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    const cv::Mat disp = readImage(GET_PARAM(1), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(disp.empty());

    const int ndisp = 128;

    if (PERF_RUN_GPU())
    {
        cv::gpu::DisparityBilateralFilter d_filter(ndisp);

        const cv::gpu::GpuMat d_img(img);
        const cv::gpu::GpuMat d_disp(disp);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() d_filter(d_disp, d_img, dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// TransformPoints

DEF_PARAM_TEST_1(Count, int);

PERF_TEST_P(Count, Calib3D_TransformPoints,
            Values(5000, 10000, 20000))
{
    const int count = GetParam();

    cv::Mat src(1, count, CV_32FC3);
    declare.in(src, WARMUP_RNG);

    const cv::Mat rvec = cv::Mat::ones(1, 3, CV_32FC1);
    const cv::Mat tvec = cv::Mat::ones(1, 3, CV_32FC1);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::transformPoints(d_src, rvec, tvec, dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

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

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::projectPoints(d_src, rvec, tvec, camera_mat, cv::Mat(), dst);

        GPU_SANITY_CHECK(dst);
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

    if (PERF_RUN_GPU())
    {
        TEST_CYCLE() cv::gpu::solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);

        GPU_SANITY_CHECK(rvec, 1e-3);
        GPU_SANITY_CHECK(tvec, 1e-3);
    }
    else
    {
        TEST_CYCLE() cv::solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);

        CPU_SANITY_CHECK(rvec, 1e-6);
        CPU_SANITY_CHECK(tvec, 1e-6);
    }
}

//////////////////////////////////////////////////////////////////////
// ReprojectImageTo3D

PERF_TEST_P(Sz_Depth, Calib3D_ReprojectImageTo3D,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    cv::Mat Q(4, 4, CV_32FC1);
    cv::randu(Q, 0.1, 1.0);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::reprojectImageTo3D(d_src, dst, Q);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::reprojectImageTo3D(src, dst, Q);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// DrawColorDisp

PERF_TEST_P(Sz_Depth, Calib3D_DrawColorDisp,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16S)))
{
    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::drawColorDisp(d_src, dst, 255);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}
