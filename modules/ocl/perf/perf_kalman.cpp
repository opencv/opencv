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
//    Jin Ma, jin@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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
#include "precomp.hpp"

///////////// Kalman Filter////////////////////////
static int kfOcl(int iteration, int dim, cv::ocl::oclMat sample_, cv::ocl::oclMat& output, const double max_noise)
{
    cv::ocl::KalmanFilter kalman_filter;

    const double EPSILON = 1.000;

    cv::ocl::oclMat Temp(dim, 1, CV_32F);

    kalman_filter.init(dim, dim);

    cv::ocl::setIdentity(kalman_filter.errorCovPre, 1);
    cv::ocl::setIdentity(kalman_filter.measurementMatrix, 1);
    cv::ocl::setIdentity(kalman_filter.errorCovPost, 1);

    kalman_filter.measurementNoiseCov.setTo(Scalar::all(0));
    kalman_filter.statePre.setTo(Scalar::all(0));
    kalman_filter.statePost.setTo(Scalar::all(0));

    kalman_filter.correct(sample_);

    for(int i = 0; i<iteration; i++)
    {
        kalman_filter.predict();
        
        cv::ocl::gemm(kalman_filter.transitionMatrix, sample_, 1, cv::ocl::oclMat(), 0, Temp);

        Size size1(Temp.cols, Temp.rows);
        Mat temp;
        gen(temp, Temp.rows, Temp.cols, Temp.type(), 0, 0xffff);
        cv::ocl::oclMat temp2(temp);
        cv::ocl::multiply(2, temp2, temp2);
        cv::ocl::subtract(temp2, 1, temp2);
        cv::ocl::multiply(max_noise, temp2, temp2);
        cv::ocl::add(temp2, Temp, Temp);

        Temp.copyTo(sample_);

        kalman_filter.correct(Temp);
    }

    output = kalman_filter.statePost;

    double diff = 0;
    vector<int> idx;

    int code = cvtest::cmpEps(cv::Mat(sample_), cv::Mat(kalman_filter.statePost), &diff, EPSILON, &idx, false);
    
    return code;
}

static int kfCpu(int iteration, int dim, Mat sample_, Mat& output, const double max_noise)
{
    cv::KalmanFilter kalman_filter;

    const double EPSILON = 1.000;

    Mat Temp(dim, 1, CV_32F);

    kalman_filter.init(dim, dim);

    cv::setIdentity(kalman_filter.errorCovPre, 1);
    cv::setIdentity(kalman_filter.measurementMatrix, 1);
    cv::setIdentity(kalman_filter.errorCovPost, 1);

    kalman_filter.measurementNoiseCov.setTo(Scalar::all(0));
    kalman_filter.statePre.setTo(Scalar::all(0));
    kalman_filter.statePost.setTo(Scalar::all(0));

    kalman_filter.correct(sample_);
    
    for(int i = 0; i<iteration; i++)
    {
        kalman_filter.predict();

        cv::gemm(kalman_filter.transitionMatrix, sample_, 1, cv::Mat(), 0, Temp);

        Size size1(Temp.cols, Temp.rows);
        Mat temp;
        gen(temp, Temp.rows, Temp.cols, Temp.type(), 0, 0xffff);

        cv::multiply(2, temp, temp);
        cv::subtract(temp, 1, temp);
        cv::multiply(max_noise, temp, temp);
        cv::add(temp, Temp, Temp);

        Temp.copyTo(sample_);

        kalman_filter.correct(Temp);
    }

    output = kalman_filter.statePost;
    double diff = 0;
    vector<int> idx;

    int code = cvtest::cmpEps(sample_, kalman_filter.statePost, &diff, EPSILON, &idx, false);

    return code;
}

PERFTEST(KalmanFilter)
{
    RNG rng;
    int dim = 1000;

    int iteration = 100;
    const double max_init = 1;
    const double max_noise = 0.1;
    Mat sample_;
    gen(sample_, dim, 1, CV_32F, -max_init, max_init);

    Mat op_cpu;

    int cpu_code= 10, ocl_code = 10;

    kfCpu(iteration, dim, sample_, op_cpu, max_noise);

    CPU_ON;
    cpu_code = kfCpu(iteration, dim, sample_, op_cpu, max_noise);
    CPU_OFF;

    cv::ocl::oclMat sample;
    sample.upload(sample_);
    cv::ocl::oclMat op_;
    
    WARMUP_ON;
    kfOcl(iteration, dim, sample, op_, max_noise);
    WARMUP_OFF;

    GPU_ON;
    ocl_code = kfOcl(iteration, dim, sample, op_, max_noise);
    GPU_OFF;

    if((cpu_code >= 0) && (ocl_code >= 0))
        TestSystem::instance().setAccurate(1, 0.f);
    else
        TestSystem::instance().setAccurate(0, std::abs(cpu_code - ocl_code));

    GPU_FULL_ON;
    kfOcl(iteration, dim, sample, op_, max_noise);
    GPU_FULL_OFF;
}
