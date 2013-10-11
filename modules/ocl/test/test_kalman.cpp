///////////////////////////////////////////////////////////////////////////////////////
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

#include "test_precomp.hpp"

#ifdef HAVE_OPENCL

using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
using namespace std;

//////////////////////////////////////////////////////////////////////////

PARAM_TEST_CASE(Kalman, int, int)
{
    int size_;
    int iteration;
    virtual void SetUp()
    {
        size_ = GET_PARAM(0);
        iteration = GET_PARAM(1);
    }
};

OCL_TEST_P(Kalman, Accuracy)
{
    const int Dim = size_;
    const int Steps = iteration;
    const double max_init = 1;
    const double max_noise = 0.1;

    Mat sample_mat(Dim, 1, CV_32F), temp_mat;
    oclMat Sample(Dim, 1, CV_32F);
    oclMat Temp(Dim, 1, CV_32F);
    Mat Temp_cpu(Dim, 1, CV_32F);

    Size size(Sample.cols, Sample.rows);

    sample_mat =  randomMat(size, Sample.type(), -max_init, max_init, false);
    Sample.upload(sample_mat);

    //ocl start
    cv::ocl::KalmanFilter kalman_filter_ocl;
    kalman_filter_ocl.init(Dim, Dim);

    cv::ocl::setIdentity(kalman_filter_ocl.errorCovPre, 1);
    cv::ocl::setIdentity(kalman_filter_ocl.measurementMatrix, 1);
    cv::ocl::setIdentity(kalman_filter_ocl.errorCovPost, 1);

    kalman_filter_ocl.measurementNoiseCov.setTo(Scalar::all(0));
    kalman_filter_ocl.statePre.setTo(Scalar::all(0));
    kalman_filter_ocl.statePost.setTo(Scalar::all(0));

    kalman_filter_ocl.correct(Sample);
    //ocl end

    //cpu start
    cv::KalmanFilter kalman_filter_cpu;

    kalman_filter_cpu.init(Dim, Dim);

    cv::setIdentity(kalman_filter_cpu.errorCovPre, 1);
    cv::setIdentity(kalman_filter_cpu.measurementMatrix, 1);
    cv::setIdentity(kalman_filter_cpu.errorCovPost, 1);

    kalman_filter_cpu.measurementNoiseCov.setTo(Scalar::all(0));
    kalman_filter_cpu.statePre.setTo(Scalar::all(0));
    kalman_filter_cpu.statePost.setTo(Scalar::all(0));

    kalman_filter_cpu.correct(sample_mat);
    //cpu end
    //test begin
    for(int i = 0; i<Steps; i++)
    {
        kalman_filter_ocl.predict();
        kalman_filter_cpu.predict();

        cv::gemm(kalman_filter_cpu.transitionMatrix, sample_mat, 1, cv::Mat(), 0, Temp_cpu);

        Size size1(Temp.cols, Temp.rows);
        Mat temp = randomMat(size1, Temp.type(), 0, 0xffff, false);


        cv::multiply(2, temp, temp);

        cv::subtract(temp, 1, temp);

        cv::multiply(max_noise, temp, temp);

        cv::add(temp, Temp_cpu, Temp_cpu);

        Temp.upload(Temp_cpu);
        Temp.copyTo(Sample);
        Temp_cpu.copyTo(sample_mat);

        kalman_filter_ocl.correct(Temp);
        kalman_filter_cpu.correct(Temp_cpu);
    }
    //test end
    EXPECT_MAT_NEAR(kalman_filter_cpu.statePost, kalman_filter_ocl.statePost, 0);
}

INSTANTIATE_TEST_CASE_P(OCL_Video, Kalman, Combine(Values(3, 7), Values(30)));

#endif // HAVE_OPENCL
