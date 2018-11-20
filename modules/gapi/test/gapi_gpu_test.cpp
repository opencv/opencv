// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"


#include "gapi_gpu_test_kernels.hpp"
#include "logger.hpp"
#include "common/gapi_tests_common.hpp"


namespace opencv_test
{

#ifdef HAVE_OPENCL

using namespace cv::gapi_test_kernels;

TEST(GPU, Symm7x7_test)
{
    const auto sz = cv::Size(1280, 720);
    cv::Mat in_mat = cv::Mat::eye(sz, CV_8UC1);
    cv::Mat out_mat_gapi(sz, CV_8UC1);
    cv::Mat out_mat_ocv(sz, CV_8UC1);
    cv::Scalar mean = cv::Scalar(127.0f);
    cv::Scalar stddev = cv::Scalar(40.f);
    cv::randn(in_mat, mean, stddev);

    //Symm7x7 coefficients and shift
    int coefficients_symm7x7[10] = { 1140, -118, 526, 290, -236, 64, -128, -5, -87, -7 };
    int shift = 10;
    cv::Mat kernel_coeff(10, 1, CV_32S);
    int* ci = kernel_coeff.ptr<int>();
    for (int i = 0; i < 10; i++)
    {
        ci[i] = coefficients_symm7x7[i];
    }

    // Run G-API
    cv::GMat in;
    auto out = TSymm7x7_test::on(in, kernel_coeff, shift);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    auto cc = comp.compile(cv::descr_of(in_mat), cv::compile_args(gpuTestPackage));
    cc(cv::gin(in_mat), cv::gout(out_mat_gapi));

    // Run OpenCV
    reference_symm7x7_CPU(in_mat, kernel_coeff, shift, out_mat_ocv);

    compare_f cmpF = AbsSimilarPoints(1, 0.05).to_compare_f();

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}
#endif

} // namespace opencv_test
