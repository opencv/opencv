// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include "opencv2/gapi/core.hpp"

//#include "opencv2/gapi/fluid/gfluidbuffer.hpp"
//#include "opencv2/gapi/fluid/gfluidkernel.hpp"

 // FIXME: move these tests with priv() to internal suite
//#include "backends/fluid/gfluidbuffer_priv.hpp"

#include "gapi_gpu_test_kernels.hpp"
#include "logger.hpp"
#include "common/gapi_tests_common.hpp"


namespace opencv_test
{

#ifdef HAVE_OPENCL

using namespace cv::gapi_test_kernels;

TEST(GPU, Symm7x7_test)
{
    cv::GMat in;
    auto out = TSymm7x7_test::on(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    const auto sz = cv::Size(1280, 720);
    cv::Mat in_mat = cv::Mat::eye(sz, CV_8UC1);
    cv::Mat out_mat_gapi(sz, CV_8UC1);
    cv::Mat out_mat_ocv(sz, CV_8UC1);
    cv::Scalar mean = cv::Scalar(127.0f);
    cv::Scalar stddev = cv::Scalar(40.f);
    cv::randn(in_mat, mean, stddev);

    // Run G-API
    auto cc = comp.compile(cv::descr_of(in_mat), cv::compile_args(gpuTestPackage));
    cc(cv::gin(in_mat), cv::gout(out_mat_gapi));

    // Run OpenCV
    cv::Point anchor = { -1, -1 };
    double delta = 0;

    int c_int[10] = { 1140, -118, 526, 290, -236, 64, -128, -5, -87, -7 };
    float c_float[10];
    for (int i = 0; i < 10; i++)
    {
        c_float[i] = c_int[i] / 1024.0f;
    }
    // J & I & H & G & H & I & J
    // I & F & E & D & E & F & I
    // H & E & C & B & C & E & H
    // G & D & B & A & B & D & G
    // H & E & C & B & C & E & H
    // I & F & E & D & E & F & I
    // J & I & H & G & H & I & J

    // A & B & C & D & E & F & G & H & I & J

    // 9 & 8 & 7 & 6 & 7 & 8 & 9
    // 8 & 5 & 4 & 3 & 4 & 5 & 8
    // 7 & 4 & 2 & 1 & 2 & 4 & 7
    // 6 & 3 & 1 & 0 & 1 & 3 & 6
    // 7 & 4 & 2 & 1 & 2 & 4 & 7
    // 8 & 5 & 4 & 3 & 4 & 5 & 8
    // 9 & 8 & 7 & 6 & 7 & 8 & 9

    float coefficients[49] =
    {
        c_float[9], c_float[8], c_float[7], c_float[6], c_float[7], c_float[8], c_float[9],
        c_float[8], c_float[5], c_float[4], c_float[3], c_float[4], c_float[5], c_float[8],
        c_float[7], c_float[4], c_float[2], c_float[1], c_float[2], c_float[4], c_float[7],
        c_float[6], c_float[3], c_float[1], c_float[0], c_float[1], c_float[3], c_float[6],
        c_float[7], c_float[4], c_float[2], c_float[1], c_float[2], c_float[4], c_float[7],
        c_float[8], c_float[5], c_float[4], c_float[3], c_float[4], c_float[5], c_float[8],
        c_float[9], c_float[8], c_float[7], c_float[6], c_float[7], c_float[8], c_float[9]
    };

    cv::Mat kernel = cv::Mat(7, 7, CV_32FC1);
    float* cf = kernel.ptr<float>();
    for (int i = 0; i < 49; i++)
    {
        cf[i] = coefficients[i];
    }

    cv::filter2D(in_mat, out_mat_ocv, CV_8UC1, kernel, anchor, delta, BORDER_REPLICATE);

    //cv::imshow("Output OCV", out_mat_ocv);
    //cv::imshow("Output GAPI", out_mat_gapi);
    //cv::Mat diff = out_mat_ocv - out_mat_gapi;
    //cv::imshow("DIFF", diff);
    //int key = cv::waitKey(0);


    compare_f cmpF = AbsSimilarPoints(1, 0.05).to_compare_f();

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}
#endif

} // namespace opencv_test
