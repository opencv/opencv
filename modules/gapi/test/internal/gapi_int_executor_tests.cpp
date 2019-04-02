// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"

namespace opencv_test
{

// FIXME: avoid code duplication
// The below graph and config is taken from ComplexIslands test suite
TEST(GExecutor, SmokeTest)
{
    cv::GMat    in[2];
    cv::GMat    tmp[4];
    cv::GScalar scl;
    cv::GMat    out[2];

    tmp[0] = cv::gapi::bitwise_not(cv::gapi::bitwise_not(in[0]));
    tmp[1] = cv::gapi::boxFilter(in[1], -1, cv::Size(3,3));
    tmp[2] = tmp[0] + tmp[1]; // FIXME: handle tmp[2] = tmp[0]+tmp[2] typo
    scl    = cv::gapi::sum(tmp[1]);
    tmp[3] = cv::gapi::medianBlur(tmp[1], 3);
    out[0] = tmp[2] + scl;
    out[1] = cv::gapi::boxFilter(tmp[3], -1, cv::Size(3,3));

    //       isl0                                         #internal1
    //       ...........................                  .........
    // (in1) -> NotNot ->(tmp0) --> Add ---------> (tmp2) --> AddC -------> (out1)
    //       :.....................^...:                  :..^....:
    //                             :                         :
    //                             :                         :
    //      #internal0             :                         :
    //        .....................:.........                :
    // (in2) -> Blur -> (tmp1) ----'--> Sum ----> (scl0) ----'
    //        :..........:..................:                  isl1
    //                   :           ..............................
    //                   `------------> Median -> (tmp3) --> Blur -------> (out2)
    //                               :............................:

    cv::gapi::island("isl0", cv::GIn(in[0], tmp[1]),  cv::GOut(tmp[2]));
    cv::gapi::island("isl1", cv::GIn(tmp[1]), cv::GOut(out[1]));

    cv::Mat in_mat1 = cv::Mat::eye(32, 32, CV_8UC1);
    cv::Mat in_mat2 = cv::Mat::eye(32, 32, CV_8UC1);
    cv::Mat out_gapi[2];

    // Run G-API:
    cv::GComputation(cv::GIn(in[0],   in[1]),    cv::GOut(out[0],      out[1]))
              .apply(cv::gin(in_mat1, in_mat2),  cv::gout(out_gapi[0], out_gapi[1]));

    // Run OpenCV
    cv::Mat out_ocv[2];
    {
        cv::Mat    ocv_tmp0;
        cv::Mat    ocv_tmp1;
        cv::Mat    ocv_tmp2;
        cv::Mat    ocv_tmp3;
        cv::Scalar ocv_scl;

        ocv_tmp0 = in_mat1; // skip !(!)
        cv::boxFilter(in_mat2, ocv_tmp1, -1, cv::Size(3,3));
        ocv_tmp2 = ocv_tmp0 + ocv_tmp1;
        ocv_scl  = cv::sum(ocv_tmp1);
        cv::medianBlur(ocv_tmp1, ocv_tmp3, 3);
        out_ocv[0] = ocv_tmp2 + ocv_scl;
        cv::boxFilter(ocv_tmp3, out_ocv[1], -1, cv::Size(3,3));
    }

    EXPECT_EQ(0, cv::countNonZero(out_gapi[0] != out_ocv[0]));
    EXPECT_EQ(0, cv::countNonZero(out_gapi[1] != out_ocv[1]));

    // FIXME: check that GIslandModel has more than 1 island (e.g. fusion
    // with breakdown worked)
}

// FIXME: Add explicit tests on GMat/GScalar/GArray<T> being connectors
// between executed islands

} // namespace opencv_test
