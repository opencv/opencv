// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STEREO_TESTS_INL_HPP
#define OPENCV_GAPI_STEREO_TESTS_INL_HPP


#include <opencv2/gapi/stereo.hpp>
#include <opencv2/gapi/cpu/stereo.hpp>
#include "gapi_stereo_tests.hpp"

#ifdef HAVE_OPENCV_CALIB3D

#include <opencv2/calib3d.hpp>

namespace opencv_test {

TEST_P(TestGAPIStereo, DisparityDepthTest)
{
    using format = cv::gapi::StereoOutputFormat;
    switch(oF) {
        case format::DEPTH_FLOAT16: dtype = CV_16FC1; break;
        case format::DEPTH_FLOAT32: dtype = CV_32FC1; break;
        case format::DISPARITY_FIXED16_12_4: dtype = CV_16SC1; break;
        default: GAPI_Assert(false && "Unsupported format in test");
    }
    initOutMats(sz, dtype);

    // G-API
    cv::GMat inL, inR;
    cv::GMat out = cv::gapi::stereo(inL, inR, oF);

    cv::GComputation(cv::GIn(inL, inR), cv::GOut(out))
        .apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi),
        cv::compile_args(cv::gapi::calib3d::cpu::kernels(),
                         cv::gapi::calib3d::cpu::StereoInitParam {
                             numDisparities,
                             blockSize,
                             baseline,
                             focus}));

    // OpenCV
    cv::StereoBM::create(numDisparities, blockSize)->compute(in_mat1,
                                                             in_mat2,
                                                             out_mat_ocv);

    static const int DISPARITY_SHIFT_16S = 4;
    switch(oF) {
        case format::DEPTH_FLOAT16:
            out_mat_ocv.convertTo(out_mat_ocv, CV_32FC1, 1./(1 << DISPARITY_SHIFT_16S), 0);
            out_mat_ocv = (focus * baseline) / out_mat_ocv;
            out_mat_ocv.convertTo(out_mat_ocv, CV_16FC1);
            break;
        case format::DEPTH_FLOAT32:
            out_mat_ocv.convertTo(out_mat_ocv, CV_32FC1, 1./(1 << DISPARITY_SHIFT_16S), 0);
            out_mat_ocv = (focus * baseline) / out_mat_ocv;
            break;
        case format::DISPARITY_FIXED16_12_4:
            break;
        default:
            GAPI_Assert(false && "Unsupported format in test");
    }

    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
}

} // namespace opencv_test

#endif // HAVE_OPENCV_CALIB3D

#endif // OPENCV_GAPI_STEREO_TESTS_INL_HPP
