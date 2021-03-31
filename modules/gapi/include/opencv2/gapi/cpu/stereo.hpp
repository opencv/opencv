// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_CPU_STEREO_API_HPP
#define OPENCV_GAPI_CPU_STEREO_API_HPP

#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

namespace cv {
namespace gapi {
namespace calib3d {
namespace cpu {

GAPI_EXPORTS GKernelPackage kernels();

/** @brief Structure for the Stereo operation initialization parameters.*/
struct GAPI_EXPORTS StereoInitParam {
    StereoInitParam(int nD, int bS, double bL, double f):
        numDisparities(nD), blockSize(bS), baseline(bL), focus(f) {}

    StereoInitParam() = default;

    int numDisparities = 0;
    int blockSize = 21;
    double baseline = 70.;
    double focus = 1000.;
};

} // namespace cpu
} // namespace calib3d
} // namespace gapi

namespace detail {

    template<> struct CompileArgTag<cv::gapi::calib3d::cpu::StereoInitParam> {
    static const char* tag() {
        return "org.opencv.stereoInit";
    }
};

} // namespace detail
} // namespace cv


#endif // OPENCV_GAPI_CPU_STEREO_API_HPP
