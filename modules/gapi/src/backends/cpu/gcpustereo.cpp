// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/stereo.hpp>
#include <opencv2/gapi/cpu/stereo.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#ifdef HAVE_OPENCV_CALIB3D
#include <opencv2/calib3d.hpp>
#endif // HAVE_OPENCV_CALIB3D

#ifdef HAVE_OPENCV_CALIB3D

/** @brief Structure for the Stereo operation setup parameters.*/
struct GAPI_EXPORTS StereoSetup {
    double baseline;
    double focus;
    cv::Ptr<cv::StereoBM> stereoBM;
};

namespace {
cv::Mat calcDepth(const cv::Mat &left, const cv::Mat &right,
                  const StereoSetup &ss) {
    constexpr int DISPARITY_SHIFT_16S = 4;
    cv::Mat disp;
    ss.stereoBM->compute(left, right, disp);
    disp.convertTo(disp, CV_32FC1, 1./(1 << DISPARITY_SHIFT_16S), 0);
    return (ss.focus * ss.baseline) / disp;
}
} // anonymous namespace

GAPI_OCV_KERNEL_ST(GCPUStereo, cv::gapi::calib3d::GStereo, StereoSetup)
{
    static void setup(const cv::GMatDesc&, const cv::GMatDesc&,
                      const cv::gapi::StereoOutputFormat,
                      std::shared_ptr<StereoSetup> &stereoSetup,
                      const cv::GCompileArgs &compileArgs) {
        auto stereoInit = cv::gapi::getCompileArg<cv::gapi::calib3d::cpu::StereoInitParam>(compileArgs)
            .value_or(cv::gapi::calib3d::cpu::StereoInitParam{});

        StereoSetup ss{stereoInit.baseline,
                       stereoInit.focus,
                       cv::StereoBM::create(stereoInit.numDisparities,
                       stereoInit.blockSize)};
        stereoSetup = std::make_shared<StereoSetup>(ss);
    }
    static void run(const cv::Mat& left,
                    const cv::Mat& right,
                    const cv::gapi::StereoOutputFormat oF,
                    cv::Mat& out_mat,
                    const StereoSetup &stereoSetup) {
        switch(oF){
            case cv::gapi::StereoOutputFormat::DEPTH_FLOAT16:
                calcDepth(left, right, stereoSetup).convertTo(out_mat, CV_16FC1);
                break;
            case cv::gapi::StereoOutputFormat::DEPTH_FLOAT32:
                calcDepth(left, right, stereoSetup).copyTo(out_mat);
                break;
            case cv::gapi::StereoOutputFormat::DISPARITY_FIXED16_12_4:
                stereoSetup.stereoBM->compute(left, right, out_mat);
                break;
            case cv::gapi::StereoOutputFormat::DISPARITY_FIXED16_11_5:
                GAPI_Assert(false && "This case may be supported in future.");
            default:
                GAPI_Assert(false && "Unknown output format!");
        }
    }
};

cv::GKernelPackage cv::gapi::calib3d::cpu::kernels() {
    static auto pkg = cv::gapi::kernels<GCPUStereo>();
    return pkg;
}

#else

cv::GKernelPackage cv::gapi::calib3d::cpu::kernels()
{
    return GKernelPackage();
}

#endif // HAVE_OPENCV_CALIB3D
