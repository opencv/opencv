// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/oak/oak.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include "oak_memory_adapters.hpp"

#include <thread>
#include <chrono>

namespace cv {
namespace gapi {
namespace oak {

GArray<uint8_t> encode(const GFrame& in, const EncoderConfig& cfg) {
    return GEncFrame::on(in, cfg);
}

GFrame sobelXY(const GFrame& in, const cv::Mat& hk, const cv::Mat& vk) {
    return GSobelXY::on(in, hk, vk);
}

GFrame copy(const GFrame& in) {
    return GCopy::on(in);
}

// This is a dummy oak::ColorCamera class that just makes our pipelining
// machinery work. The real data comes from the physical camera which
// is handled by DepthAI library.
ColorCamera::ColorCamera()
    : m_dummy(cv::MediaFrame::Create<cv::gapi::oak::OAKMediaAdapter>()) {
}

ColorCamera::ColorCamera(const ColorCameraParams& params)
    : m_dummy(cv::MediaFrame::Create<cv::gapi::oak::OAKMediaAdapter>()),
      m_params(params) {
}

bool ColorCamera::pull(cv::gapi::wip::Data &data) {
    // FIXME: Avoid passing this formal frame to the pipeline
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    data = m_dummy;
    return true;
}

cv::GMetaArg ColorCamera::descr_of() const {
    // FIXME: support other resolutions
    GAPI_Assert(m_params.resolution == ColorCameraParams::Resolution::THE_1080_P);
    return cv::GMetaArg{cv::GFrameDesc{cv::MediaFormat::NV12, cv::Size{1920, 1080}}};
}

} // namespace oak
} // namespace gapi
} // namespace cv
