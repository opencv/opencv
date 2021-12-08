// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/oak/oak.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include "oak_media_adapter.hpp"

#include <thread>
#include <chrono>

namespace cv {
namespace gimpl {
namespace oak {

// Internal kernel package defined by the backend itself.
// The function is defined in goakbackend.cpp
cv::gapi::GKernelPackage kernels();

} // namespace oak
} // namespace gimpl

namespace gapi {
namespace oak {

// Forward declaration in case was built w/o OAK
class OAKMediaAdapter;

GArray<uint8_t> encode(const GMat& in, const EncoderConfig& cfg) {
    return GEncMat::on(in, cfg);
}

GArray<uint8_t> encode(const GFrame& in, const EncoderConfig& cfg) {
    return GEncFrame::on(in, cfg);
}

cv::gapi::GKernelPackage kernels() {
    return cv::gimpl::oak::kernels();
}

// This is a dummy oak::ColorCamera class that just makes our pipelining
// machinery work. The real data comes from the physical camera which
// is handled by DepthAI library.
ColorCamera::ColorCamera()
    : m_dummy(cv::MediaFrame::Create<cv::gapi::oak::OAKMediaAdapter>()) {
}

bool ColorCamera::pull(cv::gapi::wip::Data &data) {
    // FIXME: Avoid passing this formal frame to the pipeline
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    data = m_dummy;
    return true;
}

cv::GMetaArg ColorCamera::descr_of() const {
    return cv::GMetaArg{cv::descr_of(m_dummy)};
}

} // namespace oak
} // namespace gapi
} // namespace cv
