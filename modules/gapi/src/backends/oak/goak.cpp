// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/oak/oak.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/meta.hpp>

#include <thread>
#include <chrono>

namespace cv {
namespace gimpl {
namespace oak {

// Internal kernel package defined by the backend itself.
// The function is defined in oakbackend.cpp
cv::gapi::GKernelPackage kernels();

} // namespace oak
} // namespace gimpl

#ifdef WITH_OAK_BACKEND

namespace gapi {
namespace oak {

cv::GFrame encode(const GFrame& in, const EncoderConfig& cfg) {
    return GEnc::on(in, cfg);
}

cv::gapi::GKernelPackage kernels() {
    return cv::gimpl::oak::kernels();
}

// This is a dummy oak::ColorCamera class that just makes our pipelining
// machinery work. The real data comes from the physical camera which
// is handled by firmware (and so, by the DepthAI library).
ColorCamera::ColorCamera()
    : m_dummy(cv::Mat::eye(cv::Size(3840,2160), CV_8UC3)) {
}

bool ColorCamera::pull(cv::gapi::wip::Data &data) {
    // FIXME:
    // Avoid passing this formal mat to the pipeline
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    data = m_dummy.clone();
    return true;
}

cv::GMetaArg ColorCamera::descr_of() const {
    return cv::GMetaArg{cv::descr_of(m_dummy)};
}

} // namespace oak
} // namespace gapi

#else

// fixme: add proper impls with asserts inside
#error 42

#endif // WITH_OAK_BACKEND
} // namespace cv
