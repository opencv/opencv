// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_GSTREAMINGBACKEND_HPP
#define OPENCV_GAPI_GSTREAMINGBACKEND_HPP

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/streaming/format.hpp>
#include "gstreamingkernel.hpp"

namespace cv {
namespace gimpl {
namespace streaming {

cv::gapi::GKernelPackage kernels();

struct GCopy final : public cv::detail::NoTag
{
    static constexpr const char* id() { return "org.opencv.streaming.copy"; }

    static GMetaArgs getOutMeta(const GMetaArgs &in_meta, const GArgs&) {
        GAPI_Assert(in_meta.size() == 1u);
        return in_meta;
    }

    template<typename T> static T on(const T& arg) {
        return cv::GKernelType<GCopy, std::function<T(T)>>::on(arg);
    }
};

} // namespace streaming
} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMINGBACKEND_HPP
