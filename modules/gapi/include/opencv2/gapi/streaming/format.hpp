// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_GSTREAMING_FORMAT_HPP
#define OPENCV_GAPI_GSTREAMING_FORMAT_HPP

#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

namespace cv {
namespace gapi {
namespace streaming {

GAPI_EXPORTS cv::gapi::GKernelPackage kernels();

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

G_API_OP(GBGR, <GMat(GFrame)>, "org.opencv.streaming.BGR")
{
    static GMatDesc outMeta(const GFrameDesc& in) { return GMatDesc{CV_8U, 3, in.size}; }
};

/** @brief Gets copy of the input

@note Function textual ID is "org.opencv.streaming.copy"

@param in G-type input
@return Copy of the input
*/
template<typename T,
         typename std::enable_if<!cv::detail::is_nongapi_type<T>::value, int>::type = 0>
GAPI_EXPORTS T copy(const T& in) { return GCopy::on<T>(in); }

/** @brief Gets bgr plane from input frame

@note Function textual ID is "org.opencv.streaming.BGR"

@param in Input frame
@return Image in BGR format
*/
GAPI_EXPORTS cv::GMat BGR(const cv::GFrame& in);

} // namespace streaming
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMING_FORMAT_HPP
