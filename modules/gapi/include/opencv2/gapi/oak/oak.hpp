// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_OAK_HPP
#define OPENCV_GAPI_OAK_HPP

#include <opencv2/gapi/garg.hpp>       // IStreamSource
#include <opencv2/gapi/gkernel.hpp>    // GKernelPackage
#include <opencv2/gapi/gstreaming.hpp> // GOptRunArgsP

// FIXME: remove?
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/streaming/meta.hpp>
#include <opencv2/gapi/infer/parsers.hpp>
#include <opencv2/gapi/infer.hpp>   // Generic

namespace cv {
namespace gapi {
namespace oak {

// TODO: cover all of dai parameters
struct EncoderConfig {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t framerate = 0;
    // h265 as the default enc type
};

G_API_OP(GEnc, <GFrame(GFrame, EncoderConfig)>, "org.opencv.oak.enc") {
    static GFrameDesc outMeta(const GFrameDesc&, const EncoderConfig&) {
        return cv::empty_gframe_desc();
    }
};

GAPI_EXPORTS GFrame encode(const GFrame& in, const EncoderConfig& = {});

// OAK backend & kernels ////////////////////////////////////////////////////////
GAPI_EXPORTS cv::gapi::GBackend backend();
GAPI_EXPORTS_W cv::gapi::GKernelPackage kernels();

// Camera object ///////////////////////////////////////////////////////////////

struct GAPI_EXPORTS ColorCameraParams {};

class GAPI_EXPORTS ColorCamera: public cv::gapi::wip::IStreamSource {
    cv::MediaFrame m_dummy;

    virtual bool pull(cv::gapi::wip::Data &data) override;
    virtual GMetaArg descr_of() const override;

public:
    ColorCamera();
};

} // namespace oak
} // namespace gapi

// FIXME: remove
//inline GOptRunArgsP& operator+= (      cv::GOptRunArgsP &lhs,
//                                 const cv::GOptRunArgsP &rhs) {
//    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
//    return lhs;
//}

namespace detail {
template<> struct CompileArgTag<gapi::oak::ColorCameraParams> {
    static const char* tag() { return "gapi.oak.colorCameraParams"; }
};

template<> struct CompileArgTag<gapi::oak::EncoderConfig> {
    static const char* tag() { return "gapi.oak.encoderConfig"; }
};
} // namespace detail

} // namespace cv

#endif // OPENCV_GAPI_OAK_HPP
