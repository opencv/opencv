// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "streaming/onevpl/accelerators/surface/cpu_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "logger.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

VPLMediaFrameCPUAdapter::VPLMediaFrameCPUAdapter(std::shared_ptr<Surface> surface,
                                                 SessionHandle assoc_handle):
    BaseFrameAdapter(surface, assoc_handle, AccelType::HOST) {
}

VPLMediaFrameCPUAdapter::~VPLMediaFrameCPUAdapter() = default;

MediaFrame::View VPLMediaFrameCPUAdapter::access(MediaFrame::Access) {
    const Surface::data_t& data = get_surface()->get_data();
    const Surface::info_t& info = get_surface()->get_info();
    using stride_t = typename cv::MediaFrame::View::Strides::value_type;

    stride_t pitch = static_cast<stride_t>(data.Pitch);
    switch(info.FourCC) {
        case MFX_FOURCC_I420:
        {
            cv::MediaFrame::View::Ptrs pp = {
                data.Y,
                data.U,
                data.V,
                nullptr
                };
            cv::MediaFrame::View::Strides ss = {
                    pitch,
                    pitch / 2,
                    pitch / 2, 0u
                };
            return cv::MediaFrame::View(std::move(pp), std::move(ss));
        }
        case MFX_FOURCC_NV12:
        {
            cv::MediaFrame::View::Ptrs pp = {
                data.Y,
                data.UV, nullptr, nullptr
                };
            cv::MediaFrame::View::Strides ss = {
                    pitch,
                    pitch, 0u, 0u
                };
            return cv::MediaFrame::View(std::move(pp), std::move(ss));
        }
            break;
        default:
            throw std::runtime_error("MediaFrame unknown 'fmt' type: " + std::to_string(info.FourCC));
    }
}

cv::util::any VPLMediaFrameCPUAdapter::blobParams() const {
    GAPI_Assert("VPLMediaFrameCPUAdapter::blobParams() is not implemented");
    return {};
}

void VPLMediaFrameCPUAdapter::serialize(cv::gapi::s11n::IOStream&) {
    GAPI_Assert("VPLMediaFrameCPUAdapter::serialize() is not implemented");
}

void VPLMediaFrameCPUAdapter::deserialize(cv::gapi::s11n::IIStream&) {
    GAPI_Assert("VPLMediaFrameCPUAdapter::deserialize() is not implemented");
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
