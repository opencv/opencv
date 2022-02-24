// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_CPU_FRAME_ADAPTER_HPP
#define GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_CPU_FRAME_ADAPTER_HPP

#include "streaming/onevpl/accelerators/surface/base_frame_adapter.hpp"

#ifdef HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

class VPLMediaFrameCPUAdapter : public BaseFrameAdapter {
public:
    // GAPI_EXPORTS for tests
    GAPI_EXPORTS explicit VPLMediaFrameCPUAdapter(std::shared_ptr<Surface> assoc_surface,
                                                  SessionHandle assoc_handle);
    GAPI_EXPORTS ~VPLMediaFrameCPUAdapter();

    MediaFrame::View access(MediaFrame::Access) override;

    // The default implementation does nothing
    cv::util::any blobParams() const override;
    void serialize(cv::gapi::s11n::IOStream&) override;
    void deserialize(cv::gapi::s11n::IIStream&) override;

};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_CPU_FRAME_ADAPTER_HPP
