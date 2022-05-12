// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_BASE_FRAME_ADAPTER_HPP
#define GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_BASE_FRAME_ADAPTER_HPP
#include <memory>

#include <opencv2/gapi/media.hpp>
#include <opencv2/gapi/streaming/onevpl/device_selector_interface.hpp>
#include "streaming/onevpl/accelerators/surface/surface.hpp"

#ifdef HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
class BaseFrameAdapter : public cv::MediaFrame::IAdapter {
public:
    using SessionHandle = mfxSession;

    const std::shared_ptr<Surface>& get_surface() const;
    SessionHandle get_session_handle() const;

    cv::GFrameDesc meta() const override;
    AccelType accel_type() const;
protected:
    BaseFrameAdapter(std::shared_ptr<Surface> assoc_surface, SessionHandle assoc_handle,
                     AccelType accel);
    ~BaseFrameAdapter();
    std::shared_ptr<Surface> surface();

    std::shared_ptr<Surface> parent_surface_ptr;
    SessionHandle parent_handle;
    GFrameDesc frame_desc;
    AccelType acceleration_type;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_BASE_FRAME_ADAPTER_HPP
