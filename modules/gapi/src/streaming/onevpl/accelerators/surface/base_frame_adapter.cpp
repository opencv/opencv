// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#include "streaming/onevpl/accelerators/surface/base_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "logger.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
BaseFrameAdapter::BaseFrameAdapter(std::shared_ptr<Surface> surface,
                                   SessionHandle assoc_handle,
                                   AccelType accel):
    parent_surface_ptr(surface), parent_handle(assoc_handle),
    acceleration_type(accel) {
    GAPI_Assert(parent_surface_ptr && "Surface is nullptr");
    GAPI_Assert(parent_handle && "mfxSession is nullptr");

    const Surface::info_t& info = parent_surface_ptr->get_info();
    GAPI_LOG_DEBUG(nullptr, "surface: " << parent_surface_ptr->get_handle() <<
                            ", w: " << info.Width << ", h: " << info.Height <<
                            ", p: " << parent_surface_ptr->get_data().Pitch <<
                            ", frame id: " << reinterpret_cast<void*>(this));
    switch(info.FourCC) {
        case MFX_FOURCC_I420:
            throw std::runtime_error("MediaFrame doesn't support I420 type");
            break;
        case MFX_FOURCC_NV12:
            frame_desc.fmt = MediaFormat::NV12;
            break;
        default:
            throw std::runtime_error("MediaFrame unknown 'fmt' type: " + std::to_string(info.FourCC));
    }

    frame_desc.size = cv::Size{info.Width, info.Height};
    parent_surface_ptr->obtain_lock();
}

BaseFrameAdapter::~BaseFrameAdapter() {
    // Each BaseFrameAdapter releases mfx surface counter
    // The last BaseFrameAdapter releases shared Surface pointer
    // The last surface pointer releases workspace memory
    GAPI_LOG_DEBUG(nullptr, "destroy frame id: " << reinterpret_cast<void*>(this));
    parent_surface_ptr->release_lock();
}

const std::shared_ptr<Surface>& BaseFrameAdapter::get_surface() const {
    return parent_surface_ptr;
}

std::shared_ptr<Surface> BaseFrameAdapter::surface() {
    return parent_surface_ptr;
}

BaseFrameAdapter::SessionHandle BaseFrameAdapter::get_session_handle() const {
    return parent_handle;
}

cv::GFrameDesc BaseFrameAdapter::meta() const {
    return frame_desc;
}
AccelType BaseFrameAdapter::accel_type() const {
    return acceleration_type;
}

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
