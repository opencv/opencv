// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_INTERFACE_HPP
#define GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_INTERFACE_HPP

#include <functional>
#include <memory>
#include <type_traits>

#include <opencv2/gapi/media.hpp>
#include <opencv2/gapi/streaming/onevpl/device_selector_interface.hpp>

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

class Surface;
struct VPLAccelerationPolicy
{
    using device_selector_ptr_t = std::shared_ptr<IDeviceSelector>;

    VPLAccelerationPolicy(device_selector_ptr_t selector) : device_selector(selector) {}
    virtual ~VPLAccelerationPolicy() = default;

    using pool_key_t = void*;

    using session_t = mfxSession;
    using surface_t = Surface;
    using surface_ptr_t = std::shared_ptr<surface_t>;
    using surface_weak_ptr_t = std::weak_ptr<surface_t>;
    using surface_ptr_ctr_t = std::function<surface_ptr_t(std::shared_ptr<void> out_buf_ptr,
                                                          size_t out_buf_ptr_offset,
                                                          size_t out_buf_ptr_size)>;

    device_selector_ptr_t get_device_selector() {
        return device_selector;
    }
    const device_selector_ptr_t get_device_selector() const {
        return device_selector;
    }

    virtual void init(session_t session) = 0;
    virtual void deinit(session_t session) = 0;

    // Limitation: cannot give guarantee in succesful memory realloccation
    // for existing workspace in existing pool (see realloc)
    // thus it is not implemented,
    // PLEASE provide initial memory area large enough
    virtual pool_key_t create_surface_pool(const mfxFrameAllocRequest& alloc_request, mfxVideoParam& param) = 0;

    virtual surface_weak_ptr_t get_free_surface(pool_key_t key) = 0;
    virtual size_t get_free_surface_count(pool_key_t key) const = 0;
    virtual size_t get_surface_count(pool_key_t key) const = 0;

    virtual cv::MediaFrame::AdapterPtr create_frame_adapter(pool_key_t key,
                                                            mfxFrameSurface1* surface) = 0;
private:
    device_selector_ptr_t device_selector;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_INTERFACE_HPP
