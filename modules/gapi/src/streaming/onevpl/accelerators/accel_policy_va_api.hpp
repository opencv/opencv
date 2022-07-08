// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_VA_API_HPP
#define GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_VA_API_HPP

#include <map>
#include <vector>

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#include "streaming/onevpl/accelerators/surface/surface_pool.hpp"

#ifdef __linux__
#if defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#include "va/va.h"
#include "va/va_drm.h"
#else
    typedef void* VADisplay;
#endif // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#endif // __linux__

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

// GAPI_EXPORTS for tests
struct GAPI_EXPORTS VPLVAAPIAccelerationPolicy final : public VPLAccelerationPolicy
{
    VPLVAAPIAccelerationPolicy(device_selector_ptr_t selector);
    ~VPLVAAPIAccelerationPolicy();

    using pool_t = CachedPool;

    void init(session_t session) override;
    void deinit(session_t session) override;
    pool_key_t create_surface_pool(const mfxFrameAllocRequest& alloc_request, mfxFrameInfo& info) override;
    surface_weak_ptr_t get_free_surface(pool_key_t key) override;
    size_t get_free_surface_count(pool_key_t key) const override;
    size_t get_surface_count(pool_key_t key) const override;

    cv::MediaFrame::AdapterPtr create_frame_adapter(pool_key_t key,
                                                    const FrameConstructorArgs& args) override;

private:
    std::unique_ptr<VPLAccelerationPolicy> cpu_dispatcher;
#ifdef __linux__
    VADisplay va_handle;
#endif // __linux__
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_VA_API_HPP
