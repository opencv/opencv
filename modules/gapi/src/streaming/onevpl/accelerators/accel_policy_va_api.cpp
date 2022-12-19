// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifdef HAVE_ONEVPL
#include <cstdlib>
#include <exception>
#include <stdint.h>

#ifdef __linux__
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif // __linux__

#include "streaming/onevpl/accelerators/accel_policy_va_api.hpp"
#include "streaming/onevpl/accelerators/accel_policy_cpu.hpp"
#include "streaming/onevpl/utils.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
#ifdef __linux__
VPLVAAPIAccelerationPolicy::VPLVAAPIAccelerationPolicy(device_selector_ptr_t selector) :
    VPLAccelerationPolicy(selector),
    cpu_dispatcher(new VPLCPUAccelerationPolicy(selector)),
    va_handle() {
    GAPI_LOG_INFO(nullptr, "created - TODO dispatchered on CPU acceleration");
#if defined(HAVE_VA) || defined(HAVE_VA_INTEL)
    // setup VAAPI device
    IDeviceSelector::DeviceScoreTable devices = get_device_selector()->select_devices();
    GAPI_Assert(devices.size() == 1 && "Multiple(or zero) acceleration  VAAPI devices are not unsupported");
    AccelType accel_type = devices.begin()->second.get_type();
    GAPI_Assert(accel_type == AccelType::VAAPI &&
                "Unexpected device AccelType while is waiting AccelType::VAAPI");

    va_handle = reinterpret_cast<VADisplay>(devices.begin()->second.get_ptr());
#else  // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
    GAPI_Error("VPLVAAPIAccelerationPolicy unavailable in current configuration");
#endif // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
}
#else // __linux__
VPLVAAPIAccelerationPolicy::VPLVAAPIAccelerationPolicy(device_selector_ptr_t selector) :
    VPLAccelerationPolicy(selector) {
    GAPI_Error("VPLVAAPIAccelerationPolicy unavailable in current configuration");
}
#endif // __linux__

#if defined(HAVE_VA) || defined(HAVE_VA_INTEL)
VPLVAAPIAccelerationPolicy::~VPLVAAPIAccelerationPolicy() {
    vaTerminate(va_handle);
    GAPI_LOG_INFO(nullptr, "destroyed");
}

void VPLVAAPIAccelerationPolicy::init(session_t session) {
    GAPI_LOG_INFO(nullptr, "session: " << session);

    cpu_dispatcher->init(session);
    mfxStatus sts = MFXVideoCORE_SetHandle(session,
                                           static_cast<mfxHandleType>(MFX_HANDLE_VA_DISPLAY),
                                           va_handle);
    if (sts != MFX_ERR_NONE)
    {
        throw std::logic_error("Cannot create VPLVAAPIAccelerationPolicy, MFXVideoCORE_SetHandle error: " +
                               mfxstatus_to_string(sts));
    }
    GAPI_LOG_INFO(nullptr, "finished successfully, session: " << session);
}

void VPLVAAPIAccelerationPolicy::deinit(session_t session) {
    GAPI_LOG_INFO(nullptr, "session: " << session);
}

VPLVAAPIAccelerationPolicy::pool_key_t
VPLVAAPIAccelerationPolicy::create_surface_pool(const mfxFrameAllocRequest& alloc_request, mfxFrameInfo& info) {

    return cpu_dispatcher->create_surface_pool(alloc_request, info);
}

VPLVAAPIAccelerationPolicy::surface_weak_ptr_t VPLVAAPIAccelerationPolicy::get_free_surface(pool_key_t key) {
    return cpu_dispatcher->get_free_surface(key);
}

size_t VPLVAAPIAccelerationPolicy::get_free_surface_count(pool_key_t key) const {
    return cpu_dispatcher->get_free_surface_count(key);
}

size_t VPLVAAPIAccelerationPolicy::get_surface_count(pool_key_t key) const {
    return cpu_dispatcher->get_surface_count(key);
}

cv::MediaFrame::AdapterPtr VPLVAAPIAccelerationPolicy::create_frame_adapter(pool_key_t key,
                                                                          const FrameConstructorArgs &params) {
    return cpu_dispatcher->create_frame_adapter(key, params);
}

#else // defined(HAVE_VA) || defined(HAVE_VA_INTEL)

VPLVAAPIAccelerationPolicy::~VPLVAAPIAccelerationPolicy() = default;

void VPLVAAPIAccelerationPolicy::init(session_t ) {
    GAPI_Error("VPLVAAPIAccelerationPolicy unavailable in current configuration");
}

void VPLVAAPIAccelerationPolicy::deinit(session_t) {
    GAPI_Error("VPLVAAPIAccelerationPolicy unavailable in current configuration");
}

VPLVAAPIAccelerationPolicy::pool_key_t VPLVAAPIAccelerationPolicy::create_surface_pool(const mfxFrameAllocRequest&,
                                                                                     mfxFrameInfo&) {
    GAPI_Error("VPLVAAPIAccelerationPolicy unavailable in current configuration");
}

VPLVAAPIAccelerationPolicy::surface_weak_ptr_t VPLVAAPIAccelerationPolicy::get_free_surface(pool_key_t) {
    GAPI_Error("VPLVAAPIAccelerationPolicy unavailable in current configuration");
}

size_t VPLVAAPIAccelerationPolicy::get_free_surface_count(pool_key_t) const {
    GAPI_Error("VPLVAAPIAccelerationPolicy unavailable in current configuration");
}

size_t VPLVAAPIAccelerationPolicy::get_surface_count(pool_key_t) const {
    GAPI_Error("VPLVAAPIAccelerationPolicy unavailable in current configuration");
}

cv::MediaFrame::AdapterPtr VPLVAAPIAccelerationPolicy::create_frame_adapter(pool_key_t,
                                                                          const FrameConstructorArgs &) {
    GAPI_Error("VPLVAAPIAccelerationPolicy unavailable in current configuration");
}
#endif // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
