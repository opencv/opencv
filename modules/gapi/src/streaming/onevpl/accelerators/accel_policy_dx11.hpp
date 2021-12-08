// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_DX11_HPP
#define GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_DX11_HPP
#include <map>

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#include "streaming/onevpl/accelerators/surface/surface_pool.hpp"
#include "streaming/onevpl/accelerators/dx11_alloc_resource.hpp"

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
#define D3D11_NO_HELPERS
#define NOMINMAX
#include <d3d11.h>
#include <codecvt>
#include "opencv2/core/directx.hpp"
#ifdef HAVE_OPENCL
#include <CL/cl_d3d11.h>
#endif

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

// GAPI_EXPORTS for tests
struct GAPI_EXPORTS VPLDX11AccelerationPolicy final: public VPLAccelerationPolicy
{
    // GAPI_EXPORTS for tests
    VPLDX11AccelerationPolicy(device_selector_ptr_t selector);
    ~VPLDX11AccelerationPolicy();

    using pool_t = CachedPool;

    void init(session_t session) override;
    void deinit(session_t session) override;
    pool_key_t create_surface_pool(const mfxFrameAllocRequest& alloc_request,
                                   mfxVideoParam& param) override;
    surface_weak_ptr_t get_free_surface(pool_key_t key) override;
    size_t get_free_surface_count(pool_key_t key) const override;
    size_t get_surface_count(pool_key_t key) const override;

    cv::MediaFrame::AdapterPtr create_frame_adapter(pool_key_t key,
                                                    mfxFrameSurface1* surface) override;
private:
    ID3D11Device *hw_handle;
    ID3D11DeviceContext* device_context;

    mfxFrameAllocator allocator;
    static mfxStatus MFX_CDECL alloc_cb(mfxHDL pthis,
                                        mfxFrameAllocRequest *request,
                                        mfxFrameAllocResponse *response);
    static mfxStatus MFX_CDECL lock_cb(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus MFX_CDECL unlock_cb(mfxHDL pthis, mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus MFX_CDECL get_hdl_cb(mfxHDL pthis, mfxMemId mid, mfxHDL *handle);
    static mfxStatus MFX_CDECL free_cb(mfxHDL pthis, mfxFrameAllocResponse *response);

    virtual mfxStatus on_alloc(const mfxFrameAllocRequest *request,
                               mfxFrameAllocResponse *response);
    static mfxStatus on_lock(mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus on_unlock(mfxMemId mid, mfxFrameData *ptr);
    static mfxStatus on_get_hdl(mfxMemId mid, mfxHDL *handle);
    virtual mfxStatus on_free(mfxFrameAllocResponse *response);

    using alloc_id_t = mfxU32;
    using allocation_t = std::shared_ptr<DX11AllocationRecord>;
    std::map<alloc_id_t, allocation_t> allocation_table;

    std::map<pool_key_t, pool_t> pool_table;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#undef NOMINMAX
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_DX11_HPP
