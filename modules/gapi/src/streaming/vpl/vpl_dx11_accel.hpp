#ifndef GAPI_VPL_DX11_ACCEL_HPP
#define GAPI_VPL_DX11_ACCEL_HPP

//TODO
#define  CPU_ACCEL_ADAPTER

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>
#include "streaming/vpl/vpl_accel_policy.hpp"

#ifdef CPU_ACCEL_ADAPTER
#include "streaming/vpl/vpl_cpu_accel.hpp"
#endif

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
        #define D3D11_NO_HELPERS
        #include <d3d11.h>
        #include <codecvt>
        #include "opencv2/core/directx.hpp"
        #ifdef HAVE_OPENCL
            #include <CL/cl_d3d11.h>
        #endif

namespace cv {
namespace gapi {
namespace wip {

struct VPLDX11AccelerationPolicy final: public VPLAccelerationPolicy
{
    VPLDX11AccelerationPolicy();
    ~VPLDX11AccelerationPolicy();

    void init(session_t session) override;
    void deinit(session_t session) override;
    pool_key_t create_surface_pool(size_t pool_size, size_t surface_size_bytes, surface_ptr_ctr_t creator) override;
    surface_weak_ptr_t get_free_surface(pool_key_t key) override;
    size_t get_free_surface_count(pool_key_t key) const override;
    size_t get_surface_count(pool_key_t key) const override;

    cv::MediaFrame::AdapterPtr create_frame_adapter(pool_key_t key,
                                                    mfxFrameSurface1* surface) override;

private:
    ID3D11Device *hw_handle;

#ifdef CPU_ACCEL_ADAPTER
    std::unique_ptr<VPLCPUAccelerationPolicy> adapter;
#endif
};
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX

#endif // HAVE_ONEVPL
#endif // GAPI_VPL_DX11_ACCEL_HPP
