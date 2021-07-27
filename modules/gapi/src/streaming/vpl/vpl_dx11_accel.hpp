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
    VPLDX11AccelerationPolicy(mfxSession session);
    ~VPLDX11AccelerationPolicy();

    cv::MediaFrame::AdapterPtr create_frame_adapter(mfxFrameSurface1* surface_ptr) override;

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
