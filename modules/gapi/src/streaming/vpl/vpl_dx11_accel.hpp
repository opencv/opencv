#ifndef GAPI_VPL_DX11_ACCEL_HPP
#define GAPI_VPL_DX11_ACCEL_HPP

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>

namespace cv {
namespace gapi {
namespace wip {

struct VPLAccelerationPolicy
{
    ~VPLAccelerationPolicy() {}
};
} // namespace wip
} // namespace gapi
} // namespace cv

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

struct VPLDX11AccelerationPolicy : public VPLAccelerationPolicy
{
    VPLDX11AccelerationPolicy(mfxSession session);
    ~VPLDX11AccelerationPolicy();

private:
    ID3D11Device *hw_handle;
};
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX

#endif // HAVE_ONEVPL
#endif // GAPI_VPL_DX11_ACCEL_HPP
