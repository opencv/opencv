#ifdef HAVE_ONEVPL
#include "streaming/vpl/vpl_dx11_accel.hpp"
#include "logger.hpp"

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
VPLDX11AccelerationPolicy::VPLDX11AccelerationPolicy(mfxSession session)
{
    mfxStatus sts = MFXVideoCORE_GetHandle(session, MFX_HANDLE_D3D11_DEVICE, reinterpret_cast<mfxHDL*>(&hw_handle));
    if (sts != MFX_ERR_NONE)
    {
        throw std::logic_error("Cannot create VPLDX11AccelerationPolicy, error: " + std::to_string(sts));
    }

    GAPI_LOG_INFO(nullptr, "VPLDX11AccelerationPolicy initialized");
}

VPLDX11AccelerationPolicy::~VPLDX11AccelerationPolicy()
{
    if (hw_handle)
    {
        GAPI_LOG_INFO(nullptr, "VPLDX11AccelerationPolicy release ID3D11Device");
        hw_handle->Release();
    }
}
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_ONEVPL
