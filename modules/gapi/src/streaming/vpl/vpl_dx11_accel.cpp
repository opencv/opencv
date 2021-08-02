#ifdef HAVE_ONEVPL
#include "streaming/vpl/vpl_dx11_accel.hpp"
#include "streaming/vpl/vpl_utils.hpp"
#include "streaming/vpl/surface/frame_adapter.hpp"
#include "streaming/vpl/surface/surface.hpp"
#include "logger.hpp"

#ifdef HAVE_DIRECTX
    #ifdef HAVE_D3D11
        #pragma comment(lib,"d3d11.lib")
        //#pragma comment(lib,"d3d10.lib")

        #define D3D11_NO_HELPERS
        #include <d3d11.h>
        #include <d3d11_4.h>
        #include <codecvt>
        #include "opencv2/core/directx.hpp"
        #ifdef HAVE_OPENCL
            #include <CL/cl_d3d11.h>
        #endif

namespace cv {
namespace gapi {
namespace wip {
VPLDX11AccelerationPolicy::VPLDX11AccelerationPolicy()
{
#ifdef CPU_ACCEL_ADAPTER
    adapter.reset(new VPLCPUAccelerationPolicy);
#endif
}

VPLDX11AccelerationPolicy::~VPLDX11AccelerationPolicy()
{
    if (hw_handle)
    {
        GAPI_LOG_INFO(nullptr, "VPLDX11AccelerationPolicy release ID3D11Device");
        hw_handle->Release();
    }
}

void VPLDX11AccelerationPolicy::init(session_t session) {
#if 0 /* Activate it for LEGACY API*/
//Create device
    UINT creationFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

#if defined(_DEBUG)
    // If the project is in a debug build, enable debugging via SDK Layers with this flag.
    creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    // This array defines the set of DirectX hardware feature levels this app will support.
    // Note the ordering should be preserved.
    // Don't forget to declare your application's minimum required feature level in its
    // description.  All applications are assumed to support 9.1 unless otherwise stated.
    D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3
    };
    D3D_FEATURE_LEVEL featureLevel;

    // Create the Direct3D 11 API device object and a corresponding context.
    ID3D11DeviceContext* context;
    HRESULT err =
        D3D11CreateDevice(
            nullptr, // Specify nullptr to use the default adapter.
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            creationFlags, // Set set debug and Direct2D compatibility flags.
            featureLevels, // List of feature levels this app can support.
            ARRAYSIZE(featureLevels),
            D3D11_SDK_VERSION, // Always set this to D3D11_SDK_VERSION.
            &hw_handle, // Returns the Direct3D device created.
            &featureLevel, // Returns feature level of device created.
            &context // Returns the device immediate context.
            );
    if(FAILED(err))
    {
        throw std::logic_error("Cannot create D3D11CreateDevice, error: " + std::to_string(HRESULT_CODE(err)));
    }

    // oneVPL recommendation
    ID3D11DeviceContext     *pD11Context;
    ID3D11Multithread       *pD11Multithread;
    hw_handle->GetImmediateContext(&pD11Context);
    pD11Context->QueryInterface(IID_PPV_ARGS(&pD11Multithread));
    pD11Multithread->SetMultithreadProtected(true);
 
    mfxStatus sts = MFXVideoCORE_SetHandle(session, MFX_HANDLE_D3D11_DEVICE, (mfxHDL) hw_handle);
    if (sts != MFX_ERR_NONE)
    {
        throw std::logic_error("Cannot create VPLDX11AccelerationPolicy, MFXVideoCORE_SetHandle error: " +
                               mfxstatus_to_string(sts));
    }
#else
    mfxStatus sts;
#endif
    sts = MFXVideoCORE_GetHandle(session, MFX_HANDLE_D3D11_DEVICE, reinterpret_cast<mfxHDL*>(&hw_handle));
    if (sts != MFX_ERR_NONE)
    {
        throw std::logic_error("Cannot create VPLDX11AccelerationPolicy, MFXVideoCORE_GetHandle error: " +
                               mfxstatus_to_string(sts));
    }

    //MFXVideoCORE_SetFrameAllocator(session, mfxFrameAllocator instance)
    GAPI_LOG_INFO(nullptr, "VPLDX11AccelerationPolicy initialized, session: " << session);
}

void VPLDX11AccelerationPolicy::deinit(session_t session) {
    (void)session;
    GAPI_LOG_INFO(nullptr, "deinitialize session: " << session);
}

VPLDX11AccelerationPolicy::pool_key_t
VPLDX11AccelerationPolicy::create_surface_pool(size_t pool_size, size_t surface_size_bytes,
                                               surface_ptr_ctr_t creator) {
    GAPI_LOG_DEBUG(nullptr, "pool size: " << pool_size << ", surface size bytes: " << surface_size_bytes);

#ifdef CPU_ACCEL_ADAPTER
    return adapter->create_surface_pool(pool_size, surface_size_bytes, creator);
#endif
    (void)pool_size;
    (void)surface_size_bytes;
    (void)creator;
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}

VPLDX11AccelerationPolicy::surface_weak_ptr_t VPLDX11AccelerationPolicy::get_free_surface(pool_key_t key) const
{
#ifdef CPU_ACCEL_ADAPTER
    return adapter->get_free_surface(key);
#endif
    (void)key;
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}

size_t VPLDX11AccelerationPolicy::get_free_surface_count(pool_key_t key) const {
    #ifdef CPU_ACCEL_ADAPTER
    return adapter->get_free_surface_count(key);
#endif
    (void)key;
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}

size_t VPLDX11AccelerationPolicy::get_surface_count(pool_key_t key) const {
#ifdef CPU_ACCEL_ADAPTER
    return adapter->get_surface_count(key);
#endif
    (void)key;
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}

cv::MediaFrame::AdapterPtr VPLDX11AccelerationPolicy::create_frame_adapter(pool_key_t key,
                                                                           mfxFrameSurface1* surface) {

#ifdef CPU_ACCEL_ADAPTER
    return adapter->create_frame_adapter(key, surface);
#endif
    (void)key;
    (void)surface;
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_ONEVPL
