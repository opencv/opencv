// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"
#include <opencv2/gapi/own/assert.hpp>
#include <opencv2/gapi/util/variant.hpp>

#include <opencv2/gapi/streaming/onevpl/device_selector_interface.hpp>
#include "streaming/onevpl/cfg_param_device_selector.hpp"
#include "streaming/onevpl/cfg_params_parser.hpp"
#include "streaming/onevpl/utils.hpp"
#include "logger.hpp"

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11

// get rid of generate macro max/min/etc from DX side
#define D3D11_NO_HELPERS
#define NOMINMAX
#include <d3d11.h>
#include <d3d11_4.h>
#undef D3D11_NO_HELPERS
#undef NOMINMAX
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#if defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#include "va/va.h"
#include "va/va_drm.h"
#endif // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#endif // __linux__

#include <codecvt>
#include "opencv2/core/directx.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
#ifdef __linux__
struct PlatformSpecificParams {
    ~PlatformSpecificParams() {
        for (int fd : fds) {
            close(fd);
        }
    }

    void track_fd(int fd) {
        fds.insert(fd);
    }
private:
    std::set<int> fds;
};
#else
struct PlatformSpecificParams {};
#endif

std::vector<CfgParam> update_param_with_accel_type(std::vector<CfgParam> &&param_array, AccelType type) {
    switch (type) {
        case AccelType::HOST:
            break;
        case AccelType::DX11:
            param_array.push_back(CfgParam::create_acceleration_mode(MFX_ACCEL_MODE_VIA_D3D11));
            break;
        case AccelType::VAAPI:
            param_array.push_back(CfgParam::create_acceleration_mode(MFX_IMPL_VIA_VAAPI));
            break;
        default:
            GAPI_DbgAssert(false && "Unexpected AccelType");
            break;
    }
    return std::move(param_array);
}

CfgParamDeviceSelector::CfgParamDeviceSelector(const CfgParams& cfg_params) :
    suggested_device(IDeviceSelector::create<Device>(nullptr, "CPU", AccelType::HOST)),
    suggested_context(IDeviceSelector::create<Context>(nullptr, AccelType::HOST)) {

    auto accel_mode_it =
        std::find_if(cfg_params.begin(), cfg_params.end(), [] (const CfgParam& value) {
            return value.get_name() == CfgParam::acceleration_mode_name();
        });
    if (accel_mode_it == cfg_params.end())
    {
        GAPI_LOG_DEBUG(nullptr, "No HW Accel requested. Use default CPU");
        return;
    }

    GAPI_LOG_DEBUG(nullptr, "Add HW acceleration support");
    mfxVariant accel_mode = cfg_param_to_mfx_variant(*accel_mode_it);

    switch(accel_mode.Data.U32) {
        case MFX_ACCEL_MODE_VIA_D3D11: {
#if defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
            ID3D11Device *hw_handle = nullptr;
            ID3D11DeviceContext* device_context = nullptr;

            //Create device
            UINT creationFlags = 0;//D3D11_CREATE_DEVICE_BGRA_SUPPORT;

#if !defined(NDEBUG) || defined(CV_STATIC_ANALYSIS)
            // If the project is in a debug build, enable debugging via SDK Layers with this flag.
            creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

            D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_1,
                                                  D3D_FEATURE_LEVEL_11_0,
                                                };
            D3D_FEATURE_LEVEL featureLevel;

            auto adapter_factory = createCOMPtrGuard<IDXGIFactory>();
            {
                IDXGIFactory* out_factory = nullptr;
                HRESULT err = CreateDXGIFactory(__uuidof(IDXGIFactory),
                                                reinterpret_cast<void**>(&out_factory));
                if (FAILED(err)) {
                    throw std::runtime_error("Cannot create CreateDXGIFactory, error: " + std::to_string(HRESULT_CODE(err)));
                }
                adapter_factory = createCOMPtrGuard(out_factory);
            }

            auto intel_adapter = createCOMPtrGuard<IDXGIAdapter>();
            UINT adapter_index = 0;
            const unsigned int refIntelVendorID = 0x8086;
            IDXGIAdapter* out_adapter = nullptr;

            while (adapter_factory->EnumAdapters(adapter_index, &out_adapter) != DXGI_ERROR_NOT_FOUND) {
                DXGI_ADAPTER_DESC desc{};
                out_adapter->GetDesc(&desc);
                if (desc.VendorId == refIntelVendorID) {
                    intel_adapter = createCOMPtrGuard(out_adapter);
                    break;
                }
                ++adapter_index;
            }

            if (!intel_adapter) {
                throw std::runtime_error("No Intel GPU adapter on aboard");
            }

            // Create the Direct3D 11 API device object and a corresponding context.
            HRESULT err = D3D11CreateDevice(intel_adapter.get(),
                                            D3D_DRIVER_TYPE_UNKNOWN,
                                            nullptr, creationFlags,
                                            featureLevels, ARRAYSIZE(featureLevels),
                                            D3D11_SDK_VERSION,
                                            &hw_handle, &featureLevel,
                                            &device_context);
            if(FAILED(err)) {
                throw std::logic_error("Cannot create D3D11CreateDevice, error: " + std::to_string(HRESULT_CODE(err)));
            }

            // oneVPL recommendation
            {
                ID3D11Multithread *pD11Multithread = nullptr;
                device_context->QueryInterface(IID_PPV_ARGS(&pD11Multithread));
                pD11Multithread->SetMultithreadProtected(true);
                pD11Multithread->Release();
            }

            suggested_device = IDeviceSelector::create<Device>(hw_handle, "GPU", AccelType::DX11);
            suggested_context = IDeviceSelector::create<Context>(device_context, AccelType::DX11);
#else  // defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
            GAPI_LOG_WARNING(nullptr, "Unavailable \"" <<  CfgParam::acceleration_mode_name() << ": MFX_ACCEL_MODE_VIA_D3D11\""
                                      "was chosen for current project configuration");
            throw std::logic_error(std::string("Unsupported \"") +
                                   CfgParam::acceleration_mode_name() +
                                   ": MFX_ACCEL_MODE_VIA_D3D11\"");
#endif // defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
            break;
        }
        case MFX_IMPL_VIA_VAAPI : {
#ifdef __linux__
#if defined(HAVE_VA) || defined(HAVE_VA_INTEL)
            static const char *predefined_vaapi_devices_list[] {"/dev/dri/renderD128",
                                                                "/dev/dri/renderD129",
                                                                "/dev/dri/card0",
                                                                "/dev/dri/card1",
                                                                nullptr};
            std::stringstream ss;
            int device_fd = -1;
            VADisplay va_handle;va_handle = nullptr;
            for (const char **device_path = predefined_vaapi_devices_list;
                *device_path != nullptr; device_path++) {
                device_fd = open(*device_path, O_RDWR);
                if (device_fd < 0) {
                    std::string info("Cannot open GPU file: \"");
                    info = info + *device_path + "\", error: " + strerror(errno);
                    GAPI_LOG_DEBUG(nullptr, info);
                    ss << info << std::endl;
                    continue;
                }
                va_handle = vaGetDisplayDRM(device_fd);
                if (!va_handle) {
                    close(device_fd);

                    std::string info("VAAPI device vaGetDisplayDRM failed, error: ");
                    info += strerror(errno);
                    GAPI_LOG_DEBUG(nullptr, info);
                    ss << info << std::endl;
                    continue;
                }
                int major_version = 0, minor_version = 0;
                VAStatus status {};
                status = vaInitialize(va_handle, &major_version, &minor_version);
                if (VA_STATUS_SUCCESS != status) {
                    close(device_fd);
                    va_handle = nullptr;

                    std::string info("Cannot initialize VAAPI device, error: ");
                    info += vaErrorStr(status);
                    GAPI_LOG_DEBUG(nullptr, info);
                    ss << info << std::endl;
                    continue;
                }
                GAPI_LOG_INFO(nullptr, "VAAPI created for device: " << *device_path);
                break;
            }

            // check device creation
            if (!va_handle) {
                GAPI_LOG_WARNING(nullptr, "Cannot create VAAPI device. Log:\n" << ss.str());
                throw std::logic_error(std::string("Cannot create device for \"") +
                                   CfgParam::acceleration_mode_name() +
                                   ": MFX_IMPL_VIA_VAAPI\"");
            }

            // Unfortunately VAAPI doesn't provide API for extracting initial FD value from VADisplay, which
            // value is stored as VADisplay fields, by the way. But, because we here are only one creator
            // of VAAPI device then we will need make cleanup for all allocated resources by ourselfs
            //and FD is definitely must be utilized. So, let's use complementary struct `PlatformSpecificParams` which
            // represent some kind of 'platform specific data' and which will store opened FD for
            // future utilization
            platform_specific_data.reset (new PlatformSpecificParams);
            platform_specific_data->track_fd(device_fd);

            suggested_device = IDeviceSelector::create<Device>(va_handle, "GPU", AccelType::VAAPI);
            suggested_context = IDeviceSelector::create<Context>(nullptr, AccelType::VAAPI);
#else  // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
            GAPI_Error("VPLVAAPIAccelerationPolicy unavailable in current linux configuration");
#endif // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#else // #ifdef __linux__
            GAPI_Error("MFX_IMPL_VIA_VAAPI is supported on linux only");
#endif // #ifdef __linux__
            break;
        }
        case MFX_ACCEL_MODE_NA: {
            // nothing to do
            break;
        }
        default:
            throw std::logic_error(std::string("Unsupported \"") +
                                   CfgParam::acceleration_mode_name() +"\" requested: " +
                                   std::to_string(accel_mode.Data.U32));
            break;
    }
}

CfgParamDeviceSelector::CfgParamDeviceSelector(Device::Ptr device_ptr,
                                               const std::string& device_id,
                                               Context::Ptr ctx_ptr,
                                               const CfgParams& cfg_params) :
    suggested_device(IDeviceSelector::create<Device>(nullptr, "CPU", AccelType::HOST)),
    suggested_context(IDeviceSelector::create<Context>(nullptr, AccelType::HOST)) {
    auto accel_mode_it =
        std::find_if(cfg_params.begin(), cfg_params.end(), [] (const CfgParam& value) {
            return value.get_name() == CfgParam::acceleration_mode_name();
        });
    if (accel_mode_it == cfg_params.end()) {
        GAPI_LOG_WARNING(nullptr, "Cannot deternime \"device_ptr\" type. "
                         "Make sure a param \"" << CfgParam::acceleration_mode_name() << "\" "
                         "presents in configurations and has correct value according to "
                         "\"device_ptr\" type");
        throw std::logic_error(std::string("Missing \"") +
                               CfgParam::acceleration_mode_name() +
                               "\" param");
    }

    GAPI_LOG_DEBUG(nullptr, "Turn on HW acceleration support for device: " <<
                            device_ptr <<
                            ", context: " << ctx_ptr);
    if (!device_ptr) {
        GAPI_LOG_WARNING(nullptr, "Empty \"device_ptr\" is not allowed when "
                         "param \"" <<  CfgParam::acceleration_mode_name() << "\" existed");
        throw std::logic_error("Invalid param: \"device_ptr\"");
    }

     if (!ctx_ptr) {
        GAPI_LOG_WARNING(nullptr, "Empty \"ctx_ptr\" is not allowed");
        throw std::logic_error("Invalid  param: \"ctx_ptr\"");
    }
    mfxVariant accel_mode = cfg_param_to_mfx_variant(*accel_mode_it);

    cv::util::suppress_unused_warning(device_id);
    switch(accel_mode.Data.U32) {
        case MFX_ACCEL_MODE_VIA_D3D11: {
#if defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
            suggested_device = IDeviceSelector::create<Device>(device_ptr, device_id, AccelType::DX11);
            ID3D11Device* dx_device_ptr =
                reinterpret_cast<ID3D11Device*>(suggested_device.get_ptr());
            dx_device_ptr->AddRef();

            suggested_context = IDeviceSelector::create<Context>(ctx_ptr, AccelType::DX11);
            ID3D11DeviceContext* dx_ctx_ptr =
                reinterpret_cast<ID3D11DeviceContext*>(suggested_context.get_ptr());

            // oneVPL recommendation
            {
                ID3D11Multithread *pD11Multithread = nullptr;
                dx_ctx_ptr->QueryInterface(IID_PPV_ARGS(&pD11Multithread));
                pD11Multithread->SetMultithreadProtected(true);
                pD11Multithread->Release();
            }

            dx_ctx_ptr->AddRef();
#else  // defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
            GAPI_LOG_WARNING(nullptr, "Unavailable \"" <<  CfgParam::acceleration_mode_name() <<
                                      ": MFX_ACCEL_MODE_VIA_D3D11\""
                                      "was chosen for current project configuration");
            throw std::logic_error(std::string("Unsupported \"") +
                                   CfgParam::acceleration_mode_name() + ": MFX_ACCEL_MODE_VIA_D3D11\"");
#endif // #if defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
            break;
        }
        case MFX_IMPL_VIA_VAAPI : {
#ifdef __linux__
#if defined(HAVE_VA) || defined(HAVE_VA_INTEL)
            suggested_device = IDeviceSelector::create<Device>(device_ptr, device_id, AccelType::VAAPI);
            suggested_context = IDeviceSelector::create<Context>(nullptr, AccelType::VAAPI);
#else  // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
            GAPI_Error("VPLVAAPIAccelerationPolicy unavailable in current linux configuration");
#endif // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#else // #ifdef __linux__
            GAPI_Error("MFX_IMPL_VIA_VAAPI is supported on linux only");
#endif // #ifdef __linux__
            break;
        }
        case MFX_ACCEL_MODE_NA: {
            GAPI_LOG_WARNING(nullptr, "Incompatible \"" <<  CfgParam::acceleration_mode_name() <<
                                      ": MFX_ACCEL_MODE_NA\" with "
                                      "\"device_ptr\" and \"ctx_ptr\" arguments. "
                                      "You should not clarify these arguments with \"MFX_ACCEL_MODE_NA\" mode");
            throw std::logic_error("Incompatible param: MFX_ACCEL_MODE_NA");
        }
        default:
            throw std::logic_error(std::string("Unsupported \"") +  CfgParam::acceleration_mode_name() +
                                   "\" requested: " +
                                   std::to_string(accel_mode.Data.U32));
            break;
    }
}

CfgParamDeviceSelector::CfgParamDeviceSelector(const Device &device,
                                               const Context &ctx,
                                               CfgParams) :
    suggested_device(device),
    suggested_context(ctx) {

    switch(device.get_type()) {
        case AccelType::DX11: {
#if defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
            ID3D11Device* dx_device_ptr =
                reinterpret_cast<ID3D11Device*>(suggested_device.get_ptr());
            dx_device_ptr->AddRef();

            ID3D11DeviceContext* dx_ctx_ptr =
                reinterpret_cast<ID3D11DeviceContext*>(suggested_context.get_ptr());

            // oneVPL recommendation
            {
                ID3D11Multithread *pD11Multithread = nullptr;
                dx_ctx_ptr->QueryInterface(IID_PPV_ARGS(&pD11Multithread));
                pD11Multithread->SetMultithreadProtected(true);
                pD11Multithread->Release();
            }

            dx_ctx_ptr->AddRef();
            break;
#else // defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
            GAPI_LOG_WARNING(nullptr, "Unavailable \"" <<  CfgParam::acceleration_mode_name() <<
                                      ": MFX_ACCEL_MODE_VIA_D3D11\""
                                      "was chosen for current project configuration");
            throw std::logic_error(std::string("Unsupported \"") +
                                   CfgParam::acceleration_mode_name() + ": MFX_ACCEL_MODE_VIA_D3D11\"");
#endif // defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
        }
        case AccelType::VAAPI:
#ifdef __linux__
#if !defined(HAVE_VA) || !defined(HAVE_VA_INTEL)
            GAPI_Error("VPLVAAPIAccelerationPolicy unavailable in current linux configuration");
#endif // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#else // #ifdef __linux__
            GAPI_Error("MFX_IMPL_VIA_VAAPI is supported on linux only");
#endif // #ifdef __linux__
            break;
        case AccelType::HOST:
            break;
        default:
            throw std::logic_error(std::string("Unsupported \"") +  CfgParam::acceleration_mode_name() +
                                   "\" requested: " +
                                   to_cstring(device.get_type()));
            break;
    }
}

CfgParamDeviceSelector::~CfgParamDeviceSelector() {
    GAPI_LOG_INFO(nullptr, "release context: " << suggested_context.get_ptr());
    AccelType ctype = suggested_context.get_type();
    switch(ctype) {
        case AccelType::HOST:
            //nothing to do
            break;
        case AccelType::DX11: {
#if defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
            ID3D11DeviceContext* device_ctx_ptr =
                reinterpret_cast<ID3D11DeviceContext*>(suggested_context.get_ptr());
            device_ctx_ptr->Release();
            device_ctx_ptr = nullptr;
#endif // defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
            break;
        }
        default:
            break;
    }

    GAPI_LOG_INFO(nullptr, "release device by name: " <<
                           suggested_device.get_name() <<
                           ", ptr: " << suggested_device.get_ptr());
    AccelType dtype = suggested_device.get_type();
    switch(dtype) {
        case AccelType::HOST:
            //nothing to do
            break;
        case AccelType::DX11: {
#if defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
            ID3D11Device* device_ptr = reinterpret_cast<ID3D11Device*>(suggested_device.get_ptr());
            device_ptr->Release();
            device_ptr = nullptr;
#endif // defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
            break;
        }
        case AccelType::VAAPI: {
#ifdef __linux__
#if defined(HAVE_VA) || defined(HAVE_VA_INTEL)
            VADisplay va_handle = reinterpret_cast<VADisplay>(suggested_device.get_ptr());
            vaTerminate(va_handle);
            platform_specific_data.reset();
#endif // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#endif // #ifdef __linux__
        }
        default:
            break;
    }
}

CfgParamDeviceSelector::DeviceScoreTable CfgParamDeviceSelector::select_devices() const {
    return {std::make_pair(Score::Type(Score::MaxActivePriority), suggested_device)};
}

CfgParamDeviceSelector::DeviceContexts CfgParamDeviceSelector::select_context() {
    return {suggested_context};
}

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
