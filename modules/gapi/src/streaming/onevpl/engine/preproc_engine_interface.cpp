// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#include <opencv2/gapi/streaming/onevpl/device_selector_interface.hpp>
#include "streaming/onevpl/engine/preproc_engine_interface.hpp"
#include "streaming/onevpl/engine/preproc/preproc_dispatcher.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"
#include "streaming/onevpl/engine/preproc/preproc_engine.hpp"

#include "streaming/onevpl/accelerators/accel_policy_dx11.hpp"
#include "streaming/onevpl/accelerators/accel_policy_cpu.hpp"
#include "streaming/onevpl/accelerators/accel_policy_va_api.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/cfg_param_device_selector.hpp"
#include "streaming/onevpl/cfg_params_parser.hpp"

#endif //HAVE_ONEVPL

#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {

template<typename SpecificPreprocEngine, typename ...PreprocEngineArgs >
std::unique_ptr<SpecificPreprocEngine>
IPreprocEngine::create_preproc_engine_impl(const PreprocEngineArgs& ...) {
    GAPI_Error("Unsupported ");
}

template <>
std::unique_ptr<onevpl::VPPPreprocDispatcher>
IPreprocEngine::create_preproc_engine_impl(const onevpl::Device &device,
                                           const onevpl::Context &context) {
    using namespace onevpl;
    cv::util::suppress_unused_warning(device);
    cv::util::suppress_unused_warning(context);
    std::unique_ptr<VPPPreprocDispatcher> dispatcher(new VPPPreprocDispatcher);
#ifdef HAVE_ONEVPL
    bool pp_is_created = false;
    switch (device.get_type()) {
        case onevpl::AccelType::DX11: {
            GAPI_LOG_INFO(nullptr, "Creating DX11 VPP preprocessing engine");
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
            // create GPU VPP preproc engine
            dispatcher->insert_worker<VPPPreprocEngine>(
                                std::unique_ptr<VPLAccelerationPolicy>{
                                        new VPLDX11AccelerationPolicy(
                                            std::make_shared<CfgParamDeviceSelector>(
                                                    device, context, CfgParams{}))
                                });
            GAPI_LOG_INFO(nullptr, "DX11 VPP preprocessing engine created");
            pp_is_created = true;
#endif
#endif
            break;
        }
        case onevpl::AccelType::VAAPI: {
            GAPI_LOG_INFO(nullptr, "Creating VAAPI VPP preprocessing engine");
#ifdef __linux__
#if defined(HAVE_VA) || defined(HAVE_VA_INTEL)
            // create GPU VPP preproc engine
            dispatcher->insert_worker<VPPPreprocEngine>(
                                std::unique_ptr<VPLAccelerationPolicy>{
                                        new VPLVAAPIAccelerationPolicy(
                                            std::make_shared<CfgParamDeviceSelector>(
                                                    device, context, CfgParams{}))
                                });
            GAPI_LOG_INFO(nullptr, "VAAPI VPP preprocessing engine created");
            pp_is_created = true;
#endif // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#endif // #ifdef __linux__
            break;
        }
        default: {
            GAPI_LOG_INFO(nullptr, "Creating CPU VPP preprocessing engine");
            dispatcher->insert_worker<VPPPreprocEngine>(
                        std::unique_ptr<VPLAccelerationPolicy>{
                                new VPLCPUAccelerationPolicy(
                                    std::make_shared<CfgParamDeviceSelector>(CfgParams{}))});
            GAPI_LOG_INFO(nullptr, "CPU VPP preprocessing engine created");
            pp_is_created = true;
            break;
        }
    }
    if (!pp_is_created) {
        GAPI_LOG_WARNING(nullptr, "Cannot create VPP preprocessing engine: configuration unsupported");
        GAPI_Error("VPP preproc unsupported");
    }
#endif // HAVE_ONEVPL
    return dispatcher;
}


// Force instantiation
template
std::unique_ptr<onevpl::VPPPreprocDispatcher>
IPreprocEngine::create_preproc_engine_impl<onevpl::VPPPreprocDispatcher,
                                           const onevpl::Device &,const onevpl::Context &>
                                          (const onevpl::Device &device,
                                           const onevpl::Context &ctx);
} // namespace wip
} // namespace gapi
} // namespace cv
