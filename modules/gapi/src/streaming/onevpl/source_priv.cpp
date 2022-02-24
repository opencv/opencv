// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <algorithm>
#include <sstream>

#include "streaming/onevpl/engine/decode/decode_engine_legacy.hpp"
#include "streaming/onevpl/engine/transcode/transcode_engine_legacy.hpp"
#include "streaming/onevpl/accelerators/accel_policy_dx11.hpp"
#include "streaming/onevpl/accelerators/accel_policy_cpu.hpp"
#include "streaming/onevpl/utils.hpp"
#include "streaming/onevpl/cfg_params_parser.hpp"
#include "streaming/onevpl/data_provider_defines.hpp"

#include "streaming/onevpl/source_priv.hpp"
#include "logger.hpp"

#ifndef HAVE_ONEVPL
namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
bool GSource::Priv::pull(cv::gapi::wip::Data&) {
    return true;
}
GMetaArg GSource::Priv::descr_of() const {
    return {};
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#else // HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

enum {
    VPL_NEW_API_MAJOR_VERSION = 2,
    VPL_NEW_API_MINOR_VERSION = 2
};

GSource::Priv::Priv() :
    mfx_handle(MFXLoad()),
    mfx_impl_description(),
    mfx_handle_configs(),
    cfg_params(),
    mfx_session(),
    description(),
    description_is_valid(false),
    engine(),
    consumed_frames_count()
{
    GAPI_LOG_INFO(nullptr, "Initialized MFX handle: " << mfx_handle);
}

GSource::Priv::Priv(std::shared_ptr<IDataProvider> provider,
                    const std::vector<CfgParam>& params,
                    std::shared_ptr<IDeviceSelector> device_selector) :
     GSource::Priv()
{
    // Enable Config
    if (params.empty())
    {
        GAPI_LOG_INFO(nullptr, "No special cfg params requested - use default");
        this->cfg_params = getDefaultCfgParams();
    }
    else
    {
         this->cfg_params = params;
    }

    GAPI_LOG_DEBUG(nullptr, "Requested cfg params count: " << cfg_params.size());
    this->mfx_handle_configs.resize(cfg_params.size());

    // Build VPL handle config from major input params
    // VPL dispatcher then uses this config handle to look up for all existing VPL impl
    // satisfying major input params and available in the system
    GAPI_LOG_INFO(nullptr, "Creating VPL config from input params");
    auto cfg_param_it = cfg_params.begin();
    for (mfxConfig& cfg_inst : mfx_handle_configs) {
         cfg_inst = MFXCreateConfig(mfx_handle);
         GAPI_Assert(cfg_inst && "MFXCreateConfig failed");

        if (!cfg_param_it->is_major()) {
            GAPI_LOG_DEBUG(nullptr, "Skip not major param: " << cfg_param_it->get_name());
            ++cfg_param_it;
            continue;
        }

        GAPI_LOG_DEBUG(nullptr, "Apply major param: " << cfg_param_it->get_name());
        mfxVariant mfx_param = cfg_param_to_mfx_variant(*cfg_param_it);
        mfxStatus sts = MFXSetConfigFilterProperty(cfg_inst,
                                                   (mfxU8 *)cfg_param_it->get_name().c_str(),
                                                   mfx_param);
        if (sts != MFX_ERR_NONE )
        {
            GAPI_LOG_WARNING(nullptr, "MFXSetConfigFilterProperty failed, error: " <<
                                      mfxstatus_to_string(sts) <<
                                      " - for \"" << cfg_param_it->get_name() << "\"");
            GAPI_Assert(false && "MFXSetConfigFilterProperty failed");
        }

        mfx_param.Type     = MFX_VARIANT_TYPE_U32;
        mfx_param.Data.U32 = MFX_EXTBUFF_VPP_SCALING;
        sts = MFXSetConfigFilterProperty(cfg_inst,
        (mfxU8 *)"mfxImplDescription.mfxVPPDescription.filter.FilterFourCC",
        mfx_param);

        if (sts != MFX_ERR_NONE )
        {
            GAPI_LOG_WARNING(nullptr, "MFXSetConfigFilterProperty failed, error: " <<
                                      mfxstatus_to_string(sts) <<
                                      " - for \"mfxImplDescription.mfxVPPDescription.filter.FilterFourCC\"");
            GAPI_Assert(false && "MFXSetConfigFilterProperty failed");
        }

        ++cfg_param_it;
    }

    // collect optional-preferred input parameters from input params
    // which may (optionally) or may not be used to choose the most preferrable
    // VPL implementation (for example, specific API version or Debug/Release VPL build)
    std::vector<CfgParam> preferred_params;
    std::copy_if(cfg_params.begin(), cfg_params.end(), std::back_inserter(preferred_params),
                 [] (const CfgParam& param) { return !param.is_major(); });
    std::sort(preferred_params.begin(), preferred_params.end());

    GAPI_LOG_DEBUG(nullptr, "Find MFX better implementation from handle: " << mfx_handle <<
                            " is satisfying preferrable params count: " << preferred_params.size());
    int i = 0;
    mfxImplDescription *idesc = nullptr;
    std::vector<mfxImplDescription*> available_impl_descriptions;
    std::map<size_t/*matches count*/, int /*impl index*/> matches_count;
    while (MFX_ERR_NONE == MFXEnumImplementations(mfx_handle,
                                                  i,
                                                  MFX_IMPLCAPS_IMPLDESCSTRUCTURE,
                                                  reinterpret_cast<mfxHDL *>(&idesc))) {

        available_impl_descriptions.push_back(idesc);

        std::stringstream ss;
        mfxHDL hImplPath = nullptr;
        if (MFX_ERR_NONE == MFXEnumImplementations(mfx_handle, i, MFX_IMPLCAPS_IMPLPATH, &hImplPath)) {
            if (hImplPath) {
                ss << "Implementation path: "  <<  reinterpret_cast<mfxChar *>(hImplPath) << std::endl;
                MFXDispReleaseImplDescription(mfx_handle, hImplPath);
            }
        }
        ss << *idesc << std::endl;

        GAPI_LOG_INFO(nullptr, "Implementation index: " << i << "\n" << ss.str());

        // Only one VPL implementation is required for GSource here.
        // Let's find intersection params from available impl with preferrable input params
        // to find best match.
        // An available VPL implementation with max matching count
        std::vector<CfgParam> impl_params = get_params_from_string<CfgParam>(ss.str());
        std::sort(impl_params.begin(), impl_params.end());
        GAPI_LOG_DEBUG(nullptr, "Find implementation cfg params count: " << impl_params.size());

        std::vector<CfgParam> matched_params;
        std::set_intersection(impl_params.begin(), impl_params.end(),
                                  preferred_params.begin(), preferred_params.end(),
                                  std::back_inserter(matched_params));

        if (preferred_params.empty()) {
            // in case of no input preferrance we consider all params are matched
            // for the first available VPL implementation. It will be a chosen one
            matches_count.emplace(impl_params.size(), i++);
            GAPI_LOG_DEBUG(nullptr, "No preferrable params, use the first one implementation");
            break;
        } else {
            GAPI_LOG_DEBUG(nullptr, "Equal param intersection count: " << matched_params.size());
            matches_count.emplace(matches_count.size(), i++);
        }
    }

    // Extract the most suitable VPL implementation by max score
    auto max_match_it = matches_count.rbegin();
    GAPI_Assert(max_match_it != matches_count.rend() &&
                "Cannot find matched MFX implementation for requested configuration");

    int impl_number = max_match_it->second;
    GAPI_LOG_INFO(nullptr, "Chosen implementation index: " << impl_number);

    // release unusable impl available_impl_descriptions
    std::swap(mfx_impl_description, available_impl_descriptions[impl_number]);
    for (mfxImplDescription* unusable_impl_descr : available_impl_descriptions) {
        if (unusable_impl_descr) {
            MFXDispReleaseImplDescription(mfx_handle, unusable_impl_descr);
        }
    }
    available_impl_descriptions.clear();

    // create session for implementation
    mfxStatus sts = MFXCreateSession(mfx_handle, impl_number, &mfx_session);
    if (MFX_ERR_NONE != sts) {
        GAPI_LOG_WARNING(nullptr, "Cannot create MFX Session for implementation index:" <<
                                   std::to_string(impl_number) <<
                                   ", error: " << mfxstatus_to_string(sts));
    }

    GAPI_LOG_INFO(nullptr, "Initialized MFX session: " << mfx_session);

    // create session driving engine if required
    if (!engine) {
        std::unique_ptr<VPLAccelerationPolicy> acceleration = initializeHWAccel(device_selector);

        // TODO  Add factory static method in ProcessingEngineBase
        if (mfx_impl_description->ApiVersion.Major >= VPL_NEW_API_MAJOR_VERSION) {
            GAPI_Assert(false &&
                        "GSource mfx_impl_description->ApiVersion.Major >= VPL_NEW_API_MAJOR_VERSION"
                        " - is not implemented");
        } else {
            const auto& transcode_params = VPLLegacyTranscodeEngine::get_vpp_params(preferred_params);
            if (!transcode_params.empty()) {
                engine.reset(new VPLLegacyTranscodeEngine(std::move(acceleration)));
            } else {
                engine.reset(new VPLLegacyDecodeEngine(std::move(acceleration)));
            }
        }
    }

    // create engine session for processing mfx session pipeline
    auto engine_session_ptr = engine->initialize_session(mfx_session, cfg_params,
                                                         provider);

    const mfxFrameInfo& video_param = engine_session_ptr->get_video_param();

    // set valid description
    description.size = cv::Size {
                            video_param.Width,
                            video_param.Height};
    switch(video_param.FourCC) {
        case MFX_FOURCC_I420:
            throw std::runtime_error("Cannot parse GMetaArg description: MediaFrame doesn't support I420 type");
        case MFX_FOURCC_NV12:
            description.fmt = cv::MediaFormat::NV12;
            break;
        default:
            throw std::runtime_error("Cannot parse GMetaArg description: MediaFrame unknown 'fmt' type: " +
                                     std::to_string(video_param.FourCC));
    }
    description_is_valid = true;

    //prepare session for processing
    engine->process(mfx_session);
}

GSource::Priv::~Priv() {
    engine.reset();

    GAPI_LOG_INFO(nullptr, "consumed frames count: " << consumed_frames_count);
    GAPI_LOG_INFO(nullptr, "Unload MFX implementation description: " << mfx_impl_description);
    MFXDispReleaseImplDescription(mfx_handle, mfx_impl_description);
    GAPI_LOG_INFO(nullptr, "Unload MFX handle: " << mfx_handle);
    MFXUnload(mfx_handle);
}

std::unique_ptr<VPLAccelerationPolicy> GSource::Priv::initializeHWAccel(std::shared_ptr<IDeviceSelector> selector)
{
    std::unique_ptr<VPLAccelerationPolicy> ret;

    auto accel_mode_it = std::find_if(cfg_params.begin(), cfg_params.end(), [] (const CfgParam& value) {
        return value.get_name() ==  CfgParam::acceleration_mode_name();
    });
    if (accel_mode_it == cfg_params.end())
    {
        GAPI_LOG_DEBUG(nullptr, "No HW Accel requested. Use CPU");

        ret.reset(new VPLCPUAccelerationPolicy(selector));
        return ret;
    }

    GAPI_LOG_DEBUG(nullptr, "Add HW acceleration support");
    mfxVariant accel_mode = cfg_param_to_mfx_variant(*accel_mode_it);

    switch(accel_mode.Data.U32) {
        case MFX_ACCEL_MODE_VIA_D3D11:
        {
            std::unique_ptr<VPLDX11AccelerationPolicy> cand(new VPLDX11AccelerationPolicy(selector));
            ret = std::move(cand);
            break;
        }
        case MFX_ACCEL_MODE_NA:
        {
            std::unique_ptr<VPLCPUAccelerationPolicy> cand(new VPLCPUAccelerationPolicy(selector));
            ret = std::move(cand);
            break;
        }
        default:
        {
            GAPI_LOG_WARNING(nullptr, "Cannot initialize HW Accel: "
                                      "invalid accelerator type: " <<
                                      std::to_string(accel_mode.Data.U32));
            GAPI_Assert(false && "Cannot initialize HW Accel");
        }
    }

    return ret;
}

const std::vector<CfgParam>& GSource::Priv::getDefaultCfgParams()
{
    static const std::vector<CfgParam> def_params =
        get_params_from_string<CfgParam>(
                    "mfxImplDescription.Impl: MFX_IMPL_TYPE_HARDWARE\n"
                    "mfxImplDescription.AccelerationMode: MFX_ACCEL_MODE_VIA_D3D11\n");

    return def_params;
}

const std::vector<CfgParam>& GSource::Priv::getCfgParams() const
{
    return cfg_params;
}

bool GSource::Priv::pull(cv::gapi::wip::Data& data)
{
    ProcessingEngineBase::ExecutionStatus status = ProcessingEngineBase::ExecutionStatus::Continue;
    while (0 == engine->get_ready_frames_count() &&
           status == ProcessingEngineBase::ExecutionStatus::Continue) {
        status = engine->process(mfx_session);
    }

    if (engine->get_ready_frames_count()) {
        engine->get_frame(data);
        consumed_frames_count++;
        return true;
    } else {
        return false;
    }
}

GMetaArg GSource::Priv::descr_of() const
{
    GAPI_Assert(description_is_valid);
    GMetaArg arg(description);
    return arg;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
