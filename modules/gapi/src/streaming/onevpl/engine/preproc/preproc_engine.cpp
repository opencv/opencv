// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL

#include <algorithm>
#include <exception>

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include "streaming/onevpl/data_provider_defines.hpp"

#include "streaming/onevpl/engine/preproc/preproc_engine.hpp"
#include "streaming/onevpl/engine/preproc/preproc_session.hpp"
//#include "streaming/onevpl/engine/preproc/utils.hpp"

#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/cfg_params_parser.hpp"
#include "streaming/onevpl/utils.hpp"
#include "logger.hpp"

#ifdef HAVE_INF_ENGINE
#include <inference_engine.hpp>

#define ALIGN16(value)           (((value + 15) >> 4) << 4)

bool operator< (const mfxFrameInfo &lhs, const mfxFrameInfo &rhs) {
    return memcmp(&lhs, &rhs, sizeof(mfxFrameInfo)) < 0;
}

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

VPPPreprocEngine::VPPPreprocEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel) :
    ProcessingEngineBase(std::move(accel)) {
}

cv::util::optional<PreprocParams> VPPPreprocEngine::is_applicable(const cv::MediaFrame& in_frame) {
    // TODO consider something smarter
    cv::util::optional<PreprocParams> ret;
    BaseFrameAdapter *vpl_adapter = in_frame.get<BaseFrameAdapter>();
    if (vpl_adapter) {
        ret = cv::util::make_optional<PreprocParams>({vpl_adapter->get_session_handle(),
                                                      vpl_adapter->get_surface()->get_info()});
    }
    return ret;
}
/*
mfxFrameInfo VPPPreprocEngine::to_mfxFrameInfo(const cv::GFrameDesc& frame_info) {
    mfxFrameInfo ret {0};
    ret.FourCC        = utils::MediaFormat_to_fourcc(frame_info.fmt);
    ret.ChromaFormat  = utils::MediaFormat_to_chroma(frame_info.fmt);
    ret.Width         = frame_info.size.width;
    ret.Height        = frame_info.size.height;
    ret.CropX         = 0;
    ret.CropY         = 0;
    ret.CropW         = 0;
    ret.CropH         = 0;
    ret.PicStruct     = MFX_PICSTRUCT_UNKNOWN;
    ret.FrameRateExtN = 0;
    ret.FrameRateExtD = 0;
    return ret;
}
*/
std::shared_ptr<PreprocSession>
VPPPreprocEngine::initialize_preproc(const PreprocParams& params,
                                     const InferenceEngine::InputInfo::CPtr& net_input) {
    // adjust preprocessing settings
    mfxVideoParam mfxVPPParams{0};
    // NB: IN params for VPP session must be equal to decoded surface params
    mfxVPPParams.vpp.In = params.info;

    // NB: OUT params must refer to IN params of a network
    const InferenceEngine::SizeVector& inDims = net_input->getTensorDesc().getDims();
    mfxVPPParams.vpp.Out = mfxVPPParams.vpp.In;
    mfxVPPParams.vpp.Out.FourCC        = MFX_FOURCC_NV12;
    mfxVPPParams.vpp.Out.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    mfxVPPParams.vpp.Out.Width         = static_cast<mfxU16>(ALIGN16(inDims[2]));
    mfxVPPParams.vpp.Out.Height        = static_cast<mfxU16>(ALIGN16(inDims[3]));
    mfxVPPParams.vpp.Out.CropW         = mfxVPPParams.vpp.Out.Width;
    mfxVPPParams.vpp.Out.CropH         = mfxVPPParams.vpp.Out.Height;

    // find existing session
    GAPI_LOG_DEBUG(nullptr, "Find existing PreprocSession for requested frame params"
                            ", total sessions: " << preproc_session_map.size());
    auto it = preproc_session_map.find(mfxVPPParams.vpp.In);
    if (it != preproc_session_map.end()) {
        GAPI_LOG_DEBUG(nullptr, "[" << it->second->session << "] found");
        return it->second;
    }

    // NB: make some sanity checks
    IDeviceSelector::DeviceScoreTable devices = acceleration_policy->get_device_selector()->select_devices();
    GAPI_Assert(devices.size() == 1 && "Multiple(or zero) acceleration devices case is unsupported");
    AccelType accel_type = devices.begin()->second.get_type();
    // assign acceleration
    if (accel_type == AccelType::DX11) {
        mfxVPPParams.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    } else {
        mfxVPPParams.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    }

    // clone existing VPL session to inherit VPL loader configuration
    // and avoid refer to any global state
    mfxSession mfx_vpp_session{};
    mfxStatus sts = MFXCloneSession(params.handle, &mfx_vpp_session);
    if (sts != MFX_ERR_NONE) {
        GAPI_LOG_WARNING(nullptr, "Cannot clone VPP session, error: " << mfxstatus_to_string(sts));
        GAPI_Assert(false && "Cannot continue VPP preprocessing");
    }

    GAPI_LOG_INFO(nullptr, "[" << mfx_vpp_session << "] starting allocation");
    VPLAccelerationPolicy::pool_key_t vpp_out_pool_key {};
    try {
        // assing HW acceleration processor
        acceleration_policy->init(mfx_vpp_session);
        try {
            // ask to allocate external memory pool
            mfxFrameAllocRequest vppRequests[2];
            memset(&vppRequests, 0, sizeof(mfxFrameAllocRequest) * 2);
            sts = MFXVideoVPP_QueryIOSurf(mfx_vpp_session, &mfxVPPParams, vppRequests);
            if (MFX_ERR_NONE != sts) {
                GAPI_LOG_WARNING(nullptr, "cannot execute MFXVideoVPP_QueryIOSurf, error: " <<
                                          mfxstatus_to_string(sts));
                throw std::runtime_error("Cannot execute MFXVideoVPP_QueryIOSurf");
            }

            // NB: Assing ID as upper limit descendant to distinguish specific VPP allocation
            // from decode allocations witch started from 0: by local module convention

            static uint16_t request_id = 0;
            vppRequests[1].AllocId = std::numeric_limits<uint16_t>::max() - request_id++;
            GAPI_Assert(request_id != std::numeric_limits<uint16_t>::max() && "Something wrong");

            vppRequests[1].Type |= MFX_MEMTYPE_FROM_VPPIN;
            vpp_out_pool_key = acceleration_policy->create_surface_pool(vppRequests[1],
                                                                        mfxVPPParams.vpp.Out);

            sts = MFXVideoVPP_Init(mfx_vpp_session, &mfxVPPParams);
            if (MFX_ERR_NONE != sts) {
                GAPI_LOG_WARNING(nullptr, "cannot Init VPP, error: " <<
                                          mfxstatus_to_string(sts));
                // TODO consider deallocate pool
                // but not necessary now cause every fail processed as GAPI_Assert
                throw std::runtime_error("Cannot init VPP, error: " +
                                         mfxstatus_to_string(sts));
            }
        } catch (const std::exception&) {
            GAPI_LOG_WARNING(nullptr, "[" << mfx_vpp_session << "] allocation failed, rollback");
            acceleration_policy->deinit(mfx_vpp_session);
            throw;
        }
    } catch (const std::exception&) {
        MFXClose(mfx_vpp_session);
        GAPI_Assert(false && "Cannot init preproc resources");
    }

    // create engine session after all
    std::shared_ptr<PreprocSession> sess_ptr =
                        register_session<PreprocSession>(mfx_vpp_session, mfxVPPParams);
    sess_ptr->init_surface_pool(vpp_out_pool_key);
    sess_ptr->swap_surface(*this);


    bool inserted = preproc_session_map.emplace(mfxVPPParams.vpp.In, sess_ptr).second;
    GAPI_Assert(inserted && "preproc session is exist");
    GAPI_LOG_INFO(nullptr, "PreprocSession created, total sessions: " << preproc_session_map.size());
    return sess_ptr;
}

VPPPreprocEngine::session_ptr
VPPPreprocEngine::initialize_session(mfxSession,
                                     const std::vector<CfgParam>&,
                                     std::shared_ptr<IDataProvider>) {
    return {};
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // HAVE_INF_ENGINE
