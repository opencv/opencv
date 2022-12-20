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

#include "streaming/onevpl/engine/transcode/transcode_engine_legacy.hpp"
#include "streaming/onevpl/engine/transcode/transcode_session.hpp"
#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/cfg_params_parser.hpp"
#include "streaming/onevpl/utils.hpp"
#include "logger.hpp"

#define ALIGN16(value)           (((value + 15) >> 4) << 4)

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
using vpp_param_storage = const std::map<std::string, mfxVariant>;
using vpp_param_storage_cit = typename vpp_param_storage::const_iterator;

template<typename Type>
Type get_mfx_value(const vpp_param_storage_cit &cit);

template<>
uint16_t get_mfx_value<uint16_t>(const vpp_param_storage_cit& cit) {
    return cit->second.Data.U16;
}

template<>
uint32_t get_mfx_value<uint32_t>(const vpp_param_storage_cit& cit) {
    return cit->second.Data.U32;
}

template<typename Type>
bool set_vpp_param(const char* name, Type& out_vpp_param,
                   const vpp_param_storage &params_storage,
                   mfxSession session) {
    auto it = params_storage.find(name);
    if (it != params_storage.end()) {
        auto value = get_mfx_value<Type>(it);
        GAPI_LOG_INFO(nullptr, "[" << session << "] set \"" << name <<
                               "\": " << value);
        out_vpp_param = value;
        return true;
    }
    return false;
}

std::map<std::string, mfxVariant>
    VPLLegacyTranscodeEngine::get_vpp_params(const std::vector<CfgParam> &cfg_params) {
    std::map<std::string, mfxVariant> ret;
    static const char* vpp_param_prefix {"vpp."};
    for (const auto &param : cfg_params) {
        const char *param_name_cptr = param.get_name().c_str();
        if (strstr(param_name_cptr, vpp_param_prefix) == param_name_cptr) {
            ret.emplace(param.get_name(), cfg_param_to_mfx_variant(param));
        }
    }
    GAPI_LOG_INFO(nullptr, "Detected VPP params count: [" << ret.size() <<
                            "/" << cfg_params.size() << "]");
    return ret;
}

VPLLegacyTranscodeEngine::VPLLegacyTranscodeEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel)
 : VPLLegacyDecodeEngine(std::move(accel)) {

    GAPI_LOG_INFO(nullptr, "Create Legacy Transcode Engine");
    create_pipeline(
        // 1) Read File
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyTranscodeSession &my_sess = static_cast<LegacyTranscodeSession&>(sess);
            if (!my_sess.data_provider) {
                my_sess.last_status = MFX_ERR_MORE_DATA;
                return ExecutionStatus::Continue;
            }

            my_sess.last_status = MFX_ERR_NONE;
            if (!my_sess.data_provider->fetch_bitstream_data(my_sess.stream)) {
                my_sess.last_status = MFX_ERR_MORE_DATA;
                my_sess.data_provider.reset(); //close source
            }
            return ExecutionStatus::Continue;
        },
        // 2) enqueue ASYNC decode operation
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyTranscodeSession &my_sess = static_cast<LegacyTranscodeSession&>(sess);

            // prepare sync object for new surface
            LegacyTranscodeSession::op_handle_t sync_pair{};

            // enqueue decode operation with current session surface
            my_sess.last_status =
                    MFXVideoDECODE_DecodeFrameAsync(my_sess.session,
                                                    my_sess.get_mfx_bitstream_ptr(),
                                                    my_sess.processing_surface_ptr.lock()->get_handle(),
                                                    &sync_pair.second,
                                                    &sync_pair.first);

            GAPI_LOG_DEBUG(nullptr, "START decode: " <<
                                    ", sync id:  " <<
                                    sync_pair.first <<
                                    ", dec in surface:  " <<
                                    my_sess.processing_surface_ptr.lock()->get_handle() <<
                                    ", dec out surface: " << sync_pair.second <<
                                    ", status: " <<
                                    mfxstatus_to_string(my_sess.last_status));

            // process wait-like statuses in-place:
            // It had better to use up all VPL decoding resources in pipeline
            // as soon as possible. So waiting more free-surface or device free
            while (my_sess.last_status == MFX_ERR_MORE_SURFACE ||
                   my_sess.last_status == MFX_WRN_DEVICE_BUSY) {
                try {
                    if (my_sess.last_status == MFX_ERR_MORE_SURFACE) {
                        my_sess.swap_decode_surface(*this);
                    }
                    my_sess.last_status =
                    MFXVideoDECODE_DecodeFrameAsync(my_sess.session,
                                                    my_sess.get_mfx_bitstream_ptr(),
                                                    my_sess.processing_surface_ptr.lock()->get_handle(),
                                                    &sync_pair.second,
                                                    &sync_pair.first);

                } catch (const std::runtime_error& ex) {
                    // NB: not an error, yield CPU ticks to check
                    // surface availability at a next phase.
                    // But print WARNING to notify user about pipeline stuck
                    GAPI_LOG_WARNING(nullptr, "[" << my_sess.session <<
                                               "] has no surface, reason: " <<
                                               ex.what());
                    break;
                }
            }

            if (my_sess.last_status == MFX_ERR_NONE) {
                my_sess.sync_queue.emplace(sync_pair);
            } else if (my_sess.last_status != MFX_ERR_MORE_DATA) /* suppress MFX_ERR_MORE_DATA warning */ {
                GAPI_LOG_WARNING(nullptr, "decode pending ops count: " <<
                                          my_sess.sync_queue.size() <<
                                          ", sync id: " << sync_pair.first <<
                                          ", status: " <<
                                          mfxstatus_to_string(my_sess.last_status));
            }
            return ExecutionStatus::Continue;
        },
        // 3) transcode
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyTranscodeSession &my_sess = static_cast<LegacyTranscodeSession&>(sess);

            LegacyDecodeSession::op_handle_t last_op {};
            while (!my_sess.sync_queue.empty()) {
                do {
                    if (!my_sess.vpp_surface_ptr.expired()) {
                        LegacyDecodeSession::op_handle_t pending_op = my_sess.sync_queue.front();
                        GAPI_LOG_DEBUG(nullptr, "pending DEC ops count: " <<
                                                my_sess.sync_queue.size() <<
                                                ", sync id:  " <<
                                                pending_op.first <<
                                                ", surface:  " <<
                                                pending_op.second <<
                                                ", status: " <<
                                                mfxstatus_to_string(my_sess.last_status));

                        my_sess.sync_queue.pop();
                        auto *dec_surface = pending_op.second;
                        auto *vpp_suface = my_sess.vpp_surface_ptr.lock()->get_handle();
                        my_sess.last_status = MFXVideoVPP_RunFrameVPPAsync(my_sess.session,
                                                                           dec_surface,
                                                                           vpp_suface,
                                                                           nullptr, &pending_op.first);
                        pending_op.second = vpp_suface;

                        GAPI_LOG_DEBUG(nullptr, "START transcode ops count: " <<
                                                my_sess.vpp_queue.size() <<
                                                ", sync id:  " <<
                                                pending_op.first <<
                                                ", dec surface:  " <<
                                                dec_surface <<
                                                ", trans surface: " << pending_op.second <<
                                                ", status: " <<
                                                mfxstatus_to_string(my_sess.last_status));

                        if (my_sess.last_status == MFX_ERR_MORE_SURFACE ||
                            my_sess.last_status == MFX_ERR_NONE) {
                            pending_op.second->Data.Locked++; // TODO -S- workaround
                            my_sess.vpp_queue.emplace(pending_op);
                        }
                    }

                    try {
                        my_sess.swap_transcode_surface(*this);
                    } catch (const std::runtime_error& ex) {
                        // NB: not an error, yield CPU ticks to check
                        // surface availability at a next phase.
                        // But print WARNING to notify user about pipeline stuck
                        GAPI_LOG_WARNING(nullptr, "[" << my_sess.session <<
                                                    "] has no VPP surface, reason: " <<
                                                  ex.what());
                        my_sess.vpp_surface_ptr.reset();
                        break;
                    }
                } while(my_sess.last_status == MFX_ERR_MORE_SURFACE);

                if (my_sess.vpp_surface_ptr.expired()) {
                    // TODO break main loop
                    break;
                }
            }
            return ExecutionStatus::Continue;
        },
        // 4) Wait for ASYNC decode result
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyTranscodeSession& my_sess = static_cast<LegacyTranscodeSession&>(sess);
            do {
                if (!my_sess.vpp_queue.empty()) { // FIFO: check the oldest async operation complete
                    LegacyDecodeSession::op_handle_t& pending_op = my_sess.vpp_queue.front();
                    sess.last_status = MFXVideoCORE_SyncOperation(sess.session, pending_op.first, 0);

                    GAPI_LOG_DEBUG(nullptr, "pending VPP ops count: " <<
                                            my_sess.vpp_queue.size() <<
                                            ", sync id:  " <<
                                            pending_op.first <<
                                            ", surface:  " <<
                                            pending_op.second <<
                                            ", status: " <<
                                            mfxstatus_to_string(my_sess.last_status));

                    // put frames in ready queue on success
                    if (MFX_ERR_NONE == sess.last_status) {
                        on_frame_ready(my_sess, pending_op.second);
                    }
                }
            } while (MFX_ERR_NONE == sess.last_status && !my_sess.vpp_queue.empty());
            return ExecutionStatus::Continue;
        },
        // 5) Falls back on generic status processing
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            return this->process_error(sess.last_status, static_cast<LegacyDecodeSession&>(sess));
        }
    );
}

ProcessingEngineBase::session_ptr
VPLLegacyTranscodeEngine::initialize_session(mfxSession mfx_session,
                                             const std::vector<CfgParam>& cfg_params,
                                             std::shared_ptr<IDataProvider> provider) {
    // NB: obtain decoder params
    VPLLegacyDecodeEngine::SessionParam decode_params =
                        prepare_session_param(mfx_session, cfg_params, provider);


    // NB: create transcode params
    const auto& mfxDecParams = decode_params.decoder_params.param;

    // NB: create transcode params: Out = In by default, In = initially decoded
    mfxVideoParam mfxVPPParams{};
    memset(&mfxVPPParams, 0, sizeof(mfxVPPParams));
    mfxVPPParams.vpp.In = mfxDecParams.mfx.FrameInfo;
    mfxVPPParams.vpp.Out = mfxVPPParams.vpp.In;

    std::map<std::string, mfxVariant> cfg_vpp_params =
                        VPLLegacyTranscodeEngine::get_vpp_params(cfg_params);

    // override some in-params
    if (set_vpp_param(CfgParam::vpp_in_width_name(), mfxVPPParams.vpp.In.Width,
                      cfg_vpp_params, mfx_session)) {
        mfxVPPParams.vpp.In.Width = ALIGN16(mfxVPPParams.vpp.In.Width);
    }
    if (set_vpp_param(CfgParam::vpp_in_height_name(), mfxVPPParams.vpp.In.Height,
                      cfg_vpp_params, mfx_session)) {
        mfxVPPParams.vpp.In.Height = ALIGN16(mfxVPPParams.vpp.In.Height);
    }
    set_vpp_param(CfgParam::vpp_in_crop_x_name(), mfxVPPParams.vpp.In.CropX,
                  cfg_vpp_params, mfx_session);
    set_vpp_param(CfgParam::vpp_in_crop_y_name(), mfxVPPParams.vpp.In.CropY,
                  cfg_vpp_params, mfx_session);
    set_vpp_param(CfgParam::vpp_in_crop_w_name(), mfxVPPParams.vpp.In.CropW,
                  cfg_vpp_params, mfx_session);
    set_vpp_param(CfgParam::vpp_in_crop_h_name(), mfxVPPParams.vpp.In.CropH,
                  cfg_vpp_params, mfx_session);

    // override out params
    set_vpp_param(CfgParam::vpp_out_fourcc_name(), mfxVPPParams.vpp.Out.FourCC,
                  cfg_vpp_params, mfx_session);
    set_vpp_param(CfgParam::vpp_out_chroma_format_name(), mfxVPPParams.vpp.Out.ChromaFormat,
                  cfg_vpp_params, mfx_session);
    if (set_vpp_param(CfgParam::vpp_out_width_name(), mfxVPPParams.vpp.Out.Width,
                      cfg_vpp_params, mfx_session)) {
        mfxVPPParams.vpp.Out.Width = ALIGN16(mfxVPPParams.vpp.Out.Width);
    }
    if (set_vpp_param(CfgParam::vpp_out_height_name(), mfxVPPParams.vpp.Out.Height,
                      cfg_vpp_params, mfx_session)) {
        mfxVPPParams.vpp.Out.Height = ALIGN16(mfxVPPParams.vpp.Out.Height);
    }
    set_vpp_param(CfgParam::vpp_out_crop_x_name(), mfxVPPParams.vpp.Out.CropX,
                  cfg_vpp_params, mfx_session);
    set_vpp_param(CfgParam::vpp_out_crop_y_name(), mfxVPPParams.vpp.Out.CropY,
                  cfg_vpp_params, mfx_session);
    set_vpp_param(CfgParam::vpp_out_crop_w_name(), mfxVPPParams.vpp.Out.CropW,
                  cfg_vpp_params, mfx_session);
    set_vpp_param(CfgParam::vpp_out_crop_h_name(), mfxVPPParams.vpp.Out.CropH,
                  cfg_vpp_params, mfx_session);
    set_vpp_param(CfgParam::vpp_out_pic_struct_name(), mfxVPPParams.vpp.Out.PicStruct,
                  cfg_vpp_params, mfx_session);
    set_vpp_param(CfgParam::vpp_out_framerate_n_name(), mfxVPPParams.vpp.Out.FrameRateExtN,
                  cfg_vpp_params, mfx_session);
    set_vpp_param(CfgParam::vpp_out_framerate_d_name(), mfxVPPParams.vpp.Out.FrameRateExtD,
                  cfg_vpp_params, mfx_session);

    VPLLegacyTranscodeEngine::validate_vpp_param(mfxVPPParams);

    if (mfxDecParams.IOPattern == MFX_IOPATTERN_OUT_VIDEO_MEMORY) {
        mfxVPPParams.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    } else {
        mfxVPPParams.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    }
    GAPI_LOG_INFO(nullptr, "Starting VPP initialization");

    mfxFrameAllocRequest vppRequests[2];
    memset(&vppRequests, 0, sizeof(mfxFrameAllocRequest) * 2);
    mfxStatus sts = MFXVideoVPP_QueryIOSurf(mfx_session, &mfxVPPParams, vppRequests);
    if (MFX_ERR_NONE != sts) {
        GAPI_LOG_WARNING(nullptr, "cannot execute MFXVideoVPP_QueryIOSurf");
        throw std::runtime_error("Cannot execute MFXVideoVPP_QueryIOSurf, error: " +
                                  mfxstatus_to_string(sts));
    }

    // NB: override NumFrameSuggested preallocation size (how many frames we can hold)
    // if you see bunch of WARNING about "cannot get free surface from pool"
    // and have abundant RAM size then increase `CfgParam::vpp_frames_pool_size_name()`
    // to keep more free surfaces in a round. Otherwise VPL decode pipeline will be waiting
    // till application is freeing unusable surface on its side.
     cv::optional<size_t> preallocated_frames_count_cfg;
    extract_optional_param_by_name(CfgParam::vpp_frames_pool_size_name(),
                                   cfg_params,
                                   preallocated_frames_count_cfg);
    if (preallocated_frames_count_cfg.has_value()) {
        GAPI_LOG_INFO(nullptr, "Try to use CfgParam \"" << CfgParam::vpp_frames_pool_size_name() << "\": " <<
                      preallocated_frames_count_cfg.value() << ", for session: " << mfx_session);
        try_modify_pool_size_request_param(CfgParam::vpp_frames_pool_size_name(),
                                           preallocated_frames_count_cfg.value(),
                                           vppRequests[1]);

    }

    // NB: Assign ID as upper limit descendant to distinguish specific VPP allocation
    // from decode allocations witch started from 0: by local module convention
    vppRequests[1].AllocId = std::numeric_limits<uint16_t>::max();

    vppRequests[1].Type |= MFX_MEMTYPE_FROM_VPPIN;
    VPLAccelerationPolicy::pool_key_t vpp_out_pool_key =
                acceleration_policy->create_surface_pool(vppRequests[1], mfxVPPParams.vpp.Out);

    GAPI_LOG_INFO(nullptr, "Initialize VPP for session: " << mfx_session <<
                           ", out frame info: " << mfx_frame_info_to_string(mfxVPPParams.vpp.Out));
    sts = MFXVideoVPP_Init(mfx_session, &mfxVPPParams);
    if (MFX_ERR_NONE != sts) {
        GAPI_LOG_WARNING(nullptr, "cannot Init VPP");
        throw std::runtime_error("Cannot init VPP, error: " +
                                  mfxstatus_to_string(sts));
    }

    // create engine session
    TranscoderParams transcoder_param {mfxVPPParams};
    std::shared_ptr<LegacyTranscodeSession> sess_ptr =
                register_session<LegacyTranscodeSession>(mfx_session,
                                                         std::move(decode_params.decoder_params),
                                                         std::move(transcoder_param),
                                                         provider);

    sess_ptr->init_surface_pool(decode_params.decode_pool_key);
    sess_ptr->init_transcode_surface_pool(vpp_out_pool_key);

    // prepare working surfaces
    sess_ptr->swap_decode_surface(*this);
    sess_ptr->swap_transcode_surface(*this);
    return sess_ptr;
}

void VPLLegacyTranscodeEngine::validate_vpp_param(const mfxVideoParam& mfxVPPParams) {
    GAPI_LOG_INFO(nullptr, "Starting VPP param validation");
    if (mfxVPPParams.vpp.In.Width < mfxVPPParams.vpp.In.CropW + mfxVPPParams.vpp.In.CropX) {
        GAPI_LOG_WARNING(nullptr, "Invalid vonfiguration params: sum \"" <<
                                  CfgParam::vpp_in_crop_w_name() <<
                                  "\": " << mfxVPPParams.vpp.In.CropW << " and \"" <<
                                  CfgParam::vpp_in_crop_x_name() <<
                                  "\": " << mfxVPPParams.vpp.In.CropX <<
                                  " must be less or equal to \"" <<
                                  CfgParam::vpp_in_width_name() << "\": " <<
                                  mfxVPPParams.vpp.In.Width);
        GAPI_Error("Invalid VPP params combination: Width & Crop");
    }

    if (mfxVPPParams.vpp.In.Height < mfxVPPParams.vpp.In.CropH + mfxVPPParams.vpp.In.CropY) {
        GAPI_LOG_WARNING(nullptr, "Invalid vonfiguration params: sum \"" <<
                                  CfgParam::vpp_in_crop_h_name() <<
                                  "\": " << mfxVPPParams.vpp.In.CropH << " and \"" <<
                                  CfgParam::vpp_in_crop_y_name() <<
                                  "\": " << mfxVPPParams.vpp.In.CropY <<
                                  " must be less or equal to \"" <<
                                  CfgParam::vpp_in_height_name() << "\": " <<
                                  mfxVPPParams.vpp.In.Height);
        GAPI_Error("Invalid VPP params combination: Height & Crop");
    }

    if (mfxVPPParams.vpp.Out.Width < mfxVPPParams.vpp.Out.CropW + mfxVPPParams.vpp.Out.CropX) {
        GAPI_LOG_WARNING(nullptr, "Invalid vonfiguration params: sum \"" <<
                                  CfgParam::vpp_out_crop_w_name() <<
                                  "\": " << mfxVPPParams.vpp.Out.CropW << " and \"" <<
                                  CfgParam::vpp_out_crop_x_name() <<
                                  "\": " << mfxVPPParams.vpp.Out.CropX <<
                                  " must be less or equal to \"" <<
                                  CfgParam::vpp_out_width_name() << "\": " <<
                                  mfxVPPParams.vpp.Out.Width);
        GAPI_Error("Invalid VPP params combination: Width & Crop");
    }

    if (mfxVPPParams.vpp.Out.Height < mfxVPPParams.vpp.Out.CropH + mfxVPPParams.vpp.Out.CropY) {
        GAPI_LOG_WARNING(nullptr, "Invalid vonfiguration params: sum \"" <<
                                  CfgParam::vpp_out_crop_h_name() <<
                                  "\": " << mfxVPPParams.vpp.Out.CropH << " and \"" <<
                                  CfgParam::vpp_out_crop_y_name() <<
                                  "\": " << mfxVPPParams.vpp.Out.CropY <<
                                  " must be less or equal to \"" <<
                                  CfgParam::vpp_out_height_name() << "\": " <<
                                  mfxVPPParams.vpp.Out.Height);
        GAPI_Error("Invalid VPP params combination: Height & Crop");
    }

    GAPI_LOG_INFO(nullptr, "Finished VPP param validation");
}

void VPLLegacyTranscodeEngine::on_frame_ready(LegacyTranscodeSession& sess,
                                              mfxFrameSurface1* ready_surface)
{
    GAPI_LOG_DEBUG(nullptr, "[" << sess.session << "], frame ready");

    // manage memory ownership rely on acceleration policy
    ready_surface->Data.Locked--;  // TODO -S- workaround

    VPLAccelerationPolicy::FrameConstructorArgs args{ready_surface, sess.session};
    auto frame_adapter = acceleration_policy->create_frame_adapter(sess.vpp_out_pool_id,
                                                                   args);
    ready_frames.emplace(cv::MediaFrame(std::move(frame_adapter)), sess.generate_frame_meta());

    // pop away synced out object
    sess.vpp_queue.pop();
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
