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

VPLLegacyTranscodeEngine::VPLLegacyTranscodeEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel)
 : VPLLegacyDecodeEngine(std::move(accel)) {

    GAPI_LOG_INFO(nullptr, "Create Legacy Transcode Engine");
    //inject_pipeline_operations(2,
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
                                                    (my_sess.data_provider || (my_sess.stream && my_sess.stream->DataLength))
                                                        ? my_sess.stream.get()

                                                        : nullptr, /* No more data to read, start decode draining mode*/
                                                    my_sess.procesing_surface_ptr.lock()->get_handle(),
                                                    &sync_pair.second,
                                                    &sync_pair.first);

            GAPI_LOG_INFO(nullptr, "START decode: " <<
                                            ", sync id:  " <<
                                            sync_pair.first <<
                                            ", dec in surface:  " <<
                                            my_sess.procesing_surface_ptr.lock()->get_handle() <<
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
                        my_sess.swap_surface(*this);
                    }
                    my_sess.last_status =
                    MFXVideoDECODE_DecodeFrameAsync(my_sess.session,
                                                    my_sess.stream.get(),
                                                    my_sess.procesing_surface_ptr.lock()->get_handle(),
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
            my_sess.sync_pair = sync_pair;
            return ExecutionStatus::Continue;
        },
        // 4) transcode
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyTranscodeSession &my_sess = static_cast<LegacyTranscodeSession&>(sess);

            //if (!my_sess.vpp_queue.empty())
            { // FIFO: check the oldest async decode operation complete
                LegacyDecodeSession::op_handle_t pending_op = my_sess.sync_pair;//my_sess.vpp_queue.front();
                auto *dec_surface = pending_op.second;

                my_sess.last_status = MFXVideoCORE_SyncOperation(sess.session, pending_op.first, 11000);


                //my_sess.vpp_surface_ptr.lock()->get_handle()
                if(my_sess.vpp_surface_ptr.lock())
                {
                    mfxFrameSurface1* out_surf = my_sess.vpp_surface_ptr.lock()->get_handle();
                    sess.last_status = MFXVideoVPP_RunFrameVPPAsync(sess.session, dec_surface,
                                                                    my_sess.vpp_surface_ptr.lock()->get_handle(),
                                                                    nullptr, &pending_op.first);
                    pending_op.second = my_sess.vpp_surface_ptr.lock()->get_handle();
                    GAPI_LOG_INFO(nullptr, "START transcode ops count: " <<
                                            my_sess.vpp_queue.size() <<
                                            ", sync id:  " <<
                                            pending_op.first <<
                                            ", dec surface:  " <<
                                            dec_surface <<
                                            ", trans surface: " << pending_op.second <<
                                            ", status: " <<
                                            mfxstatus_to_string(my_sess.last_status));
                    ///////////////////
                    /*my_sess.last_status = MFXVideoCORE_SyncOperation(sess.session, pending_op.first, 110000);

                    GAPI_LOG_DEBUG(nullptr, "SSSSSS pending ops count: " <<
                                            my_sess.sync_queue.size() <<
                                            ", sync id:  " <<
                                            pending_op.first <<
                                            ", surface:  " <<
                                            pending_op.second <<
                                            ", status: " <<
                                            mfxstatus_to_string(my_sess.last_status));
                    *////////////////////

                    while(sess.last_status == MFX_ERR_MORE_SURFACE)
                    {
                        //TODO put each operation ?
                        my_sess.sync_queue.emplace(pending_op);


                        try {
                            if (my_sess.last_status == MFX_ERR_MORE_SURFACE) {
                                my_sess.swap_transcode_surface(*this);
                            }
                            my_sess.last_status =
                            MFXVideoVPP_RunFrameVPPAsync(sess.session, dec_surface,
                                                         my_sess.vpp_surface_ptr.lock()->get_handle(),
                                                         nullptr, &pending_op.first);

                            // TODO uncommented pending op
                            pending_op.second = my_sess.vpp_surface_ptr.lock()->get_handle();
                            GAPI_LOG_INFO(nullptr, "MID transcode ops count: " <<
                                            my_sess.vpp_queue.size() <<
                                            ", sync id:  " <<
                                            pending_op.first <<
                                            ", dec surface:  " <<
                                            dec_surface <<
                                            ", trans surface: " << pending_op.second <<
                                            ", status: " <<
                                            mfxstatus_to_string(my_sess.last_status));
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


                    GAPI_LOG_INFO(nullptr, "END transcode ops count: " <<
                                            my_sess.vpp_queue.size() <<
                                            ", sync id:  " <<
                                            pending_op.first <<
                                            ", dec surface:  " <<
                                            dec_surface <<
                                            ", trans surface: " << pending_op.second <<
                                            ", status: " <<
                                            mfxstatus_to_string(my_sess.last_status));

                    if (sess.last_status == MFX_ERR_NONE ) {

                        // TODO not necessary now???
                        my_sess.sync_pair = pending_op;
                        /*
                        my_sess.sync_queue.emplace(std::move(pending_op));
                        my_sess.vpp_queue.pop();
                        */
                    }
                }
                else
                {
                    abort();
                }

                try {
                    my_sess.swap_transcode_surface(*this);
                }catch (... )
                {
                    my_sess.vpp_surface_ptr.reset();
                }
            }
            return ExecutionStatus::Continue;
        },
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyTranscodeSession &my_sess = static_cast<LegacyTranscodeSession&>(sess);
            if (my_sess.last_status == MFX_ERR_NONE) {
                my_sess.sync_queue.emplace(my_sess.sync_pair);
            } else if (my_sess.last_status != MFX_ERR_MORE_DATA) /* suppress MFX_ERR_MORE_DATA warning */ {
                GAPI_LOG_WARNING(nullptr, "decode pending ops count: " <<
                                          my_sess.sync_queue.size() <<
                                          ", sync id: " << my_sess.sync_pair.first <<
                                          ", status: " <<
                                          mfxstatus_to_string(my_sess.last_status));
            }
            return ExecutionStatus::Continue;
        },
        // 4) Wait for ASYNC decode result
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyTranscodeSession& my_sess = static_cast<LegacyTranscodeSession&>(sess);
            do {
                if (!my_sess.sync_queue.empty()) { // FIFO: check the oldest async operation complete
                    LegacyDecodeSession::op_handle_t& pending_op = my_sess.sync_queue.front();
                    sess.last_status = MFXVideoCORE_SyncOperation(sess.session, pending_op.first, 11000);

                    GAPI_LOG_DEBUG(nullptr, "pending ops count: " <<
                                            my_sess.sync_queue.size() <<
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
            } while (MFX_ERR_NONE == sess.last_status && !my_sess.sync_queue.empty());
            return ExecutionStatus::Continue;
        },
        // 5) Falls back on generic status procesing
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


    // NB:: create transcode params
    const auto& mfxDecParams = decode_params.decoder_params.param;

    auto vppOutImgWidth  = 672;
    auto vppOutImgHeight = 382;

    mfxVideoParam mfxVPPParams{0};
    //mfxVPPParams = mfxDecParams;
/*
    mfxVPPParams.vpp.In.FourCC        = mfxDecParams.mfx.FrameInfo.FourCC;
    mfxVPPParams.vpp.In.ChromaFormat  = mfxDecParams.mfx.FrameInfo.ChromaFormat;
    mfxVPPParams.vpp.In.Width         = vppInImgWidth;
    mfxVPPParams.vpp.In.Height        = vppInImgHeight;
    mfxVPPParams.vpp.In.CropW         = vppInImgWidth;
    mfxVPPParams.vpp.In.CropH         = vppInImgHeight;
    mfxVPPParams.vpp.In.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    mfxVPPParams.vpp.In.FrameRateExtN = 30;
    mfxVPPParams.vpp.In.FrameRateExtD = 1;
*/
    mfxVPPParams.vpp.In = mfxDecParams.mfx.FrameInfo;
    mfxVPPParams.vpp.Out.FourCC        = MFX_FOURCC_NV12;
    mfxVPPParams.vpp.Out.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    mfxVPPParams.vpp.Out.Width         = ALIGN16(vppOutImgWidth);
    mfxVPPParams.vpp.Out.Height        = ALIGN16(vppOutImgHeight);
    mfxVPPParams.vpp.Out.CropX = 0;
    mfxVPPParams.vpp.Out.CropY = 0;
    mfxVPPParams.vpp.Out.CropW         = vppOutImgWidth;
    mfxVPPParams.vpp.Out.CropH         = vppOutImgHeight;
    mfxVPPParams.vpp.Out.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    mfxVPPParams.vpp.Out.FrameRateExtN = 30;
    mfxVPPParams.vpp.Out.FrameRateExtD = 1;

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
        GAPI_LOG_WARNING(nullptr, "cannot MFXVideoVPP_QueryIOSurf");
        throw std::runtime_error("Cannot MFXVideoVPP_QueryIOSurf, error: " +
                                  mfxstatus_to_string(sts));
    }

    // TODO  enough
    //vppRequests[1].NumFrameSuggested *= 2;
/*
    auto tmpparam = mfxVPPParams;
    tmpparam.mfx.FrameInfo = mfxDecParams.vpp.Out;*/

    vppRequests[1].AllocId = 666;

    vppRequests[1].Type |= MFX_MEMTYPE_FROM_VPPIN;
    VPLAccelerationPolicy::pool_key_t vpp_out_pool_key =
                acceleration_policy->create_surface_pool(vppRequests[1], mfxVPPParams.vpp.Out);

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
    sess_ptr->swap_surface(*this);
    sess_ptr->swap_transcode_surface(*this);
    return sess_ptr;
}

ProcessingEngineBase::ExecutionStatus VPLLegacyTranscodeEngine::execute_op(operation_t& op, EngineSession& sess) {
    return op(sess);
}

void VPLLegacyTranscodeEngine::on_frame_ready(LegacyTranscodeSession& sess,
                                           mfxFrameSurface1* ready_surface)
{
    GAPI_LOG_DEBUG(nullptr, "[" << sess.session << "], frame ready");

    // manage memory ownership rely on acceleration policy
    auto frame_adapter = acceleration_policy->create_frame_adapter(sess.vpp_out_pool_id,
                                                                   ready_surface);
    ready_frames.emplace(cv::MediaFrame(std::move(frame_adapter)), sess.generate_frame_meta());

    // pop away synced out object
    sess.sync_queue.pop();
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
