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

#include "streaming/onevpl/engine/decode/decode_engine_legacy.hpp"
#include "streaming/onevpl/engine/decode/decode_session.hpp"
#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/cfg_params_parser.hpp"
#include "streaming/onevpl/utils.hpp"
#include "logger.hpp"


namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

VPLLegacyDecodeEngine::VPLLegacyDecodeEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel)
 : ProcessingEngineBase(std::move(accel)) {

    GAPI_LOG_INFO(nullptr, "Create Legacy Decode Engine");
    create_pipeline(
        // 1) Read File
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyDecodeSession &my_sess = static_cast<LegacyDecodeSession&>(sess);
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
            LegacyDecodeSession &my_sess = static_cast<LegacyDecodeSession&>(sess);

            // prepare sync object for new surface
            LegacyDecodeSession::op_handle_t sync_pair{};

            // enqueue decode operation with current session surface
            my_sess.last_status =
                    MFXVideoDECODE_DecodeFrameAsync(my_sess.session,
                                                    my_sess.last_status == MFX_ERR_NONE
                                                        ? my_sess.stream.get()
                                                        : nullptr, /* No more data to read, start decode draining mode*/
                                                    my_sess.procesing_surface_ptr.lock()->get_handle(),
                                                    &sync_pair.second,
                                                    &sync_pair.first);

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
                                                    &my_sess.stream,
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
        // 3) Wait for ASYNC decode result
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyDecodeSession& my_sess = static_cast<LegacyDecodeSession&>(sess);
            do {
                if (!my_sess.sync_queue.empty()) { // FIFO: check the oldest async operation complete
                    LegacyDecodeSession::op_handle_t& pending_op = my_sess.sync_queue.front();
                    sess.last_status = MFXVideoCORE_SyncOperation(sess.session, pending_op.first, 0);

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
        // 4) Falls back on generic status procesing
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            return this->process_error(sess.last_status, static_cast<LegacyDecodeSession&>(sess));
        }
    );
}

ProcessingEngineBase::session_ptr
VPLLegacyDecodeEngine::initialize_session(mfxSession mfx_session,
                                          const std::vector<CfgParam>& cfg_params,
                                          std::shared_ptr<IDataProvider> provider) {
    GAPI_DbgAssert(provider && "Cannot create decoder, data provider is nullptr");

    // Find codec ID from config
    auto dec_it = std::find_if(cfg_params.begin(), cfg_params.end(), [] (const CfgParam& value) {
        return value.get_name() == "mfxImplDescription.mfxDecoderDescription.decoder.CodecID";
    });
    if (dec_it == cfg_params.end()) {
        throw std::logic_error("Cannot determine DecoderID from oneVPL config. Abort");
    }

    mfxVariant decoder = cfg_param_to_mfx_variant(*dec_it);

    // fill input bitstream
    mfxBitstream bitstream{};
    const int BITSTREAM_BUFFER_SIZE = 2000000;
    bitstream.MaxLength = BITSTREAM_BUFFER_SIZE;
    bitstream.Data = (mfxU8 *)calloc(bitstream.MaxLength, sizeof(mfxU8));
    if(!bitstream.Data) {
        throw std::runtime_error("Cannot allocate bitstream.Data bytes: " +
                                 std::to_string(bitstream.MaxLength * sizeof(mfxU8)));
    }

    bitstream.CodecId = decoder.Data.U32;
    mfxStatus sts = ReadEncodedStream(bitstream, provider);
    if(MFX_ERR_NONE != sts) {
        throw std::runtime_error("Error reading bitstream, error: " +
                                 mfxstatus_to_string(sts));
    }

    // init session
    acceleration_policy->init(mfx_session);

    // Retrieve the frame information from input stream
    mfxVideoParam mfxDecParams {};
    mfxDecParams.mfx.CodecId = decoder.Data.U32;

    // set memory stream direction accroding to accelearion policy device type
    IDeviceSelector::DeviceScoreTable devices = acceleration_policy->get_device_selector()->select_devices();
    GAPI_Assert(devices.size() == 1 && "Multiple(or zero) acceleration devices case is unsupported");
    AccelType accel_type = devices.begin()->second.get_type();
    if (accel_type == AccelType::DX11) {
        mfxDecParams.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    } else if (accel_type == AccelType::HOST) {
        mfxDecParams.IOPattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    } else {
        GAPI_Assert(false && "unsupported AccelType from device selector");
    }

    sts = MFXVideoDECODE_DecodeHeader(mfx_session, &bitstream, &mfxDecParams);
    if(MFX_ERR_NONE != sts) {
        throw std::runtime_error("Error decoding header, error: " +
                                 mfxstatus_to_string(sts));
    }

    mfxFrameAllocRequest decRequest {};

    // Query number required surfaces for decoder
    MFXVideoDECODE_QueryIOSurf(mfx_session, &mfxDecParams, &decRequest);

    // External (application) allocation of decode surfaces
    GAPI_LOG_DEBUG(nullptr, "Query IOSurf for session: " << mfx_session <<
                            ", mfxFrameAllocRequest.NumFrameMin: " << decRequest.NumFrameMin <<
                            ", mfxFrameAllocRequest.NumFrameSuggested: " << decRequest.NumFrameSuggested <<
                            ", mfxFrameAllocRequest.Type: " << decRequest.Type);

    VPLAccelerationPolicy::pool_key_t decode_pool_key =
                acceleration_policy->create_surface_pool(decRequest, mfxDecParams);

    // Input parameters finished, now initialize decode
    // create decoder for session accoring to header recovered from source file
    sts = MFXVideoDECODE_Init(mfx_session, &mfxDecParams);
    if (MFX_ERR_NONE != sts) {
        throw std::runtime_error("Error initializing Decode, error: " +
                                 mfxstatus_to_string(sts));
    }

    DecoderParams decoder_param {bitstream, mfxDecParams};

    // create session
    std::shared_ptr<LegacyDecodeSession> sess_ptr =
                register_session<LegacyDecodeSession>(mfx_session,
                                                      std::move(decoder_param),
                                                      provider);

    sess_ptr->init_surface_pool(decode_pool_key);
    // prepare working decode surface
    sess_ptr->swap_surface(*this);
    return sess_ptr;
}

ProcessingEngineBase::ExecutionStatus VPLLegacyDecodeEngine::execute_op(operation_t& op, EngineSession& sess) {
    return op(sess);
}

void VPLLegacyDecodeEngine::on_frame_ready(LegacyDecodeSession& sess,
                                           mfxFrameSurface1* ready_surface)
{
    GAPI_LOG_DEBUG(nullptr, "[" << sess.session << "], frame ready");

    // manage memory ownership rely on acceleration policy
    auto frame_adapter = acceleration_policy->create_frame_adapter(sess.decoder_pool_id,
                                                                   ready_surface);
    ready_frames.emplace(cv::MediaFrame(std::move(frame_adapter)), sess.generate_frame_meta());

    // pop away synced out object
    sess.sync_queue.pop();
}

ProcessingEngineBase::ExecutionStatus VPLLegacyDecodeEngine::process_error(mfxStatus status, LegacyDecodeSession& sess)
{
    GAPI_LOG_DEBUG(nullptr, "status: " << mfxstatus_to_string(status));

    switch (status) {
        case MFX_ERR_NONE:
        {
            // prepare sync object for new surface
            try {
                sess.swap_surface(*this);
                return ExecutionStatus::Continue;
            } catch (const std::runtime_error& ex) {
                GAPI_LOG_WARNING(nullptr, "[" << sess.session << "] error: " << ex.what());
                return ExecutionStatus::Continue; // read more data
            }
        }
        case MFX_ERR_MORE_DATA: // The function requires more bitstream at input before decoding can proceed
            if (!sess.data_provider || sess.data_provider->empty()) {
                // No more data to drain from decoder, start encode draining mode
                return ExecutionStatus::Processed;
            }
            else
                return ExecutionStatus::Continue; // read more data
            break;
        case MFX_ERR_MORE_SURFACE:
        {
            // The function requires more frame surface at output before decoding can proceed.
            // This applies to external memory allocations and should not be expected for
            // a simple internal allocation case like this
            try {
                sess.swap_surface(*this);
                return ExecutionStatus::Continue;
            } catch (const std::runtime_error& ex) {
                GAPI_LOG_WARNING(nullptr, "[" << sess.session << "] error: " << ex.what());
                 return ExecutionStatus::Continue; // read more data
            }
            break;
        }
        case MFX_ERR_DEVICE_LOST:
            // For non-CPU implementations,
            // Cleanup if device is lost
            GAPI_DbgAssert(false && "VPLLegacyDecodeEngine::process_error - "
                                    "MFX_ERR_DEVICE_LOST is not processed");
            break;
        case MFX_WRN_DEVICE_BUSY:
            // For non-CPU implementations,
            // Wait a few milliseconds then try again
            GAPI_DbgAssert(false && "VPLLegacyDecodeEngine::process_error - "
                                    "MFX_WRN_DEVICE_BUSY is not processed");
            break;
        case MFX_WRN_VIDEO_PARAM_CHANGED:
            // The decoder detected a new sequence header in the bitstream.
            // Video parameters may have changed.
            // In external memory allocation case, might need to reallocate the output surface
            /*GAPI_DbgAssert(false && "VPLLegacyDecodeEngine::process_error - "
                                    "MFX_WRN_VIDEO_PARAM_CHANGED is not processed");
            */
            return ExecutionStatus::Continue;
            break;
        case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM:
            // The function detected that video parameters provided by the application
            // are incompatible with initialization parameters.
            // The application should close the component and then reinitialize it
            GAPI_DbgAssert(false && "VPLLegacyDecodeEngine::process_error - "
                                    "MFX_ERR_INCOMPATIBLE_VIDEO_PARAM is not processed");
            break;
        case MFX_ERR_REALLOC_SURFACE:
            // Bigger surface_work required. May be returned only if
            // mfxInfoMFX::EnableReallocRequest was set to ON during initialization.
            // This applies to external memory allocations and should not be expected for
            // a simple internal allocation case like this
            GAPI_DbgAssert(false && "VPLLegacyDecodeEngine::process_error - "
                                    "MFX_ERR_REALLOC_SURFACE is not processed");
            break;
        case MFX_WRN_IN_EXECUTION:
            try {
                sess.swap_surface(*this);
                return ExecutionStatus::Continue;
            } catch (const std::runtime_error& ex) {
                GAPI_LOG_WARNING(nullptr, "[" << sess.session << "] error: " << ex.what());
                return ExecutionStatus::Continue;
            }
        default:
            GAPI_LOG_WARNING(nullptr, "Unknown status code: " << mfxstatus_to_string(status) <<
                                      ", decoded frames: " << sess.decoded_frames_count);
            break;
    }

    return ExecutionStatus::Failed;
}

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
