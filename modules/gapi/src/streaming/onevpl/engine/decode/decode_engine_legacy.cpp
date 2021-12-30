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
                                                    (my_sess.data_provider || (my_sess.stream && my_sess.stream->DataLength))
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

    // init session
    acceleration_policy->init(mfx_session);

    // Get codec ID from data provider
    IDataProvider::mfx_codec_id_type decoder_id_name = provider->get_mfx_codec_id();

    // Prepare video param
    mfxVideoParam mfxDecParams {};
    mfxDecParams.mfx.CodecId = decoder_id_name;

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

    // try fetch & decode input data
    mfxStatus sts = MFX_ERR_NONE;
    std::shared_ptr<IDataProvider::mfx_bitstream> bitstream{};
    bool can_fetch_data = false;
    do {
        can_fetch_data = provider->fetch_bitstream_data(bitstream);
        if (!can_fetch_data) {
            // must fetch data always because EOF critical at this point
            GAPI_LOG_WARNING(nullptr, "cannot decode header from provider: " << provider.get() <<
                                      ". Unexpected EOF");
            throw std::runtime_error("Error reading bitstream: EOF");
        }

        sts = MFXVideoDECODE_DecodeHeader(mfx_session, bitstream.get(), &mfxDecParams);
        if(MFX_ERR_NONE != sts && MFX_ERR_MORE_DATA != sts) {
            throw std::runtime_error("Error decoding header, error: " +
                                     mfxstatus_to_string(sts));
        }
    } while (sts == MFX_ERR_MORE_DATA && !provider->empty());

    if (MFX_ERR_NONE != sts) {
        GAPI_LOG_WARNING(nullptr, "cannot decode header from provider: " << provider.get()
                                  << ". Make sure data source is valid and/or "
                                  "\"" << CfgParam::decoder_id_name() << "\""
                                  " has correct value in case of demultiplexed raw input");
         throw std::runtime_error("Error decode header, error: " +
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

    // NB: override NumFrameSuggested preallocation size (how many frames we can hold)
    size_t preallocated_frames_count = decRequest.NumFrameSuggested;
    // NB: if you see bunch of WARNING about "cannot get free surface from pool"
    // and have abundant RAM size then increase `preallocated_frames_count`
    // to keep more free surfaces in a round. Otherwise VPL decode pipeline will be waiting
    // till application is freeing unusable surface on its side.
    //
    auto queue_capacity_it = std::find_if(cfg_params.begin(), cfg_params.end(), [] (const CfgParam& value) {
        return value.get_name() == CfgParam::frames_pool_size_name();
    });
    if (queue_capacity_it != cfg_params.end()) {
        cv::util::visit(cv::util::overload_lambdas(
            [&preallocated_frames_count](uint8_t value)   { preallocated_frames_count = static_cast<size_t>(value);   },
            [&preallocated_frames_count](int8_t value)    { preallocated_frames_count = static_cast<size_t>(value);   },
            [&preallocated_frames_count](uint16_t value)  { preallocated_frames_count = static_cast<size_t>(value);   },
            [&preallocated_frames_count](int16_t value)   { preallocated_frames_count = static_cast<size_t>(value);   },
            [&preallocated_frames_count](uint32_t value)  { preallocated_frames_count = static_cast<size_t>(value);   },
            [&preallocated_frames_count](int32_t value)   { preallocated_frames_count = static_cast<size_t>(value);   },
            [&preallocated_frames_count](uint64_t value)  { preallocated_frames_count = static_cast<size_t>(value);   },
            [&preallocated_frames_count](int64_t value)   { preallocated_frames_count = static_cast<size_t>(value);   },
            [&preallocated_frames_count](float_t value)   { preallocated_frames_count = static_cast<size_t>(value);   },
            [&preallocated_frames_count](double_t value)  { preallocated_frames_count = static_cast<size_t>(value);   },
            [&preallocated_frames_count](void*)     { GAPI_Assert(false && "`void*` is unsupported type");  },
            [&preallocated_frames_count](const std::string& value) {
                preallocated_frames_count = strtoull_or_throw(value.c_str());
            }),
            queue_capacity_it->get_value());

        GAPI_LOG_INFO(nullptr, "Try to use CfgParam \"" << CfgParam::frames_pool_size_name() << "\": " <<
                      preallocated_frames_count << ", for session: " << mfx_session);

    }
    if (preallocated_frames_count < decRequest.NumFrameMin) {
        GAPI_LOG_WARNING(nullptr, "Cannot proceed with CfgParam \"" << CfgParam::frames_pool_size_name() << "\": " <<
                                  preallocated_frames_count << ". It must be equal or greater than "
                                  "mfxFrameAllocRequest.NumFrameMin: " << decRequest.NumFrameMin);
        throw std::runtime_error(std::string("Invalid value of param: ") +
                                 CfgParam::frames_pool_size_name() + ", underflow");
    } else {
        if (static_cast<size_t>(std::numeric_limits<mfxU16>::max()) < preallocated_frames_count) {
            GAPI_LOG_WARNING(nullptr, "Cannot proceed with CfgParam \"" << CfgParam::frames_pool_size_name() << "\": " <<
                                  preallocated_frames_count << ". It must not be equal than " <<
                                  std::numeric_limits<mfxU16>::max());
            throw std::runtime_error(std::string("Invalid value of param: ") +
                                 CfgParam::frames_pool_size_name() + ", overflow");
        }
        decRequest.NumFrameSuggested = static_cast<mfxU16>(preallocated_frames_count);
        GAPI_LOG_DEBUG(nullptr, "mfxFrameAllocRequest overriden by user input for session: " << mfx_session <<
                            ", mfxFrameAllocRequest.NumFrameMin: " << decRequest.NumFrameMin <<
                            ", mfxFrameAllocRequest.NumFrameSuggested: " << decRequest.NumFrameSuggested <<
                            ", mfxFrameAllocRequest.Type: " << decRequest.Type);
    }

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
            if (!(sess.data_provider || (sess.stream && sess.stream->DataLength))) {
                // No more data to drain from decoder
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
