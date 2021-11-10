// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL

#include <algorithm>
#include <exception>

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>

#include "streaming/onevpl/engine/decode/decode_engine_legacy.hpp"
#include "streaming/onevpl/engine/decode/decode_session.hpp"
#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/utils.hpp"
#include "logger.hpp"


namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
/* UTILS */
mfxU32 GetSurfaceSize(mfxU32 FourCC, mfxU32 width, mfxU32 height) {
    mfxU32 nbytes = 0;

    mfxU32 half_width = width / 2;
    mfxU32 half_height = height / 2;
    switch (FourCC) {
        case MFX_FOURCC_I420:
        case MFX_FOURCC_NV12:
            nbytes = width * height +  2 * half_width * half_height;
            break;
        case MFX_FOURCC_I010:
        case MFX_FOURCC_P010:
            nbytes = width * height + 2 * half_width * half_height;
            nbytes *= 2;
            break;
        case MFX_FOURCC_RGB4:
            nbytes = width * height * 4;
            break;
        default:
            GAPI_LOG_WARNING(nullptr, "Unsupported FourCC requested: " << FourCC);
            GAPI_Assert(false && "Unsupported FourCC requested");
            break;
    }
    return nbytes;
}

surface_ptr_t create_surface_RGB4(mfxFrameInfo frameInfo,
                                  std::shared_ptr<void> out_buf_ptr,
                                  size_t out_buf_ptr_offset,
                                  size_t out_buf_size)
{
    mfxU8* buf = reinterpret_cast<mfxU8*>(out_buf_ptr.get());
    mfxU16 surfW = frameInfo.Width * 4;
    mfxU16 surfH = frameInfo.Height;
    (void)surfH;

    // TODO more intelligent check
    if (out_buf_size <= out_buf_ptr_offset) {
        throw std::runtime_error(std::string("Insufficient buffer size: ") +
                                 std::to_string(out_buf_size) + ", buffer offset: " +
                                 std::to_string(out_buf_ptr_offset) +
                                 ", expected surface width: " + std::to_string(surfW) +
                                 ", height: " + std::to_string(surfH));
    }

    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1);
    memset(handle.get(), 0, sizeof(mfxFrameSurface1));

    handle->Info = frameInfo;
    handle->Data.B = buf + out_buf_ptr_offset;
    handle->Data.G = handle->Data.B + 1;
    handle->Data.R = handle->Data.B + 2;
    handle->Data.A = handle->Data.B + 3;
    handle->Data.Pitch = surfW;

    return Surface::create_surface(std::move(handle), out_buf_ptr);
}

surface_ptr_t create_surface_other(mfxFrameInfo frameInfo,
                                   std::shared_ptr<void> out_buf_ptr,
                                   size_t out_buf_ptr_offset,
                                   size_t out_buf_size)
{
    mfxU8* buf = reinterpret_cast<mfxU8*>(out_buf_ptr.get());
    mfxU16 surfH = frameInfo.Height;
    mfxU16 surfW = (frameInfo.FourCC == MFX_FOURCC_P010) ? frameInfo.Width * 2 : frameInfo.Width;

    // TODO more intelligent check
    if (out_buf_size <=
        out_buf_ptr_offset + (surfW * surfH) + ((surfW / 2) * (surfH / 2))) {
        throw std::runtime_error(std::string("Insufficient buffer size: ") +
                                 std::to_string(out_buf_size) + ", buffer offset: " +
                                 std::to_string(out_buf_ptr_offset) +
                                 ", expected surface width: " + std::to_string(surfW) +
                                 ", height: " + std::to_string(surfH));
    }

    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1);
    memset(handle.get(), 0, sizeof(mfxFrameSurface1));

    handle->Info = frameInfo;
    handle->Data.Y     = buf + out_buf_ptr_offset;
    handle->Data.U     = buf + out_buf_ptr_offset + (surfW * surfH);
    handle->Data.V     = handle->Data.U + ((surfW / 2) * (surfH / 2));
    handle->Data.Pitch = surfW;

    return Surface::create_surface(std::move(handle), out_buf_ptr);
}

VPLLegacyDecodeEngine::VPLLegacyDecodeEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel)
 : ProcessingEngineBase(std::move(accel)) {

    GAPI_LOG_INFO(nullptr, "Create Legacy Decode Engine");
    create_pipeline(
        // 1) Read File
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyDecodeSession &my_sess = static_cast<LegacyDecodeSession&>(sess);
            my_sess.last_status = ReadEncodedStream(my_sess.stream, my_sess.data_provider);
            if (my_sess.last_status != MFX_ERR_NONE) {
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
                                                        ? &my_sess.stream
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
                    GAPI_LOG_WARNING(nullptr, "[" << my_sess.session <<
                                               "] has no surface, reason: " <<
                                               ex.what());
                    // TODO it is supposed to place `break;` here
                    // to simulate `yield`-like behavior.
                    // Further DX11 intergation logic claims more strict rules
                    // for enqueue surfaces. If no free surface
                    // is available it had better to wait free one by checking
                    // for async result than waste time in spinning.
                    //
                    // Put it as-is at now to not break
                    // current compatibility and avoid further merge-conflicts
                }
            }

            if (my_sess.last_status == MFX_ERR_NONE) {
                my_sess.sync_queue.emplace(sync_pair);
            } else if (my_sess.last_status != MFX_ERR_MORE_DATA) /* suppress MFX_ERR_MORE_DATA warning */ {
                GAPI_LOG_WARNING(nullptr, "pending ops count: " << my_sess.sync_queue.size() <<
                                          ", sync id: " << sync_pair.first <<
                                          ", status: " << mfxstatus_to_string(my_sess.last_status));
            }
            return ExecutionStatus::Continue;
        },
        // 3) Wait for ASYNC decode result
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyDecodeSession& my_sess = static_cast<LegacyDecodeSession&>(sess);
            if (!my_sess.sync_queue.empty()) // FIFO: check the oldest async operation complete
            {
                LegacyDecodeSession::op_handle_t& pending_op = my_sess.sync_queue.front();
                sess.last_status = MFXVideoCORE_SyncOperation(sess.session, pending_op.first, 0);

                GAPI_LOG_DEBUG(nullptr, "pending ops count: " << my_sess.sync_queue.size() <<
                                        ", sync id:  " << pending_op.first <<
                                        ", status: " << mfxstatus_to_string(my_sess.last_status));

                // put frames in ready queue on success
                if (MFX_ERR_NONE == sess.last_status) {
                    on_frame_ready(my_sess, pending_op.second);
                }
            }
            return ExecutionStatus::Continue;
        },
        // 4) Falls back on generic status procesing
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            return this->process_error(sess.last_status, static_cast<LegacyDecodeSession&>(sess));
        }
    );
}

void VPLLegacyDecodeEngine::initialize_session(mfxSession mfx_session,
                                               DecoderParams&& decoder_param,
                                               std::shared_ptr<onevpl::IDataProvider> provider)
{
    mfxFrameAllocRequest decRequest = {};
    // Query number required surfaces for decoder
    MFXVideoDECODE_QueryIOSurf(mfx_session, &decoder_param.param, &decRequest);

    // External (application) allocation of decode surfaces
    GAPI_LOG_DEBUG(nullptr, "Query IOSurf for session: " << mfx_session <<
                            ", mfxFrameAllocRequest.NumFrameSuggested: " << decRequest.NumFrameSuggested <<
                            ", mfxFrameAllocRequest.Type: " << decRequest.Type);

    mfxU32 singleSurfaceSize = GetSurfaceSize(decoder_param.param.mfx.FrameInfo.FourCC,
                                              decoder_param.param.mfx.FrameInfo.Width,
                                              decoder_param.param.mfx.FrameInfo.Height);
    if (!singleSurfaceSize) {
        throw std::runtime_error("Cannot determine surface size for: fourCC" +
                                 std::to_string(decoder_param.param.mfx.FrameInfo.FourCC) +
                                 ", width: " + std::to_string(decoder_param.param.mfx.FrameInfo.Width) +
                                 ", height: " + std::to_string(decoder_param.param.mfx.FrameInfo.Height));
    }

    const auto &frameInfo = decoder_param.param.mfx.FrameInfo;
    auto surface_creator =
            [&frameInfo] (std::shared_ptr<void> out_buf_ptr, size_t out_buf_ptr_offset,
                          size_t out_buf_size) -> surface_ptr_t {
                return (frameInfo.FourCC == MFX_FOURCC_RGB4) ?
                        create_surface_RGB4(frameInfo, out_buf_ptr, out_buf_ptr_offset,
                                            out_buf_size) :
                        create_surface_other(frameInfo, out_buf_ptr, out_buf_ptr_offset,
                                             out_buf_size);};

    //TODO Configure preallocation size (how many frames we can hold)
    const size_t preallocated_frames_count = 30;
    VPLAccelerationPolicy::pool_key_t decode_pool_key =
                acceleration_policy->create_surface_pool(decRequest.NumFrameSuggested * preallocated_frames_count,
                                                         singleSurfaceSize,
                                                         surface_creator);

    // create session
    std::shared_ptr<LegacyDecodeSession> sess_ptr =
                register_session<LegacyDecodeSession>(mfx_session,
                                                      std::move(decoder_param),
                                                      provider);

    sess_ptr->init_surface_pool(decode_pool_key);
    // prepare working decode surface
    sess_ptr->swap_surface(*this);
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
                GAPI_LOG_WARNING(nullptr, "[" << sess.session << "] error: " << ex.what() <<
                                          "Abort");
                // TODO it is supposed to be `break;` here in future PR
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
                // TODO it is supposed to be `break;` here in future PR
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
            GAPI_DbgAssert(false && "VPLLegacyDecodeEngine::process_error - "
                                    "MFX_WRN_VIDEO_PARAM_CHANGED is not processed");
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
                GAPI_LOG_WARNING(nullptr, "[" << sess.session << "] error: " << ex.what() <<
                                          "Abort");
                // TODO it is supposed to be `break;` here in future PR
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
