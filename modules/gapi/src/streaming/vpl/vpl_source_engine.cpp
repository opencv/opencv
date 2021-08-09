// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "streaming/vpl/vpl_source_engine.hpp"
#include "streaming/vpl/vpl_utils.hpp"
#include "streaming/vpl/vpl_dx11_accel.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {

DecodeSession::DecodeSession(mfxSession sess, mfxBitstream&& str, std::shared_ptr<IDataProvider> provider) :
    EngineSession(sess, std::move(str)),
    data_provider(std::move(provider)),
    stop_processing(false),
    dec_surface_out()
{
}

VPLDecodeEngine::VPLDecodeEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel)
 : VPLProcessingEngine(std::move(accel)) {

    GAPI_LOG_INFO(nullptr, "Create VPL Decode Engine");
    create_pipeline(
        // 1) Reade File
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            DecodeSession &my_sess = static_cast<DecodeSession&>(sess);
            my_sess.last_status = ReadEncodedStream(my_sess.stream, my_sess.data_provider);
            if (my_sess.last_status != MFX_ERR_NONE) {
                my_sess.data_provider.reset(); //close source
            }
            return ExecutionStatus::Continue;
        },
        // 2) enqueue ASYNC decode
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            DecodeSession &my_sess = static_cast<DecodeSession&>(sess);
            sess.last_status =
                    MFXVideoDECODE_DecodeFrameAsync(my_sess.session,
                                                    my_sess.last_status == MFX_ERR_NONE
                                                        ? &my_sess.stream
                                                        : nullptr, /* No more data to read, start decode draining mode*/
                                                        nullptr,
                                                    &my_sess.dec_surface_out,
                                                    &my_sess.sync);
            GAPI_LOG_DEBUG(nullptr, "[" << my_sess.session << "], dec_surface_out: " << my_sess.dec_surface_out);
            return ExecutionStatus::Continue;
        },
        // 3) Wait for ASYNC decode result
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            if (sess.last_status == MFX_ERR_NONE) // Got 1 decoded frame
            {
                do {
                    sess.last_status = MFXVideoCORE_SyncOperation(sess.session, sess.sync, 100);
                    if (MFX_ERR_NONE == sess.last_status) {

                        DecodeSession& my_sess = static_cast<DecodeSession&>(sess);
                        on_frame_ready(my_sess);
                    }
                } while (sess.last_status == MFX_WRN_IN_EXECUTION);
            }

            return this->process_error(sess.last_status, static_cast<DecodeSession&>(sess));
        }
    );
}

void VPLDecodeEngine::initialize_session(mfxSession mfx_session,
                                         DecoderParams&& decoder_param,
                                         std::shared_ptr<IDataProvider> provider)
{
    register_session<DecodeSession>(mfx_session,
                                    std::move(decoder_param.stream),
                                    provider);
    acceleration_policy->init(mfx_session);
}

VPLProcessingEngine::ExecutionStatus VPLDecodeEngine::execute_op(operation_t& op, EngineSession& sess) {
    return op(sess);
}
    
void VPLDecodeEngine::on_frame_ready(DecodeSession& sess)
{
    mfxFrameInfo *info = &sess.dec_surface_out->Info;
    mfxFrameData *data = &sess.dec_surface_out->Data;
    size_t w = info->Width;
    size_t h = info->Height;
    size_t p = data->Pitch;

    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
    
    GAPI_LOG_DEBUG(nullptr, "[" << sess.session << "], surface: " << sess.dec_surface_out <<
                            ", w: " << w << ", h: " << h << ", p: " << p);
    const int cols = info->CropW;
    const int rows = info->CropH;

    switch (info->FourCC) {
        case MFX_FOURCC_I420: {
        } break;

        case MFX_FOURCC_NV12: {
        } break;

        default:
            GAPI_LOG_WARNING(nullptr, "Unsupported FourCC code: " << info->FourCC << ". Skip");
            return;
    }

    ready_frames.push(cv::Mat(cv::Size{rows, cols}, CV_8UC3));
}

VPLProcessingEngine::ExecutionStatus VPLDecodeEngine::process_error(mfxStatus status, DecodeSession& sess)
{
    GAPI_LOG_DEBUG(nullptr, "status: " << status);
    switch (status) {
        case MFX_ERR_NONE:
            return ExecutionStatus::Continue; 
        case MFX_ERR_MORE_DATA: // The function requires more bitstream at input before decoding can proceed
            if (!sess.data_provider || sess.data_provider->empty()) {
                // No more data to drain from decoder, start encode draining mode
                return ExecutionStatus::Processed;
            }
            else
                return ExecutionStatus::Continue; // read more data
            break;
        case MFX_ERR_MORE_SURFACE:
            // The function requires more frame surface at output before decoding can proceed.
            // This applies to external memory allocations and should not be expected for
            // a simple internal allocation case like this
            break;
        case MFX_ERR_DEVICE_LOST:
            // For non-CPU implementations,
            // Cleanup if device is lost
            break;
        case MFX_WRN_DEVICE_BUSY:
            // For non-CPU implementations,
            // Wait a few milliseconds then try again
            break;
        case MFX_WRN_VIDEO_PARAM_CHANGED:
            // The decoder detected a new sequence header in the bitstream.
            // Video parameters may have changed.
            // In external memory allocation case, might need to reallocate the output surface
            break;
        case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM:
            // The function detected that video parameters provided by the application
            // are incompatible with initialization parameters.
            // The application should close the component and then reinitialize it
            break;
        case MFX_ERR_REALLOC_SURFACE:
            // Bigger surface_work required. May be returned only if
            // mfxInfoMFX::EnableReallocRequest was set to ON during initialization.
            // This applies to external memory allocations and should not be expected for
            // a simple internal allocation case like this
            break;
        default:
            GAPI_LOG_WARNING(nullptr, "Unknown status code: " << status);
            break;
    }

    return ExecutionStatus::Failed;
}
} // namespace wip
} // namespace gapi
} // namespace cv
