#include "streaming/vpl/vpl_source_engine.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {

// Read encoded stream from file
mfxStatus ReadEncodedStream(mfxBitstream &bs, FILE *f) {
    mfxU8 *p0 = bs.Data;
    mfxU8 *p1 = bs.Data + bs.DataOffset;
    if (bs.DataOffset > bs.MaxLength - 1) {
        return MFX_ERR_NOT_ENOUGH_BUFFER;
    }
    if (bs.DataLength + bs.DataOffset > bs.MaxLength) {
        return MFX_ERR_NOT_ENOUGH_BUFFER;
    }
    for (mfxU32 i = 0; i < bs.DataLength; i++) {
        *(p0++) = *(p1++);
    }
    bs.DataOffset = 0;
    bs.DataLength += (mfxU32)fread(bs.Data + bs.DataLength, 1, bs.MaxLength - bs.DataLength, f);
    if (bs.DataLength == 0)
        return MFX_ERR_MORE_DATA;

    return MFX_ERR_NONE;
}

EngineSession::ExecutionStatus DecodeSession::execute_op(operation_t& op)
{
    return op(*this);
}
    

VPLDecodeEngine::VPLDecodeEngine(file_ptr&& src_ptr) :
    source_handle(std::move(src_ptr)),
    dec_surface_out()
{
    
}

void VPLDecodeEngine::initialize_session(mfxSession mfx_session, mfxBitstream&& mfx_session_bitstream)
{
    auto sess_ptr = register_session<DecodeSession>(mfx_session, std::move(mfx_session_bitstream),
        // 1) Reade File
        [this] (EngineSession& sess) -> EngineSession::ExecutionStatus
        {
            DecodeSession &my_sess = static_cast<DecodeSession&>(sess);
            my_sess.last_status = ReadEncodedStream(my_sess.stream, my_sess.source_ptr);
            if (my_sess.last_status != MFX_ERR_NONE) {
                my_sess.source_ptr = nullptr; //close source
            }
            return EngineSession::ExecutionStatus::Continue;
        },
        // 2) enqueue ASYNC decode
        [this] (EngineSession& sess) -> EngineSession::ExecutionStatus
        {
            GAPI_LOG_INFO/*DEBUG*/(nullptr, "session: " << sess.session << ", sess.last_status: " << sess.last_status);
            sess.last_status =
                    MFXVideoDECODE_DecodeFrameAsync(sess.session,
                                                    sess.last_status == MFX_ERR_NONE
                                                        ? &sess.stream
                                                        : nullptr, /* No more data to read, start decode draining mode*/
                                                        nullptr,
                                                    &dec_surface_out,
                                                    &sess.sync);
            GAPI_LOG_INFO/*DEBUG*/(nullptr, "session: " << sess.session << ", dec_surface_out: " << dec_surface_out <<
                                   ", sess.last_status: " << sess.last_status);
            return EngineSession::ExecutionStatus::Continue;
        },
        // 3) Wait for ASYNC decode result
        [this] (EngineSession& sess) -> EngineSession::ExecutionStatus
        {
            if (sess.last_status == MFX_ERR_NONE) // Got 1 decoded frame
            {
                do {
                    sess.last_status = MFXVideoCORE_SyncOperation(sess.session, sess.sync, 100);
                    if (MFX_ERR_NONE == sess.last_status) {
                        on_frame_ready(static_cast<DecodeSession&>(sess), dec_surface_out);
                    }
                } while (sess.last_status == MFX_WRN_IN_EXECUTION);
            }

            return this->process_error(sess.last_status, static_cast<DecodeSession&>(sess));
        });


    sess_ptr->source_ptr = source_handle.get();
}

void VPLDecodeEngine::on_frame_ready(DecodeSession& sess, mfxFrameSurface1* surface)
{
    mfxFrameInfo *info = &surface->Info;
    mfxFrameData *data = &surface->Data;
    size_t w = info->Width;
    size_t h = info->Height;
    size_t p = data->Pitch;


    GAPI_LOG_INFO/*DEBUG*/(nullptr, "session: " << sess.session << ", surface: " << surface <<
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

EngineSession::ExecutionStatus VPLDecodeEngine::process_error(mfxStatus status, DecodeSession& sess)
{
    GAPI_LOG_INFO/*DEBUG*/(nullptr, "status: " << status);
    switch (status) {
        case MFX_ERR_MORE_DATA: // The function requires more bitstream at input before decoding can proceed
            if (!sess.source_ptr) {
                // No more data to drain from decoder, start encode draining mode
                return EngineSession::ExecutionStatus::Processed;
            }
            else
                return EngineSession::ExecutionStatus::Continue; // read more data
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

    return EngineSession::ExecutionStatus::Failed;
}
} // namespace wip
} // namespace gapi
} // namespace cv
