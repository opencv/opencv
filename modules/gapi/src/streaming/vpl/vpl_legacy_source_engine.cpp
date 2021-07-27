#include <algorithm>
#include <exception>

#include "streaming/vpl/vpl_legacy_source_engine.hpp"
#include "streaming/vpl/vpl_utils.hpp"
#include "streaming/vpl/vpl_dx11_accel.hpp"

#include "logger.hpp"

#if (MFX_VERSION >= 2000)
    #include <vpl/mfxdispatcher.h>
#endif

namespace cv {
namespace gapi {
namespace wip {
/* UTILS */
mfxU32 GetSurfaceSize(mfxU32 FourCC, mfxU32 width, mfxU32 height) {
    mfxU32 nbytes = 0;

    switch (FourCC) {
        case MFX_FOURCC_I420:
        case MFX_FOURCC_NV12:
            nbytes = width * height + (width >> 1) * (height >> 1) + (width >> 1) * (height >> 1);
            break;
        case MFX_FOURCC_I010:
        case MFX_FOURCC_P010:
            nbytes = width * height + (width >> 1) * (height >> 1) + (width >> 1) * (height >> 1);
            nbytes *= 2;
            break;
        case MFX_FOURCC_RGB4:
            nbytes = width * height * 4;
            break;
        default:
            break;
    }

    return nbytes;
}

std::shared_ptr<mfxFrameSurface1> create_surface_RGB4(mfxFrameInfo frameInfo,
                                                      mfxU8* out_buf_ptr,
                                                      size_t out_buf_ptr_offset)
{
    mfxU16 surfW = frameInfo.Width * 4;
    mfxU16 surfH = frameInfo.Height;
    (void)surfH;

    std::shared_ptr<mfxFrameSurface1> ret(new mfxFrameSurface1);
    memset(ret.get(), 0, sizeof(mfxFrameSurface1));

    ret->Info = frameInfo ;
    ret->Data.B = out_buf_ptr + out_buf_ptr_offset;
    ret->Data.G = ret->Data.B + 1;
    ret->Data.R = ret->Data.B + 2;
    ret->Data.A = ret->Data.B + 3;
    ret->Data.Pitch = surfW;

    return ret;
}

std::shared_ptr<mfxFrameSurface1> create_surface_other(mfxFrameInfo frameInfo,
                                                       mfxU8* out_buf_ptr,
                                                       size_t out_buf_ptr_offset = 0)
{
    mfxU16 surfH = frameInfo.Height;
    mfxU16 surfW = (frameInfo.FourCC == MFX_FOURCC_P010) ? frameInfo.Width * 2 : frameInfo.Width;

    std::shared_ptr<mfxFrameSurface1> ret(new mfxFrameSurface1);
    memset(ret.get(), 0, sizeof(mfxFrameSurface1));
    
    ret->Info = frameInfo;
    ret->Data.Y     = out_buf_ptr + out_buf_ptr_offset;
    ret->Data.U     = out_buf_ptr + out_buf_ptr_offset + (surfW * surfH);
    ret->Data.V     = ret->Data.U + ((surfW / 2) * (surfH / 2));
    ret->Data.Pitch = surfW;

    return ret;
}


    
LegacyDecodeSession::LegacyDecodeSession(mfxSession sess,
                                         DecoderParams&& decoder_param,
                                         VPLLegacyDecodeEngine::file_ptr&& source) :
    EngineSession(sess, std::move(decoder_param.stream)),
    mfx_decoder_param(std::move(decoder_param.param)),
    source_handle(std::move(source)),
    stop_processing(false),
    curr_surface_ptr(),
    dec_surface_out()
{
}

void LegacyDecodeSession::init_surface_pool(std::vector<std::shared_ptr<mfxFrameSurface1>>&& surf_pool,
                                            mfxFrameAllocRequest&& decRequest) {

    assert(!surf_pool.empty() && "Empty surf pool");
    decoder_surf_pool = std::move(surf_pool);
    request = std::move(decRequest);

    auto surf_ptr = get_free_surface();
    curr_surface_ptr = surf_ptr.get();
}

std::shared_ptr<mfxFrameSurface1> LegacyDecodeSession::get_free_surface() const {

    auto it =
        std::find_if(decoder_surf_pool.begin(), decoder_surf_pool.end(),
                     [](const std::shared_ptr<mfxFrameSurface1>& val) {
        //assert(val && "Surface must exist");
        return !val->Data.Locked;
    });

    // TODO realloc pool
    if (it == decoder_surf_pool.end()) {
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 " - cannot get free surface from pool, size: " +
                                 std::to_string(decoder_surf_pool.size()));
    }

    return *it;
}
VPLLegacyDecodeEngine::VPLLegacyDecodeEngine() {

    GAPI_LOG_INFO(nullptr, "Create Legacy Decode Engine");
    create_pipeline(
        // 1) Reade File
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyDecodeSession &my_sess = static_cast<LegacyDecodeSession&>(sess);
            my_sess.last_status = ReadEncodedStream(my_sess.stream, my_sess.source_handle.get());

            GAPI_LOG_INFO(nullptr, "ReadEncodedStream, session: " << my_sess.session <<
                                   ", error: " <<  my_sess.last_status);
            if (my_sess.last_status != MFX_ERR_NONE) {
                my_sess.source_handle.reset(); //close source
                GAPI_LOG_INFO(nullptr, "Close source, session: " << my_sess.session <<
                                   ", source: " <<  my_sess.source_handle);
            }
            return ExecutionStatus::Continue;
        },
        // 2) enqueue ASYNC decode
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            LegacyDecodeSession &my_sess = static_cast<LegacyDecodeSession&>(sess);
            GAPI_LOG_INFO/*DEBUG*/(nullptr, "session: " << my_sess.session <<
                                            ", sess.last_status: " << my_sess.last_status <<
                                            ", surf_ptr: " << my_sess.curr_surface_ptr);

            sess.last_status =
                    MFXVideoDECODE_DecodeFrameAsync(my_sess.session,
                                                    my_sess.last_status == MFX_ERR_NONE
                                                        ? &my_sess.stream
                                                        : nullptr, /* No more data to read, start decode draining mode*/
                                                    my_sess.curr_surface_ptr,
                                                    &my_sess.dec_surface_out,
                                                    &my_sess.sync);
            GAPI_LOG_INFO/*DEBUG*/(nullptr, "session: " << my_sess.session << ", dec_surface_out: " << my_sess.dec_surface_out <<
                                   ", sess.last_status: " << my_sess.last_status);
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

                        LegacyDecodeSession& my_sess = static_cast<LegacyDecodeSession&>(sess);
                        on_frame_ready(my_sess);
                    }
                } while (sess.last_status == MFX_WRN_IN_EXECUTION);
            }

            return this->process_error(sess.last_status, static_cast<LegacyDecodeSession&>(sess));
        }
    );
}

void VPLLegacyDecodeEngine::initialize_session(mfxSession mfx_session,
                                         DecoderParams&& decoder_param,
                                         file_ptr&& source_handle,
                                         std::unique_ptr<VPLAccelerationPolicy>&& acceleration_policy)
{
    mfxFrameAllocRequest decRequest = {};
    mfxU8 *decOutBuf                = nullptr;
    // Query number required surfaces for decoder
    MFXVideoDECODE_QueryIOSurf(mfx_session, &decoder_param.param, &decRequest);

    // External (application) allocation of decode surfaces
    GAPI_LOG_INFO/*DEBUG*/(nullptr, "Query IOSurf for session: " << mfx_session <<
                                    ", mfxFrameAllocRequest.NumFrameSuggested: " << decRequest.NumFrameSuggested <<
                                    ", mfxFrameAllocRequest.Type: " << decRequest.Type);

    std::vector<std::shared_ptr<mfxFrameSurface1>> decoder_surf_pool (decRequest.NumFrameSuggested * 3);
    
    mfxU32 singleSurfaceSize = GetSurfaceSize(decoder_param.param.mfx.FrameInfo.FourCC,
                                        decoder_param.param.mfx.FrameInfo.Width,
                                        decoder_param.param.mfx.FrameInfo.Height);
    if (!singleSurfaceSize) {
        throw std::runtime_error("Cannot determine surface size for: fourCC" +
                                 std::to_string(decoder_param.param.mfx.FrameInfo.FourCC) +
                                 ", width: " + std::to_string(decoder_param.param.mfx.FrameInfo.Width) +
                                 ", height: " + std::to_string(decoder_param.param.mfx.FrameInfo.Height));
    }

    size_t framePoolBufSize = static_cast<size_t>(singleSurfaceSize) * decoder_surf_pool.size();

    GAPI_LOG_INFO/*DEBUG*/(nullptr, "Session: " << mfx_session <<
                                    ". Allocate OutBuf memory "
                                    "singleSurfaceSize: " << singleSurfaceSize <<
                                    ", framePoolBufSize: " << framePoolBufSize);

    // TODO use accel policy
    (void)acceleration_policy;
    decOutBuf = reinterpret_cast<mfxU8 *>(calloc(framePoolBufSize, 1));

    // create surfaces
    for (size_t i = 0; i < decoder_surf_pool.size(); i++) {
        size_t buf_offset = static_cast<size_t>(i) * singleSurfaceSize;

        std::shared_ptr<mfxFrameSurface1> surf_ptr;
        if (decoder_param.param.mfx.FrameInfo.FourCC == MFX_FOURCC_RGB4) {
            surf_ptr = create_surface_RGB4(decoder_param.param.mfx.FrameInfo,
                                           decOutBuf, buf_offset);
        } else {
            surf_ptr = create_surface_other(decoder_param.param.mfx.FrameInfo,
                                            decOutBuf, buf_offset);
        }
        
        decoder_surf_pool[i] = std::move(surf_ptr);
    }

    // create session
    std::shared_ptr<LegacyDecodeSession> sess_ptr =
                register_session<LegacyDecodeSession>(mfx_session,
                                                      std::move(decoder_param),
                                                      std::move(source_handle));

    // TODO Use common pool for all sessions ?
    sess_ptr->init_surface_pool(std::move(decoder_surf_pool), std::move(decRequest));
    sess_ptr->acceleration_policy = std::move(acceleration_policy);
}

VPLProcessingEngine::ExecutionStatus VPLLegacyDecodeEngine::execute_op(operation_t& op, EngineSession& sess) {
    return op(sess);
}
    
void VPLLegacyDecodeEngine::on_frame_ready(LegacyDecodeSession& sess)
{
    mfxFrameInfo *info = &sess.dec_surface_out->Info;
    mfxFrameData *data = &sess.dec_surface_out->Data;
    size_t w = info->Width;
    size_t h = info->Height;
    size_t p = data->Pitch;


    GAPI_LOG_INFO/*DEBUG*/(nullptr, "session: " << sess.session << ", surface: " << sess.dec_surface_out <<
                           ", w: " << w << ", h: " << h << ", p: " << p);
    
    // manage memory ownership rely on acceleration policy 
    auto frame_adapter = sess.acceleration_policy->create_frame_adapter(sess.dec_surface_out);
    ready_frames.push(cv::MediaFrame(std::move(frame_adapter)));
}

VPLProcessingEngine::ExecutionStatus VPLLegacyDecodeEngine::process_error(mfxStatus status, LegacyDecodeSession& sess)
{
    GAPI_LOG_INFO/*DEBUG*/(nullptr, "status: " << status);

    switch (status) {
        case MFX_ERR_NONE:
            return ExecutionStatus::Continue; 
        case MFX_ERR_MORE_DATA: // The function requires more bitstream at input before decoding can proceed
            GAPI_LOG_INFO/*DEBUG*/(nullptr, "MFX_ERR_MORE_DATA for session: " << sess.session <<
                                            ", source: " << sess.source_handle);
            if (!sess.source_handle) {
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

            mfxFrameSurface1 *oldSurface = sess.curr_surface_ptr;
            try
            {
                auto surf_ptr = sess.get_free_surface();
                sess.curr_surface_ptr = surf_ptr.get();

                GAPI_LOG_INFO/*DEBUG*/(nullptr, "MFX_ERR_MORE_SURFACE for session: " << sess.session <<
                                             ", old surface: " << oldSurface <<
                                             ", new surface: "<< sess.curr_surface_ptr);
                return ExecutionStatus::Continue; 
            } catch (const std::exception& ex)
            {
                GAPI_LOG_WARNING(nullptr, "MFX_ERR_MORE_SURFACE for session: " << sess.session <<
                                             ", Not processed, error: " << ex.what());
            }
            break;
        }
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
