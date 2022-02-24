// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL

#include <chrono>
#include <exception>

#include "streaming/onevpl/engine/decode/decode_session.hpp"
#include "streaming/onevpl/engine/decode/decode_engine_legacy.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/utils.hpp"

#include "logger.hpp"
namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
LegacyDecodeSession::LegacyDecodeSession(mfxSession sess,
                                         DecoderParams&& decoder_param,
                                         std::shared_ptr<IDataProvider> provider) :
    EngineSession(sess),
    mfx_decoder_param(std::move(decoder_param.param)),
    data_provider(std::move(provider)),
    stream(std::move(decoder_param.stream)),
    processing_surface_ptr(),
    sync_queue(),
    decoded_frames_count()
{
}

LegacyDecodeSession::~LegacyDecodeSession()
{
    GAPI_LOG_INFO(nullptr, "Close Decode for session: " << session);
    MFXVideoDECODE_Close(session);
}

void LegacyDecodeSession::swap_decode_surface(VPLLegacyDecodeEngine& engine) {
    VPLAccelerationPolicy* acceleration_policy = engine.get_accel();
    GAPI_Assert(acceleration_policy && "Empty acceleration_policy");
    request_free_surface(session, decoder_pool_id, *acceleration_policy, processing_surface_ptr);
}

void LegacyDecodeSession::init_surface_pool(VPLAccelerationPolicy::pool_key_t key) {
    GAPI_Assert(key && "Init decode pull with empty key");
    decoder_pool_id = key;
}

Data::Meta LegacyDecodeSession::generate_frame_meta() {
    const auto now = std::chrono::system_clock::now();
    const auto dur = std::chrono::duration_cast<std::chrono::microseconds>
                (now.time_since_epoch());
    Data::Meta meta {
                        {cv::gapi::streaming::meta_tag::timestamp, int64_t{dur.count()} },
                        {cv::gapi::streaming::meta_tag::seq_id, int64_t{decoded_frames_count++}}
                    };
    return meta;
}

const mfxFrameInfo& LegacyDecodeSession::get_video_param() const {
    return mfx_decoder_param.mfx.FrameInfo;
}

IDataProvider::mfx_bitstream *LegacyDecodeSession::get_mfx_bitstream_ptr() {
    return (data_provider || (stream && stream->DataLength)) ?
            stream.get() : nullptr;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
