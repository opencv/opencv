// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL

#include <chrono>
#include <exception>

#include "streaming/onevpl/engine/transcode/transcode_session.hpp"
#include "streaming/onevpl/engine/transcode/transcode_engine_legacy.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/utils.hpp"

#include "logger.hpp"
namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
LegacyTranscodeSession::LegacyTranscodeSession(mfxSession sess,
                                               DecoderParams&& decoder_param,
                                               TranscoderParams&& transcoder_param,
                                               std::shared_ptr<IDataProvider> provider) :
    LegacyDecodeSession(sess, std::move(decoder_param), std::move(provider)),
    mfx_transcoder_param(std::move(transcoder_param.param))
{
}

LegacyTranscodeSession::~LegacyTranscodeSession()
{
    GAPI_LOG_INFO(nullptr, "Close Transcode for session: " << session);
    MFXVideoVPP_Close(session);
}

void LegacyTranscodeSession::init_transcode_surface_pool(VPLAccelerationPolicy::pool_key_t key) {
    GAPI_Assert(key && "Init transcode pull with empty key");
    vpp_out_pool_id = key;
}

void LegacyTranscodeSession::swap_transcode_surface(VPLLegacyTranscodeEngine& engine) {
    VPLAccelerationPolicy* acceleration_policy = engine.get_accel();
    GAPI_Assert(acceleration_policy && "Empty acceleration_policy");
    request_free_surface(session, vpp_out_pool_id, *acceleration_policy, vpp_surface_ptr);
}

const mfxFrameInfo& LegacyTranscodeSession::get_video_param() const {
    return mfx_transcoder_param.vpp.Out;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
