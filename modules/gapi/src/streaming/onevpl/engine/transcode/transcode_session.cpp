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
    try {
        auto cand = acceleration_policy->get_free_surface(vpp_out_pool_id).lock();

        GAPI_LOG_DEBUG(nullptr, "[" << session << "] swap surface"
                                ", old: " << (!vpp_surface_ptr.expired()
                                              ? vpp_surface_ptr.lock()->get_handle()
                                              : nullptr) <<
                                ", new: "<< cand->get_handle());

        vpp_surface_ptr = cand;
    } catch (const std::runtime_error& ex) {
        GAPI_LOG_WARNING(nullptr, "[" << session << "] error: " << ex.what());

        // Delegate exception processing on caller
        throw;
    }
}

const mfxFrameInfo& LegacyTranscodeSession::get_video_param() const {
    return mfx_transcoder_param.vpp.Out;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
