// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL

#include <chrono>
#include <exception>

#include "streaming/onevpl/engine/preproc/preproc_session.hpp"
#include "streaming/onevpl/engine/preproc/preproc_engine.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/utils.hpp"

#include "logger.hpp"
namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
PreprocSession::PreprocSession(mfxSession sess, const mfxVideoParam& vpp_out_param) :
    EngineSession(sess, {}),
    mfx_vpp_out_param(vpp_out_param),
    procesing_surface_ptr(),
    sync_queue()
{
}

PreprocSession::~PreprocSession() {
    GAPI_LOG_INFO(nullptr, "Close VPP for session: " << session);
    MFXVideoVPP_Close(session);
}

void PreprocSession::swap_surface(VPPPreprocEngine& engine) {
    VPLAccelerationPolicy* acceleration_policy = engine.get_accel();
    GAPI_Assert(acceleration_policy && "Empty acceleration_policy");
    try {
        auto cand = acceleration_policy->get_free_surface(vpp_pool_id).lock();

        GAPI_LOG_DEBUG(nullptr, "[" << session << "] swap surface"
                                ", old: " << (!procesing_surface_ptr.expired()
                                              ? procesing_surface_ptr.lock()->get_handle()
                                              : nullptr) <<
                                ", new: "<< cand->get_handle());

        procesing_surface_ptr = cand;
    } catch (const std::runtime_error& ex) {
        GAPI_LOG_WARNING(nullptr, "[" << session << "] error: " << ex.what());

        // Delegate exception processing on caller
        throw;
    }
}

void PreprocSession::init_surface_pool(VPLAccelerationPolicy::pool_key_t key) {
    GAPI_Assert(key && "Init preproc pull with empty key");
    vpp_pool_id = key;
}

const mfxFrameInfo& PreprocSession::get_video_param() const {
    return mfx_vpp_out_param.vpp.Out;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
