// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL
#ifdef HAVE_INF_ENGINE

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
vpp_pp_session::vpp_pp_session(mfxSession sess, const mfxVideoParam& vpp_out_param) :
    EngineSession(sess),
    mfx_vpp_out_param(vpp_out_param),
    procesing_surface_ptr(),
    sync_in_queue(),
    vpp_out_queue(),
    preprocessed_frames_count()
{
}

vpp_pp_session::~vpp_pp_session() {
    GAPI_LOG_INFO(nullptr, "Close VPP for session: " << session);
    MFXVideoVPP_Close(session);
}

Data::Meta vpp_pp_session::generate_frame_meta() {
    const auto now = std::chrono::system_clock::now();
    const auto dur = std::chrono::duration_cast<std::chrono::microseconds>
                (now.time_since_epoch());
    Data::Meta meta {
                        {cv::gapi::streaming::meta_tag::timestamp, int64_t{dur.count()} },
                        {cv::gapi::streaming::meta_tag::seq_id, int64_t{preprocessed_frames_count++}}
                    };
    return meta;
}

void vpp_pp_session::swap_surface(VPPPreprocEngine& engine) {
    VPLAccelerationPolicy* acceleration_policy = engine.get_accel();
    GAPI_Assert(acceleration_policy && "Empty acceleration_policy");
    request_free_surface(session, vpp_pool_id, *acceleration_policy,
                         procesing_surface_ptr, true);
}

void vpp_pp_session::init_surface_pool(VPLAccelerationPolicy::pool_key_t key) {
    GAPI_Assert(key && "Init preproc pull with empty key");
    vpp_pool_id = key;
}

const mfxFrameInfo& vpp_pp_session::get_video_param() const {
    return mfx_vpp_out_param.vpp.Out;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_INF_ENGINE
#endif // HAVE_ONEVPL
