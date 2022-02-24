// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation
#ifdef HAVE_ONEVPL

#include "streaming/onevpl/engine/engine_session.hpp"
#include "streaming/onevpl/utils.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

EngineSession::EngineSession(mfxSession sess) :
    session(sess) {
}

EngineSession::~EngineSession()
{
    GAPI_LOG_INFO(nullptr, "Close session: " << session);
    MFXClose(session);
}

std::string EngineSession::error_code_to_str() const
{
    return mfxstatus_to_string(last_status);
}

void EngineSession::request_free_surface(mfxSession session,
                                         VPLAccelerationPolicy::pool_key_t key,
                                         VPLAccelerationPolicy &acceleration_policy,
                                         std::weak_ptr<Surface> &surface_to_exchange,
                                         bool reset_if_not_found) {
    try {
        auto cand = acceleration_policy.get_free_surface(key).lock();

        GAPI_LOG_DEBUG(nullptr, "[" << session << "] swap surface"
                                ", old: " << (!surface_to_exchange.expired()
                                              ? surface_to_exchange.lock()->get_handle()
                                              : nullptr) <<
                                ", new: "<< cand->get_handle());

        surface_to_exchange = cand;
    } catch (const std::runtime_error& ex) {
        GAPI_LOG_WARNING(nullptr, "[" << session << "] error: " << ex.what());
        if (reset_if_not_found) {
            surface_to_exchange.reset();
        }
        // Delegate exception processing on caller side
        throw;
    }
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
