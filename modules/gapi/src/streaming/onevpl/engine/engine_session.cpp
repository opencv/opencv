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

EngineSession::EngineSession(mfxSession sess, std::shared_ptr<IDataProvider::mfx_bitstream>&& str) :
        session(sess), stream(std::move(str)) {}
EngineSession::~EngineSession()
{
    GAPI_LOG_INFO(nullptr, "Close session: " << session);
    MFXClose(session);
}

std::string EngineSession::error_code_to_str() const
{
    return mfxstatus_to_string(last_status);
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
