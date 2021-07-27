#include <iterator>

#include "streaming/engine/engine_session.hpp"
#include "streaming/vpl/vpl_utils.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {

    
EngineSession::EngineSession(mfxSession sess, mfxBitstream&& str) :
        session(sess), stream(std::move(str)) {}
EngineSession::~EngineSession()
{
    GAPI_LOG_INFO(nullptr, "Close Decode for session: " << session);
    MFXVideoDECODE_Close(session);

    GAPI_LOG_INFO(nullptr, "Close session: " << session);
    MFXClose(session);
}

std::string EngineSession::error_code_to_str() const
{
    return mfxstatus_to_string(last_status);
}
} // namespace wip
} // namespace gapi
} // namespace cv
