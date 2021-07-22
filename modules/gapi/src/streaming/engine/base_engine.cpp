#include "streaming/engine/base_engine.hpp"

namespace cv {
namespace gapi {
namespace wip {

void VPLProcessingEngine::process(mfxSession session) {
    auto sess_it = sessions.find(session);
    if (sess_it == sessions.end()) { abort();}

    EngineSession &processing_session = sess_it->second;
    processing_session.execute();
}
} // namespace wip
} // namespace gapi
} // namespace cv
