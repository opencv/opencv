#include "streaming/engine/base_engine.hpp"

namespace cv {
namespace gapi {
namespace wip {

void VPLProcessingEngine::process(mfxSession session) {
    auto sess_it = sessions.find(session);
    if (sess_it == sessions.end()) { abort();}

    session_ptr processing_session = sess_it->second;
    EngineSession::ExecutionStatus status = processing_session->execute();

    if (status == EngineSession::ExecutionStatus::Failed) {
        sessions.erase(sess_it);
    }
}

size_t VPLProcessingEngine::get_ready_frames_count() const
{
    return ready_frames.size();
}
void VPLProcessingEngine::get_frame(Data &data)
{
    data = ready_frames.front();
    ready_frames.pop();
}
} // namespace wip
} // namespace gapi
} // namespace cv
