#ifndef GAPI_STREAMING_BASE_ENGINE_HPP
#define GAPI_STREAMING_BASE_ENGINE_HPP

#include <queue>
#include "streaming/engine/engine_session.hpp"

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>

namespace cv {
namespace gapi {
namespace wip {

class VPLProcessingEngine {
public:
    using session_ptr = std::shared_ptr<EngineSession>;
    using SessionsTable = std::map<mfxSession, session_ptr>;

    using frame_t = cv::gapi::wip::Data;
    using frames_container_t = std::queue<frame_t>;
    
    template<class SpecificSession, class ...SessionArgs>
    std::shared_ptr<SpecificSession> register_session(mfxSession session, mfxBitstream&& stream, SessionArgs&& ...args)
    {
        auto sess_impl = std::make_shared<SpecificSession>(session, std::move(stream));
        sess_impl->create_operations(std::forward<SessionArgs>(args)...);
        
        sessions.emplace(session, sess_impl);
        return sess_impl;
    }

    void process(mfxSession session);

    size_t get_ready_frames_count() const;
    void get_frame(Data &data);
protected:
    SessionsTable sessions;
    frames_container_t ready_frames;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_BASE_ENGINE_HPP
