#ifndef GAPI_STREAMING_BASE_ENGINE_HPP
#define GAPI_STREAMING_BASE_ENGINE_HPP

#include "streaming/engine/engine_session.hpp"

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>

namespace cv {
namespace gapi {
namespace wip {

class VPLProcessingEngine {
public:
    using SessionsTable = std::map<mfxSession, EngineSession>;
    SessionsTable sessions;

    template<class ...SessionArgs>
    void register_session(mfxSession session, mfxBitstream stream, SessionArgs&& ...args)
    {
        auto it = sessions.emplace(std::piecewise_construct, 
                                   std::forward_as_tuple(session),
                                   std::forward_as_tuple(session, stream)).first;
        it->second.create_operations(std::forward<SessionArgs>(args)...);
    }

    void process(mfxSession session);
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_BASE_ENGINE_HPP
