#ifndef GAPI_STREAMING_ENGINE_SESSION_HPP
#define GAPI_STREAMING_ENGINE_SESSION_HPP

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>

namespace cv {
namespace gapi {
namespace wip {

struct EngineSession {
    mfxSession session;
    mfxBitstream stream;
    mfxSyncPoint sync;
    mfxStatus last_status;

    using operation_t = std::function<void(EngineSession&)>;
    EngineSession(mfxSession sess, mfxBitstream str) :
        session(sess), stream(str) {}
		
    template<class ...Ops>
    void create_operations(Ops&&...ops)
    {
        operations = std::vector<operation_t>({ops...});
        cur_op_it = operations.begin();
    }

    void execute() 
    {
        (*cur_op_it) (*this);

        ++cur_op_it;
        if (cur_op_it == operations.end())
        {
            cur_op_it = operations.begin();
        }
    }

private:
    std::vector<operation_t> operations;
    typename std::vector<operation_t>::iterator cur_op_it;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ENGINE_SESSION_HPP
