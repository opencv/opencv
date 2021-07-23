#ifndef GAPI_STREAMING_ENGINE_SESSION_HPP
#define GAPI_STREAMING_ENGINE_SESSION_HPP

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
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

    enum class ExecutionStatus {
        Continue,
        Processed,
        Failed
    };
    
    using operation_t = std::function<ExecutionStatus(EngineSession&)>;
    EngineSession(mfxSession sess, mfxBitstream&& str);
    virtual ~EngineSession();
    
    template<class ...Ops>
    void create_operations(Ops&&...ops)
    {
        std::vector<operation_t>({std::forward<Ops>(ops)...}).swap(operations);
        cur_op_it = operations.begin();
    }

    ExecutionStatus execute();

    static const char * status_to_string(ExecutionStatus);
private:
    virtual ExecutionStatus execute_op(operation_t& op);
    std::vector<operation_t> operations;
    typename std::vector<operation_t>::iterator cur_op_it;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ENGINE_SESSION_HPP
