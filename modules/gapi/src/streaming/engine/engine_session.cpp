#include <iterator>

#include "streaming/engine/engine_session.hpp"
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

const char * EngineSession::status_to_string(EngineSession::ExecutionStatus status)
{
    switch(status) {
        case ExecutionStatus::Continue: return "Continue";
        case ExecutionStatus::Processed: return "Processed";
        case ExecutionStatus::Failed: return "Failed";
        default:
            return "UNKNOWN";
    }
}

EngineSession::ExecutionStatus EngineSession::execute() 
{
    GAPI_LOG_INFO/*DEBUG*/(nullptr, "Execute operation for session: " << session <<
                                    ", op id: " << std::distance(operations.begin(), cur_op_it));
    ExecutionStatus status = execute_op(*cur_op_it);

    ++cur_op_it;
    if (cur_op_it == operations.end())
    {
        cur_op_it = operations.begin();
    }
    GAPI_LOG_INFO/*DEBUG*/(nullptr, "Operation for session: " << session <<
                                    ", got result: " << EngineSession::status_to_string (status) <<
                                    ", next op id: " << std::distance(operations.begin(), cur_op_it));
    return status;
}

EngineSession::ExecutionStatus EngineSession::execute_op(EngineSession::operation_t& op)
{
     return op(*this);
}
} // namespace wip
} // namespace gapi
} // namespace cv
