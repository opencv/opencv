#include "streaming/engine/base_engine.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {

VPLProcessingEngine::ExecutionStatus VPLProcessingEngine::process(mfxSession session) {
    auto sess_it = sessions.find(session);
    if (sess_it == sessions.end()) {

        // TODO remember the last session status
        return ExecutionStatus::Processed;
    }

    session_ptr processing_session = sess_it->second;
    ExecutionData& exec_data = execution_table[session];
    
    GAPI_LOG_INFO/*DEBUG*/(nullptr, "Execute operation for session: " << session <<
                                    ", op id: " << exec_data.op_id);
    ExecutionStatus status = execute_op(pipeline.at(exec_data.op_id), *processing_session);

    ++exec_data.op_id;
    if (exec_data.op_id == pipeline.size())
    {
        exec_data.op_id = 0;
    }
    GAPI_LOG_INFO/*DEBUG*/(nullptr, "Operation for session: " << session <<
                                    ", got result: " << VPLProcessingEngine::status_to_string (status) <<
                                    ", next op id: " << exec_data.op_id);
    if (status == ExecutionStatus::Failed) {
        sessions.erase(sess_it);
        execution_table.erase(session);
    }

    if(status == ExecutionStatus::Processed) {
        sessions.erase(sess_it);
        execution_table.erase(session);
    }

    return status;
}

const char * VPLProcessingEngine::status_to_string(ExecutionStatus status)
{
    switch(status) {
        case ExecutionStatus::Continue: return "Continue";
        case ExecutionStatus::Processed: return "Processed";
        case ExecutionStatus::Failed: return "Failed";
        default:
            return "UNKNOWN";
    }
}

VPLProcessingEngine::ExecutionStatus VPLProcessingEngine::execute_op(operation_t& op, EngineSession& sess)
{
     return op(sess);
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
