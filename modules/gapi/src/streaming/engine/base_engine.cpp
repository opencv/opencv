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

    GAPI_LOG_INFO/*DEBUG*/(nullptr, "[" << session <<"] start op id: " << exec_data.op_id);
    ExecutionStatus status = execute_op(pipeline.at(exec_data.op_id), *processing_session);
    size_t old_op_id = exec_data.op_id++;
    if (exec_data.op_id == pipeline.size())
    {
        exec_data.op_id = 0;
    }
    GAPI_LOG_INFO/*DEBUG*/(nullptr, "[" << session <<"] finish op id: " << old_op_id <<
                                    ", " << processing_session->error_code_to_str() <<
                                    ", " << VPLProcessingEngine::status_to_string(status) <<
                                    ", next op id: " << exec_data.op_id);

    if (status == ExecutionStatus::Failed) {

        GAPI_LOG_WARNING(nullptr, "Operation for session: " << session <<
                                  ", " << VPLProcessingEngine::status_to_string(status) <<
                                  " - remove it");
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
        case ExecutionStatus::Continue: return "CONTINUE";
        case ExecutionStatus::Processed: return "PROCESSED";
        case ExecutionStatus::Failed: return "FAILED";
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
