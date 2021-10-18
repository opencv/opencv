// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ENGINE_PROCESSING_ENGINE_BASE_HPP
#define GAPI_STREAMING_ONEVPL_ENGINE_PROCESSING_ENGINE_BASE_HPP

#include <queue>
#include "streaming/onevpl/engine/engine_session.hpp"
#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

#ifdef HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

struct VPLAccelerationPolicy;
struct IDataProvider;

// GAPI_EXPORTS for tests
class GAPI_EXPORTS ProcessingEngineBase {
public:
    enum class ExecutionStatus {
        Continue,
        Processed,
        SessionNotFound,
        Failed
    };
    struct ExecutionData {
        size_t op_id = 0;
    };

    using file_ptr = std::unique_ptr<FILE, decltype(&fclose)>;

    using session_ptr = std::shared_ptr<EngineSession>;
    using SessionsTable = std::map<mfxSession, session_ptr>;
    using ExecutionDataTable = std::map<mfxSession, ExecutionData>;

    using frame_t = cv::gapi::wip::Data;
    using frames_container_t = std::queue<frame_t>;
    using operation_t = std::function<ExecutionStatus(EngineSession&)>;

    static const char * status_to_string(ExecutionStatus);

    ProcessingEngineBase(std::unique_ptr<VPLAccelerationPolicy>&& accel);
    virtual ~ProcessingEngineBase();

    virtual void initialize_session(mfxSession mfx_session,
                                    DecoderParams&& decoder_param,
                                    std::shared_ptr<IDataProvider> provider) = 0;

    ExecutionStatus process(mfxSession session);
    size_t get_ready_frames_count() const;
    void get_frame(Data &data);

    const VPLAccelerationPolicy* get_accel() const;
    VPLAccelerationPolicy* get_accel();
protected:
    SessionsTable sessions;
    frames_container_t ready_frames;
    ExecutionDataTable execution_table;

    std::vector<operation_t> pipeline;
    std::unique_ptr<VPLAccelerationPolicy> acceleration_policy;

    virtual ExecutionStatus execute_op(operation_t& op, EngineSession& sess);

    template<class ...Ops>
    void create_pipeline(Ops&&...ops)
    {
        GAPI_DbgAssert(pipeline.empty() && "Pipeline must be empty");
        std::vector<operation_t>({std::forward<Ops>(ops)...}).swap(pipeline);
    }

    template<class SpecificSession, class ...SessionArgs>
    std::shared_ptr<SpecificSession> register_session(mfxSession key,
                                                      SessionArgs&& ...args)
    {
        auto sess_impl = std::make_shared<SpecificSession>(key,
                                                           std::forward<SessionArgs>(args)...);
        sessions.emplace(key, sess_impl);
        execution_table.emplace(key, ExecutionData{});
        return sess_impl;
    }
};


mfxStatus ReadEncodedStream(mfxBitstream &bs, std::shared_ptr<IDataProvider>& data_provider);
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ENGINE_PROCESSING_ENGINE_BASE_HPP
