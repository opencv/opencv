// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2022 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_EXECUTOR_HPP
#define OPENCV_GAPI_STREAMING_EXECUTOR_HPP

#ifdef _MSC_VER
#pragma warning(disable: 4503)  // "decorated name length exceeded"
                                // on concurrent_bounded_queue
#endif

#include <memory> // unique_ptr, shared_ptr
#include <vector>

#include "executor/last_value.hpp"

#include <ade/graph.hpp>

#include "backends/common/gbackend.hpp"

namespace cv {
namespace gimpl {

// FIXME: Currently all GExecutor comments apply also
// to this one. Please document it separately in the future.

// GStreamingExecutor is a state machine described as follows
//
//              setSource() called
//   STOPPED:  - - - - - - - - - ->READY:
//   --------                      ------
//   Initial state                 Input data specified
//   No threads running            Threads are created and IDLE
//   ^                             (Currently our emitter threads
//   :                             are bounded to input data)
//   : stop() called               No processing happending
//   : OR                          :
//   : end-of-stream reached       :  start() called
//   : during pull()/try_pull()    V
//   :                             RUNNING:
//   :                             --------
//   :                             Actual pipeline execution
//    - - - - - - - - - - - - - -  Threads are running
//
class GStreamingExecutor {
protected:
    enum class State {
        STOPPED,
        READY,
        RUNNING,
    } state = State::STOPPED;

    std::unique_ptr<ade::Graph> m_orig_graph;
    std::shared_ptr<ade::Graph> m_island_graph;
    cv::GCompileArgs m_comp_args;
    cv::GMetaArgs m_last_metas;
    util::optional<bool> m_reshapable;

    cv::gimpl::GIslandModel::Graph m_gim; // FIXME: make const?

    cv::GRunArgs m_const_vals;

    void virtual start_impl() = 0;
    bool virtual pull_impl(cv::GRunArgs& this_result) = 0;
    bool virtual try_pull_impl(cv::GRunArgs& this_result) = 0;
    void virtual stop_impl() = 0;
    void virtual wait_shutdown() = 0;

public:
    explicit GStreamingExecutor(std::unique_ptr<ade::Graph> &&g_model,
                                const cv::GCompileArgs &comp_args);
    virtual ~GStreamingExecutor();
    void virtual setSource(GRunArgs &&args) = 0;
    void start();
    bool pull(cv::GRunArgsP &&outs);
    bool try_pull(cv::GRunArgsP &&outs);
    void stop();
    bool running() const;
};

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_EXECUTOR_HPP
