// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024 Intel Corporation


#ifndef OPENCV_GAPI_GTHREADEDEXECUTOR_HPP
#define OPENCV_GAPI_GTHREADEDEXECUTOR_HPP

#include <utility> // tuple, required by magazine
#include <unordered_map> // required by magazine

#include "executor/gabstractexecutor.hpp"
#include "executor/thread_pool.hpp"

namespace cv {
namespace gimpl {

class Task;
class TaskManager {
public:
    using F = std::function<void()>;

    std::shared_ptr<Task> createTask(F &&f, std::vector<std::shared_ptr<Task>> &&producers);
    void scheduleAndWait(cv::gapi::own::ThreadPool& tp);

private:
    std::vector<std::shared_ptr<Task>> m_all_tasks;
    std::vector<std::shared_ptr<Task>> m_initial_tasks;
};

struct GraphState {
    Mag mag;
    std::mutex m;
};

class IslandActor;
class GThreadedExecutor final: public GAbstractExecutor {
public:
    class Input;
    class Output;

    explicit GThreadedExecutor(const uint32_t num_threads,
                               std::unique_ptr<ade::Graph> &&g_model);
    void run(cv::gimpl::GRuntimeArgs &&args) override;

    bool canReshape() const override;
    void reshape(const GMetaArgs& inMetas, const GCompileArgs& args) override;

    void prepareForNewStream() override;

private:
    struct DataDesc
    {
        ade::NodeHandle slot_nh;
        ade::NodeHandle data_nh;
    };

    void initResource(const ade::NodeHandle &nh, const ade::NodeHandle &orig_nh);

    GraphState                                m_state;
    std::vector<DataDesc>                     m_slots;
    cv::gapi::own::ThreadPool                 m_thread_pool;
    TaskManager                               m_task_manager;
    std::vector<std::shared_ptr<IslandActor>> m_actors;
};

class GThreadedExecutor::Input final: public GIslandExecutable::IInput
{
public:
    Input(GraphState& state, const std::vector<RcDesc> &rcs);

private:
    virtual StreamMsg get() override;
    virtual StreamMsg try_get() override { return get(); }

private:
    GraphState& m_state;
};

class GThreadedExecutor::Output final: public GIslandExecutable::IOutput
{
public:
    Output(GraphState &state, const std::vector<RcDesc> &rcs);
    void verify();

private:
    GRunArgP get(int idx) override;
    void post(cv::GRunArgP&&, const std::exception_ptr& e) override;
    void post(Exception&& ex) override;
    void post(EndOfStream&&) override {};
    void meta(const GRunArgP &out, const GRunArg::Meta &m) override;

private:
    GraphState& m_state;
    std::unordered_map<const void*, int> m_out_idx;
    std::exception_ptr m_eptr;
};

class IslandActor {
public:
    using Ptr = std::shared_ptr<IslandActor>;
    IslandActor(const std::vector<RcDesc>          &in_objects,
                const std::vector<RcDesc>          &out_objects,
                std::shared_ptr<GIslandExecutable> isl_exec,
                GraphState                         &state);

    void run();
    void verify();
    std::shared_ptr<GIslandExecutable> exec() { return m_isl_exec; }

private:
    std::shared_ptr<GIslandExecutable> m_isl_exec;
    GThreadedExecutor::Input           m_inputs;
    GThreadedExecutor::Output          m_outputs;
};


} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GTHREADEDEXECUTOR_HPP
