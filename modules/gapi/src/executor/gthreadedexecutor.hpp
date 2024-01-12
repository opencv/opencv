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

class IslandActor {
public:
    using Ptr = std::shared_ptr<IslandActor>;
    IslandActor(const std::vector<RcDesc>          &in_objects,
                const std::vector<RcDesc>          &out_objects,
                std::shared_ptr<GIslandExecutable> isl_exec,
                Mag                                &res,
                std::mutex                         &m);

    void run();
    void verify();
    std::shared_ptr<GIslandExecutable> exec() { return m_isl_exec; }

private:
    std::vector<RcDesc>                m_in_objs;
    std::vector<RcDesc>                m_out_objs;
    std::shared_ptr<GIslandExecutable> m_isl_exec;
    Mag                                &m_res;
    std::exception_ptr                 m_e;
    std::mutex                         &m_mutex; // Mag protection
};

class Task;
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

    Mag                       m_res;
    std::vector<DataDesc>     m_slots;
    cv::gapi::own::ThreadPool m_tp;
    std::mutex                m_mutex;

    std::vector<IslandActor::Ptr>      m_actors;
    std::vector<std::shared_ptr<Task>> m_initial_tasks;
    std::unordered_map< ade::NodeHandle
                      , std::shared_ptr<Task>
                      , ade::HandleHasher<ade::Node>> m_tasks;
};

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GTHREADEDEXECUTOR_HPP
