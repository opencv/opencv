// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_GEXECUTOR_HPP
#define OPENCV_GAPI_GEXECUTOR_HPP

#include <utility> // tuple, required by magazine
#include <unordered_map> // required by magazine

#include "executor/gabstractexecutor.hpp"
#include "executor/thread_pool.hpp"

namespace cv {
namespace gimpl {

class GExecutor final: public GAbstractExecutor
{
protected:
    Mag m_res;

    // FIXME: Naive executor details are here for now
    // but then it should be moved to another place
    struct OpDesc
    {
        std::vector<RcDesc> in_objects;
        std::vector<RcDesc> out_objects;
        std::shared_ptr<GIslandExecutable> isl_exec;
    };
    std::vector<OpDesc> m_ops;

    struct DataDesc
    {
        ade::NodeHandle slot_nh;
        ade::NodeHandle data_nh;
    };
    std::vector<DataDesc> m_slots;

    class Input;
    class Output;

    void initResource(const ade::NodeHandle &nh, const ade::NodeHandle &orig_nh); // FIXME: shouldn't it be RcDesc?

public:
    explicit GExecutor(std::unique_ptr<ade::Graph> &&g_model);
    void run(cv::gimpl::GRuntimeArgs &&args) override;

    bool canReshape() const override;
    void reshape(const GMetaArgs& inMetas, const GCompileArgs& args) override;

    void prepareForNewStream() override;
};

namespace wip {

struct Task {
    void operator()();

    std::vector<RcDesc> in_objects;
    std::vector<RcDesc> out_objects;
    std::shared_ptr<GIslandExecutable> isl_exec;
    Mag& res;
    std::mutex &m;
    cv::gapi::own::ThreadPool &tp;

    uint32_t num_deps;
    uint32_t ready_deps;
    std::unordered_set<Task*> dependents;
};


class GExecutor final: public GAbstractExecutor
{
protected:

    // FIXME: Naive executor details are here for now
    // but then it should be moved to another place
    struct DataDesc
    {
        ade::NodeHandle slot_nh;
        ade::NodeHandle data_nh;
    };

    void initResource(const ade::NodeHandle &nh, const ade::NodeHandle &orig_nh); // FIXME: shouldn't it be RcDesc?

    Mag                       m_res;
    std::vector<DataDesc>     m_slots;
    cv::gapi::own::ThreadPool m_tp{4u};
    std::mutex                m_mutex;

    std::unordered_map<ade::NodeHandle, Task, ade::HandleHasher<ade::Node>> m_tasks;
public:
    class Input;
    class Output;

    explicit GExecutor(std::unique_ptr<ade::Graph> &&g_model);
    void run(cv::gimpl::GRuntimeArgs &&args) override;

    bool canReshape() const override;
    void reshape(const GMetaArgs& inMetas, const GCompileArgs& args) override;

    void prepareForNewStream() override;
};
} // namespace wip

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GEXECUTOR_HPP
