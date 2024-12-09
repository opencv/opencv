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

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GEXECUTOR_HPP
