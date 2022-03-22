// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2022 Intel Corporation


#ifndef OPENCV_GAPI_SERIAL_EXECUTOR_HPP
#define OPENCV_GAPI_SERIAL_EXECUTOR_HPP

#include <memory> // unique_ptr, shared_ptr

#include <ade/graph.hpp>

#include "backends/common/gbackend.hpp"
#include "executor/gexecutor.hpp"

namespace cv {
namespace gimpl {

// Graph-level executor serial implementation.
//
// This class implements naive Island graph execution model which is similar
// to current CPU (OpenCV) plugin execution model.
// The execution steps are:
// 1. Allocate all internal resources first (NB - CPU plugin doesn't do it)
// 2. Put input/output GComputation arguments to the storage
// 3. For every Island, prepare vectors of input/output parameter descs
// 4. Iterate over a list of operations (sorted in the topological order)
// 5. For every operation, form a list of input/output data objects
// 6. Run GIslandExecutable
// 7. writeBack
//
class GSerialExecutor final : public GExecutor
{
protected:
    Mag m_res;

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

private:
    void runImpl(cv::gimpl::GRuntimeArgs &&args) override;

public:
    explicit GSerialExecutor(std::unique_ptr<ade::Graph> &&g_model);

    bool canReshape() const override;
    void reshape(const GMetaArgs& inMetas, const GCompileArgs& args) override;

    void prepareForNewStream() override;

};

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GEXECUTOR_HPP
