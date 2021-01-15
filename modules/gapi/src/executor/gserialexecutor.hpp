// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_SERIAL_EXECUTOR_HPP
#define OPENCV_GAPI_SERIAL_EXECUTOR_HPP

#include <memory> // unique_ptr, shared_ptr

#include <utility> // tuple, required by magazine
#include <unordered_map> // required by magazine

#include <ade/graph.hpp>

#include "backends/common/gbackend.hpp"
#include "executor/gexecutor.hpp"

namespace cv {
namespace gimpl {

// Graph-level executor interface.
//
// This class specifies API for a "super-executor" which orchestrates
// the overall Island graph execution.
//
// Every Island (subgraph) execution is delegated to a particular
// backend and is done opaquely to the GExecutor.
//
// Inputs to a GExecutor instance are:
// - GIslandModel - a high-level graph model which may be seen as a
//   "procedure" to execute.
//   - GModel - a low-level graph of operations (from which a GIslandModel
//     is projected)
// - GComputation runtime arguments - vectors of input/output objects
//
// Every GExecutor is responsible for
// a. Maintaining non-island (intermediate) data objects within graph
// b. Providing GIslandExecutables with input/output data according to
//    their protocols
// c. Triggering execution of GIslandExecutables when task/data dependencies
//    are met.
//
// By default G-API stores all data on host, and cross-Island
// exchange happens via host buffers (and CV data objects).
//
// Today's exchange data objects are:
// - cv::Mat               - for image buffers
// - cv::Scalar            - for single values (with up to four components inside)
// - cv::detail::VectorRef - an untyped wrapper over std::vector<T>
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
