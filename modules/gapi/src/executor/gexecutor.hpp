// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_GEXECUTOR_HPP
#define OPENCV_GAPI_GEXECUTOR_HPP

#include <memory> // unique_ptr, shared_ptr

#include <utility> // tuple, required by magazine
#include <unordered_map> // required by magazine

#include <ade/graph.hpp>

#include "backends/common/gbackend.hpp"

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

class GExecutor
{
protected:
    Mag m_res;
    std::unique_ptr<ade::Graph> m_orig_graph;
    std::shared_ptr<ade::Graph> m_island_graph;

    cv::gimpl::GModel::Graph       m_gm;  // FIXME: make const?
    cv::gimpl::GIslandModel::Graph m_gim; // FIXME: make const?

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
    void run(cv::gimpl::GRuntimeArgs &&args);

    bool canReshape() const;
    void reshape(const GMetaArgs& inMetas, const GCompileArgs& args);

    void prepareForNewStream();

    const GModel::Graph& model() const; // FIXME: make it ConstGraph?
};

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_GEXECUTOR_HPP
