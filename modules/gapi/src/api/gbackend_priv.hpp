// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef GAPI_API_GBACKEND_PRIV_HPP
#define GAPI_API_GBACKEND_PRIV_HPP

#include <memory>
#include <unordered_set>

#include <ade/graph.hpp>
#include <ade/passes/pass_base.hpp> // passes::PassContext
#include <ade/execution_engine/execution_engine.hpp> // ..SetupContext

#include "opencv2/gapi/gcommon.hpp"
#include "opencv2/gapi/gkernel.hpp"

#include "compiler/gmodel.hpp"
#include "compiler/gislandmodel.hpp"

namespace cv
{
namespace gimpl
{
    class GBackend;
    class GIslandExecutable;
} // namespace gimpl
} // namespace cv

// GAPI_EXPORTS is here to make tests build on Windows
class GAPI_EXPORTS cv::gapi::GBackend::Priv
{
public:
    using EPtr = std::unique_ptr<cv::gimpl::GIslandExecutable>;

    virtual void unpackKernel(ade::Graph            &graph,
                              const ade::NodeHandle &op_node,
                              const GKernelImpl     &impl);

    // FIXME: since backends are not passed to ADE anymore,
    // there's no need in having both cv::gimpl::GBackend
    // and cv::gapi::GBackend - these two things can be unified
    // NOTE - nodes are guaranteed to be topologically sorted.

    // NB: This method is deprecated
    virtual EPtr compile(const ade::Graph   &graph,
                         const GCompileArgs &args,
                         const std::vector<ade::NodeHandle> &nodes) const;


    virtual EPtr compile(const ade::Graph   &graph,
                         const GCompileArgs &args,
                         const std::vector<ade::NodeHandle> &nodes,
                         const std::vector<cv::gimpl::Data>& ins_data,
                         const std::vector<cv::gimpl::Data>& outs_data) const;

    // Ask backend to provide general backend-specific compiler passes
    virtual void addBackendPasses(ade::ExecutionEngineSetupContext &);

    // Ask backend to put extra meta-sensitive backend passes Since
    // the inception of Streaming API one can compile graph without
    // meta information, so if some passes depend on this information,
    // they are called when meta information becomes available.
    virtual void addMetaSensitiveBackendPasses(ade::ExecutionEngineSetupContext &);

    virtual cv::GKernelPackage auxiliaryKernels() const;

    // Ask backend if it has a custom control over island fusion process
    // This method is quite redundant but there's nothing better fits
    // the current fusion process. By default, [existing] backends don't
    // control the merge.
    // FIXME: Refactor to a single entity?
    virtual bool controlsMerge() const;

    // Ask backend if it is ok to merge these two islands connected
    // via a data slot. By default, [existing] backends allow to merge everything.
    // FIXME: Refactor to a single entity?
    // FIXME: Strip down the type details form graph? (make it ade::Graph?)
    virtual bool allowsMerge(const cv::gimpl::GIslandModel::Graph &g,
                             const ade::NodeHandle &a_nh,
                             const ade::NodeHandle &slot_nh,
                             const ade::NodeHandle &b_nh) const;

    virtual ~Priv() = default;
};

#endif // GAPI_API_GBACKEND_PRIV_HPP
