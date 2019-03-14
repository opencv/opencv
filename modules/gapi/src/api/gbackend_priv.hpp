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
    virtual EPtr compile(const ade::Graph   &graph,
                         const GCompileArgs &args,
                         const std::vector<ade::NodeHandle> &nodes) const;

    virtual void addBackendPasses(ade::ExecutionEngineSetupContext &);

    virtual ~Priv() = default;
};

#endif // GAPI_API_GBACKEND_PRIV_HPP
