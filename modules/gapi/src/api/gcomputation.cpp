// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"
#include <algorithm> // remove_if
#include <cctype>    // isspace (non-locale version)
#include <ade/util/algorithm.hpp>
#include <ade/util/zip_range.hpp>   // util::indexed

#include "logger.hpp" // GAPI_LOG

#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gkernel.hpp>

#include "api/gcomputation_priv.hpp"
#include "api/gcall_priv.hpp"
#include "api/gnode_priv.hpp"

#include "compiler/gmodelbuilder.hpp"
#include "compiler/gcompiler.hpp"
#include "compiler/gcompiled_priv.hpp"
#include "compiler/gstreaming_priv.hpp"

static cv::GTypesInfo collectInfo(const cv::gimpl::GModel::ConstGraph& g,
                                  const std::vector<ade::NodeHandle>& nhs) {
    cv::GTypesInfo info;
    info.reserve(nhs.size());

    ade::util::transform(nhs, std::back_inserter(info), [&g](const ade::NodeHandle& nh) {
        const auto& data = g.metadata(nh).get<cv::gimpl::Data>();
        return cv::GTypeInfo{data.shape, data.kind, data.ctor};
    });

    return info;
}

// NB: This function is used to collect graph input/output info.
// Needed for python bridge to unpack inputs and constructs outputs properly.
static cv::GraphInfo::Ptr collectGraphInfo(const cv::GComputation::Priv& priv)
{
    auto g = cv::gimpl::GCompiler::makeGraph(priv);
    cv::gimpl::GModel::ConstGraph cgr(*g);
    auto in_info  = collectInfo(cgr, cgr.metadata().get<cv::gimpl::Protocol>().in_nhs);
    auto out_info = collectInfo(cgr, cgr.metadata().get<cv::gimpl::Protocol>().out_nhs);
    return cv::GraphInfo::Ptr(new cv::GraphInfo{std::move(in_info), std::move(out_info)});
}

// cv::GComputation private implementation /////////////////////////////////////
// <none>

// cv::GComputation public implementation //////////////////////////////////////
cv::GComputation::GComputation(const Generator& gen)
    : m_priv(gen().m_priv)
{
}

cv::GComputation::GComputation(GMat in, GMat out)
    : cv::GComputation(cv::GIn(in), cv::GOut(out))
{
}


cv::GComputation::GComputation(GMat in, GScalar out)
    : cv::GComputation(cv::GIn(in), cv::GOut(out))
{
}

cv::GComputation::GComputation(GMat in1, GMat in2, GMat out)
    : cv::GComputation(cv::GIn(in1, in2), cv::GOut(out))
{
}

cv::GComputation::GComputation(GMat in1, GMat in2, GScalar out)
    : cv::GComputation(cv::GIn(in1, in2), cv::GOut(out))
{
}

cv::GComputation::GComputation(const std::vector<GMat> &ins,
                               const std::vector<GMat> &outs)
    : m_priv(new Priv())
{
    Priv::Expr e;
    const auto wrap = [](cv::GMat m) { return GProtoArg(m); };
    ade::util::transform(ins,  std::back_inserter(e.m_ins),  wrap);
    ade::util::transform(outs, std::back_inserter(e.m_outs), wrap);
    m_priv->m_shape = std::move(e);
}

cv::GComputation::GComputation(cv::GProtoInputArgs &&ins,
                               cv::GProtoOutputArgs &&outs)
    : m_priv(new Priv())
{
    m_priv->m_shape = Priv::Expr{
          std::move(ins.m_args)
        , std::move(outs.m_args)
    };
}

cv::GComputation::GComputation(cv::gapi::s11n::IIStream &is)
    : m_priv(new Priv())
{
    m_priv->m_shape = gapi::s11n::deserialize(is);
}

void cv::GComputation::serialize(cv::gapi::s11n::IOStream &os) const
{
    // Build a basic GModel and write the whole thing to the stream
    auto pG = cv::gimpl::GCompiler::makeGraph(*m_priv);
    std::vector<ade::NodeHandle> nhs(pG->nodes().begin(), pG->nodes().end());
    gapi::s11n::serialize(os, *pG, nhs);
}


cv::GCompiled cv::GComputation::compile(GMetaArgs &&metas, GCompileArgs &&args)
{
    // FIXME: Cache gcompiled per parameters here?
    cv::gimpl::GCompiler comp(*this, std::move(metas), std::move(args));
    return comp.compile();
}

cv::GStreamingCompiled cv::GComputation::compileStreaming(GMetaArgs &&metas, GCompileArgs &&args)
{
    cv::gimpl::GCompiler comp(*this, std::move(metas), std::move(args));
    return comp.compileStreaming();
}

cv::GStreamingCompiled cv::GComputation::compileStreaming(GCompileArgs &&args)
{
    // NB: Used by python bridge
    if (!m_priv->m_info)
    {
        m_priv->m_info = collectGraphInfo(*m_priv);
    }

    cv::gimpl::GCompiler comp(*this, {}, std::move(args));
    auto compiled = comp.compileStreaming();

    compiled.priv().setInInfo(m_priv->m_info->inputs);
    compiled.priv().setOutInfo(m_priv->m_info->outputs);

    return compiled;
}

cv::GStreamingCompiled cv::GComputation::compileStreaming(const cv::detail::ExtractMetaCallback &callback,
                                                                GCompileArgs                   &&args)
{
    // NB: Used by python bridge
    if (!m_priv->m_info)
    {
        m_priv->m_info = collectGraphInfo(*m_priv);
    }

    auto ins = callback(m_priv->m_info->inputs);
    cv::gimpl::GCompiler comp(*this, std::move(ins), std::move(args));
    auto compiled = comp.compileStreaming();
    compiled.priv().setInInfo(m_priv->m_info->inputs);
    compiled.priv().setOutInfo(m_priv->m_info->outputs);

    return compiled;
}

// FIXME: Introduce similar query/test method for GMetaArgs as a building block
// for functions like this?
static bool formats_are_same(const cv::GMetaArgs& metas1, const cv::GMetaArgs& metas2)
{
    return std::equal(metas1.cbegin(), metas1.cend(), metas2.cbegin(),
                      [](const cv::GMetaArg& meta1, const cv::GMetaArg& meta2) {
                          if (meta1.index() == meta2.index() && meta1.index() == cv::GMetaArg::index_of<cv::GMatDesc>())
                          {
                              const auto& desc1 = cv::util::get<cv::GMatDesc>(meta1);
                              const auto& desc2 = cv::util::get<cv::GMatDesc>(meta2);

                              // comparison by size is omitted
                              return (desc1.chan  == desc2.chan &&
                                      desc1.depth == desc2.depth);
                          }
                          else
                          {
                              return meta1 == meta2;
                          }
                     });
}

void cv::GComputation::recompile(GMetaArgs&& in_metas, GCompileArgs &&args)
{
    // FIXME Graph should be recompiled when GCompileArgs have changed
    if (m_priv->m_lastMetas != in_metas)
    {
        if (m_priv->m_lastCompiled &&
            m_priv->m_lastCompiled.canReshape() &&
            formats_are_same(m_priv->m_lastMetas, in_metas))
        {
            m_priv->m_lastCompiled.reshape(in_metas, args);
        }
        else
        {
            // FIXME: Had to construct temporary object as compile() takes && (r-value)
            m_priv->m_lastCompiled = compile(GMetaArgs(in_metas), std::move(args));
        }
        m_priv->m_lastMetas = in_metas;
    }
    else if (in_metas.size() == 0) {
        // Happens when the graph is head-less (e.g. starts with const-vals only)
        // always compile ad-hoc
        m_priv->m_lastCompiled = compile(GMetaArgs(in_metas), std::move(args));
    }
}

void cv::GComputation::apply(GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args)
{
    recompile(descr_of(ins), std::move(args));
    m_priv->m_lastCompiled(std::move(ins), std::move(outs));
}

void cv::GComputation::apply(const std::vector<cv::Mat> &ins,
                             const std::vector<cv::Mat> &outs,
                             GCompileArgs &&args)
{
    GRunArgs call_ins;
    GRunArgsP call_outs;

    auto tmp = outs;
    for (const cv::Mat &m : ins) { call_ins.emplace_back(m);   }
    for (      cv::Mat &m : tmp) { call_outs.emplace_back(&m); }

    apply(std::move(call_ins), std::move(call_outs), std::move(args));
}

// NB: This overload is called from python code
cv::GRunArgs cv::GComputation::apply(const cv::detail::ExtractArgsCallback &callback,
                                           GCompileArgs                   &&args)
{
    // NB: Used by python bridge
    if (!m_priv->m_info)
    {
        m_priv->m_info = collectGraphInfo(*m_priv);
    }

    auto ins = callback(m_priv->m_info->inputs);
    recompile(descr_of(ins), std::move(args));

    GRunArgs run_args;
    GRunArgsP outs;
    run_args.reserve(m_priv->m_info->outputs.size());
    outs.reserve(m_priv->m_info->outputs.size());

    cv::detail::constructGraphOutputs(m_priv->m_info->outputs, run_args, outs);

    m_priv->m_lastCompiled(std::move(ins), std::move(outs));
    return run_args;
}

#if !defined(GAPI_STANDALONE)
void cv::GComputation::apply(cv::Mat in, cv::Mat &out, GCompileArgs &&args)
{
    apply(cv::gin(in), cv::gout(out), std::move(args));
    // FIXME: The following doesn't work!
    // Operation result is not replicated into user's object
    // apply({GRunArg(in)}, {GRunArg(out)});
}

void cv::GComputation::apply(cv::Mat in, cv::Scalar &out, GCompileArgs &&args)
{
    apply(cv::gin(in), cv::gout(out), std::move(args));
}

void cv::GComputation::apply(cv::Mat in1, cv::Mat in2, cv::Mat &out, GCompileArgs &&args)
{
    apply(cv::gin(in1, in2), cv::gout(out), std::move(args));
}

void cv::GComputation::apply(cv::Mat in1, cv::Mat in2, cv::Scalar &out, GCompileArgs &&args)
{
    apply(cv::gin(in1, in2), cv::gout(out), std::move(args));
}

void cv::GComputation::apply(const std::vector<cv::Mat> &ins,
                                   std::vector<cv::Mat> &outs,
                             GCompileArgs &&args)
{
    GRunArgs call_ins;
    GRunArgsP call_outs;

    for (const cv::Mat &m : ins)  { call_ins.emplace_back(m);   }
    for (      cv::Mat &m : outs) { call_outs.emplace_back(&m); }

    apply(std::move(call_ins), std::move(call_outs), std::move(args));
}
#endif // !defined(GAPI_STANDALONE)

cv::GComputation::Priv& cv::GComputation::priv()
{
    return *m_priv;
}

const cv::GComputation::Priv& cv::GComputation::priv() const
{
    return *m_priv;
}

// Islands /////////////////////////////////////////////////////////////////////

void cv::gapi::island(const std::string       &name,
                            GProtoInputArgs  &&ins,
                            GProtoOutputArgs &&outs)
{
    {
        // Island must have a printable name.
        // Forbid names which contain only spaces.
        GAPI_Assert(!name.empty());
        const auto first_printable_it = std::find_if_not(name.begin(), name.end(), isspace);
        const bool likely_printable   = first_printable_it != name.end();
        GAPI_Assert(likely_printable);
    }
    // Even if the name contains spaces, keep it unmodified as user will
    // then use this string to assign affinity, etc.

    // First, set island tags on all operations from `ins` to `outs`
    auto island = cv::gimpl::unrollExpr(ins.m_args, outs.m_args);
    if (island.all_ops.empty())
    {
        util::throw_error(std::logic_error("Operation range is empty"));
    }
    for (auto &op_expr_node : island.all_ops)
    {
        auto &op_expr_node_p = op_expr_node.priv();

        GAPI_Assert(op_expr_node.shape() == GNode::NodeShape::CALL);
        const GCall&       call   = op_expr_node.call();
        const GCall::Priv& call_p = call.priv();

        if (!op_expr_node_p.m_island.empty())
        {
            util::throw_error(std::logic_error
                              (  "Operation " + call_p.m_k.name
                               + " is already assigned to island \""
                               + op_expr_node_p.m_island + "\""));
        }
        else
        {
            op_expr_node_p.m_island = name;
            GAPI_LOG_INFO(NULL,
                          "Assigned " << call_p.m_k.name << "_" << &call_p <<
                          " to island \"" << name << "\"");
        }
    }

    // Note - this function only sets islands to all operations in
    // expression tree, it is just a first step.
    // The second step is assigning intermediate data objects to Islands,
    // see passes::initIslands for details.
}
