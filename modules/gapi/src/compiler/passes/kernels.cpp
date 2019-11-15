// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <ade/util/zip_range.hpp>   // util::indexed
#include <ade/graph.hpp>
#include <ade/passes/check_cycles.hpp>

#include <opencv2/gapi/gcompoundkernel.hpp> // compound::backend()
#include <opencv2/gapi/gkernel.hpp>         // GKernelPackage
#include <opencv2/gapi/infer.hpp>           // GNetPackage

#include "compiler/gmodel.hpp"
#include "compiler/passes/passes.hpp"

#include "api/gbackend_priv.hpp"
#include "backends/common/gbackend.hpp"
#include "compiler/gmodelbuilder.hpp"
#include "logger.hpp"    // GAPI_LOG
#include "api/gproto_priv.hpp" // is_dynamic, rewrap

namespace
{
    struct ImplInfo
    {
        cv::GKernelImpl impl;
        cv::GArgs       in_args;
    };

    // Generally the algorithm is following
    //
    // 1. Get GCompoundKernel implementation
    // 2. Create GCompoundContext
    // 3. Run GCompoundKernel with GCompoundContext
    // 4. Build subgraph from inputs/outputs GCompoundKernel
    // 5. Replace compound node to subgraph

    void expand(ade::Graph& g, ade::NodeHandle nh, const ImplInfo& impl_info)
    {
        cv::gimpl::GModel::Graph gr(g);
        auto compound_impl = cv::util::any_cast<cv::detail::GCompoundKernel>(impl_info.impl.opaque);

        // GCompoundContext instantiates its own objects
        // in accordance with the RcDescs from in_args
        cv::detail::GCompoundContext context(impl_info.in_args);
        compound_impl.apply(context);

        cv::GProtoArgs ins, outs;
        ins.reserve(context.m_args.size());
        outs.reserve(context.m_results.size());

        // Inputs can be non-dynamic types.
        // Such inputs are not used when building a graph
        for (const auto& arg : context.m_args)
        {
            if (cv::gimpl::proto::is_dynamic(arg))
            {
                ins.emplace_back(cv::gimpl::proto::rewrap(arg));
            }
        }

        ade::util::transform(context.m_results, std::back_inserter(outs), &cv::gimpl::proto::rewrap);

        cv::gimpl::GModelBuilder builder(g);

        // Build the subgraph graph which will need to replace the compound node
        const auto& proto_slots = builder.put(ins, outs);

        const auto& in_nhs  = std::get<2>(proto_slots);
        const auto& out_nhs = std::get<3>(proto_slots);

        auto sorted_in_nhs  = cv::gimpl::GModel::orderedInputs(gr, nh);
        auto sorted_out_nhs = cv::gimpl::GModel::orderedOutputs(gr, nh);

        // Reconnect expanded kernels from graph data objects
        // to subgraph data objects, then drop that graph data objects
        for (const auto& it : ade::util::zip(in_nhs, sorted_in_nhs))
        {
            const auto& subgr_in_nh = std::get<0>(it);
            const auto& comp_in_nh  = std::get<1>(it);

            cv::gimpl::GModel::redirectReaders(gr, subgr_in_nh, comp_in_nh);
            gr.erase(subgr_in_nh);
        }

        gr.erase(nh);

        for (const auto& it : ade::util::zip(out_nhs, sorted_out_nhs))
        {
            const auto& subgr_out_nh = std::get<0>(it);
            const auto& comp_out_nh  = std::get<1>(it);

            cv::gimpl::GModel::redirectWriter(gr, subgr_out_nh, comp_out_nh);
            gr.erase(subgr_out_nh);
        }
    }
} // anonymous namespace

// This pass, given the network package, associates every infer[list] node
// with particular inference backend and its parameters.
void cv::gimpl::passes::bindNetParams(ade::passes::PassContext &ctx,
                                      const gapi::GNetPackage  &pkg)
{
    GModel::Graph gr(ctx.graph);
    ade::TypedGraph<NetworkParams> pgr(ctx.graph);

    for (const auto &nh : gr.nodes())
    {
        if (gr.metadata(nh).get<NodeType>().t == NodeType::OP)
        {
            auto &op = gr.metadata(nh).get<Op>();
            if (op.k.tag.empty())
                continue;

            // FIXME: What if there's more than one???
            const auto it = ade::util::find_if(pkg.networks,
                                               [&](const cv::gapi::GNetParam &p) {
                                                   return p.tag == op.k.tag;
                                               });
            if (it == std::end(pkg.networks))
                continue;

            pgr.metadata(nh).set(NetworkParams{it->params});
        }
    }
}

// This pass, given the kernel package, selects a kernel implementation
// for every operation in the graph
void cv::gimpl::passes::resolveKernels(ade::passes::PassContext   &ctx,
                                       const gapi::GKernelPackage &kernels)
{
    std::unordered_set<cv::gapi::GBackend> active_backends;

    GModel::Graph gr(ctx.graph);
    for (const auto &nh : gr.nodes())
    {
        if (gr.metadata(nh).get<NodeType>().t == NodeType::OP)
        {
            auto &op = gr.metadata(nh).get<Op>();
            cv::gapi::GBackend selected_backend;
            cv::GKernelImpl    selected_impl;
            std::tie(selected_backend, selected_impl) = kernels.lookup(op.k.name);

            selected_backend.priv().unpackKernel(ctx.graph, nh, selected_impl);
            op.backend = selected_backend;
            active_backends.insert(selected_backend);
        }
    }
    gr.metadata().set(ActiveBackends{active_backends});
}

void cv::gimpl::passes::expandKernels(ade::passes::PassContext &ctx, const gapi::GKernelPackage &kernels)
{
    GModel::Graph gr(ctx.graph);

    // Repeat the loop while there are compound kernels.
    // Restart procedure after every successful unrolling
    bool has_compound_kernel = true;
    while (has_compound_kernel)
    {
        has_compound_kernel = false;
        for (const auto& nh : gr.nodes())
        {
            if (gr.metadata(nh).get<NodeType>().t == NodeType::OP)
            {
                const auto& op = gr.metadata(nh).get<Op>();

                cv::gapi::GBackend selected_backend;
                cv::GKernelImpl    selected_impl;
                std::tie(selected_backend, selected_impl) = kernels.lookup(op.k.name);

                if (selected_backend == cv::gapi::compound::backend())
                {
                    has_compound_kernel = true;
                    expand(ctx.graph, nh, ImplInfo{selected_impl, op.args});
                    break;
                }
            }
        }
    }
    GAPI_LOG_INFO(NULL, "Final graph: " << ctx.graph.nodes().size() << " nodes" << std::endl);
}
