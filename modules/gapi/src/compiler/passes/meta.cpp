// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <ade/util/zip_range.hpp>   // util::indexed
#include <ade/graph.hpp>
#include <ade/passes/check_cycles.hpp>

#include "compiler/gmodel.hpp"
#include "compiler/passes/passes.hpp"
#include "logger.hpp"    // GAPI_LOG


// Iterate over all nodes and initialize meta of objects taken from the
// outside (i.e., computation input/output arguments)
void cv::gimpl::passes::initMeta(ade::passes::PassContext &ctx, const GMetaArgs &metas)
{
    GModel::Graph gr(ctx.graph);

    const auto &proto = gr.metadata().get<Protocol>();

    for (const auto& it : ade::util::indexed(proto.in_nhs))
    {
        auto& data = gr.metadata(ade::util::value(it)).get<Data>();
        data.meta = metas.at(ade::util::index(it));
    }
}

// Iterate over all operations in the topological order, trigger kernels
// validate() function, update output objects metadata.
void cv::gimpl::passes::inferMeta(ade::passes::PassContext &ctx, bool meta_is_initialized)
{
    // FIXME: ADE pass dependency on topo_sort?
    // FIXME: ADE pass dependency on initMeta?
    GModel::Graph gr(ctx.graph);

    const auto sorted = gr.metadata().get<ade::passes::TopologicalSortData>() ;
    for (const auto &nh : sorted.nodes())
    {
        if (gr.metadata(nh).get<NodeType>().t == NodeType::OP)
        {
            const auto& op = gr.metadata(nh).get<Op>();
            GAPI_Assert(op.k.outMeta != nullptr);

            // Prepare operation's input metadata vector
            // Note that it's size is usually different from nh.inEdges.size(),
            // and its element count is equal to operation's arguments count
            // (which may contain graph-construction-time parameters like integers, etc)
            GMetaArgs input_meta_args(op.args.size());

            // Iterate through input edges, update input_meta_args's slots
            // appropriately. Not all of them will be updated due to (see above).
            GAPI_Assert(nh->inEdges().size() > 0);
            for (const auto &in_eh : nh->inEdges())
            {
                const auto& input_port = gr.metadata(in_eh).get<Input>().port;
                const auto& input_nh   = in_eh->srcNode();
                GAPI_Assert(gr.metadata(input_nh).get<NodeType>().t == NodeType::DATA);

                const auto& input_meta = gr.metadata(input_nh).get<Data>().meta;
                if (util::holds_alternative<util::monostate>(input_meta))
                {
                    // No meta in an input argument - a fatal error
                    // (note graph is traversed here in topoligcal order)
                    util::throw_error(std::logic_error("Fatal: input object's metadata "
                                                       "not found!"));
                    // FIXME: Add more details!!!
                }
                input_meta_args.at(input_port) = input_meta;
            }

            // Now ask kernel for it's output meta.
            // Resulting out_args may have a larger size than op.outs, since some
            // outputs could stay unused (unconnected)
            const auto out_metas = gr.metadata(nh).contains<CustomMetaFunction>()
                ? gr.metadata(nh).get<CustomMetaFunction>().customOutMeta(ctx.graph,
                                                                          nh,
                                                                          input_meta_args,
                                                                          op.args)
                : op.k.outMeta(input_meta_args, op.args);

            // Walk through operation's outputs, update meta of output objects
            // appropriately
            GAPI_Assert(nh->outEdges().size() > 0);
            for (const auto &out_eh : nh->outEdges())
            {
                const auto &output_port = gr.metadata(out_eh).get<Output>().port;
                const auto &output_nh   = out_eh->dstNode();
                GAPI_Assert(gr.metadata(output_nh).get<NodeType>().t == NodeType::DATA);

                auto       &output_meta = gr.metadata(output_nh).get<Data>().meta;
                if (!meta_is_initialized && !util::holds_alternative<util::monostate>(output_meta))
                {
                    GAPI_LOG_INFO(NULL,
                                  "!!! Output object has an initialized meta - "
                                  "how it is possible today?" << std::endl; );
                    if (output_meta != out_metas.at(output_port))
                    {
                      util::throw_error(std::logic_error("Fatal: meta mismatch"));
                        // FIXME: New exception type?
                        // FIXME: More details!
                    }
                }
                // Store meta in graph
                output_meta = out_metas.at(output_port);
            }
        } // if(OP)
    } // for(sorted)
}

// After all metadata in graph is inferred, store a vector of inferred metas
// for computation output values.
void cv::gimpl::passes::storeResultingMeta(ade::passes::PassContext &ctx)
{
    GModel::Graph gr(ctx.graph);

    const auto &proto = gr.metadata().get<Protocol>();
    GMetaArgs output_metas(proto.out_nhs.size());

    for (const auto& it : ade::util::indexed(proto.out_nhs))
    {
        auto& data = gr.metadata(ade::util::value(it)).get<Data>();
        output_metas[ade::util::index(it)] = data.meta;
    }

    gr.metadata().set(OutputMeta{output_metas});
}
