// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <iostream>                              // cout
#include <sstream>                               // stringstream
#include <fstream>                               // ofstream
#include <map>

#include <ade/passes/check_cycles.hpp>

#include <opencv2/gapi/gproto.hpp>
#include "compiler/gmodel.hpp"
#include "compiler/gislandmodel.hpp"
#include "compiler/passes/passes.hpp"

namespace cv { namespace gimpl { namespace passes {

// TODO: FIXME: Ideally all this low-level stuff with accessing ADE APIs directly
// should be incapsulated somewhere into GModel, so here we'd operate not
// with raw nodes and edges, but with Operations and Data it produce/consume.
void dumpDot(const ade::Graph &g, std::ostream& os)
{
    GModel::ConstGraph gr(g);

    const std::unordered_map<cv::GShape, std::string> data_labels = {
        {cv::GShape::GMAT,    "GMat"},
        {cv::GShape::GSCALAR, "GScalar"},
        {cv::GShape::GARRAY,  "GArray"},
    };

    auto format_op_label  = [&gr](ade::NodeHandle nh) -> std::string {
        std::stringstream ss;
        const cv::GKernel k = gr.metadata(nh).get<Op>().k;
        ss << k.name << "_" << nh;
        return ss.str();
    };

    auto format_op  = [&format_op_label](ade::NodeHandle nh) -> std::string {
        return "\"" + format_op_label(nh) + "\"";
    };

    auto format_obj = [&gr, &data_labels](ade::NodeHandle nh) -> std::string {
        std::stringstream ss;
        const auto &data = gr.metadata(nh).get<Data>();
        ss << data_labels.at(data.shape) << "_" << data.rc;
        return ss.str();
    };

    auto format_log = [&gr](ade::NodeHandle nh, const std::string &obj_name) {
        std::stringstream ss;
        const auto &msgs = gr.metadata(nh).get<Journal>().messages;
        ss << "xlabel=\"";
        if (!obj_name.empty()) { ss << "*** " << obj_name << " ***:\n"; };
        for (const auto &msg : msgs) { ss << msg << "\n"; }
        ss << "\"";
        return ss.str();
    };

    // FIXME:
    // Unify with format_log
    auto format_log_e = [&gr](ade::EdgeHandle nh) {
        std::stringstream ss;
        const auto &msgs = gr.metadata(nh).get<Journal>().messages;
        for (const auto &msg : msgs) { ss << "\n" << msg; }
        return ss.str();
    };

    auto sorted = gr.metadata().get<ade::passes::TopologicalSortData>();

    os << "digraph GAPI_Computation {\n";

    // Prior to dumping the graph itself, list Data and Op nodes individually
    // and put type information in labels.
    // Also prepare list of nodes in islands, if any
    std::map<std::string, std::vector<std::string> > islands;
    for (auto &nh : sorted.nodes())
    {
        const auto node_type = gr.metadata(nh).get<NodeType>().t;
        if (NodeType::DATA == node_type)
        {
            const auto obj_data = gr.metadata(nh).get<Data>();
            const auto obj_name = format_obj(nh);

            os << obj_name << " [label=\"" << obj_name << "\n" << obj_data.meta << "\"";
            if (gr.metadata(nh).contains<Journal>()) { os << ", " << format_log(nh, obj_name); }
            os << " ]\n";

            if (gr.metadata(nh).contains<Island>())
                islands[gr.metadata(nh).get<Island>().island].push_back(obj_name);
        }
        else if (NodeType::OP == gr.metadata(nh).get<NodeType>().t)
        {
            const auto obj_name       = format_op(nh);
            const auto obj_name_label = format_op_label(nh);

            os << obj_name << " [label=\"" << obj_name_label << "\"";
            if (gr.metadata(nh).contains<Journal>()) { os << ", " << format_log(nh, obj_name_label); }
            os << " ]\n";

            if (gr.metadata(nh).contains<Island>())
                islands[gr.metadata(nh).get<Island>().island].push_back(obj_name);
        }
    }

    // Then, dump Islands (only nodes, operations and data, without links)
    for (const auto &isl : islands)
    {
        os << "subgraph \"cluster " + isl.first << "\" {\n";
        for(auto isl_node : isl.second) os << isl_node << ";\n";
        os << "label=\"" << isl.first << "\";";
        os << "}\n";
    }

    // Now dump the graph
    for (auto &nh : sorted.nodes())
    {
        // FIXME: Alan Kay probably hates me.
        switch (gr.metadata(nh).get<NodeType>().t)
        {
        case NodeType::DATA:
        {
            const auto obj_name = format_obj(nh);
            for (const auto &eh : nh->outEdges())
            {
                os << obj_name << " -> " << format_op(eh->dstNode())
                   << " [ label = \"in_port: "
                   << gr.metadata(eh).get<Input>().port;
                   if (gr.metadata(eh).contains<Journal>()) { os << format_log_e(eh); }
                   os << "\" ] \n";
            }
        }
        break;
        case NodeType::OP:
        {
            for (const auto &eh : nh->outEdges())
            {
                os << format_op(nh) << " -> " << format_obj(eh->dstNode())
                   << " [ label = \"out_port: "
                   << gr.metadata(eh).get<Output>().port
                   << " \" ]; \n";
            }
        }
        break;
        default: GAPI_Assert(false);
        }
    }

    // And finally dump a GIslandModel (not connected with GModel directly,
    // but projected in the same .dot file side-by-side)
    auto pIG = gr.metadata().get<IslandModel>().model;
    GIslandModel::Graph gim(*pIG);
    for (auto nh : gim.nodes())
    {
        switch (gim.metadata(nh).get<NodeKind>().k)
        {
        case NodeKind::ISLAND:
            {
                const auto island   = gim.metadata(nh).get<FusedIsland>().object;
                const auto isl_name = "\"" + island->name() + "\"";
                for (auto out_nh : nh->outNodes())
                {
                    os << isl_name << " -> \"slot:"
                       << format_obj(gim.metadata(out_nh).get<DataSlot>()
                                     .original_data_node)
                       << "\"\n";
                }
            }
            break;
        case NodeKind::SLOT:
            {
                const auto obj_name = format_obj(gim.metadata(nh).get<DataSlot>()
                                                 .original_data_node);
                for (auto cons_nh : nh->outNodes())
                {
                    if (gim.metadata(cons_nh).get<NodeKind>().k == NodeKind::ISLAND) {
                        os << "\"slot:" << obj_name << "\" -> \""
                           << gim.metadata(cons_nh).get<FusedIsland>().object->name()
                           << "\"\n";
                    } // other data consumers -- sinks -- are processed separately
                }
            }
            break;
        case NodeKind::EMIT:
            {
                for (auto out_nh : nh->outNodes())
                {
                    const auto obj_name = format_obj(gim.metadata(out_nh).get<DataSlot>()
                                                     .original_data_node);
                    os << "\"emit:" << nh << "\" -> \"slot:" << obj_name << "\"\n";
                }
            }
            break;
        case NodeKind::SINK:
            {
                for (auto in_nh : nh->inNodes())
                {
                    const auto obj_name = format_obj(gim.metadata(in_nh).get<DataSlot>()
                                                     .original_data_node);
                    os << "\"slot:" << obj_name << "\" -> \"sink:" << nh << "\"\n";
                }
            }
            break;
        default:
            GAPI_Assert(false);
            break;
        }
    }

    os << "}" << std::endl;
}

void dumpDot(ade::passes::PassContext &ctx, std::ostream& os)
{
    dumpDot(ctx.graph, os);
}

void dumpDotStdout(ade::passes::PassContext &ctx)
{
    dumpDot(ctx, std::cout);
}

void dumpDotToFile(ade::passes::PassContext &ctx, const std::string& dump_path)
{
    std::ofstream dump_file(dump_path);

    if (dump_file.is_open())
    {
        dumpDot(ctx, dump_file);
        dump_file << std::endl;
    }
}

void dumpGraph(ade::passes::PassContext &ctx, const std::string& dump_path)
{
    dump_path.empty() ? dumpDotStdout(ctx) : dumpDotToFile(ctx, dump_path);
}

}}} // cv::gimpl::passes
