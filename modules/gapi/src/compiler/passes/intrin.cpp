// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "precomp.hpp"

#include <ade/util/algorithm.hpp>
#include <ade/util/zip_range.hpp>
#include <opencv2/gapi/streaming/desync.hpp>// GDesync intrinsic

#include "compiler/gmodel.hpp"
#include "compiler/passes/passes.hpp"

namespace {
namespace desync {

void trace(cv::gimpl::GModel::Graph &g) {
}

// Probably the simplest case: desync makes no sense in the regular
// compilation process, so just drop all its occurences in the graph,
// reconnecting nodes properly.
void drop(cv::gimpl::GModel::Graph &g) {
    using namespace cv::gimpl;
    std::vector<ade::NodeHandle> nodes(g.nodes().begin(), g.nodes().end());
    for (auto &&nh : nodes) {
        if (nh == nullptr) {
            // Some nodes could be dropped already during the procedure
            // thanks ADE their NodeHandles updated automatically
            continue;
        }
        if (g.metadata(nh).get<NodeType>().t == NodeType::OP) {
            const auto &op = g.metadata(nh).get<Op>();
            if (op.k.name == cv::gapi::streaming::detail::GDesync::id()) {
                // What we need to do here:
                // 1. Connect the readers of its produced data objects
                //    to the input data objects of desync;
                // 2. Drop the data object it produces.
                // 3. Drop the desync operation itself;
                std::vector<ade::NodeHandle> in_data_objs = GModel::orderedInputs(g, nh);
                std::vector<ade::NodeHandle> out_data_objs = GModel::orderedOutputs(g, nh);
                GAPI_Assert(in_data_objs.size() == out_data_objs.size());
                GAPI_DbgAssert(ade::util::all_of
                               (out_data_objs,
                                [&](const ade::NodeHandle &oh) {
                                   return g.metadata(oh).contains<Data>();
                               }));
                // (1)
                for (auto &&it: ade::util::zip(ade::util::toRange(in_data_objs),
                                               ade::util::toRange(out_data_objs))) {
                    GModel::redirectReaders(g, std::get<1>(it), std::get<0>(it));
                }
                // (2)
                for (auto &&old_out_nh : out_data_objs) {
                    g.erase(old_out_nh);
                }
                // (3)
                g.erase(nh);
            } // if (desync)
        } // if (Op)
    } // for(nodes)
}

} // namespace desync
} // anonymous namespace

void cv::gimpl::passes::intrinDesync(ade::passes::PassContext &ctx) {
    GModel::Graph gr(ctx.graph);
    if (!gr.metadata().contains<HasIntrinsics>())
        return;

    gr.metadata().contains<Streaming>()
        ? desync::trace(gr) // Streaming compilation
        : desync::drop(gr); // Regular compilation
}

// Clears the HasIntrinsics flag if all intrinsics have been handled.
void cv::gimpl::passes::intrinFinalize(ade::passes::PassContext &ctx) {
    GModel::Graph gr(ctx.graph);
    for (auto &&nh : gr.nodes()) {
        if (gr.metadata(nh).get<NodeType>().t == NodeType::OP) {
            const auto &op = gr.metadata(nh).get<Op>();
            if (is_intrinsic(op.k.name)) {
                return;
            }
        }
    }
    // If reached here, really clear the flag
    gr.metadata().erase<HasIntrinsics>();
}
