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

namespace desync {
namespace {

// Drop the desynchronized node `nh` from the graph, reconnect the
// graph structure properly.  This is a helper function which is used
// in both drop(g) and apply(g) passes.
//
// @return a vector of new edge handles connecting the "main" graph
// with its desynchronized part.
std::vector<ade::EdgeHandle> drop(cv::gimpl::GModel::Graph &g,
                                  ade::NodeHandle nh) {
    using namespace cv::gimpl;

    // What we need to do here:
    // 1. Connect the readers of its produced data objects
    //    to the input data objects of desync;
    // 2. Drop the data object it produces.
    // 3. Drop the desync operation itself;
    std::vector<ade::NodeHandle> in_data_objs = GModel::orderedInputs(g, nh);
    std::vector<ade::NodeHandle> out_data_objs = GModel::orderedOutputs(g, nh);
    std::vector<ade::EdgeHandle> new_links;
    GAPI_Assert(in_data_objs.size() == out_data_objs.size());
    GAPI_DbgAssert(ade::util::all_of
                   (out_data_objs,
                    [&](const ade::NodeHandle &oh) {
                       return g.metadata(oh).contains<Data>();
                   }));
    // (1)
    for (auto &&it: ade::util::zip(ade::util::toRange(in_data_objs),
                                   ade::util::toRange(out_data_objs))) {
        auto these_new_links = GModel::redirectReaders(g,
                                                       std::get<1>(it),
                                                       std::get<0>(it));
        new_links.insert(new_links.end(),
                         these_new_links.begin(),
                         these_new_links.end());
    }
    // (2)
    for (auto &&old_out_nh : out_data_objs) {
        g.erase(old_out_nh);
    }
    // (3)
    g.erase(nh);

    return new_links;
}

// Tracing a desynchronizing subgraph is somewhat tricky and happens
// in both directions: downwards and upwards.
//
// The downward process is the basic one: we start with a "desync"
// OP node and go down to the graph using the "output" edges. We check
// if all nodes on this path [can] belong to this desynchronized path
// and don't overlap with others.
//
// An important contract to maintain is that the desynchronized part
// can't have any input references from the "main" graph part or any
// other desynchronized part in the graph. This contract is validated
// by checking every node's input which must belong to the same
// desynchronized part.
//
// Here is the pitfall of this check:
//
//       v
//     GMat_0
//       v
//   +----------+
//   | desync() |      <- This point originates the traceDown process
//   +----------+
//       v
//     GMat_0'         <- This node will be tagged for this desync at
//       :--------.       step 0/1
//       v        :    <- The order how output nodes are visited is not
//   +----------+ :       specified, we can visit Op2() first (as there
//   | Op1()    | :       is a direct link) bypassing visiting and tagging
//   +----------+ :       Op1() and GMat_1
//       v        :
//     GMat_1     :
//       :    .---'
//       v    v        <- When we visit Op2() via the 2nd edge on this
//   +----------+         graph, we check if all inputs belong to the same
//   | Op2()    |         desynchronized graph and GMat_1 fails this check
//   +----------+         (since the traceDown() process haven't visited
//                        it yet).
//
// Cases like this originate the traceUp() process: if we find an
// input node in our desynchronized path which doesn't belong to this
// path YET, it is not 100% a problem, and we need to trace it back
// (upwards) to see if it is really a case.

// This recursive function checks the desync_id in the graph upwards.
// The process doesn't continue for nodes which have a valid
// desync_id already.
// The process only continues for nodes which have no desync_id
// assigned. If there's no such nodes anymore, the procedure is
// considered complete and a list of nodes to tag is returned to the
// caller.
//
// If NO inputs of this node have a valid desync_id, the desync
// invariant is broken and the function throws.
void traceUp(cv::gimpl::GModel::Graph &g,
             const ade::NodeHandle &nh,
             int desync_id,
             std::vector<ade::NodeHandle> &path) {
    using namespace cv::gimpl;

    GAPI_Assert(!nh->inNodes().empty()
                && "traceUp: a desynchronized part of the graph is not isolated?");

    if (g.metadata(nh).contains<DesyncPath>()) {
        // We may face nodes which have DesyncPath already visited during
        // this recursive process (e.g. via some other output or branch in the
        // subgraph)
        if (g.metadata(nh).get<DesyncPath>().index != desync_id) {
            GAPI_Assert(false && "Desynchronization can't be nested!");
        }
        return; // This object belongs to the desync path - exit early.
    }

    // Regardless of the result, put this nh to the path
    path.push_back(nh);

    // Check if the input nodes are OK
    std::vector<ade::NodeHandle> nodes_to_trace;
    nodes_to_trace.reserve(nh->inNodes().size());
    for (auto &&in_nh : nh->inNodes()) {
        if (g.metadata(in_nh).contains<DesyncPath>()) {
            // We may face nodes which have DesyncPath already visited during
            // this recursive process (e.g. via some other output or branch in the
            // subgraph)
            GAPI_Assert(g.metadata(in_nh).get<DesyncPath>().index == desync_id
                        && "Desynchronization can't be nested!");
        } else {
            nodes_to_trace.push_back(in_nh);
        }
    }

    // If there are nodes to trace, continue the recursion
    for (auto &&up_nh : nodes_to_trace) {
        traceUp(g, up_nh, desync_id, path);
    }
}

// This recursive function propagates the desync_id down to the graph
// starting at nh, and also checks:
// - if this desync path is isolated;
// - if this desync path is not overlapped.
// It also originates the traceUp() process at the points of
// uncertainty (as described in the comment above).
void traceDown(cv::gimpl::GModel::Graph &g,
               const ade::NodeHandle &nh,
               int desync_id) {
    using namespace cv::gimpl;

    if (g.metadata(nh).contains<DesyncPath>()) {
        // We may face nodes which have DesyncPath already visited during
        // this recursive process (e.g. via some other output or branch in the
        // subgraph)
        GAPI_Assert(g.metadata(nh).get<DesyncPath>().index == desync_id
                    && "Desynchronization can't be nested!");
    } else {
        g.metadata(nh).set(DesyncPath{desync_id});
    }

    // All inputs of this data object must belong to the same
    // desync path.
    for (auto &&in_nh : nh->inNodes()) {
        // If an input object is not assigned to this desync path,
        // it does not means that the object doesn't belong to
        // this path. Check it.
        std::vector<ade::NodeHandle> path_up;
        traceUp(g, in_nh, desync_id, path_up);
        // We get here on success. Just set the proper tags for
        // the identified input path.
        for (auto &&up_nh : path_up) {
            g.metadata(up_nh).set(DesyncPath{desync_id});
        }
    }

    // Propagate the tag & check down
    for (auto &&out_nh : nh->outNodes()) {
        traceDown(g, out_nh, desync_id);
    }
}

// Streaming case: ensure the graph has proper isolation of the
// desynchronized parts, set proper Edge metadata hints for
// GStreamingIntrinExecutable
void apply(cv::gimpl::GModel::Graph &g) {
    using namespace cv::gimpl;

    // Stage 0. Trace down the desync operations in the graph.
    // Tag them with their unique (per graph) identifiers.
    int total_desync = 0;
    for (auto &&nh : g.nodes()) {
        if (g.metadata(nh).get<NodeType>().t == NodeType::OP) {
            const auto &op = g.metadata(nh).get<Op>();
            if (op.k.name == cv::gapi::streaming::detail::GDesync::id()) {
                GAPI_Assert(!g.metadata(nh).contains<DesyncPath>()
                            && "Desynchronization can't be nested!");
                const int this_desync_id = total_desync++;
                g.metadata(nh).set(DesyncPath{this_desync_id});
                for (auto &&out_nh: nh->outNodes()) {
                    traceDown(g, out_nh, this_desync_id);
                }
            } // if (desync)
        } // if(OP)
    } // for(nodes)

    // Tracing is done for all desync ops in the graph now.
    // Stage 1. Drop the desync operations from the graph, but mark
    // the desynchronized edges a special way.
    // The desynchronized edge is the edge which connects a main
    // subgraph data with a desynchronized subgraph data.
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
                auto index = g.metadata(nh).get<DesyncPath>().index;
                auto new_links = drop(g, nh);
                for (auto &&eh : new_links) {
                    g.metadata(eh).set(DesyncEdge{index});
                }
            } // if (desync)
        } // if (Op)
    } // for(nodes)

    // Stage 2. Put a synchronized tag if there were changes applied
    if (total_desync > 0) {
        g.metadata().set(Desynchronized{});
    }
}

// Probably the simplest case: desync makes no sense in the regular
// compilation process, so just drop all its occurrences in the graph,
// reconnecting nodes properly.
void drop(cv::gimpl::GModel::Graph &g) {
    // FIXME: LOG here that we're dropping the desync operations as
    // they have no sense when compiling in the regular mode.
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
                drop(g, nh);
            } // if (desync)
        } // if (Op)
    } // for(nodes)
}

} // anonymous namespace
} // namespace desync

void cv::gimpl::passes::intrinDesync(ade::passes::PassContext &ctx) {
    GModel::Graph gr(ctx.graph);
    if (!gr.metadata().contains<HasIntrinsics>())
        return;

    gr.metadata().contains<Streaming>()
        ? desync::apply(gr) // Streaming compilation
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
