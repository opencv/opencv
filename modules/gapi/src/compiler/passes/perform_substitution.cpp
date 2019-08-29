// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "pattern_matching.hpp"

namespace cv { namespace gimpl {
namespace {
using Graph = GModel::Graph;

template<typename Container, typename Callable>
void erase(Graph& g, const Container& c, Callable getNh)
{
    for (auto first = c.begin(); first != c.end(); ++first) {
        ade::NodeHandle node = getNh(first);
        if (node == nullptr) continue;  // some nodes might already be erased
        g.erase(node);
    }
}
}  // anonymous namespace

void performSubstitution(Graph& graph,
                         const SubgraphMatch& patternToGraph,
                         const SubgraphMatch& patternToSubstitute)
{
    // substitute input nodes
    for (const auto& inputNodePair : patternToGraph.inputDataNodes) {
        // Note: we don't replace input DATA nodes here, only redirect their output edges
        const auto& patternDataNode = inputNodePair.first;
        const auto& graphDataNode = inputNodePair.second;
        const auto& substituteDataNode = patternToSubstitute.inputDataNodes.at(patternDataNode);
        GModel::redirectReaders(graph, substituteDataNode, graphDataNode);
    }

    // substitute output nodes
    for (const auto& outputNodePair : patternToGraph.outputDataNodes) {
        // Note: we don't replace output DATA nodes here, only redirect their input edges
        const auto& patternDataNode = outputNodePair.first;
        const auto& graphDataNode = outputNodePair.second;
        const auto& substituteDataNode = patternToSubstitute.outputDataNodes.at(patternDataNode);
        // delete existing edges (otherwise we cannot redirect)
        for (auto e : graphDataNode->inEdges()) {
            graph.erase(e);
        }
        GModel::redirectWriter(graph, substituteDataNode, graphDataNode);
    }

    // erase redundant nodes
    const auto get_from_node = [] (std::list<ade::NodeHandle>::const_iterator it) { return *it; };
    const auto get_from_pair = [] (SubgraphMatch::M::const_iterator it) { return it->second; };

    // erase input data nodes of __substitute__
    erase(graph, patternToSubstitute.inputDataNodes, get_from_pair);

    // erase old start OP nodes of __main graph__
    erase(graph, patternToGraph.startOpNodes, get_from_pair);

    // erase old internal nodes of __main graph__
    erase(graph, patternToGraph.internalLayers, get_from_node);

    // erase old finish OP nodes of __main graph__
    erase(graph, patternToGraph.finishOpNodes, get_from_pair);

    // erase output data nodes of __substitute__
    erase(graph, patternToSubstitute.outputDataNodes, get_from_pair);
}

}  // namespace gimpl
}  // namespace cv
