// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "pattern_matching.hpp"

#include "ade/util/zip_range.hpp"

namespace cv { namespace gimpl {
namespace {
using Graph = GModel::Graph;

template<typename Iterator>
ade::NodeHandle getNh(Iterator it) { return *it; }

template<>
ade::NodeHandle getNh(SubgraphMatch::M::const_iterator it) { return it->second; }

template<typename Container>
void erase(Graph& g, const Container& c)
{
    for (auto first = c.begin(); first != c.end(); ++first) {
        ade::NodeHandle node = getNh(first);
        if (node == nullptr) continue;  // some nodes might already be erased
        g.erase(node);
    }
}
}  // anonymous namespace

void performSubstitution(GModel::Graph& graph,
                         const Protocol& patternP,
                         const Protocol& substituteP,
                         const SubgraphMatch& patternToGraphMatch)
{
    // 1. substitute input nodes
    const auto& patternIns = patternP.in_nhs;
    const auto& substituteIns = substituteP.in_nhs;

    for (auto it : ade::util::zip(ade::util::toRange(patternIns),
                                  ade::util::toRange(substituteIns))) {
        // Note: we don't replace input DATA nodes here, only redirect their output edges
        const auto& patternDataNode = std::get<0>(it);
        const auto& substituteDataNode = std::get<1>(it);
        const auto& graphDataNode = patternToGraphMatch.inputDataNodes.at(patternDataNode);
        GModel::redirectReaders(graph, substituteDataNode, graphDataNode);
    }

    // 2. substitute output nodes
    const auto& patternOuts = patternP.out_nhs;
    const auto& substituteOuts = substituteP.out_nhs;

    for (auto it : ade::util::zip(ade::util::toRange(patternOuts),
                                  ade::util::toRange(substituteOuts))) {
        // Note: we don't replace output DATA nodes here, only redirect their input edges
        const auto& patternDataNode = std::get<0>(it);
        const auto& substituteDataNode = std::get<1>(it);
        const auto& graphDataNode = patternToGraphMatch.outputDataNodes.at(patternDataNode);

        // delete existing edges (otherwise we cannot redirect)
        auto existingEdges = graphDataNode->inEdges();
        // NB: we cannot iterate over node->inEdges() here directly because it gets modified when
        //     edges are erased. Erasing an edge supposes that src/dst nodes will remove
        //     (correspondingly) out/in edge (which is _our edge_). Now, this deleting means
        //     node->inEdges() will also get updated in the process: so, we'd iterate over a
        //     container which changes in this case. Using supplementary std::vector instead:
        std::vector<ade::EdgeHandle> handles(existingEdges.begin(), existingEdges.end());
        for (const auto& e : handles) {
            graph.erase(e);
        }

        GModel::redirectWriter(graph, substituteDataNode, graphDataNode);
    }

    // 3. erase redundant nodes:
    // erase input data nodes of __substitute__
    erase(graph, substituteIns);

    // erase old start OP nodes of __main graph__
    erase(graph, patternToGraphMatch.startOpNodes);

    // erase old internal nodes of __main graph__
    erase(graph, patternToGraphMatch.internalLayers);

    // erase old finish OP nodes of __main graph__
    erase(graph, patternToGraphMatch.finishOpNodes);

    // erase output data nodes of __substitute__
    erase(graph, substituteOuts);
}

}  // namespace gimpl
}  // namespace cv
