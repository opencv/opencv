// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "pattern_matching.hpp"

#include <ade/util/zip_range.hpp>

namespace cv { namespace gimpl {

namespace {
using Graph = GModel::Graph;
using Metadata = typename Graph::CMetadataT;

// Returns true if two DATA nodes are semantically identical:
//  - both nodes have the same GShape
//  - both nodes have the same storage
//
// @param firstMeta - metadata of first
// @param secondMeta - metadata of second
bool compareDataNodes(const Metadata& firstMeta,
                      const Metadata& secondMeta)
{
    GAPI_Assert(firstMeta.get<NodeType>().t == NodeType::DATA);
    GAPI_Assert(firstMeta.get<NodeType>().t == secondMeta.get<NodeType>().t);

    const auto& firstData = firstMeta.get<Data>();
    const auto& secondData = secondMeta.get<Data>();
    // compare shape
    if (firstData.shape != secondData.shape) {
        return false;
    }
    // compare storage
    if (firstData.storage != secondData.storage) {
        return false;
    }

    // NB: it seems enough for now to only check shape && storage
    return true;
}

// Returns matched pairs of {pattern node, substitute node}
SubgraphMatch::M matchDataNodes(const Graph& pattern,
                                const Graph& substitute,
                                const std::vector<ade::NodeHandle>& patternNodes,
                                std::vector<ade::NodeHandle> substituteNodes)
{
    SubgraphMatch::M matched;
    for (auto it : ade::util::zip(patternNodes, substituteNodes)) {
        const auto& pNode = std::get<0>(it);
        const auto& sNode = std::get<1>(it);
        if (!compareDataNodes(pattern.metadata(pNode), substitute.metadata(sNode))) {
            return {};
        }
        matched.insert({ pNode, sNode });
    }
    return matched;
}
}  // anonymous namespace

SubgraphMatch matchPatternToSubstitute(const Graph& pattern,
                                       const Graph& substitute,
                                       const Protocol& patternP,
                                       const Protocol& substituteP)
{
    //---------------------------------------------------------------
    // Match data nodes which start and end our pattern and substitute
    const auto& patternDataInputs = patternP.in_nhs;
    const auto& patternDataOutputs = patternP.out_nhs;

    const auto& substituteDataInputs = substituteP.in_nhs;
    const auto& substituteDataOutputs = substituteP.out_nhs;

    // number of data nodes must be the same
    GAPI_Assert(patternDataInputs.size() == substituteDataInputs.size());
    GAPI_Assert(patternDataOutputs.size() == substituteDataOutputs.size());

    // for each pattern input we must find a corresponding substitute input
    auto matchedDataInputs = matchDataNodes(pattern, substitute, patternDataInputs,
        substituteDataInputs);
    // data inputs must be matched in all cases
    GAPI_Assert(!matchedDataInputs.empty());

    auto matchedDataOutputs = matchDataNodes(pattern, substitute, patternDataOutputs,
        substituteDataOutputs);
    // data outputs must be matched in all cases
    GAPI_Assert(!matchedDataOutputs.empty());

    //---------------------------------------------------------------
    // Construct SubgraphMatch object
    SubgraphMatch match;
    match.inputDataNodes = std::move(matchedDataInputs);
    match.outputDataNodes = std::move(matchedDataOutputs);
    return match;
}

}  // namespace gimpl
}  // namespace cv
