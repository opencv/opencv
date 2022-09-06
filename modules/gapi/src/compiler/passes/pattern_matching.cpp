// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include <unordered_set>

#include "pattern_matching.hpp"

namespace  {
using Graph = cv::gimpl::GModel::Graph;
using Metadata = typename Graph::CMetadataT;
using VisitedMatchings = std::list<std::pair<ade::NodeHandle, ade::NodeHandle>>;

using LabeledNodes = std::unordered_map
                    < // reader node
                      ade::NodeHandle
                      // if the reader node above is:
                      //  - DATA node: then vector is 1-element vector containing port number of
                      //    the input edge
                      //  - OP node: then vector is ports' vector of current connections between
                      //    this node and an parent active DATA node
                    , std::vector<std::size_t>
                    , ade::HandleHasher<ade::Node>
                    >;

using MultipleMatchings = std::unordered_map
                          // pattern OP node
                         < ade::NodeHandle
                          // nodes in the test graph which match to the pattern OP node above
                          , std::vector<ade::NodeHandle>
                          , ade::HandleHasher<ade::Node>
                         >;

// Returns true if two DATA nodes are semantically and structurally identical:
//  - both nodes have the same GShape
//  - both nodes are produced by the same port numbers
//  - both nodes have the same number of output edges
//  (output edges' ports are not checked here)
//
// @param first - first node to compare
// @param firstPorts - a single element vector with first DATA node's producer output port
// @param firstMeta - metadata of first
// @param second - second node to compare
// @param secondPorts - a single element vector with second DATA node's producer output port
// @param secondMeta - metadata of second
bool compareDataNodes(const ade::NodeHandle& first, const std::vector<std::size_t>& firstPorts,
                      const Metadata& firstMeta,
                      const ade::NodeHandle& second, const std::vector<std::size_t>& secondPorts,
                      const Metadata& secondMeta) {
    if (secondMeta.get<cv::gimpl::NodeType>().t != cv::gimpl::NodeType::DATA) {
        throw std::logic_error("NodeType of passed node as second argument"
                               "shall be NodeType::DATA!");
    }

    if (firstMeta.get<cv::gimpl::Data>().shape != secondMeta.get<cv::gimpl::Data>().shape) {
        return false;
    }

    if (*firstPorts.begin() != *secondPorts.begin()) {
        return false;
    }

    const auto& firstOutputEdges = first->outEdges();
    const auto& secondOutputEdges = second->outEdges();

    if (firstOutputEdges.size() != secondOutputEdges.size()) {
        return false;
    }

    // FIXME: Because of new changes which introduce existence of unused DATA nodes
    // check that first and second nodes have the same type of DATA::Storage.

    return true;
};

// Returns true if two OP nodes semantically and structurally identical:
//    - both nodes have the same kernel name
//    - both nodes are produced by the same port numbers
//    - if any of the nodes are in the array with visited matchings, then:
//      first node is equal to found matching first argument and
//      second node is equal to found matching second argument
//
// @param first - first node to compare
// @param firstPorts - ports' vector of current connections between first node and an parent active
//                     DATA node
// @param firstMeta - metadata of first
// @param second - second node to compare
// @param secondPorts - ports' vector of current connections between second node and an parent
//                      active DATA node
// @param secondMeta - metadata of second
// @param [out] isAlreadyVisited - set to true if first and second nodes have been already visited
bool compareOpNodes(const VisitedMatchings& matchedVisitedNodes,
                    const ade::NodeHandle& first, std::vector<std::size_t> firstPorts,
                    const Metadata& firstMeta,
                    const ade::NodeHandle& second, std::vector<std::size_t> secondPorts,
                    const Metadata& secondMeta,
                    bool& isAlreadyVisited) {
    if (secondMeta.get<cv::gimpl::NodeType>().t != cv::gimpl::NodeType::OP) {
        throw std::logic_error("NodeType of passed node as second argument shall be NodeType::OP!");
    }

    // Assuming that if kernels names are the same then
    // output DATA nodes counts from kernels are the same.
    // Assuming that if kernels names are the same then
    // input DATA nodes counts to kernels are the same.
    if (firstMeta.get<cv::gimpl::Op>().k.name != secondMeta.get<cv::gimpl::Op>().k.name) {
        return false;
    }

    std::sort(firstPorts.begin(), firstPorts.end());
    std::sort(secondPorts.begin(), secondPorts.end());
    if (firstPorts != secondPorts) {
        return false;
    }

    // Shall work, but it is good to test on the cases where multiple start pattern OP nodes
    // maps to the test's one.
    auto foundIt = std::find_if(matchedVisitedNodes.begin(), matchedVisitedNodes.end(),
                               [&first, &second](const std::pair<ade::NodeHandle,
                                                                 ade::NodeHandle>& match)
                               {return first == match.first || second == match.second; });
    if (foundIt != matchedVisitedNodes.end()) {
        if (first != foundIt->first || second != foundIt->second) {
            return false;
        }

        isAlreadyVisited = true;
    }

    return true;
};

// Retrieves and return sample from the cartesian product of candidates sets
VisitedMatchings sampleFromProduct(std::size_t sampleIdx, // index of the sample in the product
                                   const MultipleMatchings& candidatesSets) // map of nodes to sets
                                                                            // of candidates
                                                                           {
    VisitedMatchings matchingsSample;

    std::size_t quo = sampleIdx;
    for (const auto& setForNode : candidatesSets) {
        // TODO: order is not determined: for ex., for last node.
        // May be use ordered set and map to ensure order?
        auto size = setForNode.second.size();

        // The below code block decodes sampleIdx into a particular sample from cartesian product
        // of candidates sets.
        std::size_t index = quo % size;
        quo = quo / size;
        const auto& candidate = setForNode.second[index];
        matchingsSample.push_back({ setForNode.first, candidate });
    }

    return matchingsSample;
}

// Depending on type of the node retrieve port number (IN/OUT) of the edge entering this node.
std::size_t labelOf (const ade::NodeHandle& node, // reader node
                     const ade::EdgeHandle& edge, // edge entering the reader node
                     const Graph& graph) // graph containing node and edge
                                        {

    if (graph.metadata(node).get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::OP) {
        return graph.metadata(edge).get<cv::gimpl::Input>().port;
    }
    else {
        return graph.metadata(edge).get<cv::gimpl::Output>().port;
    }
};

inline bool IS_STARTPOINT(const ade::NodeHandle& nh){
    return nh->inEdges().empty();
}

inline bool IS_ENDPOINT(const ade::NodeHandle& nh){
    // FIXME: Because of new changes which introduce existence of unused DATA nodes
    // Try to rely on the nh Data::Storage::OUTPUT
    return nh->outEdges().empty();
}
}  // anonymous namespace

// Routine relies on the logic that 1 DATA node may have only 1 input edge.
cv::gimpl::SubgraphMatch
cv::gimpl::findMatches(const cv::gimpl::GModel::Graph& patternGraph,
                       const cv::gimpl::GModel::Graph& testGraph) {

    //TODO: Possibly, we may add N^2 check whether this graph may match or not at all.
    //      Check that all pattern OP nodes exist in computational graph.

    //---------------------------------------------------------------
    // Identify operations which start and end our pattern
    SubgraphMatch::S patternStartOpNodes, patternEndOpNodes;

    const auto& patternInputDataNodes = patternGraph.metadata().get<cv::gimpl::Protocol>().in_nhs;
    const auto& patternOutputDataNodes = patternGraph.metadata().get<cv::gimpl::Protocol>().out_nhs;

    for (const auto& node : patternInputDataNodes) {
        auto opNodes = node->outNodes();
        patternStartOpNodes.insert(opNodes.begin(), opNodes.end());
    }

    for (const auto& node : patternOutputDataNodes) {
        auto opNodes = node->inNodes();
        // May be switched to patternEndOpNodes.insert(*opNodes.begin());
        patternEndOpNodes.insert(opNodes.begin(), opNodes.end());
    }

    std::unordered_map<ade::NodeHandle,              // pattern OP node
                       std::vector<ade::NodeHandle>, // nodes in the test graph which match
                                                     // to the pattern OP node
                       ade::HandleHasher<ade::Node>> allMatchingsForStartOpNodes;

    //Filling of allMatchingsForStartOpNodes
    std::size_t possibleStartPointsCount = 1;

    // For every starting OP node of pattern identify matching candidates(there may be many)
    // in test graph.
    auto testOpNodes = ade::util::filter(testGraph.nodes(),
                                         [&](const ade::NodeHandle& node) {
                                             return testGraph.metadata(node).
                                                        get<cv::gimpl::NodeType>().t
                                                    == cv::gimpl::NodeType::OP;
                                         });
    for (const auto& patternStartOpNode : patternStartOpNodes) {
        const auto& patternOpMeta = patternGraph.metadata(patternStartOpNode);

        auto& possibleMatchings = allMatchingsForStartOpNodes[patternStartOpNode];
        std::copy_if(testOpNodes.begin(), testOpNodes.end(), std::back_inserter(possibleMatchings),
            [&](const ade::NodeHandle& testOpNode) {
                const auto& testOpMeta = testGraph.metadata(testOpNode);

                bool stub = false;
                return compareOpNodes({ },
                                      patternStartOpNode, {  }, patternOpMeta,
                                      testOpNode, {  }, testOpMeta,
                                      stub);
            });

        if (possibleMatchings.size() == 0) {
            // Pattern graph is not matched
            return SubgraphMatch { };
        }

        possibleStartPointsCount *= possibleMatchings.size();
    }

    SubgraphMatch::M subgraphStartOps;
    SubgraphMatch::M subgraphEndOps;
    // FIXME: consider moving to S
    std::list<ade::NodeHandle> subgraphInternals;


    // Structural matching first, semantic matching second.

    // 'patternFound' means pattern is matched.
    bool patternFound = false;
    std::size_t i = 0;
    while (!patternFound && (i < possibleStartPointsCount)) {
        subgraphStartOps.clear();
        subgraphEndOps.clear();
        subgraphInternals.clear();

        // List of the pairs representing matchings of pattern node to the test node.
        VisitedMatchings matchedVisitedNodes;

        // Cartesian product of candidate sets for start OP nodes gives set of samples
        // as possible matchings for start OP nodes.
        // Let allMatchingsForStartOpNodes looks like:  x1 : [ y1 ]
        //                                              x2 : [ y2, y3 ]
        // Cartesian product of two these candidates sets (for x1 and x2 pattern nodes
        // correspondingly) produces two samples of matchings for x1, x2:
        //                         [ (x1, y1), (x2, y2) ]
        //                         [ (x1, y1), (x2, y3) ]
        //

        // Here we fill matchedVisitedNodes list with the next sample from the cartesian product
        // of candidates sets.
        // i is traversing full cartesian product of candidates sets.
        matchedVisitedNodes = sampleFromProduct(i, allMatchingsForStartOpNodes);

        bool stop = false;

        // matchIt is an iterator to a pair of pattern ade::NodeHandle to test's ade::nodeHandle.
        auto matchIt = matchedVisitedNodes.begin();
        std::size_t size = matchedVisitedNodes.size();

        while (!stop) {
            // The following loop traverses through the current level of matchings.
            // Every iteration we consider only one certain pair of matched nodes.
            for (std::size_t index = 0u; index < size && !stop; ++index, ++matchIt) {

                // Check if a given matchIt->first node is an pattern-ending OP node.
                // If it is just remember it in a special map.
                bool cond1 = std::find(patternEndOpNodes.begin(),
                                       patternEndOpNodes.end(),
                                       matchIt->first)
                             != patternEndOpNodes.end();
                if (cond1) {
                    subgraphEndOps[matchIt->first] = matchIt->second;
                }

                // Check if a given matchIt->first node is an pattern-starting OP node.
                // If it is just remember it in a special map.
                bool cond2 = std::find(patternStartOpNodes.begin(),
                                       patternStartOpNodes.end(),
                                       matchIt->first)
                             != patternStartOpNodes.end();
                if (cond2) {
                    subgraphStartOps[matchIt->first] = matchIt->second;
                }

                // If neither of conditions are true mark the test node as an internal one.
                if (!cond1 && !cond2) {
                    subgraphInternals.push_back(matchIt->second);
                }

                //-------------------------------------------------------------------------------
                // Given the current pattern/test matching of nodes, traverse their descendants.
                // For every descendant store the port of the edge connecting to it.
                // NOTE: the nature of port number may vary: it may be either IN for OP nodes
                // or OUT for DATA ones
                LabeledNodes patternOutputNodesLabeled;
                LabeledNodes testOutputNodesLabeled;

                auto patternOutputEdges = matchIt->first->outEdges();
                auto testOutputEdges = matchIt->second->outEdges();

                for (const auto& patternOutputEdge : patternOutputEdges) {
                    const auto& dstNh = patternOutputEdge->dstNode();
                    if (!IS_ENDPOINT(dstNh)) {
                        //Assuming that there is no case for the op node without output data nodes.
                        patternOutputNodesLabeled[dstNh].
                                push_back(labelOf(dstNh, patternOutputEdge, patternGraph));
                    }
                }

                for (const auto& testOutputEdge : testOutputEdges) {
                    const auto& dstNh = testOutputEdge->dstNode();
                    testOutputNodesLabeled[dstNh].
                            push_back(labelOf(dstNh, testOutputEdge, testGraph));
                }

                //---------------------------------------------------------------------------------
                // Traverse through labeled descendants of pattern node and for every descedant
                // find a matching in labeled descendants of corresponding test node
                for (const auto& patternNode : patternOutputNodesLabeled) {
                    bool isAlreadyVisited = false;
                    const auto& patternNodeMeta = patternGraph.metadata(patternNode.first);

                    auto testIt = std::find_if(testOutputNodesLabeled.begin(),
                                               testOutputNodesLabeled.end(),
                        [&](const std::pair<const ade::NodeHandle,
                                           std::vector<std::size_t>>& testNode) {
                        const auto& testNodeMeta = testGraph.metadata(testNode.first);

                        auto patternNodeType = patternNodeMeta.get<cv::gimpl::NodeType>().t;

                        switch(patternNodeType) {
                        case cv::gimpl::NodeType::DATA:
                            return compareDataNodes(patternNode.first, patternNode.second,
                                                    patternNodeMeta,
                                                    testNode.first, testNode.second,
                                                    testNodeMeta);
                        case cv::gimpl::NodeType::OP:
                            return compareOpNodes(matchedVisitedNodes,
                                                  patternNode.first, patternNode.second,
                                                  patternNodeMeta,
                                                  testNode.first, testNode.second,
                                                  testNodeMeta,
                                                  isAlreadyVisited);
                        default:
                            break;
                        }
                        GAPI_Error("Unsupported Node type!");
                    });

                    if (testIt == testOutputNodesLabeled.end()) {
                        stop = true;
                        break;
                    }

                    // Update matchedVisitedNodes list with found pair of nodes if the pair
                    // has not been visited before.
                    if (!isAlreadyVisited) {
                        matchedVisitedNodes.push_back({ patternNode.first, testIt->first });
                    }
                } // Loop traversed patternOutputNodesLabeled
            } // Loop traversed matchedVisitedNodes

            // Suppose, pattern and test graphs' structures without input DATA nodes look like:
            //         Pattern graph                Test graph
            //        op1       op2               t_op1      t_op2
            //      +-----+   +-----+            +-----+    +-----+
            //      v     v   v     v            v     v    v     v
            //      d1    d2  d3    d4          t_d1  t_d2 t_d3  t_d4
            //      v     v   v     v            v     v    v     v
            //     ...   ... ...   ...          ...   ...  ...   ...

            // matchedVisitedNodes content before previous loop execution:
            //     op1 <--> t_op1, op2 <--> t_op2
            // matchedVisitedNodes content after previous loop execution (extended with the next
            // level of matchings):
            //     op1 <--> t_op1, op2 <--> t_op2 | d1 <--> t_d1, d2 <--> t_d2, d3 <--> t_d3, d4 <--> t_d4
            //                                           ^
            //                                           |
            //                                      matchIt
            //
            // matchIt iterator points to the first matching in next level if the next level exists.
            // If there is no next level, matchIt == matchedVisitedNodes.end() and all pattern
            // levels (except ones for IN/OUT data nodes) have been already processed, so,
            // pattern subgraph is found.

            if (!stop) {
                // Check if pattetn subgraph is found
                if (matchIt == matchedVisitedNodes.end()) {
                    // Found
                    stop = true;
                    patternFound = true;
                }

                // Update 'size' with the size of the new level of matchings
                size = static_cast<std::size_t>(std::distance(matchIt, matchedVisitedNodes.end()));
            }
        }

        if (!patternFound){
            // Pattern subgraph is not matched.
            // Switch to the next combination of starting points
            ++i;
            continue;
        }

        SubgraphMatch::M inputApiMatch;
        SubgraphMatch::M outputApiMatch;

        // Traversing current result for starting OPs
        for (auto it = subgraphStartOps.begin();
                it != subgraphStartOps.end() && patternFound; ++it) {
            const auto& match = *it;
            auto patternInputEdges = match.first->inEdges();
            auto testInputEdges = match.second->inEdges();

            SubgraphMatch::S patternUniqInNodes(match.first->inNodes().begin(),
                                                match.first->inNodes().end());
            SubgraphMatch::S testUniqInNodes(match.second->inNodes().begin(),
                                             match.second->inNodes().end());

            if (patternUniqInNodes.size() < testUniqInNodes.size()) {
                inputApiMatch.clear();
                patternFound = false;
                break;
            }
            // Else, patternInNodes.size() > testInNodes.size() is considered as valid case.

            // Match pattern input DATA nodes with boundary matched test DATA nodes.
            for (const auto& patternInEdge : patternInputEdges) {

                // Not all start OP nodes are located in the beginning of the pattern graph
                // Start OP may have one input DATA node as an Protocol IN node and other
                // input DATA nodes produced from another operations
                if (!IS_STARTPOINT(patternInEdge->srcNode())) {
                    continue;
                }

                auto patternInputPort =
                        patternGraph.metadata(patternInEdge).get<cv::gimpl::Input>().port;

                auto matchedIt = std::find_if(testInputEdges.begin(), testInputEdges.end(),
                    [&](const ade::EdgeHandle& testInEdge) -> bool {
                    auto testInputPort =
                            testGraph.metadata(testInEdge).get<cv::gimpl::Input>().port;

                    if (patternInputPort != testInputPort) {
                        return false;
                    }

                    auto foundIt = inputApiMatch.find(patternInEdge->srcNode());
                    if (foundIt != inputApiMatch.end()) {
                        if (testInEdge->srcNode() != foundIt->second) {
                            return false;
                        }
                        return true;
                    }

                    // Update inputApiMatch map only if the pair of nodes isn't in the map already
                    inputApiMatch[patternInEdge->srcNode()] = testInEdge->srcNode();
                    return true;
                });

                if (matchedIt == testInputEdges.end()) {
                    inputApiMatch.clear();
                    patternFound  = false;
                    break;
                }
            } // Loop traversed patternInputEdges
        } // Loop traversed sugraphStartOps

        if (!patternFound) {
            // Pattern IN data nodes can not be matched.
            // Switch to the next combination of starting points
            ++i;
            continue;
        }

        // Create vector with the correctly ordered IN data nodes in the test subgraph
        std::vector<ade::NodeHandle> inputTestDataNodes;
        for (const auto& patternInNode : patternInputDataNodes) {
            inputTestDataNodes.push_back(inputApiMatch.at(patternInNode));
        }

        // Traversing current result for ending OPs
        // There is an assumption that if the pattern subgraph is matched, then
        // OUT data nodes shall be definitely matched
        for (const auto& match : subgraphEndOps) {
            auto patternOutputEdges = match.first->outEdges();
            auto testOutputEdges = match.second->outEdges();

            GAPI_Assert(patternOutputEdges.size() == testOutputEdges.size()
                        &&
                        "Ending OP nodes are matched, so OPs' outputs count shall be the same!");

            // Match pattern output DATA nodes with boundary matched test DATA nodes.
            for (const auto& patternOutEdge : patternOutputEdges) {

                // Not all end OP nodes are located in the ending of the pattern graph
                // End OP node may have one output DATA node as an Protocol OUT node and other
                // output DATA nodes as input for another operations
                if (!IS_ENDPOINT(patternOutEdge->dstNode())) {
                    continue;
                }

                auto patternOutputPort =
                        patternGraph.metadata(patternOutEdge).get<cv::gimpl::Output>().port;

                auto matchedIt = std::find_if(testOutputEdges.begin(), testOutputEdges.end(),
                    [&](const ade::EdgeHandle& testOutEdge) -> bool {
                    auto testOutputPort =
                            testGraph.metadata(testOutEdge).get<cv::gimpl::Output>().port;

                    if (patternOutputPort != testOutputPort) {
                        return false;
                    }

                    outputApiMatch[patternOutEdge->dstNode()] = testOutEdge->dstNode();
                    return true;
                });

                GAPI_Assert(matchedIt != testOutputEdges.end()
                            &&
                            "There shall be a match for every OUT data node from ending OP node,"
                            "if ending OP node matches");
            }

        }

        // Create vector with the correctly ordered OUT data nodes in the test subgraph
        std::vector<ade::NodeHandle> outputTestDataNodes;
        for (const auto& patternOutNode : patternOutputDataNodes) {
            outputTestDataNodes.push_back(outputApiMatch.at(patternOutNode));
        }

        SubgraphMatch subgraph;

        subgraph.inputDataNodes = std::move(inputApiMatch);
        subgraph.startOpNodes = std::move(subgraphStartOps);
        subgraph.internalLayers = std::move(subgraphInternals);
        subgraph.finishOpNodes = std::move(subgraphEndOps);
        subgraph.outputDataNodes = std::move(outputApiMatch);

        subgraph.inputTestDataNodes = std::move(inputTestDataNodes);
        subgraph.outputTestDataNodes = std::move(outputTestDataNodes);

        return subgraph;

    }

    return SubgraphMatch { };
}
