// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

#include "graph_simplifier.hpp"

#include <queue>

namespace cv { namespace dnn {

Subgraph::~Subgraph() {}

int Subgraph::addNodeToMatch(const std::string& op, int input_0, int input_1,
                             int input_2, int input_3)
{
    int nodeInputs[] = {input_0, input_1, input_2, input_3};
    int numInputs = 0;
    for (int i = 0; i < 4; ++i)
    {
        numInputs += (int)(nodeInputs[i] != -1);
    }
    return addNodeToMatch(op, std::vector<int>(&nodeInputs[0], &nodeInputs[0] + numInputs));
}

int Subgraph::addNodeToMatch(const std::string& op, const std::vector<int>& inputs_)
{
    for (int i = 0; i < inputs_.size(); ++i)
    {
        CV_Assert(inputs_[i] < (int)nodes.size());
    }
    nodes.push_back(op);
    inputs.push_back(inputs_);
    return nodes.size() - 1;
}

void Subgraph::setFusedNode(const std::string& op, int input_0, int input_1,
                            int input_2, int input_3, int input_4, int input_5)
{
    int nodeInputs[] = {input_0, input_1, input_2, input_3, input_4, input_5};
    int numInputs = 0;
    for (int i = 0; i < 6; ++i)
    {
        CV_Assert(nodeInputs[i] < (int)nodes.size());
        numInputs += (int)(nodeInputs[i] != -1);
    }
    setFusedNode(op, std::vector<int>(&nodeInputs[0], &nodeInputs[0] + numInputs));
}

void Subgraph::setFusedNode(const std::string& op, const std::vector<int>& inputs_)
{
    fusedNodeInputs = inputs_;
    fusedNodeOp = op;
}

int Subgraph::getInputNodeId(const Ptr<ImportGraphWrapper>& net,
                             const Ptr<ImportNodeWrapper>& node,
                             int inpId)
{
    CV_Assert(inpId < node->getNumInputs());
    std::string name = node->getInputName(inpId);
    const int numNodes = net->getNumNodes();
    for (int i = 0; i < numNodes; ++i)
    {
        const int numOutputs = net->getNumOutputs(i);
        for (int j = 0; j < numOutputs; j++)
        {
            if (net->getOutputName(i, j) == name)
                return i;
        }
    }
    CV_Error(Error::StsParseError, "Input node with name " + name + " not found");
}

bool Subgraph::match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                     std::vector<int>& matchedNodesIds)
{
    matchedNodesIds.clear();

    // Collection of all matchings states across branching.
    // If there is no commutative ops in the subgraph - there would be just a single map.
    std::vector<std::shared_ptr<std::map<int, int>>> matchCandidates;
    matchCandidates.push_back(makePtr<std::map<int, int>>());

    struct State
    {
        int nodeToMatch;
        int targetNodeId;
        // Every state refers to current matchings pairs as well as
        // matchings from parent branches produced by commutative ops.
        std::vector<std::shared_ptr<std::map<int, int>>> matchings;

        // When we register a matching pair we should register it in every parent branch.
        // This is actual for branching in case of commutative ops only.
        void addMatch(std::pair<int, int> match)
        {
            for (auto& m : matchings)
                m->insert(match);
        }
    };

    std::queue<State> states;
    states.push({nodeId, (int)nodes.size() - 1, matchCandidates});

    while (!states.empty())
    {
        auto state = states.front();
        states.pop();
        int nodeToMatch = state.nodeToMatch;
        int targetNodeId = state.targetNodeId;
        auto matchings = state.matchings.back();

        if (matchings->find(targetNodeId) != matchings->end())
            continue;

        // Empty placeholder matches with any input type
        if (nodes[targetNodeId].empty()) {
            state.addMatch({targetNodeId, nodeToMatch});
            continue;
        }

        const Ptr<ImportNodeWrapper> node = net->getNode(nodeToMatch);
        if (node->getType() != nodes[targetNodeId])
            continue;

        std::vector<int>& inputNodes = inputs[targetNodeId];
        if (inputNodes.size() != node->getNumInputs())
            continue;

        state.addMatch({targetNodeId, nodeToMatch});

        bool isCommutative = net->isCommutativeOp(node->getType());
        if (isCommutative)
        {
            if (inputNodes.size() != 2)
                CV_Error(Error::StsNotImplemented, "Commutative op fusion with more than 2 inputs");

            auto newMatchings = makePtr<std::map<int, int>>(*matchings);
            matchCandidates.push_back(newMatchings);
            state.matchings.push_back(newMatchings);
            states.push({getInputNodeId(net, node, 0), inputNodes[0], state.matchings});
            states.push({getInputNodeId(net, node, 1), inputNodes[1], state.matchings});
            state.matchings.pop_back();

            newMatchings = makePtr<std::map<int, int>>(*matchings);
            matchCandidates.push_back(newMatchings);
            state.matchings.push_back(newMatchings);
            states.push({getInputNodeId(net, node, 0), inputNodes[1], state.matchings});
            states.push({getInputNodeId(net, node, 1), inputNodes[0], state.matchings});
            state.matchings.pop_back();
        }
        else
        {
            for (int j = 0; j < inputNodes.size(); ++j)
            {
                nodeId = getInputNodeId(net, node, j);
                states.push({nodeId, inputNodes[j], state.matchings});
            }
        }
    }
    for (auto& matchings : matchCandidates)
    {
        if (matchings->size() != nodes.size())
            continue;
        matchedNodesIds.resize(matchings->size());
        for (int i = 0; i < matchings->size(); ++i)
        {
            CV_Assert(matchings->find(i) != matchings->end());
            matchedNodesIds[i] = matchings->at(i);
        }
        return true;
    }
    return false;
}

void Subgraph::replace(const Ptr<ImportGraphWrapper>& net, const std::vector<int>& matchedNodesIds)
{
    // Extract names of input nodes.
    std::vector<std::string> inputsNames(fusedNodeInputs.size());
    for (int i = 0; i < fusedNodeInputs.size(); ++i)
    {
        std::string inpName;
        // Find input node name looking at inputs of fused nodes.
        for (int j = 0; j < matchedNodesIds.size() && inpName.empty(); ++j)
        {
            Ptr<ImportNodeWrapper> node = net->getNode(matchedNodesIds[j]);
            std::vector<int>& inpIndices = inputs[j];

            CV_Assert(inpIndices.empty() || node->getNumInputs() == inpIndices.size());
            for (int k = 0; k < inpIndices.size(); ++k)
            {
                if (inpIndices[k] == fusedNodeInputs[i])
                {
                    inpName = node->getInputName(k);
                    break;
                }
            }
        }
        CV_Assert(!inpName.empty());
        inputsNames[i] = inpName;
    }

    Ptr<ImportNodeWrapper> node = net->getNode(matchedNodesIds.back());

    // Modify the last node to be a fused one.
    node->setType(fusedNodeOp);
    node->setInputNames(inputsNames);

    std::vector<Ptr<ImportNodeWrapper> > inputNodes(inputsNames.size());
    for (int i = 0; i < inputsNames.size(); ++i)
    {
        inputNodes[i] = net->getNode(getInputNodeId(net, node, i));
    }
    finalize(net, node, inputNodes);
}

void Subgraph::finalize(const Ptr<ImportGraphWrapper>& net,
                        const Ptr<ImportNodeWrapper>& fusedNode,
                        std::vector<Ptr<ImportNodeWrapper> >& inputs) {}

void simplifySubgraphs(const Ptr<ImportGraphWrapper>& net,
                       const std::vector<Ptr<Subgraph> >& patterns)
{
    int numNodes = net->getNumNodes();
    std::vector<int> matchedNodesIds;
    std::vector<int> nodesToRemove;
    for (int j = 0; j < patterns.size(); ++j)
    {
        for (int i = 0; i < numNodes; ++i)
        {
            if (patterns[j]->match(net, i, matchedNodesIds))
            {
                patterns[j]->replace(net, matchedNodesIds);
                // Remove matched nodes except the last one.
                nodesToRemove.insert(nodesToRemove.end(), matchedNodesIds.begin(), matchedNodesIds.end() - 1);
            }
        }
    }

    if (nodesToRemove.empty())
        return;

    // Collect reference counts for every node
    std::vector<int> refcounts(net->getNumNodes(), 0);
    std::map<std::string, int> nodeIds;

    // Register node outputs.
    // Every usage of one of the node's outputs should be counted.
    for (int nodeId = 0; nodeId < refcounts.size(); ++nodeId) {
        for (int i = 0; i < net->getNumOutputs(nodeId); ++i) {
            std::string name = net->getOutputName(nodeId, i);
            nodeIds[name] = nodeId;
        }
    }

    for (int nodeId = 0; nodeId < refcounts.size(); ++nodeId) {
        // Increase counters for node's inputs
        auto node = net->getNode(nodeId);
        for (int i = 0; i < node->getNumInputs(); ++i) {
            std::string inpName = node->getInputName(i);
            if (inpName.empty())
                continue;
            CV_Assert(nodeIds.find(inpName) != nodeIds.end());
            refcounts[nodeIds[inpName]] += 1;
        }
    }

    // Remove all fused nodes. Indices expected to be in descending order.
    std::sort(nodesToRemove.begin(), nodesToRemove.end(), [](int a, int b) { return a > b; });
    for (int nodeId : nodesToRemove) {
        if (refcounts[nodeId] == 0) {
            // Decrease references to node's inputs and remove node itself
            auto node = net->getNode(nodeId);
            for (int i = 0; i < node->getNumInputs(); ++i) {
                std::string inpName = node->getInputName(i);
                refcounts[nodeIds[inpName]] -= 1;
            }
            net->removeNode(nodeId);
            refcounts[nodeId] = -1;  // Same node cannot be removed twice
        }
    }
}

}}  // namespace cv::dnn
