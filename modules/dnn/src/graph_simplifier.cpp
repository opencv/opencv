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

    std::queue<int> nodesToMatch;
    std::queue<int> targetNodes;
    std::vector<std::pair<int, int> > matchings;
    matchings.reserve(nodes.size());
    nodesToMatch.push(nodeId);
    targetNodes.push(nodes.size() - 1);
    while (!nodesToMatch.empty())
    {
        int nodeToMatch = nodesToMatch.front();
        int targetNodeId = targetNodes.front();
        nodesToMatch.pop();
        targetNodes.pop();

        if (std::find_if(matchings.begin(), matchings.end(), [&](const std::pair<int, int>& match){ return match.first == targetNodeId; }) !=
            matchings.end())
            continue;

        // Empty placeholder matches with any input type
        if (nodes[targetNodeId].empty()) {
            matchings.push_back({targetNodeId, nodeToMatch});
            continue;
        }

        const Ptr<ImportNodeWrapper> node = net->getNode(nodeToMatch);
        if (node->getType() != nodes[targetNodeId])
            continue;

        std::vector<int>& inputNodes = inputs[targetNodeId];
        if (inputNodes.size() != node->getNumInputs())
            continue;

        bool isCommutative = net->isCommutativeOp(node->getType());

        for (int j = 0; j < inputNodes.size(); ++j)
        {
            // Sometimes, ONNX may have input but it's empty (see Clip layer from reduceL2_subgraph2_2 testcase)
            if (node->getInputName(j).empty())
                continue;
            nodeId = getInputNodeId(net, node, j);
            const Ptr<ImportNodeWrapper> inpNode = net->getNode(nodeId);
            if (isCommutative)
            {
                for (int i = 0; i < inputNodes.size(); ++i)
                {
                    nodesToMatch.push(nodeId);
                    targetNodes.push(inputNodes[i]);
                }
            }
            else
            {
                nodesToMatch.push(nodeId);
                targetNodes.push(inputNodes[j]);
            }
        }
        matchings.push_back({targetNodeId, nodeToMatch});
    }
    if (matchings.size() != nodes.size())
        return false;

    // Sort matched by pattern nodes order.
    std::sort(matchings.begin(), matchings.end());
    matchedNodesIds.resize(matchings.size());
    for (int i = 0; i < matchings.size(); ++i)
    {
        matchedNodesIds[i] = matchings[i].second;
    }
    return true;
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
