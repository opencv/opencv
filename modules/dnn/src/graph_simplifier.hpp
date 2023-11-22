// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_GRAPH_SIMPLIFIER_HPP__
#define __OPENCV_DNN_GRAPH_SIMPLIFIER_HPP__

#include <string>

#include <opencv2/core.hpp>

namespace cv { namespace dnn {

class ImportNodeWrapper
{
public:
    virtual ~ImportNodeWrapper() {}

    virtual int getNumInputs() const = 0;

    virtual std::string getInputName(int idx) const = 0;

    virtual std::string getType() const = 0;

    virtual void setType(const std::string& type) = 0;

    virtual void setInputNames(const std::vector<std::string>& inputs) = 0;
};

class ImportGraphWrapper
{
public:
    virtual ~ImportGraphWrapper() {}

    virtual Ptr<ImportNodeWrapper> getNode(int idx) const = 0;

    virtual int getNumNodes() const = 0;

    virtual int getNumOutputs(int nodeId) const = 0;

    virtual std::string getOutputName(int nodeId, int outId) const = 0;

    virtual void removeNode(int idx) = 0;

    virtual bool isCommutativeOp(const std::string& type) const = 0;
};

class Subgraph  // Interface to match and replace subgraphs.
{
public:
    virtual ~Subgraph();

    // Add a node to be matched in the origin graph. Specify ids of nodes that
    // are expected to be inputs. Returns id of a newly added node.
    // TODO: Replace inputs to std::vector<int> in C++11
    int addNodeToMatch(const std::string& op, int input_0 = -1, int input_1 = -1,
                       int input_2 = -1, int input_3 = -1);

    int addNodeToMatch(const std::string& op, const std::vector<int>& inputs_);

    // Specify resulting node. All the matched nodes in subgraph excluding
    // input nodes will be fused into this single node.
    // TODO: Replace inputs to std::vector<int> in C++11
    void setFusedNode(const std::string& op, int input_0 = -1, int input_1 = -1,
                      int input_2 = -1, int input_3 = -1, int input_4 = -1,
                      int input_5 = -1);

    void setFusedNode(const std::string& op, const std::vector<int>& inputs_);

    static int getInputNodeId(const Ptr<ImportGraphWrapper>& net,
                              const Ptr<ImportNodeWrapper>& node,
                              int inpId);

    // Match TensorFlow subgraph starting from <nodeId> with a set of nodes to be fused.
    // Const nodes are skipped during matching. Returns true if nodes are matched and can be fused.
    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds);

    // Fuse matched subgraph.
    void replace(const Ptr<ImportGraphWrapper>& net, const std::vector<int>& matchedNodesIds);

    virtual void finalize(const Ptr<ImportGraphWrapper>& net,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >& inputs);

private:
    std::vector<std::string> nodes;         // Nodes to be matched in the origin graph.
    std::vector<std::vector<int> > inputs;  // Connections of an every node to it's inputs.

    std::string fusedNodeOp;           // Operation name of resulting fused node.
    std::vector<int> fusedNodeInputs;  // Inputs of fused node.
};

void simplifySubgraphs(const Ptr<ImportGraphWrapper>& net,
                       const std::vector<Ptr<Subgraph> >& patterns);

}}  // namespace dnn, namespace cv

#endif  // __OPENCV_DNN_GRAPH_SIMPLIFIER_HPP__
