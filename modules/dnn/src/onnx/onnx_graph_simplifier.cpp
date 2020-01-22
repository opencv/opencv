// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#include "../graph_simplifier.hpp"
#include "onnx_graph_simplifier.hpp"

#include <queue>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// This wrapper can behave differently for fake input nodes and real graph nodes.
class ONNXNodeWrapper : public ImportNodeWrapper
{
public:
    ONNXNodeWrapper(opencv_onnx::NodeProto* _node = 0) : node(_node) {}

    virtual int getNumInputs() const CV_OVERRIDE
    {
        return node ? node->input_size() : 0;
    }

    virtual std::string getInputName(int idx) const CV_OVERRIDE
    {
        CV_Assert_N(node, idx < node->input_size());
        return node->input(idx);
    }

    virtual std::string getType() const CV_OVERRIDE
    {
        return node ? node->op_type() : "";
    }

    virtual void setType(const std::string& type) CV_OVERRIDE
    {
        CV_Assert(node);
        node->set_op_type(type);
    }

    virtual void setInputNames(const std::vector<std::string>& inputs) CV_OVERRIDE
    {
        CV_Assert(node);
        node->clear_input();
        for (int i = 0; i < inputs.size(); ++i)
            node->add_input(inputs[i]);
    }

    opencv_onnx::NodeProto* node;
};

// ONNX graph's inputs are separate from nodes so we index them before the rest of nodes.
class ONNXGraphWrapper : public ImportGraphWrapper
{
public:
    ONNXGraphWrapper(opencv_onnx::GraphProto& _net) : net(_net)
    {
        numInputs = net.input_size();
    }

    virtual Ptr<ImportNodeWrapper> getNode(int idx) const CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = 0;
        if (idx >= numInputs)
            node = net.mutable_node(idx - numInputs);
        return makePtr<ONNXNodeWrapper>(node);
    }

    virtual int getNumNodes() const CV_OVERRIDE
    {
        return numInputs + net.node_size();
    }

    virtual std::string getNodeName(int idx) const CV_OVERRIDE
    {
        if (idx < numInputs)
            return net.input(idx).name();
        else
            return net.node(idx - numInputs).output(0);
    }

    virtual void removeNode(int idx) CV_OVERRIDE
    {
        CV_Assert(idx >= numInputs);
        net.mutable_node()->DeleteSubrange(idx - numInputs, 1);
    }

private:
    int numInputs;
    opencv_onnx::GraphProto& net;
};

class SoftMaxSubgraph : public Subgraph
{
public:
    SoftMaxSubgraph() : axis(1)
    {
        int input = addNodeToMatch("");
        int inpExp = addNodeToMatch("Exp", input);
        int sum = addNodeToMatch("ReduceSum", inpExp);
        addNodeToMatch("Div", inpExp, sum);
        setFusedNode("Softmax", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            Ptr<ImportNodeWrapper> sum = net->getNode(matchedNodesIds[1]);
            opencv_onnx::NodeProto* node = sum.dynamicCast<ONNXNodeWrapper>()->node;

            for (int i = 0; i < node->attribute_size(); i++)
            {
                opencv_onnx::AttributeProto attr = node->attribute(i);
                if (attr.name() != "axes")
                    continue;
                if (attr.ints_size() != 1)
                    CV_Error(Error::StsNotImplemented, format("Unexpected number of axes: %d", attr.ints_size()));
                axis = attr.ints(0);
                return true;
            }
            CV_Error(Error::StsNotImplemented, "Missed axes attribute");
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>&,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::AttributeProto* attr = node->add_attribute();
        attr->set_name("axis");
        attr->set_i(axis);
    }

private:
    int axis;
};

void simplifySubgraphs(opencv_onnx::GraphProto& net)
{
    std::vector<Ptr<Subgraph> > subgraphs;
    subgraphs.push_back(makePtr<SoftMaxSubgraph>());

    simplifySubgraphs(Ptr<ImportGraphWrapper>(new ONNXGraphWrapper(net)), subgraphs);
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
