// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifdef HAVE_PROTOBUF

#include "tf_graph_simplifier.hpp"

namespace cv { namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

using ::google::protobuf::RepeatedField;
using ::google::protobuf::MapPair;

class Subgraph  // Interface to match and replace TensorFlow subgraphs.
{
public:
    // Add a node to be matched in the origin graph. Specify ids of nodes that
    // are expected to be inputs. Returns id of a newly added node.
    // TODO: Replace inputs to std::vector<int> in C++11
    int addNodeToMatch(const std::string& op, int input_0 = -1, int input_1 = -1,
                       int input_2 = -1, int input_3 = -1)
    {
        int nodeInputs[] = {input_0, input_1, input_2, input_3};
        int numInputs = 0;
        for (int i = 0; i < 4; ++i)
        {
            numInputs += (int)(nodeInputs[i] != -1);
        }
        return addNodeToMatch(op, std::vector<int>(&nodeInputs[0], &nodeInputs[0] + numInputs));
    }

    int addNodeToMatch(const std::string& op, const std::vector<int>& inputs_)
    {
        for (int i = 0; i < inputs_.size(); ++i)
        {
            CV_Assert(inputs_[i] < (int)nodes.size());
        }
        nodes.push_back(op);
        inputs.push_back(inputs_);
        return nodes.size() - 1;
    }

    // Specify resulting node. All the matched nodes in subgraph excluding
    // input nodes will be fused into this single node.
    // TODO: Replace inputs to std::vector<int> in C++11
    void setFusedNode(const std::string& op, int input_0 = -1, int input_1 = -1,
                      int input_2 = -1, int input_3 = -1, int input_4 = -1,
                      int input_5 = -1)
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

    void setFusedNode(const std::string& op, const std::vector<int>& inputs_)
    {
        fusedNodeInputs = inputs_;
        fusedNodeOp = op;
        nodesToFuse.clear();
        for (int i = 0; i < nodes.size(); ++i)
        {
            if (std::find(fusedNodeInputs.begin(), fusedNodeInputs.end(), i) == fusedNodeInputs.end() &&
                nodes[i] != "Const")
                nodesToFuse.push_back(i);
        }
    }

    static const tensorflow::NodeDef& getInputNode(const tensorflow::GraphDef& net,
                                                   const tensorflow::NodeDef& node,
                                                   int inpId)
    {
        CV_Assert(inpId < node.input_size());
        std::string name = node.input(inpId);
        const int numNodes = net.node_size();
        for (int i = 0; i < numNodes; ++i)
        {
            if (net.node(i).name() == name)
                return net.node(i);
        }
        CV_Error(Error::StsParseError, "Input node with name " + name + " not found");
        return net.node(0);  // just return something
    }

    // Match TensorFlow subgraph starting from <nodeId> with a set of nodes to be fused.
    // Const nodes are skipped during matching. Returns true if nodes are matched and can be fused.
    virtual bool match(const tensorflow::GraphDef& net, int nodeId, std::vector<int>& matchedNodesIds)
    {
        matchedNodesIds.clear();
        matchedNodesIds.reserve(nodesToFuse.size());

        int numNodes = net.node_size();
        for (int i = 0; i < nodesToFuse.size(); ++i)
        {
            while (nodeId < numNodes && net.node(nodeId).op() == "Const")
            {
                nodeId += 1;
            }
            if (nodeId > numNodes - 1)
                return false;

            const tensorflow::NodeDef& node = net.node(nodeId);

            if (node.op() != nodes[nodesToFuse[i]])
                return false;

            std::vector<int>& inputNodes = inputs[nodesToFuse[i]];
            if (inputNodes.size() != node.input_size())
                return false;
            for (int j = 0; j < inputNodes.size(); ++j)
            {
                if (nodes[inputNodes[j]].empty())  // Unknown input node type.
                    continue;
                const tensorflow::NodeDef& inpNode = getInputNode(net, node, j);
                if (inpNode.op() != nodes[inputNodes[j]])
                    return false;
            }

            matchedNodesIds.push_back(nodeId);
            nodeId += 1;
        }
        return true;
    }

    // Fuse matched subgraph.
    void replace(tensorflow::GraphDef& net, const std::vector<int>& matchedNodesIds)
    {
        // Extract names of input nodes.
        std::vector<std::string> inputsNames(fusedNodeInputs.size());
        for (int i = 0; i < fusedNodeInputs.size(); ++i)
        {
            std::string inpName;
            // Find input node name looking at inputs of fused nodes.
            for (int j = 0; j < matchedNodesIds.size() && inpName.empty(); ++j)
            {
                const tensorflow::NodeDef &node = net.node(matchedNodesIds[j]);
                std::vector<int>& inpIndices = inputs[nodesToFuse[j]];

                CV_Assert(node.input_size() == inpIndices.size());
                for (int k = 0; k < inpIndices.size(); ++k)
                {
                    if (inpIndices[k] == fusedNodeInputs[i])
                    {
                        inpName = node.input(k);
                        break;
                    }
                }
            }
            CV_Assert(!inpName.empty());
            inputsNames[i] = inpName;
        }

        // Remove matched nodes except the last one. Indices in ascending order are expected.
        tensorflow::NodeDef* node = net.mutable_node(matchedNodesIds.back());
        for (int i = matchedNodesIds.size() - 2; i >= 0; --i)
            net.mutable_node()->DeleteSubrange(matchedNodesIds[i], 1);

        // Modify the last node to be a fused one.
        node->set_op(fusedNodeOp);
        node->clear_input();
        for (int i = 0; i < inputsNames.size(); ++i)
        {
            node->add_input(inputsNames[i]);
        }

        std::vector<tensorflow::NodeDef*> inputNodes(inputsNames.size());
        for (int i = 0; i < inputsNames.size(); ++i)
        {
            inputNodes[i] = (tensorflow::NodeDef*)&getInputNode(net, *node, i);
        }
        finalize(net, node, inputNodes);
    }

    virtual void finalize(tensorflow::GraphDef&, tensorflow::NodeDef*,
                          std::vector<tensorflow::NodeDef*>&) {}

private:
    std::vector<std::string> nodes;         // Nodes to be matched in the origin graph.
    std::vector<std::vector<int> > inputs;  // Connections of an every node to it's inputs.

    std::string fusedNodeOp;           // Operation name of resulting fused node.
    std::vector<int> nodesToFuse;      // Set of nodes to be fused.
    std::vector<int> fusedNodeInputs;  // Inputs of fused node.
};

class BatchNormSubgraph : public Subgraph
{
public:
    BatchNormSubgraph()
    {
        int input = addNodeToMatch("");
        int epsilon = addNodeToMatch("Const");
        int moving_variance = addNodeToMatch("Const");
        int moving_mean = addNodeToMatch("Const");
        int beta = addNodeToMatch("Const");
        int gamma = addNodeToMatch("Const");
        int add = addNodeToMatch("Add", moving_variance, epsilon);
        int rsqrt = addNodeToMatch("Rsqrt", add);
        int mul = addNodeToMatch("Mul", rsqrt, gamma);
        int mul_1 = addNodeToMatch("Mul", input, mul);
        int mul_2 = addNodeToMatch("Mul", moving_mean, mul);
        int sub = addNodeToMatch("Sub", beta, mul_2);
        addNodeToMatch("Add", mul_1, sub);

        setFusedNode("FusedBatchNorm", input, gamma, beta, moving_mean, moving_variance, epsilon);
    }

    virtual void finalize(tensorflow::GraphDef&, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes)
    {
        Mat epsMat = getTensorContent(inputNodes.back()->attr().at("value").tensor());
        CV_Assert(epsMat.total() == 1, epsMat.type() == CV_32FC1);

        fusedNode->mutable_input()->ReleaseLast();
        fusedNode->clear_attr();
        tensorflow::AttrValue epsilon;
        epsilon.set_f(epsMat.at<float>(0));
        fusedNode->mutable_attr()->insert(MapPair<std::string, tensorflow::AttrValue>("epsilon", epsilon));
    }
};

class BatchNormNoGammaSubgraph : public Subgraph
{
public:
    BatchNormNoGammaSubgraph()
    {
        int input = addNodeToMatch("");
        int epsilon = addNodeToMatch("Const");
        int moving_variance = addNodeToMatch("Const");
        int moving_mean = addNodeToMatch("Const");
        int beta = addNodeToMatch("Const");
        int add = addNodeToMatch("Add", moving_variance, epsilon);
        int rsqrt = addNodeToMatch("Rsqrt", add);
        int mul = addNodeToMatch("Mul", input, rsqrt);
        int mul_1 = addNodeToMatch("Mul", moving_mean, rsqrt);
        int sub = addNodeToMatch("Sub", beta, mul_1);
        addNodeToMatch("Add", mul, sub);

        // There is a fake reference to beta that will be replaced to a new gamma tensor.
        setFusedNode("FusedBatchNorm", input, beta, beta, moving_mean, moving_variance, epsilon);
    }

    virtual void finalize(tensorflow::GraphDef& net, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes)
    {
        Mat epsMat = getTensorContent(inputNodes.back()->attr().at("value").tensor());
        CV_Assert(epsMat.total() == 1, epsMat.type() == CV_32FC1);

        fusedNode->mutable_input()->ReleaseLast();
        fusedNode->clear_attr();
        tensorflow::AttrValue epsilon;
        epsilon.set_f(epsMat.at<float>(0));
        fusedNode->mutable_attr()->insert(MapPair<std::string, tensorflow::AttrValue>("epsilon", epsilon));

        tensorflow::NodeDef* gamma = net.add_node();
        gamma->set_op("Const");
        gamma->set_name(fusedNode->name() + "/gamma");
        // Just put a single value to recognize this node as Const.
        gamma->mutable_attr()->insert(MapPair<std::string, tensorflow::AttrValue>("value", epsilon));
        fusedNode->set_input(1, gamma->name());
    }
};

// tf.contrib.layers.flatten
class FlattenSubgraph : public Subgraph
{
public:
    FlattenSubgraph()
    {
        int input = addNodeToMatch("");
        int shape = addNodeToMatch("Const");
        int stack = addNodeToMatch("Const");
        int stack_1 = addNodeToMatch("Const");
        int stack_2 = addNodeToMatch("Const");
        int strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);
        int shape_pack = addNodeToMatch("Const");
        int pack = addNodeToMatch("Pack", strided_slice, shape_pack);
        addNodeToMatch("Reshape", input, pack);

        setFusedNode("Flatten", input);
    }
};

// tf.contrib.layers.flatten in case of unknown batch size
class FlattenShapeSubgraph : public Subgraph
{
public:
    FlattenShapeSubgraph()
    {
        int input = addNodeToMatch("");
        int shape = addNodeToMatch("Shape", input);
        int stack = addNodeToMatch("Const");
        int stack_1 = addNodeToMatch("Const");
        int stack_2 = addNodeToMatch("Const");
        int strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);
        int shape_pack = addNodeToMatch("Const");
        int pack = addNodeToMatch("Pack", strided_slice, shape_pack);
        addNodeToMatch("Reshape", input, pack);

        setFusedNode("Flatten", input);
    }
};

// K.layers.Softmax
class SoftMaxKerasSubgraph : public Subgraph
{
public:
    SoftMaxKerasSubgraph()
    {
        int input = addNodeToMatch("");
        int maxReductionIndices = addNodeToMatch("Const");
        int smMax = addNodeToMatch("Max", input, maxReductionIndices);
        int smSub = addNodeToMatch("Sub", input, smMax);
        int smExp = addNodeToMatch("Exp", smSub);
        int sumReductionIndices = addNodeToMatch("Const");
        int smSum = addNodeToMatch("Sum", smExp, sumReductionIndices);
        addNodeToMatch("RealDiv", smExp, smSum);

        setFusedNode("Softmax", input);
    }
};

class ReLU6KerasSubgraph : public Subgraph
{
public:
    ReLU6KerasSubgraph()
    {
        int input = addNodeToMatch("");
        int relu = addNodeToMatch("Relu", input);
        int maxValue = addNodeToMatch("Const");
        int clipValue = addNodeToMatch("Const");
        int minimum = addNodeToMatch("Minimum", relu, maxValue);
        addNodeToMatch("Maximum", minimum, clipValue);

        setFusedNode("Relu6", input);
    }

    virtual bool match(const tensorflow::GraphDef& net, int nodeId, std::vector<int>& matchedNodesIds)
    {
        if (!Subgraph::match(net, nodeId, matchedNodesIds))
            return false;
        Mat maxValue = getTensorContent(net.node(nodeId + 1).attr().at("value").tensor());
        return maxValue.type() == CV_32FC1 && maxValue.total() == 1 && maxValue.at<float>(0) == 6;
    }
};

// Keras' reshape stores output shape in separate Const nodes by one value.
// Need to merge them into a single Const node.
class ReshapeKerasSubgraph : public Subgraph
{
public:
    ReshapeKerasSubgraph(int _numOutDims) : numOutDims(_numOutDims)
    {
        int input = addNodeToMatch("");
        int shape = addNodeToMatch("Shape", input);
        int stack = addNodeToMatch("Const");
        int stack_1 = addNodeToMatch("Const");
        int stack_2 = addNodeToMatch("Const");
        int strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);

        std::vector<int> ids(1 + numOutDims);
        ids[0] = strided_slice;
        for (int i = 0; i < numOutDims; ++i)
            ids[1 + i] = addNodeToMatch("Const");
        int pack = addNodeToMatch("Pack", ids);
        addNodeToMatch("Reshape", input, pack);

        ids[0] = input;
        setFusedNode("Reshape", ids);
    }

    virtual void finalize(tensorflow::GraphDef&, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes)
    {
        std::vector<int> shape(numOutDims + 1);  // batch size in Keras is implicit.
        shape[0] = -1;
        for (int i = 0; i < numOutDims; ++i)
        {
            shape[1 + i] = inputNodes[1 + i]->attr().at("value").tensor().int_val(0);
        }
        tensorflow::TensorProto* shapeTensor = inputNodes[1]->mutable_attr()->at("value").mutable_tensor();
        fusedNode->mutable_input()->DeleteSubrange(2, numOutDims - 1);

        shapeTensor->clear_int_val();
        for (int i = 0; i < shape.size(); ++i)
        {
            shapeTensor->add_int_val(shape[i]);
        }
    }

private:
    int numOutDims;
};

void simplifySubgraphs(tensorflow::GraphDef& net)
{
    std::vector<Ptr<Subgraph> > subgraphs;
    subgraphs.push_back(Ptr<Subgraph>(new BatchNormSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new BatchNormNoGammaSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new FlattenSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new FlattenShapeSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new SoftMaxKerasSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new ReLU6KerasSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new ReshapeKerasSubgraph(3)));

    int numNodes = net.node_size();
    std::vector<int> matchedNodesIds;
    for (int i = 0; i < numNodes; ++i)
    {
        for (int j = 0; j < subgraphs.size(); ++j)
        {
            if (subgraphs[j]->match(net, i, matchedNodesIds))
            {
                subgraphs[j]->replace(net, matchedNodesIds);
                numNodes -= matchedNodesIds.size() - 1;  // #matchedNodes removed and one added.
                break;
            }
        }
    }
}

void RemoveIdentityOps(tensorflow::GraphDef& net)
{
    typedef std::map<String, String>  IdentityOpsMap;
    IdentityOpsMap identity_ops;

    std::vector<int> identity_ops_idx;

    int layersCount = net.node_size();
    for (int li = 0; li < layersCount; li++)
    {
        const tensorflow::NodeDef &layer = net.node(li);
        String type = layer.op();

        if (type == "Identity" || type == "Dropout") {
            identity_ops_idx.push_back(li);
            identity_ops[layer.name()] = layer.input(0);
        }
    }

    for (int li = 0; li < layersCount; li++)
    {
        tensorflow::NodeDef* layer = net.mutable_node(li);
        for (int input_id = 0; input_id < layer->input_size(); input_id++) {
            String input_op_name = layer->input(input_id);
            IdentityOpsMap::iterator it = identity_ops.find(input_op_name);

            if (it != identity_ops.end()) {
                layer->set_input(input_id, it->second);
            }
        }
    }

    std::sort(identity_ops_idx.begin(), identity_ops_idx.end());

    int removed_nodes = 0;
    for(size_t i = 0; i < identity_ops_idx.size(); i++) {
        int start_id = identity_ops_idx[i] - removed_nodes;
        net.mutable_node()->DeleteSubrange(start_id, 1);
        removed_nodes++;
    }
}

Mat getTensorContent(const tensorflow::TensorProto &tensor)
{
    std::string content = tensor.tensor_content();
    switch (tensor.dtype())
    {
        case tensorflow::DT_FLOAT:
        {
            if (!content.empty())
                return Mat(1, content.size() / sizeof(float), CV_32FC1, (void*)content.c_str()).clone();
            else
            {
                const RepeatedField<float>& field = tensor.float_val();
                CV_Assert(!field.empty());
                return Mat(1, field.size(), CV_32FC1, (void*)field.data()).clone();
            }
        }
        case tensorflow::DT_DOUBLE:
        {
            if (!content.empty())
                return Mat(1, content.size() / sizeof(double), CV_64FC1, (void*)content.c_str()).clone();
            else
            {
                const RepeatedField<double>& field = tensor.double_val();
                CV_Assert(!field.empty());
                return Mat(1, field.size(), CV_64FC1, (void*)field.data()).clone();
            }
        }
        case tensorflow::DT_INT32:
        {
            if (!content.empty())
                return Mat(1, content.size() / sizeof(int32_t), CV_32SC1, (void*)content.c_str()).clone();
            else
            {
                const RepeatedField<int32_t>& field = tensor.int_val();
                CV_Assert(!field.empty());
                return Mat(1, field.size(), CV_32SC1, (void*)field.data()).clone();
            }
        }
        case tensorflow::DT_HALF:
        {
            Mat halfs;
            if (!content.empty())
            {
                static const int kHalfSize = 2;
                halfs = Mat(1, content.size() / kHalfSize, CV_16UC1, (void*)content.c_str());
            }
            else
            {
                const RepeatedField<int32_t>& field = tensor.half_val();
                CV_Assert(!field.empty());
                Mat ints(1, field.size(), CV_32SC1, (void*)field.data());
                ints.convertTo(halfs, CV_16UC1);
            }
            // Reinterpret as a signed shorts just for a convertFp16 call.
            Mat halfsSigned(halfs.size(), CV_16SC1, halfs.data);
            Mat floats(halfs.size(), CV_32FC1);
            convertFp16(halfsSigned, floats);
            return floats;
        }
        case tensorflow::DT_QUINT8:
        {
            CV_Assert(!content.empty());
            return Mat(1, content.size(), CV_8UC1, (void*)content.c_str()).clone();
        }
        default:
            CV_Error(Error::StsError, "Tensor's data type is not supported");
            break;
    }
    return Mat();
}

CV__DNN_EXPERIMENTAL_NS_END
}}  // namespace dnn, namespace cv

#endif  // HAVE_PROTOBUF
