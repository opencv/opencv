// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/core/utils/fp_control_utils.hpp>

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "net_impl.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_INF_ENGINE


/** mark input pins as outputs from other subnetworks
 * FIXIT must be done by DNN engine not ngraph.
 */
void Net::Impl::addNgraphOutputs(LayerData& ld)
{
    CV_TRACE_FUNCTION();

    CV_LOG_DEBUG(NULL, "DNN/IE: layer of new subnet: " << ld.name << "@" << ld.type);

    Ptr<InfEngineNgraphNet> layerNet;
    auto it = ld.backendNodes.find(preferableBackend);
    if (it != ld.backendNodes.end())
    {
        Ptr<BackendNode> node = it->second;
        if (!node.empty())
        {
            Ptr<InfEngineNgraphNode> ieNode = node.dynamicCast<InfEngineNgraphNode>();
            CV_Assert(!ieNode.empty());
            CV_Assert(!ieNode->net.empty());
            layerNet = ieNode->net;
        }
    }

    for (int i = 0; i < ld.inputBlobsId.size(); ++i)
    {
        LayerData& inpLd = layers[ld.inputBlobsId[i].lid];
        Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
        if (!inpNode.empty())
        {
            Ptr<InfEngineNgraphNode> ieInpNode = inpNode.dynamicCast<InfEngineNgraphNode>();
            CV_Assert(!ieInpNode.empty());
            CV_Assert(!ieInpNode->net.empty());
            if (layerNet != ieInpNode->net)
            {
                CV_LOG_DEBUG(NULL, "DNN/IE: pin output between subnets: " << ieInpNode->node->get_friendly_name());
                ieInpNode->net->addOutput(ieInpNode);
            }
        }
    }
}

void Net::Impl::initNgraphBackend(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_TRACE_FUNCTION();
    CV_CheckEQ(preferableBackend, DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, "");

    Ptr<InfEngineNgraphNet> net;

    for (MapIdToLayerData::const_iterator it = layers.begin(); it != layers.end(); ++it)
    {
        const LayerData& ld = it->second;
        if (ld.id == 0)
        {
            CV_Assert((netInputLayer->outNames.empty() && ld.outputBlobsWrappers.size() == 1) ||
                      (netInputLayer->outNames.size() == ld.outputBlobsWrappers.size()));
            for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
            {
                InferenceEngine::DataPtr dataPtr = ngraphDataNode(ld.outputBlobsWrappers[i]);
                std::string outputName = netInputLayer->outNames.empty() ? ld.name : netInputLayer->outNames[i];
                outputName = ld.outputBlobsWrappers.size() > 1 ? (outputName + "." + std::to_string(i)) : outputName;
                dataPtr->setName(outputName);
            }
        }
        else
        {
            for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
            {
                InferenceEngine::DataPtr dataPtr = ngraphDataNode(ld.outputBlobsWrappers[i]);
                std::string outputName = ld.outputBlobsWrappers.size() > 1 ? (ld.name + "." + std::to_string(i)) : ld.name;
                dataPtr->setName(outputName);
            }
        }
    }

    if (skipInfEngineInit)
    {
        Ptr<BackendNode> node = layers[lastLayerId].backendNodes[preferableBackend];
        CV_Assert(!node.empty());

        Ptr<InfEngineNgraphNode> ieNode = node.dynamicCast<InfEngineNgraphNode>();
        CV_Assert(!ieNode.empty());

        CV_Assert(ieNode->net);
        InfEngineNgraphNet& ienet = *ieNode->net;
        ienet.reset();

        for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
        {
            LayerData& ld = it->second;
            if (ld.id == 0)
            {
                for (int i = 0; i < ld.inputBlobsWrappers.size(); ++i)
                {
                    InferenceEngine::DataPtr dataPtr = ngraphDataNode(ld.inputBlobsWrappers[i]);
                    dataPtr->setName(netInputLayer->outNames[i]);
                }
            }
            else
            {
                for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
                {
                    auto it = ienet.outputsDesc.find(ld.name);
                    if (it != ienet.outputsDesc.end())
                    {
                        const InferenceEngine::TensorDesc& descriptor = it->second;
                        InferenceEngine::DataPtr dataPtr = ngraphDataOutputNode(ld.outputBlobsWrappers[i], descriptor, ld.name);
                        dataPtr->setName(ld.name);
                    }
                    else
                    {
                        InferenceEngine::DataPtr dataPtr = ngraphDataNode(ld.outputBlobsWrappers[i]);
                        dataPtr->setName(ld.name);
                    }
                }
            }
            ienet.addBlobs(ld.inputBlobsWrappers);
            ienet.addBlobs(ld.outputBlobsWrappers);
            ld.skip = true;
        }
        layers[lastLayerId].skip = false;
        ienet.init((Target)preferableTarget);
        return;
    }

    bool supportsCPUFallback = !isArmComputePlugin() && (preferableTarget == DNN_TARGET_CPU ||
                               openvino::checkTarget(DNN_TARGET_CPU));

    // Build Inference Engine networks from sets of layers that support this
    // backend. Split a whole model on several Inference Engine networks if
    // some of layers are not implemented.
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData& ld = it->second;

        CV_LOG_DEBUG(NULL, "DNN/IE: processing layer " << ld.name << "@" << ld.type << " (" << ld.id << ") ...");

        if (ld.id == 0 && ld.skip)
        {
            CV_LOG_DEBUG(NULL, "DNN/IE:    SKIP!");
            continue;
        }

        bool fused = ld.skip;
        Ptr<Layer> layer = ld.layerInstance;
        if (!fused && !layer->supportBackend(preferableBackend))
        {
            CV_LOG_DEBUG(NULL, "DNN/IE:    NOT supported!");
            bool customizable = ld.id != 0 && supportsCPUFallback;

            // TODO: there is a bug in Myriad plugin with custom layers shape infer.
            if (preferableTarget == DNN_TARGET_MYRIAD || preferableTarget == DNN_TARGET_HDDL)
            {
                for (int i = 0; customizable && i < ld.inputBlobs.size(); ++i)
                {
                    customizable = ld.inputBlobs[i]->size[0] == 1;
                }
            }

            // TODO: fix these workarounds
            if (preferableTarget == DNN_TARGET_MYRIAD ||
                preferableTarget == DNN_TARGET_HDDL ||
                preferableTarget == DNN_TARGET_OPENCL ||
                preferableTarget == DNN_TARGET_OPENCL_FP16)
                customizable &= ld.type != "Concat";

            if (preferableTarget == DNN_TARGET_OPENCL ||
                preferableTarget == DNN_TARGET_OPENCL_FP16)
                customizable &= ld.type != "Power";

            if (preferableTarget == DNN_TARGET_OPENCL)
                customizable &= ld.type != "Eltwise";

            if (!customizable)
            {
                CV_LOG_DEBUG(NULL, "DNN/IE:    NOT customizable!");
                addNgraphOutputs(ld);
                net = Ptr<InfEngineNgraphNet>();
                layer->preferableTarget = DNN_TARGET_CPU;

                for (int i = 0; i < ld.inputBlobsId.size(); ++i)
                {
                    LayerData& inpLd = layers[ld.inputBlobsId[i].lid];
                    Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
                    if (!inpNode.empty())
                    {
                        Ptr<InfEngineNgraphNode> ieNode = inpNode.dynamicCast<InfEngineNgraphNode>();
                        CV_Assert(!ieNode.empty());
                        ieNode->net->addOutput(ieNode);
                    }
                }
                continue;
            }
        }
        ld.skip = true;  // Initially skip all Inference Engine supported layers.

        // Create a new network if one of inputs from different Inference Engine graph.
        std::vector<Ptr<BackendNode>> inputNodes;
        for (int i = 0; i < ld.inputBlobsId.size(); ++i)
        {
            // Layer_Test_ROIPooling.Accuracy has 2 inputs inpLD = 0, 0 -> has 4 inputNodes (input, rois, input, rois)
            if (inputNodes.size() == ld.inputBlobsId.size())
            {
                break;
            }
            LayerData& inpLd = layers[ld.inputBlobsId[i].lid];
            Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
            if (!inpNode.empty())
            {
                Ptr<InfEngineNgraphNode> ieInpNode = inpNode.dynamicCast<InfEngineNgraphNode>();
                CV_Assert(!ieInpNode.empty());
                CV_Assert(!ieInpNode->net.empty());
                if (ieInpNode->net == net && !fused)
                {
                    inputNodes.push_back(inpNode);
                    continue;
                }
            }

            if (net.empty())
            {
                net = Ptr<InfEngineNgraphNet>(new InfEngineNgraphNet(*this));
            }

            if (!fused)
            {
                std::vector<std::string> inputNames;
                std::vector<cv::Mat> inputs;

                auto curr_pos = inpLd.consumers.begin();
                auto compare = [&ld](const LayerPin& lp) { return lp.lid == ld.id; };
                auto cons = curr_pos;
                while ((cons = std::find_if(curr_pos, inpLd.consumers.end(), compare)) !=
                        inpLd.consumers.end()) {
                    int cons_inp = cons->oid;
                    Ptr<NgraphBackendWrapper> inpWrapper = inpLd.outputBlobsWrappers[cons_inp].
                                                                 dynamicCast<NgraphBackendWrapper>();
                    CV_Assert(!inpWrapper.empty());
                    auto iter = std::find(inputNames.begin(), inputNames.end(),
                            inpWrapper->dataPtr->getName());
                    if (iter == inputNames.end())
                    {
                        inputNames.push_back(inpWrapper->dataPtr->getName());
                        inputs.push_back(inpLd.outputBlobs[cons_inp]);
                    }
                    curr_pos = cons + 1;
                }

                auto inps = net->setInputs(inputs, inputNames);
                for (auto& inp : inps)
                {
                    inputNodes.emplace_back(Ptr<BackendNode>(new InfEngineNgraphNode(inp)));
                }
            }
        }

        Ptr<BackendNode> node;
        if (!net.empty())
        {
            if (fused)
            {
                bool inPlace = ld.inputBlobsId.size() == 1 && ld.outputBlobs.size() == 1 &&
                               ld.inputBlobs[0]->data == ld.outputBlobs[0].data;
                CV_Assert(inPlace);
                node = layers[ld.inputBlobsId[0].lid].backendNodes[preferableBackend];
                ld.inputBlobsWrappers = layers[ld.inputBlobsId[0].lid].inputBlobsWrappers;
            }
        }
        else
        {
            net = Ptr<InfEngineNgraphNet>(new InfEngineNgraphNet(*this));
        }

        if (!fused)
        {
            CV_Assert(ld.inputBlobsId.size() == inputNodes.size());
            for (int i = 0; i < ld.inputBlobsId.size(); ++i)
            {
                int lid = ld.inputBlobsId[i].lid;
                int oid = ld.inputBlobsId[i].oid;
                if (oid == 0 || lid == 0)
                    continue;

                auto ieInpNode = inputNodes[i].dynamicCast<InfEngineNgraphNode>();
                const auto& ngraph_input_node = ieInpNode->node;
                CV_LOG_DEBUG(NULL, "DNN/IE: bind output port " << lid << ":" << oid << " (" << ngraph_input_node->get_friendly_name() << ":" << ngraph_input_node->get_type_info().name << ")");

                // Handle parameters from other subnets. Output port is not used in this case
                if ((ngraph::op::is_parameter(ngraph_input_node) || ngraph::op::is_constant(ngraph_input_node)) &&
                        ngraph_input_node->get_output_size() == 1)
                {
                    inputNodes[i] = Ptr<BackendNode>(new InfEngineNgraphNode(ngraph_input_node));
                    continue;
                }
                CV_CheckLT((size_t)oid, ngraph_input_node->get_output_size(), "");
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_4)
                // FIXIT refactor ".initNgraph()" API to use Output<Node>
                // WA: use Concat to emulate Identity operation with requested output port
                auto oid_node = std::make_shared<ngraph::op::Concat>(ngraph::OutputVector { ngraph_input_node->output(oid) }, 0);
                inputNodes[i] = Ptr<BackendNode>(new InfEngineNgraphNode(oid_node));
#elif INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_3)
                inputNodes[i] = Ptr<BackendNode>(new InfEngineNgraphNode(ieInpNode->node->get_output_as_single_output_node(oid)));
#else
                inputNodes[i] = Ptr<BackendNode>(new InfEngineNgraphNode(ieInpNode->node->get_output_as_single_output_node(oid, false)));
#endif
            }

            if (layer->supportBackend(preferableBackend))
            {
                CV_LOG_DEBUG(NULL, "DNN/IE: wrap layer " << ld.name << "@" << ld.type << " - outputs: " << ld.outputBlobsWrappers.size());
                node = layer->initNgraph(ld.inputBlobsWrappers, inputNodes);
#if 0  // FIXIT doesn't work with multiple outputs (set name is applied to the same node)
                for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
                {
                    InferenceEngine::DataPtr dataPtr = ngraphDataNode(ld.outputBlobsWrappers[i]);
                    node.dynamicCast<InfEngineNgraphNode>()->setName(dataPtr->getName());
                }
#else
                node.dynamicCast<InfEngineNgraphNode>()->setName(layer->name);
#endif
            }
            else
            {
                CV_LOG_DEBUG(NULL, "DNN/IE: layer is not supported: " << ld.name << "@" << ld.type);
                node = Ptr<BackendNode>(new InfEngineNgraphNode(inputNodes,
                        ld.layerInstance, ld.inputBlobs, ld.outputBlobs, ld.internals));
            }
        }
        else if (node.empty())
        {
            CV_LOG_DEBUG(NULL, "DNN/IE: node.empty() bypass...");
            continue;
        }

        ld.backendNodes[preferableBackend] = node;

        Ptr<InfEngineNgraphNode> ieNode = node.dynamicCast<InfEngineNgraphNode>();
        CV_Assert(!ieNode.empty());
        ieNode->net = net;

        for (const auto& pin : blobsToKeep_)
        {
            if (pin.lid == ld.id)
            {
                ieNode->net->addOutput(ieNode);
                break;
            }
        }
        ieNode->net->setNodePtr(&ieNode->node);

        net->addBlobs(ld.inputBlobsWrappers);
        net->addBlobs(ld.outputBlobsWrappers);
        addNgraphOutputs(ld);
    }

    // Initialize all networks.
    for (MapIdToLayerData::reverse_iterator it = layers.rbegin(); it != layers.rend(); ++it)
    {
        LayerData& ld = it->second;
        auto iter = ld.backendNodes.find(preferableBackend);
        if (iter == ld.backendNodes.end())
            continue;

        Ptr<BackendNode>& node = iter->second;
        if (node.empty())
            continue;

        Ptr<InfEngineNgraphNode> ieNode = node.dynamicCast<InfEngineNgraphNode>();
        if (ieNode.empty())
            continue;

        CV_Assert(!ieNode->net.empty());

        if (!ieNode->net->isInitialized())
        {
            ieNode->net->addOutput(ieNode);
            ieNode->net->createNet((Target)preferableTarget);
            ld.skip = false;
        }
    }
}

//}  // Net::Impl

/*static*/
Net Net::Impl::createNetworkFromModelOptimizer(InferenceEngine::CNNNetwork& ieNet)
{
    CV_TRACE_FUNCTION();

    CV_TRACE_REGION("register_inputs");

    std::vector<String> inputsNames;
    std::vector<MatShape> inp_shapes;
    for (auto& it : ieNet.getInputsInfo())
    {
        inputsNames.push_back(it.first);
        std::vector<size_t> dims = it.second->getTensorDesc().getDims();
        inp_shapes.push_back(std::vector<int>(dims.begin(), dims.end()));
    }

    Net cvNet;
    cvNet.setInputsNames(inputsNames);

    // set empty input to determine input shapes
    for (int inp_id = 0; inp_id < inputsNames.size(); ++inp_id)
    {
        cvNet.setInputShape(inputsNames[inp_id], inp_shapes[inp_id]);
    }

    CV_TRACE_REGION_NEXT("backendNode");

    Ptr<BackendNode> backendNode;
    {
        auto fake_node = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape {});
        Ptr<InfEngineNgraphNode> backendNodeNGraph(new InfEngineNgraphNode(fake_node));
        backendNodeNGraph->net = Ptr<InfEngineNgraphNet>(new InfEngineNgraphNet(*(cvNet.impl), ieNet));
        backendNode = backendNodeNGraph;
    }

    CV_TRACE_REGION_NEXT("register_outputs");

    auto ngraphFunction = ieNet.getFunction();
    CV_Assert(ngraphFunction);
    std::vector<std::shared_ptr<ngraph::Node>> ngraphOperations = ngraphFunction->get_ops();

    for (auto& it : ieNet.getOutputsInfo())
    {
        CV_TRACE_REGION("output");
        const auto& outputName = it.first;

        LayerParams lp;
        int lid = cvNet.addLayer(it.first, "", lp);

        LayerData& ld = cvNet.impl->layers[lid];

        {
            Ptr<Layer> cvLayer(new NgraphBackendLayer(ieNet));
            cvLayer->name = outputName;
            cvLayer->type = "_unknown_";

            auto process_layer = [&](const std::string& name) -> bool
            {
                CV_TRACE_REGION("ngraph_function");
                for (const auto& op : ngraphOperations)
                {
                    CV_Assert(op);
                    if (op->get_friendly_name() == name)
                    {
                        const std::string typeName = op->get_type_info().name;
                        cvLayer->type = typeName;
                        return true;
                    }
                }
                return false;
            };

            bool found = process_layer(outputName);
            if (!found)
            {
                auto pos = outputName.rfind('.');  // cut port number: ".0"
                if (pos != std::string::npos)
                {
                    std::string layerName = outputName.substr(0, pos);
                    found = process_layer(layerName);
                }
            }
            if (!found)
                CV_LOG_WARNING(NULL, "DNN/IE: Can't determine output layer type: '" << outputName << "'");

            ld.layerInstance = cvLayer;
            ld.backendNodes[DNN_BACKEND_INFERENCE_ENGINE_NGRAPH] = backendNode;
        }

        for (int i = 0; i < inputsNames.size(); ++i)
            cvNet.connect(0, i, lid, i);
    }

    CV_TRACE_REGION_NEXT("finalize");

    cvNet.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH);

    cvNet.impl->skipInfEngineInit = true;
    return cvNet;
}
#endif  // HAVE_INF_ENGINE

Net Net::readFromModelOptimizer(const String& xml, const String& bin)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_INF_ENGINE
    CV_UNUSED(xml); CV_UNUSED(bin);
    CV_Error(Error::StsError, "Build OpenCV with Inference Engine to enable loading models from Model Optimizer.");
#else

    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    InferenceEngine::Core& ie = getCore("");
    InferenceEngine::CNNNetwork ieNet = ie.ReadNetwork(xml, bin);

    return Impl::createNetworkFromModelOptimizer(ieNet);
#endif  // HAVE_INF_ENGINE
}

Net Net::readFromModelOptimizer(const std::vector<uchar>& bufferModelConfig, const std::vector<uchar>& bufferWeights)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!bufferModelConfig.empty());
    CV_Assert(!bufferWeights.empty());
    return readFromModelOptimizer(bufferModelConfig.data(), bufferModelConfig.size(),
                                           bufferWeights.data(), bufferWeights.size());
}

Net Net::readFromModelOptimizer(
        const uchar* bufferModelConfigPtr, size_t bufferModelConfigSize,
        const uchar* bufferWeightsPtr, size_t bufferWeightsSize
)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_INF_ENGINE
    CV_UNUSED(bufferModelConfigPtr); CV_UNUSED(bufferWeightsPtr);
    CV_UNUSED(bufferModelConfigSize); CV_UNUSED(bufferModelConfigSize);
    CV_Error(Error::StsError, "Build OpenCV with Inference Engine to enable loading models from Model Optimizer.");
#else

    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    InferenceEngine::Core& ie = getCore("");

    std::string model; model.assign((char*)bufferModelConfigPtr, bufferModelConfigSize);

    InferenceEngine::CNNNetwork ieNet;
    try
    {
        InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, { bufferWeightsSize }, InferenceEngine::Layout::C);
        InferenceEngine::Blob::CPtr weights_blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc, (uint8_t*)bufferWeightsPtr, bufferWeightsSize);

        ieNet = ie.ReadNetwork(model, weights_blob);
    }
    catch (const std::exception& e)
    {
        CV_Error(Error::StsError, std::string("DNN: IE failed to load model: ") + e.what());
    }

    return Impl::createNetworkFromModelOptimizer(ieNet);
#endif  // HAVE_INF_ENGINE
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
