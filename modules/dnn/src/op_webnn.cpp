// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <fstream>
#include "op_webnn.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/core/utils/filesystem.private.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#include "net_impl.hpp"

namespace cv { namespace dnn {

#ifdef HAVE_WEBNN

CV__DNN_INLINE_NS_BEGIN


void Net::Impl::addWebnnOutputs(LayerData &ld)
{
    CV_TRACE_FUNCTION();

    Ptr<WebnnNet> layerNet;
    auto it = ld.backendNodes.find(preferableBackend);
    if (it != ld.backendNodes.end())
    {
        Ptr<BackendNode> node = it->second;
        if (!node.empty())
        {
            Ptr<WebnnBackendNode> webnnNode = node.dynamicCast<WebnnBackendNode>();
            CV_Assert(!webnnNode.empty()); CV_Assert(!webnnNode->net.empty());
            layerNet = webnnNode->net;
        }
    }

    for (int i = 0; i < ld.inputBlobsId.size(); ++i)
    {
        LayerData &inpLd = layers[ld.inputBlobsId[i].lid];
        Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
        if (!inpNode.empty())
        {
            Ptr<WebnnBackendNode> webnnInpNode = inpNode.dynamicCast<WebnnBackendNode>();
            CV_Assert(!webnnInpNode.empty()); CV_Assert(!webnnInpNode->net.empty());
            if (layerNet != webnnInpNode->net)
            {
                webnnInpNode->net->addOutput(webnnInpNode->name);
                webnnInpNode->net->setUnconnectedNodes(webnnInpNode);
            }
        }
    }
}


void Net::Impl::initWebnnBackend(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_TRACE_FUNCTION();
    CV_Assert_N(preferableBackend == DNN_BACKEND_WEBNN, haveWebnn());

    Ptr<WebnnNet> net;

    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData &ld = it->second;
        if (ld.id == 0)
        {
            CV_Assert((netInputLayer->outNames.empty() && ld.outputBlobsWrappers.size() == 1) ||
                      (netInputLayer->outNames.size() == ld.outputBlobsWrappers.size()));
            for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
            {
                Ptr<WebnnBackendWrapper> wrapper = ld.outputBlobsWrappers[i].dynamicCast<WebnnBackendWrapper>();
                std::string outputName = netInputLayer->outNames.empty() ? ld.name : netInputLayer->outNames[i];
                outputName = ld.outputBlobsWrappers.size() > 1 ? (outputName + "." + std::to_string(i)) : outputName;
                wrapper->name = outputName;
            }
        }
        else
        {
            for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
            {
                Ptr<WebnnBackendWrapper> wrapper = ld.outputBlobsWrappers[i].dynamicCast<WebnnBackendWrapper>();
                std::string outputName = ld.outputBlobsWrappers.size() > 1 ? (ld.name + "." + std::to_string(i)) : ld.name;
                wrapper->name = outputName;
            }
        }
    }

    // Build WebNN networks from sets of layers that support this
    // backend. Split a whole model on several WebNN networks if
    // some of layers are not implemented.
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData &ld = it->second;

        if (ld.id == 0 && ld.skip)
            continue;

        bool fused = ld.skip;
        Ptr<Layer> layer = ld.layerInstance;
        if (!fused && !layer->supportBackend(preferableBackend))
        {
            // For test use. when not using WebNN, the test case will fail
            // with the following code.
            CV_LOG_WARNING(NULL, "Layer " + ld.type + " name " + ld.name + " is unsupported by WebNN backend.");

            addWebnnOutputs(ld);
            net = Ptr<WebnnNet>();
            layer->preferableTarget = DNN_TARGET_CPU;

            for (int i = 0; i < ld.inputBlobsId.size(); ++i)
            {
                LayerData &inpLd = layers[ld.inputBlobsId[i].lid];
                Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
                if (!inpNode.empty()) {
                    Ptr<WebnnBackendNode> webnnNode = inpNode.dynamicCast<WebnnBackendNode>();
                    CV_Assert(!webnnNode.empty());
                    webnnNode->net->setUnconnectedNodes(webnnNode);
                }
            }
            continue;
        }
        ld.skip = true; // Initially skip all WebNN supported layers.

        // Create a new network if one of inputs from different WebNN graph.
        std::vector<Ptr<BackendNode>> inputNodes;
        for (int i = 0; i < ld.inputBlobsId.size(); ++i)
        {
            // Layer_Test_ROIPooling.Accuracy has 2 inputs inpLD = 0, 0 -> has 4 inputNodes (input, rois, input, rois)
            if (inputNodes.size() == ld.inputBlobsId.size()) {
                break;
            }
            LayerData &inpLd = layers[ld.inputBlobsId[i].lid];
            Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
            if (!inpNode.empty())
            {
                 Ptr<WebnnBackendNode> webnnInpNode = inpNode.dynamicCast<WebnnBackendNode>();
                 CV_Assert(!webnnInpNode.empty()); CV_Assert(!webnnInpNode->net.empty());
                 if (webnnInpNode->net == net && !fused) {
                    inputNodes.push_back(inpNode);
                    continue;
                 }
            }

            if (net.empty()) {
                net = Ptr<WebnnNet>(new WebnnNet());
            }

            if (!fused) {
                std::vector<std::string> inputNames;
                std::vector<cv::Mat> inputs;

                auto curr_pos = inpLd.consumers.begin();
                auto compare = [&ld] (const LayerPin& lp) { return lp.lid == ld.id; };
                auto cons = curr_pos;
                while ((cons = std::find_if(curr_pos, inpLd.consumers.end(), compare)) !=
                        inpLd.consumers.end()) {
                    int cons_inp = cons->oid;
                    Ptr<WebnnBackendWrapper> inpWrapper = inpLd.outputBlobsWrappers[cons_inp].
                                                                 dynamicCast<WebnnBackendWrapper>();
                    CV_Assert(!inpWrapper.empty());
                    auto iter = std::find(inputNames.begin(), inputNames.end(),
                                          inpWrapper->name);
                    if (iter == inputNames.end()) {
                        inputNames.push_back(inpWrapper->name);
                        inputs.push_back(inpLd.outputBlobs[cons_inp]);
                    }
                    curr_pos = cons + 1;
                }

                auto inps = net->setInputs(inputs, inputNames);
                for (auto& inp : inps) {
                    WebnnBackendNode* node = new WebnnBackendNode(inp);
                    node->net = net;
                    inputNodes.emplace_back(Ptr<BackendNode>(node));
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
        else {
            net = Ptr<WebnnNet>(new WebnnNet());
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

                auto webnnInpNode = inputNodes[i].dynamicCast<WebnnBackendNode>();
                inputNodes[i] = Ptr<BackendNode>(new WebnnBackendNode(webnnInpNode->operand));
            }

            if (layer->supportBackend(preferableBackend))
            {
                if (ld.type == "Const") {
                    ml::Operand fake_operand;
                    Ptr<WebnnBackendNode> fake_input_node = Ptr<WebnnBackendNode>(new WebnnBackendNode(fake_operand));
                    fake_input_node->net = net;
                    inputNodes.push_back(fake_input_node);
                }
                node = layer->initWebnn(ld.inputBlobsWrappers, inputNodes);
                for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
                {
                    Ptr<WebnnBackendWrapper> wrapper = ld.outputBlobsWrappers[i].dynamicCast<WebnnBackendWrapper>();
                    node.dynamicCast<WebnnBackendNode>()->name = wrapper->name;
                }
            }
            else
            {
                continue;
            }
        }
        else if (node.empty())
            continue;

        ld.backendNodes[preferableBackend] = node;

        Ptr<WebnnBackendNode> webnnNode = node.dynamicCast<WebnnBackendNode>();
        CV_Assert(!webnnNode.empty());
        webnnNode->net = net;

        if (ld.consumers.empty()) {
            // TF EAST_text_detection
            webnnNode->net->setUnconnectedNodes(webnnNode);
        }
        for (const auto& pin : blobsToKeep_)
        {
            if (pin.lid == ld.id)
            {
                webnnNode->net->addOutput(webnnNode->name);
                break;
            }
        }
        net->addBlobs(ld.inputBlobsWrappers);
        net->addBlobs(ld.outputBlobsWrappers);
        addWebnnOutputs(ld);
    }

    // Initialize all networks.
    for (MapIdToLayerData::reverse_iterator it = layers.rbegin(); it != layers.rend(); ++it)
    {
        LayerData &ld = it->second;
        auto iter = ld.backendNodes.find(preferableBackend);
        if (iter == ld.backendNodes.end())
            continue;

        Ptr<BackendNode>& node = iter->second;
        if (node.empty())
            continue;

        Ptr<WebnnBackendNode> webnnNode = node.dynamicCast<WebnnBackendNode>();
        if (webnnNode.empty())
            continue;

        CV_Assert(!webnnNode->net.empty());

        if (!webnnNode->net->isInitialized())
        {
            webnnNode->net->setUnconnectedNodes(webnnNode);
            webnnNode->net->createNet((Target)preferableTarget);
            ld.skip = false;
        }
    }
}


CV__DNN_INLINE_NS_END


namespace webnn {
ml::Operand BuildConstant(const ml::GraphBuilder& builder,
                              const std::vector<int32_t>& dimensions,
                              const void* value,
                              size_t size,
                              ml::OperandType type) {
        ml::OperandDescriptor desc;
        desc.type = type;
        desc.dimensions = dimensions.data();
        desc.dimensionsCount = (uint32_t)dimensions.size();
        ml::ArrayBufferView resource;
        resource.buffer = const_cast<void*>(value);
        resource.byteLength = size;
        return builder.Constant(&desc, &resource);
    }
}

static std::string kDefaultInpLayerName = "opencv_webnn_empty_inp_layer_name";

static std::vector<Ptr<WebnnBackendWrapper> >
webnnWrappers(const std::vector<Ptr<BackendWrapper> >& ptrs)
{
    std::vector<Ptr<WebnnBackendWrapper> > wrappers(ptrs.size());
    for (int i = 0; i < ptrs.size(); ++i)
    {
        CV_Assert(!ptrs[i].empty());
        wrappers[i] = ptrs[i].dynamicCast<WebnnBackendWrapper>();
        CV_Assert(!wrappers[i].empty());
    }
    return wrappers;
}

// WebnnNet
WebnnNet::WebnnNet()
{
    hasNetOwner = false;
    device_name = "CPU";

#ifdef __EMSCRIPTEN__
    context = ml::Context(emscripten_webnn_create_context());
#else
    WebnnProcTable backendProcs = webnn_native::GetProcs();
    webnnProcSetProcs(&backendProcs);
    context = ml::Context(webnn_native::CreateContext());
#endif
    builder = ::ml::CreateGraphBuilder(context);
    namedOperands = ::ml::CreateNamedOperands();
}

void WebnnNet::addOutput(const std::string& name)
{
    requestedOutputs.push_back(name);
}

void WebnnNet::createNet(Target targetId) {
    init(targetId);
}

void WebnnNet::init(Target targetId)
{
    switch (targetId)
    {
        case DNN_TARGET_CPU:
            device_name = "CPU";
            break;
        case DNN_TARGET_OPENCL:
            device_name = "GPU";
            break;
        default:
            CV_Error(Error::StsNotImplemented, "Unknown target");
    };

    graph = builder.Build(namedOperands);
    CV_Assert(graph!=nullptr);
    isInit = true;
}

std::vector<ml::Operand> WebnnNet::setInputs(const std::vector<cv::Mat>& inputs,
                                             const std::vector<std::string>& names) {
    CV_Assert_N(inputs.size() == names.size());
    std::vector<ml::Operand> current_inp;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        auto& m = inputs[i];

        std::vector<int32_t> dimensions = webnn::getShape(m);
        ml::OperandDescriptor descriptor;
        descriptor.dimensions = dimensions.data();
        descriptor.dimensionsCount = dimensions.size();
        if (m.type() == CV_32F)
        {
            descriptor.type = ml::OperandType::Float32;
        }
        else
        {
            CV_Error(Error::StsNotImplemented, format("Unsupported data type %s", typeToString(m.type()).c_str()));
        }
        ml::Operand inputOperand = builder.Input(names[i].c_str(), &descriptor);
        current_inp.push_back(std::move(inputOperand));
    }
    inputNames = names;
    return current_inp;
}

void WebnnNet::setUnconnectedNodes(Ptr<WebnnBackendNode>& node) {
    outputNames.push_back(node->name);
    namedOperands.Set(outputNames.back().c_str(), node->operand);
}

bool WebnnNet::isInitialized()
{
    return isInit;
}

void WebnnNet::reset()
{
    allBlobs.clear();
    isInit = false;
}

void WebnnNet::addBlobs(const std::vector<cv::Ptr<BackendWrapper> >& ptrs)
{
    auto wrappers = webnnWrappers(ptrs);
    for (const auto& wrapper : wrappers)
    {
        std::string name = wrapper->name;
        name = name.empty() ? kDefaultInpLayerName : name;
        allBlobs.insert({name, wrapper});
    }
}

void WebnnNet::forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers, bool isAsync)
{
    CV_LOG_DEBUG(NULL, "WebnnNet::forward(" << (isAsync ? "async" : "sync") << ")");
    ml::NamedInputs named_inputs = ::ml::CreateNamedInputs();
    std::vector<ml::Input> inputs(inputNames.size());
    for (int i = 0; i < inputNames.size(); ++i) {
        const std::string& name = inputNames[i];
        ml::Input& input = inputs[i];
        auto blobIt = allBlobs.find(name);
        CV_Assert(blobIt != allBlobs.end());
        const Ptr<WebnnBackendWrapper> wrapper = blobIt->second;
        input.resource.buffer = wrapper->host->data;
        input.resource.byteLength = wrapper->size;
        named_inputs.Set(name.c_str(), &input);
    }
    std::vector<Ptr<WebnnBackendWrapper> > outs = webnnWrappers(outBlobsWrappers);
    ml::NamedOutputs named_outputs = ::ml::CreateNamedOutputs();
    std::vector<ml::ArrayBufferView> outputs(outs.size());
    for (int i = 0; i < outs.size(); ++i) {
        const std::string& name = outs[i]->name;
        ml::ArrayBufferView& output = outputs[i];
        output.buffer = outs[i]->host->data;
        // std::cout<<"host data size: "<<outs[i]->host->total()*outs[i]->host->elemSize()<<std::endl;
        output.byteLength = outs[i]->size;
        // std::cout<<"outs[i]->size: "<< outs[i]->size << std::endl;
        named_outputs.Set(name.c_str(), &output);
    }
    ml::ComputeGraphStatus status = graph.Compute(named_inputs, named_outputs);
    if (status != ::ml::ComputeGraphStatus::Success) {
        CV_Error(Error::StsAssert, format("Failed to compute: %d", int(status)));
    }
}

// WebnnBackendNode
WebnnBackendNode::WebnnBackendNode(ml::Operand&& _operand)
    : BackendNode(DNN_BACKEND_WEBNN), operand(std::move(_operand)) {}

WebnnBackendNode::WebnnBackendNode(ml::Operand& _operand)
    : BackendNode(DNN_BACKEND_WEBNN), operand(_operand) {}

// WebnnBackendWrapper
WebnnBackendWrapper::WebnnBackendWrapper(int targetId, cv::Mat& m)
    : BackendWrapper(DNN_BACKEND_WEBNN, targetId)
{
    size = m.total() * m.elemSize();
    // buffer.reset(new char[size]);
    // std::memcpy(buffer.get(), m.data, size);
    // dimensions = getShape<int32_t>(m);
    // descriptor.dimensions = dimensions.data();
    // descriptor.dimensionsCount = dimensions.size();
    if (m.type() == CV_32F)
    {
        descriptor.type = ml::OperandType::Float32;
    }
    else
    {
        CV_Error(Error::StsNotImplemented, format("Unsupported data type %s", typeToString(m.type()).c_str()));
    }
    host = &m;
}

WebnnBackendWrapper::~WebnnBackendWrapper()
{
    // nothing
}

void WebnnBackendWrapper::copyToHost()
{
    CV_LOG_DEBUG(NULL, "WebnnBackendWrapper::copyToHost()");
    //CV_Error(Error::StsNotImplemented, "");
}

void WebnnBackendWrapper::setHostDirty()
{
    CV_LOG_DEBUG(NULL, "WebnnBackendWrapper::setHostDirty()");
    //CV_Error(Error::StsNotImplemented, "");
}

void forwardWebnn(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                  Ptr<BackendNode>& node, bool isAsync)
{
    CV_Assert(!node.empty());
    Ptr<WebnnBackendNode> webnnNode = node.dynamicCast<WebnnBackendNode>();
    CV_Assert(!webnnNode.empty());
    webnnNode->net->forward(outBlobsWrappers, isAsync);
}


#else
void forwardWebnn(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                   Ptr<BackendNode>& operand, bool isAsync)
{
    CV_Assert(false && "WebNN is not enabled in this OpenCV build");
}

#endif

}
}