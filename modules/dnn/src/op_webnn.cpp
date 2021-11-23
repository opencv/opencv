// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <fstream>
#include "op_webnn.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/core/utils/filesystem.private.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

#ifdef HAVE_WEBNN

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