// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "op_inf_engine.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_INF_ENGINE
#include <ie_extension.h>
#include <ie_plugin_dispatcher.hpp>
#endif  // HAVE_INF_ENGINE

namespace cv { namespace dnn {

#ifdef HAVE_INF_ENGINE

InfEngineBackendNode::InfEngineBackendNode(const InferenceEngine::CNNLayerPtr& _layer)
    : BackendNode(DNN_BACKEND_INFERENCE_ENGINE), layer(_layer) {}

void InfEngineBackendNode::connect(std::vector<Ptr<BackendWrapper> >& inputs,
                                   std::vector<Ptr<BackendWrapper> >& outputs)
{
    layer->insData.resize(inputs.size());
    for (int i = 0; i < inputs.size(); ++i)
    {
        InferenceEngine::DataPtr dataPtr = infEngineDataNode(inputs[i]);
        layer->insData[i] = InferenceEngine::DataWeakPtr(dataPtr);
        dataPtr->inputTo[layer->name] = layer;
    }

    CV_Assert(!outputs.empty());

    layer->outData.resize(1);
    InferenceEngine::DataPtr dataPtr = infEngineDataNode(outputs[0]);
    dataPtr->name = layer->name;
    layer->outData[0] = dataPtr;
    dataPtr->creatorLayer = InferenceEngine::CNNLayerWeakPtr(layer);
}

static std::vector<Ptr<InfEngineBackendWrapper> >
infEngineWrappers(const std::vector<Ptr<BackendWrapper> >& ptrs)
{
    std::vector<Ptr<InfEngineBackendWrapper> > wrappers(ptrs.size());
    for (int i = 0; i < ptrs.size(); ++i)
    {
        CV_Assert(!ptrs[i].empty());
        wrappers[i] = ptrs[i].dynamicCast<InfEngineBackendWrapper>();
        CV_Assert(!wrappers[i].empty());
    }
    return wrappers;
}

static InferenceEngine::Layout estimateLayout(const Mat& m)
{
    if (m.dims == 4)
        return InferenceEngine::Layout::NCHW;
    else if (m.dims == 2)
        return InferenceEngine::Layout::NC;
    else
        return InferenceEngine::Layout::ANY;
}

static InferenceEngine::DataPtr wrapToInfEngineDataNode(const Mat& m, const std::string& name = "")
{
    std::vector<size_t> reversedShape(&m.size[0], &m.size[0] + m.dims);
    std::reverse(reversedShape.begin(), reversedShape.end());
    if (m.type() == CV_32F)
        return InferenceEngine::DataPtr(
            new InferenceEngine::Data(name, reversedShape, InferenceEngine::Precision::FP32, estimateLayout(m))
        );
    else if (m.type() == CV_8U)
        return InferenceEngine::DataPtr(
            new InferenceEngine::Data(name, reversedShape, InferenceEngine::Precision::U8, estimateLayout(m))
        );
    else
        CV_Error(Error::StsNotImplemented, format("Unsupported data type %d", m.type()));
}

InferenceEngine::Blob::Ptr wrapToInfEngineBlob(const Mat& m, const std::vector<size_t>& shape,
                                               InferenceEngine::Layout layout)
{
    if (m.type() == CV_32F)
        return InferenceEngine::make_shared_blob<float>(InferenceEngine::Precision::FP32,
                                                        layout, shape, (float*)m.data);
    else if (m.type() == CV_8U)
        return InferenceEngine::make_shared_blob<uint8_t>(InferenceEngine::Precision::U8,
                                                          layout, shape, (uint8_t*)m.data);
    else
        CV_Error(Error::StsNotImplemented, format("Unsupported data type %d", m.type()));
}

InferenceEngine::Blob::Ptr wrapToInfEngineBlob(const Mat& m, InferenceEngine::Layout layout)
{
    std::vector<size_t> reversedShape(&m.size[0], &m.size[0] + m.dims);
    std::reverse(reversedShape.begin(), reversedShape.end());
    return wrapToInfEngineBlob(m, reversedShape, layout);
}

InferenceEngine::DataPtr infEngineDataNode(const Ptr<BackendWrapper>& ptr)
{
    CV_Assert(!ptr.empty());
    Ptr<InfEngineBackendWrapper> p = ptr.dynamicCast<InfEngineBackendWrapper>();
    CV_Assert(!p.empty());
    return p->dataPtr;
}

InfEngineBackendWrapper::InfEngineBackendWrapper(int targetId, const cv::Mat& m)
    : BackendWrapper(DNN_BACKEND_INFERENCE_ENGINE, targetId)
{
    dataPtr = wrapToInfEngineDataNode(m);
    blob = wrapToInfEngineBlob(m, estimateLayout(m));
}

InfEngineBackendWrapper::InfEngineBackendWrapper(Ptr<BackendWrapper> wrapper)
    : BackendWrapper(DNN_BACKEND_INFERENCE_ENGINE, wrapper->targetId)
{
    Ptr<InfEngineBackendWrapper> ieWrapper = wrapper.dynamicCast<InfEngineBackendWrapper>();
    CV_Assert(!ieWrapper.empty());
    InferenceEngine::DataPtr srcData = ieWrapper->dataPtr;
    dataPtr = InferenceEngine::DataPtr(
        new InferenceEngine::Data(srcData->name, srcData->dims, srcData->precision,
                                  srcData->layout)
    );
    blob = ieWrapper->blob;
}

Ptr<BackendWrapper> InfEngineBackendWrapper::create(Ptr<BackendWrapper> wrapper)
{
    return Ptr<BackendWrapper>(new InfEngineBackendWrapper(wrapper));
}

InfEngineBackendWrapper::~InfEngineBackendWrapper()
{

}

void InfEngineBackendWrapper::copyToHost()
{

}

void InfEngineBackendWrapper::setHostDirty()
{

}

InfEngineBackendNet::InfEngineBackendNet()
{
    targetDevice = InferenceEngine::TargetDevice::eCPU;
    precision = InferenceEngine::Precision::FP32;
    hasNetOwner = false;
}

InfEngineBackendNet::InfEngineBackendNet(InferenceEngine::CNNNetwork& net)
{
    targetDevice = InferenceEngine::TargetDevice::eCPU;
    precision = InferenceEngine::Precision::FP32;
    inputs = net.getInputsInfo();
    outputs = net.getOutputsInfo();
    layers.resize(net.layerCount());  // A hack to execute InfEngineBackendNet::layerCount correctly.
    netOwner = net;
    hasNetOwner = true;
}

void InfEngineBackendNet::Release() noexcept
{
    layers.clear();
    inputs.clear();
    outputs.clear();
}

void InfEngineBackendNet::setPrecision(InferenceEngine::Precision p) noexcept
{
    precision = p;
}

InferenceEngine::Precision InfEngineBackendNet::getPrecision() noexcept
{
    return hasNetOwner ? netOwner.getPrecision() : precision;
}

InferenceEngine::Precision InfEngineBackendNet::getPrecision() const noexcept
{
    return hasNetOwner ? netOwner.getPrecision() : precision;
}

// Assume that outputs of network is unconnected blobs.
void InfEngineBackendNet::getOutputsInfo(InferenceEngine::OutputsDataMap &outputs_) noexcept
{
    const_cast<const InfEngineBackendNet*>(this)->getOutputsInfo(outputs_);
}
void InfEngineBackendNet::getOutputsInfo(InferenceEngine::OutputsDataMap &outputs_) const noexcept
{
    outputs_ = outputs;
}

// Returns input references that aren't connected to internal outputs.
void InfEngineBackendNet::getInputsInfo(InferenceEngine::InputsDataMap &inputs_) noexcept
{
    const_cast<const InfEngineBackendNet*>(this)->getInputsInfo(inputs_);
}

// Returns input references that aren't connected to internal outputs.
void InfEngineBackendNet::getInputsInfo(InferenceEngine::InputsDataMap &inputs_) const noexcept
{
    inputs_ = inputs;
}

InferenceEngine::InputInfo::Ptr InfEngineBackendNet::getInput(const std::string &inputName) noexcept
{
    return const_cast<const InfEngineBackendNet*>(this)->getInput(inputName);
}

InferenceEngine::InputInfo::Ptr InfEngineBackendNet::getInput(const std::string &inputName) const noexcept
{
    const auto& it = inputs.find(inputName);
    CV_Assert(it != inputs.end());
    return it->second;
}

void InfEngineBackendNet::getName(char*, size_t) noexcept
{
}

void InfEngineBackendNet::getName(char*, size_t) const noexcept
{
}

const std::string& InfEngineBackendNet::getName() const noexcept
{
    return name;
}

InferenceEngine::StatusCode InfEngineBackendNet::serialize(const std::string&, const std::string&, InferenceEngine::ResponseDesc*) const noexcept
{
    CV_Error(Error::StsNotImplemented, "");
    return InferenceEngine::StatusCode::OK;
}

size_t InfEngineBackendNet::layerCount() noexcept
{
    return const_cast<const InfEngineBackendNet*>(this)->layerCount();
}

size_t InfEngineBackendNet::layerCount() const noexcept
{
    return layers.size();
}

InferenceEngine::DataPtr& InfEngineBackendNet::getData(const char *dname) noexcept
{
    CV_Error(Error::StsNotImplemented, "");
    return outputs.begin()->second;  // Just return something.
}

void InfEngineBackendNet::addLayer(const InferenceEngine::CNNLayerPtr &layer) noexcept
{
    layers.push_back(layer);
    inputs.clear();
    outputs.clear();
}

InferenceEngine::StatusCode
InfEngineBackendNet::addOutput(const std::string &layerName, size_t outputIndex,
                               InferenceEngine::ResponseDesc *resp) noexcept
{
    for (const auto& l : layers)
    {
        for (const InferenceEngine::DataPtr& out : l->outData)
        {
            if (out->name == layerName)
            {
                outputs[out->name] = out;
                return InferenceEngine::StatusCode::OK;
            }
        }
    }
    CV_Error(Error::StsObjectNotFound, "Cannot find a layer " + layerName);
    return InferenceEngine::StatusCode::OK;
}

InferenceEngine::StatusCode
InfEngineBackendNet::getLayerByName(const char *layerName, InferenceEngine::CNNLayerPtr &out,
                                    InferenceEngine::ResponseDesc *resp) noexcept
{
    return const_cast<const InfEngineBackendNet*>(this)->getLayerByName(layerName, out, resp);
}

InferenceEngine::StatusCode InfEngineBackendNet::getLayerByName(const char *layerName,
                                                                InferenceEngine::CNNLayerPtr &out,
                                                                InferenceEngine::ResponseDesc *resp) const noexcept
{
    for (auto& l : layers)
    {
        if (l->name == layerName)
        {
            out = l;
            return InferenceEngine::StatusCode::OK;
        }
    }
    CV_Error(Error::StsObjectNotFound, cv::format("Cannot find a layer %s", layerName));
    return InferenceEngine::StatusCode::NOT_FOUND;
}

void InfEngineBackendNet::setTargetDevice(InferenceEngine::TargetDevice device) noexcept
{
    if (device != InferenceEngine::TargetDevice::eCPU &&
        device != InferenceEngine::TargetDevice::eGPU &&
        device != InferenceEngine::TargetDevice::eMYRIAD &&
        device != InferenceEngine::TargetDevice::eFPGA)
        CV_Error(Error::StsNotImplemented, "");
    targetDevice = device;
}

InferenceEngine::TargetDevice InfEngineBackendNet::getTargetDevice() noexcept
{
    return const_cast<const InfEngineBackendNet*>(this)->getTargetDevice();
}

InferenceEngine::TargetDevice InfEngineBackendNet::getTargetDevice() const noexcept
{
    return targetDevice == InferenceEngine::TargetDevice::eFPGA ?
           InferenceEngine::TargetDevice::eHETERO : targetDevice;
}

InferenceEngine::StatusCode InfEngineBackendNet::setBatchSize(const size_t) noexcept
{
    CV_Error(Error::StsNotImplemented, "");
    return InferenceEngine::StatusCode::OK;
}

InferenceEngine::StatusCode InfEngineBackendNet::setBatchSize(size_t size, InferenceEngine::ResponseDesc *responseDesc) noexcept
{
    CV_Error(Error::StsNotImplemented, "");
    return InferenceEngine::StatusCode::OK;
}

size_t InfEngineBackendNet::getBatchSize() const noexcept
{
    size_t batchSize = 0;
    for (const auto& inp : inputs)
    {
        CV_Assert(inp.second);
        std::vector<size_t> dims = inp.second->getDims();
        CV_Assert(!dims.empty());
        if (batchSize != 0)
            CV_Assert(batchSize == dims.back());
        else
            batchSize = dims.back();
    }
    return batchSize;
}

#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2018R2)
InferenceEngine::StatusCode InfEngineBackendNet::AddExtension(const InferenceEngine::IShapeInferExtensionPtr &extension, InferenceEngine::ResponseDesc *resp) noexcept
{
    CV_Error(Error::StsNotImplemented, "");
    return InferenceEngine::StatusCode::OK;
}

InferenceEngine::StatusCode InfEngineBackendNet::reshape(const InferenceEngine::ICNNNetwork::InputShapes &inputShapes, InferenceEngine::ResponseDesc *resp) noexcept
{
    CV_Error(Error::StsNotImplemented, "");
    return InferenceEngine::StatusCode::OK;
}
#endif

void InfEngineBackendNet::init(int targetId)
{
    if (inputs.empty())
    {
        // Collect all external input blobs.
        inputs.clear();
        std::map<std::string, InferenceEngine::DataPtr> internalOutputs;
        for (const auto& l : layers)
        {
            for (const InferenceEngine::DataWeakPtr& ptr : l->insData)
            {
                InferenceEngine::DataPtr inp(ptr);
                if (internalOutputs.find(inp->name) == internalOutputs.end())
                {
                    InferenceEngine::InputInfo::Ptr inpInfo(new InferenceEngine::InputInfo());
                    inpInfo->setInputData(inp);
                    if (inputs.find(inp->name) == inputs.end())
                        inputs[inp->name] = inpInfo;
                }
            }
            for (const InferenceEngine::DataPtr& out : l->outData)
            {
                // TODO: Replace to uniqueness assertion.
                if (internalOutputs.find(out->name) == internalOutputs.end())
                    internalOutputs[out->name] = out;
            }
        }
        CV_Assert(!inputs.empty());

#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2018R3)
        for (const auto& inp : inputs)
        {
            InferenceEngine::LayerParams lp;
            lp.name = inp.first;
            lp.type = "Input";
            lp.precision = InferenceEngine::Precision::FP32;
            std::shared_ptr<InferenceEngine::CNNLayer> inpLayer(new InferenceEngine::CNNLayer(lp));

            layers.push_back(inpLayer);

            InferenceEngine::DataPtr dataPtr = inp.second->getInputData();
            // TODO: remove precision dependency (see setInput.normalization tests)
            if (dataPtr->precision == InferenceEngine::Precision::FP32)
            {
                inpLayer->outData.assign(1, dataPtr);
                dataPtr->creatorLayer = InferenceEngine::CNNLayerWeakPtr(inpLayer);
            }
        }
#endif
    }

    if (outputs.empty())
    {
        // Add all unconnected blobs to output blobs.
        InferenceEngine::OutputsDataMap unconnectedOuts;
        for (const auto& l : layers)
        {
            if (l->type == "Input")
                continue;
            // Add all outputs.
            for (const InferenceEngine::DataPtr& out : l->outData)
            {
                // TODO: Replace to uniqueness assertion.
                if (unconnectedOuts.find(out->name) == unconnectedOuts.end())
                    unconnectedOuts[out->name] = out;
            }
            // Remove internally connected outputs.
            for (const InferenceEngine::DataWeakPtr& inp : l->insData)
            {
                unconnectedOuts.erase(InferenceEngine::DataPtr(inp)->name);
            }
        }
        CV_Assert(!unconnectedOuts.empty());

        for (auto it = unconnectedOuts.begin(); it != unconnectedOuts.end(); ++it)
        {
            outputs[it->first] = it->second;
        }
    }

    // Set up input blobs.
    inpBlobs.clear();
    for (const auto& it : inputs)
    {
        CV_Assert(allBlobs.find(it.first) != allBlobs.end());
        inpBlobs[it.first] = allBlobs[it.first];
        it.second->setPrecision(inpBlobs[it.first]->precision());
    }

    // Set up output blobs.
    outBlobs.clear();
    for (const auto& it : outputs)
    {
        CV_Assert(allBlobs.find(it.first) != allBlobs.end());
        outBlobs[it.first] = allBlobs[it.first];
    }

    switch (targetId)
    {
    case DNN_TARGET_CPU: setTargetDevice(InferenceEngine::TargetDevice::eCPU); break;
    case DNN_TARGET_OPENCL_FP16:
        setPrecision(InferenceEngine::Precision::FP16);
        /* Falls through. */
    case DNN_TARGET_OPENCL: setTargetDevice(InferenceEngine::TargetDevice::eGPU); break;
    case DNN_TARGET_MYRIAD:
    {
        setPrecision(InferenceEngine::Precision::FP16);
        setTargetDevice(InferenceEngine::TargetDevice::eMYRIAD); break;
    }
    case DNN_TARGET_FPGA:
    {
        setPrecision(InferenceEngine::Precision::FP16);
        setTargetDevice(InferenceEngine::TargetDevice::eFPGA); break;
    }
    default:
        CV_Error(Error::StsError, format("Unknown target identifier: %d", targetId));
    }

    if (!isInitialized())
        initPlugin(*this);
}

static std::map<InferenceEngine::TargetDevice, InferenceEngine::InferenceEnginePluginPtr> sharedPlugins;

void InfEngineBackendNet::initPlugin(InferenceEngine::ICNNNetwork& net)
{
    CV_Assert(!isInitialized());

    try
    {
        auto pluginIt = sharedPlugins.find(targetDevice);
        if (pluginIt != sharedPlugins.end())
        {
            enginePtr = pluginIt->second;
        }
        else
        {
            auto dispatcher = InferenceEngine::PluginDispatcher({""});
            if (targetDevice == InferenceEngine::TargetDevice::eFPGA)
                enginePtr = dispatcher.getPluginByDevice("HETERO:FPGA,CPU");
            else
                enginePtr = dispatcher.getSuitablePlugin(targetDevice);
            sharedPlugins[targetDevice] = enginePtr;

            if (targetDevice == InferenceEngine::TargetDevice::eCPU ||
                targetDevice == InferenceEngine::TargetDevice::eFPGA)
            {
                std::string suffixes[] = {"_avx2", "_sse4", ""};
                bool haveFeature[] = {
                    checkHardwareSupport(CPU_AVX2),
                    checkHardwareSupport(CPU_SSE4_2),
                    true
                };
                for (int i = 0; i < 3; ++i)
                {
                    if (!haveFeature[i])
                        continue;
    #ifdef _WIN32
                    std::string libName = "cpu_extension" + suffixes[i] + ".dll";
    #else
                    std::string libName = "libcpu_extension" + suffixes[i] + ".so";
    #endif  // _WIN32
                    try
                    {
                        InferenceEngine::IExtensionPtr extension =
                            InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(libName);
                        enginePtr->AddExtension(extension, 0);
                        break;
                    }
                    catch(...) {}
                }
                // Some of networks can work without a library of extra layers.
            }
        }
        plugin = InferenceEngine::InferencePlugin(enginePtr);

        netExec = plugin.LoadNetwork(net, {});
        infRequest = netExec.CreateInferRequest();
        infRequest.SetInput(inpBlobs);
        infRequest.SetOutput(outBlobs);
    }
    catch (const std::exception& ex)
    {
        CV_Error(Error::StsAssert, format("Failed to initialize Inference Engine backend: %s", ex.what()));
    }
}

bool InfEngineBackendNet::isInitialized()
{
    return (bool)enginePtr;
}

void InfEngineBackendNet::addBlobs(const std::vector<Ptr<BackendWrapper> >& ptrs)
{
    auto wrappers = infEngineWrappers(ptrs);
    for (const auto& wrapper : wrappers)
    {
        allBlobs.insert({wrapper->dataPtr->name, wrapper->blob});
    }
}

void InfEngineBackendNet::forward()
{
    infRequest.Infer();
}

Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob)
{
    // NOTE: Inference Engine sizes are reversed.
    std::vector<size_t> dims = blob->dims();
    std::vector<int> size(dims.rbegin(), dims.rend());
    return Mat(size, CV_32F, (void*)blob->buffer());
}

InfEngineBackendLayer::InfEngineBackendLayer(const InferenceEngine::DataPtr& output_)
{
    output = output_;
}

bool InfEngineBackendLayer::getMemoryShapes(const std::vector<MatShape> &inputs,
                                            const int requiredOutputs,
                                            std::vector<MatShape> &outputs,
                                            std::vector<MatShape> &internals) const
{
    std::vector<size_t> dims = output->dims;
    std::vector<int> shape(dims.rbegin(), dims.rend());
    outputs.assign(1, shape);
    return false;
}

bool InfEngineBackendLayer::supportBackend(int backendId)
{
    return backendId == DNN_BACKEND_DEFAULT ||
           (backendId == DNN_BACKEND_INFERENCE_ENGINE && haveInfEngine());
}

void InfEngineBackendLayer::forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs,
                                    OutputArrayOfArrays internals)
{
    CV_Error(Error::StsInternal, "Choose Inference Engine as a preferable backend.");
}

InferenceEngine::TBlob<int16_t>::Ptr convertFp16(const InferenceEngine::Blob::Ptr& blob)
{
    auto halfs = InferenceEngine::make_shared_blob<int16_t>(InferenceEngine::Precision::FP16, blob->layout(), blob->dims());
    halfs->allocate();
    Mat floatsData(1, blob->size(), CV_32F, blob->buffer());
    Mat halfsData(1, blob->size(), CV_16SC1, halfs->buffer());
    convertFp16(floatsData, halfsData);
    return halfs;
}

#endif  // HAVE_INF_ENGINE

bool haveInfEngine()
{
#ifdef HAVE_INF_ENGINE
    return true;
#else
    return false;
#endif  // HAVE_INF_ENGINE
}

void forwardInfEngine(Ptr<BackendNode>& node)
{
    CV_Assert(haveInfEngine());
#ifdef HAVE_INF_ENGINE
    CV_Assert(!node.empty());
    Ptr<InfEngineBackendNode> ieNode = node.dynamicCast<InfEngineBackendNode>();
    CV_Assert(!ieNode.empty());
    ieNode->net->forward();
#endif  // HAVE_INF_ENGINE
}

CV__DNN_EXPERIMENTAL_NS_BEGIN

void resetMyriadDevice()
{
#ifdef HAVE_INF_ENGINE
    sharedPlugins.erase(InferenceEngine::TargetDevice::eMYRIAD);
#endif  // HAVE_INF_ENGINE
}

CV__DNN_EXPERIMENTAL_NS_END
}}  // namespace dnn, namespace cv
