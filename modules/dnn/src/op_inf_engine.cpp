// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "op_inf_engine.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_INF_ENGINE
#include <ie_extension.h>
#include <ie_plugin_dispatcher.hpp>
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2018R5)
#include <vpu/vpu_plugin_config.hpp>
#endif
#endif  // HAVE_INF_ENGINE

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv { namespace dnn {

#ifdef HAVE_INF_ENGINE

// For networks with input layer which has an empty name, IE generates a name id[some_number].
// OpenCV lets users use an empty input name and to prevent unexpected naming,
// we can use some predefined name.
static std::string kDefaultInpLayerName = "empty_inp_layer_name";

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2018R5)
InfEngineBackendNode::InfEngineBackendNode(const InferenceEngine::Builder::Layer& _layer)
    : BackendNode(DNN_BACKEND_INFERENCE_ENGINE), layer(_layer) {}
#else
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
#endif

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

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2018R5)

InfEngineBackendNet::InfEngineBackendNet() : netBuilder("")
{
    hasNetOwner = false;
    targetDevice = InferenceEngine::TargetDevice::eCPU;
}

InfEngineBackendNet::InfEngineBackendNet(InferenceEngine::CNNNetwork& net) : netBuilder(""), cnn(net)
{
    hasNetOwner = true;
    targetDevice = InferenceEngine::TargetDevice::eCPU;
}

void InfEngineBackendNet::connect(const std::vector<Ptr<BackendWrapper> >& inputs,
                                  const std::vector<Ptr<BackendWrapper> >& outputs,
                                  const std::string& layerName)
{
    std::vector<Ptr<InfEngineBackendWrapper> > inpWrappers = infEngineWrappers(inputs);
    std::map<std::string, int>::iterator it = layers.find(layerName);
    CV_Assert(it != layers.end());

    const int layerId = it->second;
    for (size_t i = 0; i < inpWrappers.size(); ++i)
    {
        const auto& inp = inpWrappers[i];
        const std::string& inpName = inp->dataPtr->name;
        int inpId;
        it = layers.find(inpName);
        if (it == layers.end())
        {
            InferenceEngine::Builder::InputLayer inpLayer(!inpName.empty() ? inpName : kDefaultInpLayerName);

            std::vector<size_t> shape(inp->blob->dims());
            std::reverse(shape.begin(), shape.end());

            inpLayer.setPort(InferenceEngine::Port(shape));
            inpId = netBuilder.addLayer(inpLayer);

            layers.insert({inpName, inpId});
        }
        else
            inpId = it->second;

        netBuilder.connect((size_t)inpId, {(size_t)layerId, i});
        unconnectedLayersIds.erase(inpId);
    }
    CV_Assert(!outputs.empty());
    InferenceEngine::DataPtr dataPtr = infEngineDataNode(outputs[0]);
    dataPtr->name = layerName;
}

void InfEngineBackendNet::init(int targetId)
{
    if (!hasNetOwner)
    {
        CV_Assert(!unconnectedLayersIds.empty());
        for (int id : unconnectedLayersIds)
        {
            InferenceEngine::Builder::OutputLayer outLayer("myconv1");
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2019R1)
            // Inference Engine determines network precision by ports.
            InferenceEngine::Precision p = (targetId == DNN_TARGET_MYRIAD ||
                                            targetId == DNN_TARGET_OPENCL_FP16) ?
                                           InferenceEngine::Precision::FP16 :
                                           InferenceEngine::Precision::FP32;
            outLayer.setPort(InferenceEngine::Port({}, p));
#endif
            netBuilder.addLayer({InferenceEngine::PortInfo(id)}, outLayer);
        }
        cnn = InferenceEngine::CNNNetwork(InferenceEngine::Builder::convertToICNNNetwork(netBuilder.build()));
    }

    switch (targetId)
    {
    case DNN_TARGET_CPU:
        targetDevice = InferenceEngine::TargetDevice::eCPU;
        break;
    case DNN_TARGET_OPENCL: case DNN_TARGET_OPENCL_FP16:
        targetDevice = InferenceEngine::TargetDevice::eGPU;
        break;
    case DNN_TARGET_MYRIAD:
        targetDevice = InferenceEngine::TargetDevice::eMYRIAD;
        break;
    case DNN_TARGET_FPGA:
        targetDevice = InferenceEngine::TargetDevice::eFPGA;
        break;
    default:
        CV_Error(Error::StsError, format("Unknown target identifier: %d", targetId));
    }

    for (const auto& name : requestedOutputs)
    {
        cnn.addOutput(name);
    }

    for (const auto& it : cnn.getInputsInfo())
    {
        const std::string& name = it.first;
        auto blobIt = allBlobs.find(name);
        CV_Assert(blobIt != allBlobs.end());
        inpBlobs[name] = blobIt->second;
        it.second->setPrecision(blobIt->second->precision());
    }
    for (const auto& it : cnn.getOutputsInfo())
    {
        const std::string& name = it.first;
        auto blobIt = allBlobs.find(name);
        CV_Assert(blobIt != allBlobs.end());
        outBlobs[name] = blobIt->second;
        it.second->setPrecision(blobIt->second->precision());  // Should be always FP32
    }

    initPlugin(cnn);
}

void InfEngineBackendNet::addLayer(InferenceEngine::Builder::Layer& layer)
{
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2019R1)
    // Add weights to network and connect them after input blobs.
    std::map<std::string, InferenceEngine::Parameter>& params = layer.getParameters();
    std::vector<int> blobsIds;
    std::vector<int> portIds;
    for (const std::string& name : {"weights", "biases"})
    {
        bool asInput = false;
        int portId = 0;
        for (int i = 0; i < layer.getInputPorts().size(); ++i)
        {
            const auto& port = layer.getInputPorts()[i];
            auto it = port.getParameters().find("type");
            if (it != port.getParameters().end() && it->second == name)
            {
                portId = i;
                asInput = true;
                break;
            }
        }

        if (!asInput)
            continue;

        auto it = params.find(name);
        if (it != params.end())
        {
            InferenceEngine::Blob::Ptr blob = it->second.as<InferenceEngine::Blob::Ptr>();
            params.erase(it);
            int blobId = netBuilder.addLayer(InferenceEngine::Builder::ConstLayer(name).setData(blob));
            blobsIds.push_back(blobId);
            portIds.push_back(portId);
        }
    }
#endif

    int id = netBuilder.addLayer(layer);
    const std::string& layerName = layer.getName();
    CV_Assert(layers.insert({layerName, id}).second);
    unconnectedLayersIds.insert(id);

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2019R1)
    // By default, all the weights are connected to last ports ids.
    for (int i = 0; i < blobsIds.size(); ++i)
    {
        netBuilder.connect((size_t)blobsIds[i], {(size_t)id, (size_t)portIds[i]});
    }
#endif
}

void InfEngineBackendNet::addOutput(const std::string& name)
{
    requestedOutputs.push_back(name);
}

#endif  // IE >= R5

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

#if INF_ENGINE_VER_MAJOR_LT(INF_ENGINE_RELEASE_2018R5)
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

void InfEngineBackendNet::Release() CV_NOEXCEPT
{
    layers.clear();
    inputs.clear();
    outputs.clear();
}

void InfEngineBackendNet::setPrecision(InferenceEngine::Precision p) CV_NOEXCEPT
{
    precision = p;
}

InferenceEngine::Precision InfEngineBackendNet::getPrecision() CV_NOEXCEPT
{
    return hasNetOwner ? netOwner.getPrecision() : precision;
}

InferenceEngine::Precision InfEngineBackendNet::getPrecision() const CV_NOEXCEPT
{
    return hasNetOwner ? netOwner.getPrecision() : precision;
}

// Assume that outputs of network is unconnected blobs.
void InfEngineBackendNet::getOutputsInfo(InferenceEngine::OutputsDataMap &outputs_) CV_NOEXCEPT
{
    const_cast<const InfEngineBackendNet*>(this)->getOutputsInfo(outputs_);
}
void InfEngineBackendNet::getOutputsInfo(InferenceEngine::OutputsDataMap &outputs_) const CV_NOEXCEPT
{
    outputs_ = outputs;
}

// Returns input references that aren't connected to internal outputs.
void InfEngineBackendNet::getInputsInfo(InferenceEngine::InputsDataMap &inputs_) CV_NOEXCEPT
{
    const_cast<const InfEngineBackendNet*>(this)->getInputsInfo(inputs_);
}

// Returns input references that aren't connected to internal outputs.
void InfEngineBackendNet::getInputsInfo(InferenceEngine::InputsDataMap &inputs_) const CV_NOEXCEPT
{
    inputs_ = inputs;
}

InferenceEngine::InputInfo::Ptr InfEngineBackendNet::getInput(const std::string &inputName) CV_NOEXCEPT
{
    return const_cast<const InfEngineBackendNet*>(this)->getInput(inputName);
}

InferenceEngine::InputInfo::Ptr InfEngineBackendNet::getInput(const std::string &inputName) const CV_NOEXCEPT
{
    const auto& it = inputs.find(inputName);
    CV_Assert(it != inputs.end());
    return it->second;
}

void InfEngineBackendNet::getName(char*, size_t) CV_NOEXCEPT
{
}

void InfEngineBackendNet::getName(char*, size_t) const CV_NOEXCEPT
{
}

const std::string& InfEngineBackendNet::getName() const CV_NOEXCEPT
{
    return name;
}

InferenceEngine::StatusCode InfEngineBackendNet::serialize(const std::string&, const std::string&, InferenceEngine::ResponseDesc*) const CV_NOEXCEPT
{
    CV_Error(Error::StsNotImplemented, "");
    return InferenceEngine::StatusCode::OK;
}

size_t InfEngineBackendNet::layerCount() CV_NOEXCEPT
{
    return const_cast<const InfEngineBackendNet*>(this)->layerCount();
}

size_t InfEngineBackendNet::layerCount() const CV_NOEXCEPT
{
    return layers.size();
}

InferenceEngine::DataPtr& InfEngineBackendNet::getData(const char *dname) CV_NOEXCEPT
{
    CV_Error(Error::StsNotImplemented, "");
    return outputs.begin()->second;  // Just return something.
}

void InfEngineBackendNet::addLayer(const InferenceEngine::CNNLayerPtr &layer) CV_NOEXCEPT
{
    layers.push_back(layer);
    inputs.clear();
    outputs.clear();
}

InferenceEngine::StatusCode
InfEngineBackendNet::addOutput(const std::string &layerName, size_t outputIndex,
                               InferenceEngine::ResponseDesc *resp) CV_NOEXCEPT
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
                                    InferenceEngine::ResponseDesc *resp) CV_NOEXCEPT
{
    return const_cast<const InfEngineBackendNet*>(this)->getLayerByName(layerName, out, resp);
}

InferenceEngine::StatusCode InfEngineBackendNet::getLayerByName(const char *layerName,
                                                                InferenceEngine::CNNLayerPtr &out,
                                                                InferenceEngine::ResponseDesc *resp) const CV_NOEXCEPT
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

void InfEngineBackendNet::setTargetDevice(InferenceEngine::TargetDevice device) CV_NOEXCEPT
{
    if (device != InferenceEngine::TargetDevice::eCPU &&
        device != InferenceEngine::TargetDevice::eGPU &&
        device != InferenceEngine::TargetDevice::eMYRIAD &&
        device != InferenceEngine::TargetDevice::eFPGA)
        CV_Error(Error::StsNotImplemented, "");
    targetDevice = device;
}

InferenceEngine::TargetDevice InfEngineBackendNet::getTargetDevice() CV_NOEXCEPT
{
    return const_cast<const InfEngineBackendNet*>(this)->getTargetDevice();
}

InferenceEngine::TargetDevice InfEngineBackendNet::getTargetDevice() const CV_NOEXCEPT
{
    return targetDevice == InferenceEngine::TargetDevice::eFPGA ?
           InferenceEngine::TargetDevice::eHETERO : targetDevice;
}

InferenceEngine::StatusCode InfEngineBackendNet::setBatchSize(const size_t) CV_NOEXCEPT
{
    CV_Error(Error::StsNotImplemented, "");
    return InferenceEngine::StatusCode::OK;
}

InferenceEngine::StatusCode InfEngineBackendNet::setBatchSize(size_t size, InferenceEngine::ResponseDesc *responseDesc) CV_NOEXCEPT
{
    CV_Error(Error::StsNotImplemented, "");
    return InferenceEngine::StatusCode::OK;
}

size_t InfEngineBackendNet::getBatchSize() const CV_NOEXCEPT
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

InferenceEngine::StatusCode InfEngineBackendNet::AddExtension(const InferenceEngine::IShapeInferExtensionPtr &extension, InferenceEngine::ResponseDesc *resp) CV_NOEXCEPT
{
    CV_Error(Error::StsNotImplemented, "");
    return InferenceEngine::StatusCode::OK;
}

InferenceEngine::StatusCode InfEngineBackendNet::reshape(const InferenceEngine::ICNNNetwork::InputShapes &inputShapes, InferenceEngine::ResponseDesc *resp) CV_NOEXCEPT
{
    CV_Error(Error::StsNotImplemented, "");
    return InferenceEngine::StatusCode::OK;
}

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

#endif  // IE < R5

static std::map<InferenceEngine::TargetDevice, InferenceEngine::InferenceEnginePluginPtr>& getSharedPlugins()
{
    static std::map<InferenceEngine::TargetDevice, InferenceEngine::InferenceEnginePluginPtr> sharedPlugins;
    return sharedPlugins;
}


#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2018R5) && !defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)
static bool detectMyriadX_()
{
    InferenceEngine::Builder::Network builder("");
    InferenceEngine::idx_t inpId = builder.addLayer(
                                   InferenceEngine::Builder::InputLayer().setPort(InferenceEngine::Port({1})));

#if INF_ENGINE_RELEASE <= 2018050000
    InferenceEngine::idx_t clampId;
    {
        InferenceEngine::Builder::Layer l = InferenceEngine::Builder::ClampLayer();
        auto& blobs = l.getConstantData();
        auto blob = InferenceEngine::make_shared_blob<int16_t>(
                        InferenceEngine::Precision::FP16,
                        InferenceEngine::Layout::C, {1});
        blob->allocate();
        blobs[""] = blob;
        clampId = builder.addLayer({inpId}, l);
    }
    builder.addLayer({InferenceEngine::PortInfo(clampId)}, InferenceEngine::Builder::OutputLayer());
#else
    InferenceEngine::idx_t clampId = builder.addLayer({inpId}, InferenceEngine::Builder::ClampLayer());
    builder.addLayer({InferenceEngine::PortInfo(clampId)},
                      InferenceEngine::Builder::OutputLayer().setPort(InferenceEngine::Port({},
                      InferenceEngine::Precision::FP16)));
#endif

    InferenceEngine::CNNNetwork cnn = InferenceEngine::CNNNetwork(
                                      InferenceEngine::Builder::convertToICNNNetwork(builder.build()));

    InferenceEngine::TargetDevice device = InferenceEngine::TargetDevice::eMYRIAD;
    InferenceEngine::InferenceEnginePluginPtr enginePtr;
    {
        AutoLock lock(getInitializationMutex());
        auto& sharedPlugins = getSharedPlugins();
        auto pluginIt = sharedPlugins.find(device);
        if (pluginIt != sharedPlugins.end()) {
            enginePtr = pluginIt->second;
        } else {
            auto dispatcher = InferenceEngine::PluginDispatcher({""});
            enginePtr = dispatcher.getSuitablePlugin(device);
            sharedPlugins[device] = enginePtr;
        }
    }
    auto plugin = InferenceEngine::InferencePlugin(enginePtr);
    try
    {
        auto netExec = plugin.LoadNetwork(cnn, {{InferenceEngine::VPUConfigParams::KEY_VPU_PLATFORM,
                                                 InferenceEngine::VPUConfigParams::VPU_2480}});
        auto infRequest = netExec.CreateInferRequest();
    } catch(...) {
        return false;
    }
    return true;
}
#endif // >= 2018R5

void InfEngineBackendNet::initPlugin(InferenceEngine::ICNNNetwork& net)
{
    CV_Assert(!isInitialized());

    try
    {
        AutoLock lock(getInitializationMutex());
        auto& sharedPlugins = getSharedPlugins();
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
    #elif defined(__APPLE__)
                    std::string libName = "libcpu_extension" + suffixes[i] + ".dylib";
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
        std::string name = wrapper->dataPtr->name;
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2018R5)
        name = name.empty() ? kDefaultInpLayerName : name;
#endif
        allBlobs.insert({name, wrapper->blob});
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

bool InfEngineBackendLayer::getMemoryShapes(const std::vector<MatShape> &inputs,
                                            const int requiredOutputs,
                                            std::vector<MatShape> &outputs,
                                            std::vector<MatShape> &internals) const
{
#if INF_ENGINE_VER_MAJOR_EQ(INF_ENGINE_RELEASE_2018R3)
    InferenceEngine::ICNNNetwork::InputShapes inShapes = const_cast<InferenceEngine::CNNNetwork&>(t_net).getInputShapes();
#else
    InferenceEngine::ICNNNetwork::InputShapes inShapes = t_net.getInputShapes();
#endif
    InferenceEngine::ICNNNetwork::InputShapes::iterator itr;
    bool equal_flag = true;
    size_t i = 0;
    for (itr = inShapes.begin(); itr != inShapes.end(); ++itr)
    {
        InferenceEngine::SizeVector currentInShape(inputs[i].begin(), inputs[i].end());
        if (itr->second != currentInShape)
        {
            itr->second = currentInShape;
            equal_flag = false;
        }
        i++;
    }

    if (!equal_flag)
    {
        InferenceEngine::CNNNetwork curr_t_net(t_net);
        curr_t_net.reshape(inShapes);
    }
    std::vector<size_t> dims = t_net.getOutputsInfo()[name]->getDims();
    outputs.push_back(MatShape(dims.begin(), dims.end()));
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

InferenceEngine::Blob::Ptr convertFp16(const InferenceEngine::Blob::Ptr& blob)
{
    auto halfs = InferenceEngine::make_shared_blob<int16_t>(InferenceEngine::Precision::FP16, blob->layout(), blob->dims());
    halfs->allocate();
    Mat floatsData(1, blob->size(), CV_32F, blob->buffer());
    Mat halfsData(1, blob->size(), CV_16SC1, halfs->buffer());
    convertFp16(floatsData, halfsData);
    return halfs;
}

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2018R5)
void addConstantData(const std::string& name, InferenceEngine::Blob::Ptr data,
                     InferenceEngine::Builder::Layer& l)
{
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2019R1)
    l.getParameters()[name] = data;
#else
    l.addConstantData(name, data);
#endif
}
#endif

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

CV__DNN_INLINE_NS_BEGIN

void resetMyriadDevice()
{
#ifdef HAVE_INF_ENGINE
    AutoLock lock(getInitializationMutex());
    getSharedPlugins().erase(InferenceEngine::TargetDevice::eMYRIAD);
#endif  // HAVE_INF_ENGINE
}

#ifdef HAVE_INF_ENGINE
bool isMyriadX()
{
     static bool myriadX = getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X;
     return myriadX;
}

static std::string getInferenceEngineVPUType_()
{
    static std::string param_vpu_type = utils::getConfigurationParameterString("OPENCV_DNN_IE_VPU_TYPE", "");
    if (param_vpu_type == "")
    {
#if defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)
        param_vpu_type = OPENCV_DNN_IE_VPU_TYPE_DEFAULT;
#elif INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2018R5)
        CV_LOG_INFO(NULL, "OpenCV-DNN: running Inference Engine VPU autodetection: Myriad2/X. In case of other accelerator types specify 'OPENCV_DNN_IE_VPU_TYPE' parameter");
        try {
            bool isMyriadX_ = detectMyriadX_();
            if (isMyriadX_)
            {
                param_vpu_type = CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X;
            }
            else
            {
                param_vpu_type = CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_2;
            }
        }
        catch (...)
        {
            CV_LOG_WARNING(NULL, "OpenCV-DNN: Failed Inference Engine VPU autodetection. Specify 'OPENCV_DNN_IE_VPU_TYPE' parameter.");
            param_vpu_type.clear();
        }
#else
        CV_LOG_WARNING(NULL, "OpenCV-DNN: VPU auto-detection is not implemented. Consider specifying VPU type via 'OPENCV_DNN_IE_VPU_TYPE' parameter");
        param_vpu_type = CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_2;
#endif
    }
    CV_LOG_INFO(NULL, "OpenCV-DNN: Inference Engine VPU type='" << param_vpu_type << "'");
    return param_vpu_type;
}

cv::String getInferenceEngineVPUType()
{
    static cv::String vpu_type = getInferenceEngineVPUType_();
    return vpu_type;
}
#else  // HAVE_INF_ENGINE
cv::String getInferenceEngineVPUType()
{
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}
#endif  // HAVE_INF_ENGINE


CV__DNN_INLINE_NS_END
}}  // namespace dnn, namespace cv
