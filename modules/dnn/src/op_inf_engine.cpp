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
#endif  // HAVE_INF_ENGINE

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv { namespace dnn {

#ifdef HAVE_INF_ENGINE

// For networks with input layer which has an empty name, IE generates a name id[some_number].
// OpenCV lets users use an empty input name and to prevent unexpected naming,
// we can use some predefined name.
static std::string kDefaultInpLayerName = "empty_inp_layer_name";

InfEngineBackendNode::InfEngineBackendNode(const InferenceEngine::Builder::Layer& _layer)
    : BackendNode(DNN_BACKEND_INFERENCE_ENGINE), layer(_layer) {}

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
        it.second->setPrecision(blobIt->second->precision());
    }
    for (const auto& it : cnn.getOutputsInfo())
    {
        const std::string& name = it.first;
        auto blobIt = allBlobs.find(name);
        CV_Assert(blobIt != allBlobs.end());
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

InferenceEngine::Blob::Ptr cloneBlob(const InferenceEngine::Blob::Ptr& blob)
{
    InferenceEngine::Precision precision = blob->precision();
    InferenceEngine::Blob::Ptr copy;
    if (precision == InferenceEngine::Precision::FP32)
    {
        copy = InferenceEngine::make_shared_blob<float>(precision, blob->layout(), blob->dims());
    }
    else if (precision == InferenceEngine::Precision::U8)
    {
        copy = InferenceEngine::make_shared_blob<uint8_t>(precision, blob->layout(), blob->dims());
    }
    else
        CV_Error(Error::StsNotImplemented, "Unsupported blob precision");
    copy->allocate();
    return copy;
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

static std::map<InferenceEngine::TargetDevice, InferenceEngine::InferenceEnginePluginPtr>& getSharedPlugins()
{
    static std::map<InferenceEngine::TargetDevice, InferenceEngine::InferenceEnginePluginPtr> sharedPlugins;
    return sharedPlugins;
}


#if !defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)
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
        auto netExec = plugin.LoadNetwork(cnn, {{"VPU_PLATFORM", "VPU_2480"}});
        auto infRequest = netExec.CreateInferRequest();
    } catch(...) {
        return false;
    }
    return true;
}
#endif  // !defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)

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

            std::vector<std::string> candidates;

            std::string param_pluginPath = utils::getConfigurationParameterString("OPENCV_DNN_IE_EXTRA_PLUGIN_PATH", "");
            if (!param_pluginPath.empty())
            {
                candidates.push_back(param_pluginPath);
            }

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
                    candidates.push_back("cpu_extension" + suffixes[i] + ".dll");
#elif defined(__APPLE__)
                    candidates.push_back("libcpu_extension" + suffixes[i] + ".so");  // built as loadable module
                    candidates.push_back("libcpu_extension" + suffixes[i] + ".dylib");  // built as shared library
#else
                    candidates.push_back("libcpu_extension" + suffixes[i] + ".so");
#endif  // _WIN32
                }
            }
            bool found = false;
            for (size_t i = 0; i != candidates.size(); ++i)
            {
                const std::string& libName = candidates[i];
                try
                {
                    InferenceEngine::IExtensionPtr extension =
                        InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(libName);
                    enginePtr->AddExtension(extension, 0);
                    CV_LOG_INFO(NULL, "DNN-IE: Loaded extension plugin: " << libName);
                    found = true;
                    break;
                }
                catch(...) {}
            }
            if (!found && !candidates.empty())
            {
                CV_LOG_WARNING(NULL, "DNN-IE: Can't load extension plugin (extra layers for some networks). Specify path via OPENCV_DNN_IE_EXTRA_PLUGIN_PATH parameter");
            }
            // Some of networks can work without a library of extra layers.
#ifndef _WIN32
            // Limit the number of CPU threads.
            enginePtr->SetConfig({{
                InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, format("%d", getNumThreads()),
            }}, 0);
#endif
        }
        plugin = InferenceEngine::InferencePlugin(enginePtr);

        netExec = plugin.LoadNetwork(net, {});
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

void InfEngineBackendNet::addBlobs(const std::vector<cv::Ptr<BackendWrapper> >& ptrs)
{
    auto wrappers = infEngineWrappers(ptrs);
    for (const auto& wrapper : wrappers)
    {
        std::string name = wrapper->dataPtr->name;
        name = name.empty() ? kDefaultInpLayerName : name;
        allBlobs.insert({name, wrapper->blob});
    }
}

void InfEngineBackendNet::InfEngineReqWrapper::makePromises(const std::vector<Ptr<BackendWrapper> >& outsWrappers)
{
    auto outs = infEngineWrappers(outsWrappers);
    outProms.clear();
    outProms.resize(outs.size());
    outsNames.resize(outs.size());
    for (int i = 0; i < outs.size(); ++i)
    {
        outs[i]->futureMat = outProms[i].getArrayResult();
        outsNames[i] = outs[i]->dataPtr->name;
    }
}

void InfEngineBackendNet::forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                                  bool isAsync)
{
    // Look for finished requests.
    Ptr<InfEngineReqWrapper> reqWrapper;
    for (auto& wrapper : infRequests)
    {
        if (wrapper->isReady)
        {
            reqWrapper = wrapper;
            break;
        }
    }
    if (reqWrapper.empty())
    {
        reqWrapper = Ptr<InfEngineReqWrapper>(new InfEngineReqWrapper());
        try
        {
            reqWrapper->req = netExec.CreateInferRequest();
        }
        catch (const std::exception& ex)
        {
            CV_Error(Error::StsAssert, format("Failed to initialize Inference Engine backend: %s", ex.what()));
        }
        infRequests.push_back(reqWrapper);

        InferenceEngine::BlobMap inpBlobs, outBlobs;
        for (const auto& it : cnn.getInputsInfo())
        {
            const std::string& name = it.first;
            auto blobIt = allBlobs.find(name);
            CV_Assert(blobIt != allBlobs.end());
            inpBlobs[name] = isAsync ? cloneBlob(blobIt->second) : blobIt->second;
        }
        for (const auto& it : cnn.getOutputsInfo())
        {
            const std::string& name = it.first;
            auto blobIt = allBlobs.find(name);
            CV_Assert(blobIt != allBlobs.end());
            outBlobs[name] = isAsync ? cloneBlob(blobIt->second) : blobIt->second;
        }
        reqWrapper->req.SetInput(inpBlobs);
        reqWrapper->req.SetOutput(outBlobs);

        InferenceEngine::IInferRequest::Ptr infRequestPtr = reqWrapper->req;
        infRequestPtr->SetUserData(reqWrapper.get(), 0);

        infRequestPtr->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
            {
                InfEngineReqWrapper* wrapper;
                request->GetUserData((void**)&wrapper, 0);
                CV_Assert(wrapper && "Internal error");

                size_t processedOutputs = 0;
                try
                {
                    for (; processedOutputs < wrapper->outProms.size(); ++processedOutputs)
                    {
                        const std::string& name = wrapper->outsNames[processedOutputs];
                        Mat m = infEngineBlobToMat(wrapper->req.GetBlob(name));

                        try
                        {
                            CV_Assert(status == InferenceEngine::StatusCode::OK);
                            wrapper->outProms[processedOutputs].setValue(m.clone());
                        }
                        catch (...)
                        {
                            try {
                                wrapper->outProms[processedOutputs].setException(std::current_exception());
                            } catch(...) {
                                CV_LOG_ERROR(NULL, "DNN: Exception occured during async inference exception propagation");
                            }
                        }
                    }
                }
                catch (...)
                {
                    std::exception_ptr e = std::current_exception();
                    for (; processedOutputs < wrapper->outProms.size(); ++processedOutputs)
                    {
                        try {
                            wrapper->outProms[processedOutputs].setException(e);
                        } catch(...) {
                            CV_LOG_ERROR(NULL, "DNN: Exception occured during async inference exception propagation");
                        }
                    }
                }
                wrapper->isReady = true;
            }
        );
    }
    if (isAsync)
    {
        // Copy actual data to infer request's input blobs.
        for (const auto& it : cnn.getInputsInfo())
        {
            const std::string& name = it.first;
            auto blobIt = allBlobs.find(name);
            Mat srcMat = infEngineBlobToMat(blobIt->second);
            Mat dstMat = infEngineBlobToMat(reqWrapper->req.GetBlob(name));
            srcMat.copyTo(dstMat);
        }

        // Set promises to output blobs wrappers.
        reqWrapper->makePromises(outBlobsWrappers);

        reqWrapper->isReady = false;
        reqWrapper->req.StartAsync();
    }
    else
    {
        reqWrapper->req.Infer();
    }
}

Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob)
{
    // NOTE: Inference Engine sizes are reversed.
    std::vector<size_t> dims = blob->dims();
    std::vector<int> size(dims.rbegin(), dims.rend());

    int type = -1;
    switch (blob->precision())
    {
        case InferenceEngine::Precision::FP32: type = CV_32F; break;
        case InferenceEngine::Precision::U8: type = CV_8U; break;
        default:
            CV_Error(Error::StsNotImplemented, "Unsupported blob precision");
    }
    return Mat(size, type, (void*)blob->buffer());
}

bool InfEngineBackendLayer::getMemoryShapes(const std::vector<MatShape> &inputs,
                                            const int requiredOutputs,
                                            std::vector<MatShape> &outputs,
                                            std::vector<MatShape> &internals) const
{
    InferenceEngine::ICNNNetwork::InputShapes inShapes = t_net.getInputShapes();
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

void addConstantData(const std::string& name, InferenceEngine::Blob::Ptr data,
                     InferenceEngine::Builder::Layer& l)
{
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2019R1)
    l.getParameters()[name] = data;
#else
    l.addConstantData(name, data);
#endif
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

void forwardInfEngine(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                      Ptr<BackendNode>& node, bool isAsync)
{
    CV_Assert(haveInfEngine());
#ifdef HAVE_INF_ENGINE
    CV_Assert(!node.empty());
    Ptr<InfEngineBackendNode> ieNode = node.dynamicCast<InfEngineBackendNode>();
    CV_Assert(!ieNode.empty());
    ieNode->net->forward(outBlobsWrappers, isAsync);
#endif  // HAVE_INF_ENGINE
}

CV__DNN_EXPERIMENTAL_NS_BEGIN

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
#else
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


CV__DNN_EXPERIMENTAL_NS_END
}}  // namespace dnn, namespace cv
