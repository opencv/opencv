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
#endif  // HAVE_INF_ENGINE

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv { namespace dnn {

#ifdef HAVE_INF_ENGINE

static Backend parseInferenceEngineBackendType(const cv::String& backend)
{
    CV_Assert(!backend.empty());
    if (backend == CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        return DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    if (backend == CV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_API)
        return DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019;
    CV_Error(Error::StsBadArg, cv::format("Unknown IE backend: %s", backend.c_str()));
}
static const char* dumpInferenceEngineBackendType(Backend backend)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        return CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        return CV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_API;
    CV_Error(Error::StsBadArg, cv::format("Invalid backend ID for IE: %d", backend));
}
Backend& getInferenceEngineBackendTypeParam()
{
    static Backend param = parseInferenceEngineBackendType(
        utils::getConfigurationParameterString("OPENCV_DNN_BACKEND_INFERENCE_ENGINE_TYPE",
#ifdef HAVE_DNN_NGRAPH
            CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH
#elif defined(HAVE_DNN_IE_NN_BUILDER_2019)
            CV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_API
#else
#error "Build configuration error: nGraph or NN Builder API backend should be enabled"
#endif
        )
    );
    return param;
}

CV__DNN_INLINE_NS_BEGIN

cv::String getInferenceEngineBackendType()
{
    return dumpInferenceEngineBackendType(getInferenceEngineBackendTypeParam());
}
cv::String setInferenceEngineBackendType(const cv::String& newBackendType)
{
    Backend newBackend = parseInferenceEngineBackendType(newBackendType);
    Backend& param = getInferenceEngineBackendTypeParam();
    Backend old = param;
    param = newBackend;
    return dumpInferenceEngineBackendType(old);
}

CV__DNN_INLINE_NS_END


Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob)
{
    // NOTE: Inference Engine sizes are reversed.
    std::vector<size_t> dims = blob->getTensorDesc().getDims();
    std::vector<int> size(dims.begin(), dims.end());
    auto precision = blob->getTensorDesc().getPrecision();

    int type = -1;
    switch (precision)
    {
        case InferenceEngine::Precision::FP32: type = CV_32F; break;
        case InferenceEngine::Precision::U8: type = CV_8U; break;
        default:
            CV_Error(Error::StsNotImplemented, "Unsupported blob precision");
    }
    return Mat(size, type, (void*)blob->buffer());
}

void infEngineBlobsToMats(const std::vector<InferenceEngine::Blob::Ptr>& blobs,
                          std::vector<Mat>& mats)
{
    mats.resize(blobs.size());
    for (int i = 0; i < blobs.size(); ++i)
        mats[i] = infEngineBlobToMat(blobs[i]);
}


#ifdef HAVE_DNN_IE_NN_BUILDER_2019

// For networks with input layer which has an empty name, IE generates a name id[some_number].
// OpenCV lets users use an empty input name and to prevent unexpected naming,
// we can use some predefined name.
static std::string kDefaultInpLayerName = "empty_inp_layer_name";
static std::string kOpenCVLayersType = "OpenCVLayer";

static std::string shapesToStr(const std::vector<Mat>& mats)
{
    std::ostringstream shapes;
    shapes << mats.size() << " ";
    for (const Mat& m : mats)
    {
        shapes << m.dims << " ";
        for (int i = 0; i < m.dims; ++i)
            shapes << m.size[i] << " ";
    }
    return shapes.str();
}

static void strToShapes(const std::string& str, std::vector<std::vector<size_t> >& shapes)
{
    std::istringstream ss(str);
    int num, dims;
    ss >> num;
    shapes.resize(num);
    for (int i = 0; i < num; ++i)
    {
        ss >> dims;
        shapes[i].resize(dims);
        for (int j = 0; j < dims; ++j)
            ss >> shapes[i][j];
    }
}

class InfEngineCustomLayer : public InferenceEngine::ILayerExecImpl
{
public:
    explicit InfEngineCustomLayer(const InferenceEngine::CNNLayer& layer) : cnnLayer(layer)
    {
        std::istringstream iss(layer.GetParamAsString("impl"));
        size_t ptr;
        iss >> ptr;
        cvLayer = (Layer*)ptr;

        std::vector<std::vector<size_t> > shapes;
        strToShapes(layer.GetParamAsString("internals"), shapes);
        internals.resize(shapes.size());
        for (int i = 0; i < shapes.size(); ++i)
            internals[i].create(std::vector<int>(shapes[i].begin(), shapes[i].end()), CV_32F);
    }

    virtual InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs,
                                                std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                                InferenceEngine::ResponseDesc *resp) noexcept
    {
        std::vector<Mat> inpMats, outMats;
        infEngineBlobsToMats(inputs, inpMats);
        infEngineBlobsToMats(outputs, outMats);

        try
        {
            cvLayer->forward(inpMats, outMats, internals);
            return InferenceEngine::StatusCode::OK;
        }
        catch (...)
        {
            return InferenceEngine::StatusCode::GENERAL_ERROR;
        }
    }

    virtual InferenceEngine::StatusCode
    getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf,
                               InferenceEngine::ResponseDesc* resp) noexcept
    {
        std::vector<InferenceEngine::DataConfig> inDataConfig;
        std::vector<InferenceEngine::DataConfig> outDataConfig;
        for (auto& it : cnnLayer.insData)
        {
            InferenceEngine::DataConfig conf;
            conf.desc = it.lock()->getTensorDesc();
            inDataConfig.push_back(conf);
        }

        for (auto& it : cnnLayer.outData)
        {
            InferenceEngine::DataConfig conf;
            conf.desc = it->getTensorDesc();
            outDataConfig.push_back(conf);
        }

        InferenceEngine::LayerConfig layerConfig;
        layerConfig.inConfs = inDataConfig;
        layerConfig.outConfs = outDataConfig;

        conf.push_back(layerConfig);
        return InferenceEngine::StatusCode::OK;
    }

    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config,
                                     InferenceEngine::ResponseDesc *resp) noexcept
    {
        return InferenceEngine::StatusCode::OK;
    }

private:
    InferenceEngine::CNNLayer cnnLayer;
    dnn::Layer* cvLayer;
    std::vector<Mat> internals;
};

class InfEngineCustomLayerShapeInfer : public InferenceEngine::IShapeInferImpl
{
public:
      InferenceEngine::StatusCode
      inferShapes(const std::vector<InferenceEngine::Blob::CPtr>& inBlobs,
                  const std::map<std::string, std::string>& params,
                  const std::map<std::string, InferenceEngine::Blob::Ptr>& blobs,
                  std::vector<InferenceEngine::SizeVector>& outShapes,
                  InferenceEngine::ResponseDesc* desc) noexcept override
      {
          strToShapes(params.at("outputs"), outShapes);
          return InferenceEngine::StatusCode::OK;
      }
};

class InfEngineCustomLayerFactory : public InferenceEngine::ILayerImplFactory {
public:
    explicit InfEngineCustomLayerFactory(const InferenceEngine::CNNLayer* layer) : cnnLayer(*layer) {}

    InferenceEngine::StatusCode
    getImplementations(std::vector<InferenceEngine::ILayerImpl::Ptr>& impls,
                       InferenceEngine::ResponseDesc* resp) noexcept override {
        impls.push_back(std::make_shared<InfEngineCustomLayer>(cnnLayer));
        return InferenceEngine::StatusCode::OK;
    }

private:
    InferenceEngine::CNNLayer cnnLayer;
};

InferenceEngine::StatusCode InfEngineExtension::getFactoryFor(
        InferenceEngine::ILayerImplFactory*& factory,
        const InferenceEngine::CNNLayer* cnnLayer,
        InferenceEngine::ResponseDesc* resp
) noexcept
{
    if (cnnLayer->type != kOpenCVLayersType)
        return InferenceEngine::StatusCode::NOT_IMPLEMENTED;
    factory = new InfEngineCustomLayerFactory(cnnLayer);
    return InferenceEngine::StatusCode::OK;
}

InfEngineBackendNode::InfEngineBackendNode(const InferenceEngine::Builder::Layer& _layer)
    : BackendNode(DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019), layer(_layer) {}

    InfEngineBackendNode::InfEngineBackendNode(Ptr<Layer>& cvLayer_, std::vector<Mat*>& inputs,
                                               std::vector<Mat>& outputs,
                                               std::vector<Mat>& internals)
        : BackendNode(DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019), layer(cvLayer_->name),
          cvLayer(cvLayer_)
{
    CV_Assert(!cvLayer->name.empty());
    layer.setName(cvLayer->name);
    layer.setType(kOpenCVLayersType);
    layer.getParameters()["impl"] = (size_t)cvLayer.get();
    layer.getParameters()["outputs"] = shapesToStr(outputs);
    layer.getParameters()["internals"] = shapesToStr(internals);
    layer.setInputPorts(std::vector<InferenceEngine::Port>(inputs.size()));
    layer.setOutputPorts(std::vector<InferenceEngine::Port>(outputs.size()));
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

InfEngineBackendNet::InfEngineBackendNet() : netBuilder("")
{
    hasNetOwner = false;
    device_name = "CPU";
}

InfEngineBackendNet::InfEngineBackendNet(InferenceEngine::CNNNetwork& net) : netBuilder(""), cnn(net)
{
    hasNetOwner = true;
    device_name = "CPU";
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
        const std::string& inpName = inp->dataPtr->getName();

        std::string inpLayerName = inpName;
        size_t inpPortId = inpName.rfind('.');
        if (inpPortId != std::string::npos)
        {
            std::string portIdStr = inpName.substr(inpPortId + 1);
            if (std::all_of(portIdStr.begin(), portIdStr.end(), ::isdigit))
            {
                inpLayerName = inpName.substr(0, inpPortId);
                inpPortId = atoi(portIdStr.c_str());
            }
            else
                inpPortId = 0;
        }
        else
            inpPortId = 0;

        int inpId;
        it = layers.find(inpLayerName);
        if (it == layers.end())
        {
            InferenceEngine::Builder::InputLayer inpLayer(!inpLayerName.empty() ? inpLayerName : kDefaultInpLayerName);
            std::vector<size_t> shape(inp->blob->getTensorDesc().getDims());
            inpLayer.setPort(InferenceEngine::Port(shape));
            inpId = netBuilder.addLayer(inpLayer);

            layers.insert({inpName, inpId});
        }
        else
            inpId = it->second;

        netBuilder.connect({(size_t)inpId, inpPortId}, {(size_t)layerId, i});
        unconnectedPorts.erase({inpId, inpPortId});
    }
    CV_Assert(!outputs.empty());
    for (int i = 0; i < outputs.size(); ++i)
    {
        InferenceEngine::DataPtr dataPtr = infEngineDataNode(outputs[i]);
        std::string outputName = outputs.size() > 1 ? (layerName + "." + std::to_string(i)) : layerName;
#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R1)
        dataPtr->name = outputName;
#else
        dataPtr->setName(outputName);
#endif
    }
}

void InfEngineBackendNet::init(Target targetId)
{
    if (!hasNetOwner)
    {
        CV_Assert(!unconnectedPorts.empty());
        for (const auto& port : unconnectedPorts)
        {
            InferenceEngine::Builder::OutputLayer outLayer("myconv1");
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2019R1)
            // Inference Engine determines network precision by ports.
            InferenceEngine::Precision p = (targetId == DNN_TARGET_MYRIAD ||
                                            targetId == DNN_TARGET_HDDL ||
                                            targetId == DNN_TARGET_OPENCL_FP16) ?
                                           InferenceEngine::Precision::FP16 :
                                           InferenceEngine::Precision::FP32;
            outLayer.setPort(InferenceEngine::Port({}, p));
#endif
            netBuilder.addLayer({InferenceEngine::PortInfo(port.first, port.second)}, outLayer);
        }
        netBuilder.getContext().addShapeInferImpl(kOpenCVLayersType,
                            std::make_shared<InfEngineCustomLayerShapeInfer>());
        cnn = InferenceEngine::CNNNetwork(InferenceEngine::Builder::convertToICNNNetwork(netBuilder.build()));
    }

    switch (targetId)
    {
        case DNN_TARGET_CPU:
            device_name = "CPU";
            break;
        case DNN_TARGET_OPENCL:
        case DNN_TARGET_OPENCL_FP16:
            device_name = "GPU";
            break;
        case DNN_TARGET_MYRIAD:
            device_name = "MYRIAD";
            break;
        case DNN_TARGET_HDDL:
            device_name = "HDDL";
            break;
        case DNN_TARGET_FPGA:
            device_name = "FPGA";
            break;
        default:
            CV_Error(Error::StsNotImplemented, "Unknown target");
    };

    for (const auto& name : requestedOutputs)
    {
        cnn.addOutput(name);
    }

    for (const auto& it : cnn.getInputsInfo())
    {
        const std::string& name = it.first;
        auto blobIt = allBlobs.find(name);
        CV_Assert(blobIt != allBlobs.end());
        it.second->setPrecision(blobIt->second->getTensorDesc().getPrecision());
    }
    for (const auto& it : cnn.getOutputsInfo())
    {
        const std::string& name = it.first;
        auto blobIt = allBlobs.find(name);
        CV_Assert(blobIt != allBlobs.end());
        it.second->setPrecision(blobIt->second->getTensorDesc().getPrecision());  // Should be always FP32
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
    for (int i = 0; i < layer.getOutputPorts().size(); ++i)
        unconnectedPorts.insert({id, i});

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
    std::vector<size_t> shape = getShape<size_t>(m);
    if (m.type() == CV_32F)
        return InferenceEngine::DataPtr(new InferenceEngine::Data(name,
               {InferenceEngine::Precision::FP32, shape, estimateLayout(m)}));
    else if (m.type() == CV_8U)
        return InferenceEngine::DataPtr(new InferenceEngine::Data(name,
               {InferenceEngine::Precision::U8, shape, estimateLayout(m)}));
    else
        CV_Error(Error::StsNotImplemented, format("Unsupported data type %d", m.type()));
}

InferenceEngine::Blob::Ptr wrapToInfEngineBlob(const Mat& m, const std::vector<size_t>& shape,
                                               InferenceEngine::Layout layout)
{
    if (m.type() == CV_32F)
        return InferenceEngine::make_shared_blob<float>(
               {InferenceEngine::Precision::FP32, shape, layout}, (float*)m.data);
    else if (m.type() == CV_8U)
        return InferenceEngine::make_shared_blob<uint8_t>(
               {InferenceEngine::Precision::U8, shape, layout}, (uint8_t*)m.data);
    else
        CV_Error(Error::StsNotImplemented, format("Unsupported data type %d", m.type()));
}

InferenceEngine::Blob::Ptr wrapToInfEngineBlob(const Mat& m, InferenceEngine::Layout layout)
{
    std::vector<size_t> shape = getShape<size_t>(m);
    return wrapToInfEngineBlob(m, shape, layout);
}

InferenceEngine::Blob::Ptr cloneBlob(const InferenceEngine::Blob::Ptr& blob)
{
    InferenceEngine::Blob::Ptr copy;
    auto description = blob->getTensorDesc();
    InferenceEngine::Precision precision = description.getPrecision();
    if (precision == InferenceEngine::Precision::FP32)
    {
        copy = InferenceEngine::make_shared_blob<float>(description);
    }
    else if (precision == InferenceEngine::Precision::U8)
    {
        copy = InferenceEngine::make_shared_blob<uint8_t>(description);
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
    : BackendWrapper(DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019, targetId)
{
    dataPtr = wrapToInfEngineDataNode(m);
    blob = wrapToInfEngineBlob(m, estimateLayout(m));
}

InfEngineBackendWrapper::InfEngineBackendWrapper(Ptr<BackendWrapper> wrapper)
    : BackendWrapper(DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019, wrapper->targetId)
{
    Ptr<InfEngineBackendWrapper> ieWrapper = wrapper.dynamicCast<InfEngineBackendWrapper>();
    CV_Assert(!ieWrapper.empty());
    InferenceEngine::DataPtr srcData = ieWrapper->dataPtr;

    dataPtr = InferenceEngine::DataPtr(new InferenceEngine::Data(srcData->getName(), srcData->getTensorDesc()));
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

#endif // HAVE_DNN_IE_NN_BUILDER_2019

#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R1)
static std::map<std::string, InferenceEngine::InferenceEnginePluginPtr>& getSharedPlugins()
{
    static std::map<std::string, InferenceEngine::InferenceEnginePluginPtr> sharedPlugins;
    return sharedPlugins;
}
#else
static bool init_IE_plugins()
{
    // load and hold IE plugins
    static InferenceEngine::Core* init_core = new InferenceEngine::Core();  // 'delete' is never called
    (void)init_core->GetAvailableDevices();
    return true;
}
static InferenceEngine::Core& retrieveIECore(const std::string& id, std::map<std::string, std::shared_ptr<InferenceEngine::Core> >& cores)
{
    AutoLock lock(getInitializationMutex());
    std::map<std::string, std::shared_ptr<InferenceEngine::Core> >::iterator i = cores.find(id);
    if (i == cores.end())
    {
        std::shared_ptr<InferenceEngine::Core> core = std::make_shared<InferenceEngine::Core>();
        cores[id] = core;
        return *core.get();
    }
    return *(i->second).get();
}
static InferenceEngine::Core& create_IE_Core_instance(const std::string& id)
{
    static std::map<std::string, std::shared_ptr<InferenceEngine::Core> > cores;
    return retrieveIECore(id, cores);
}
static InferenceEngine::Core& create_IE_Core_pointer(const std::string& id)
{
    // load and hold IE plugins
    static std::map<std::string, std::shared_ptr<InferenceEngine::Core> >* cores =
            new std::map<std::string, std::shared_ptr<InferenceEngine::Core> >();
    return retrieveIECore(id, *cores);
}
InferenceEngine::Core& getCore(const std::string& id)
{
    // to make happy memory leak tools use:
    // - OPENCV_DNN_INFERENCE_ENGINE_HOLD_PLUGINS=0
    // - OPENCV_DNN_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND=0
    static bool param_DNN_INFERENCE_ENGINE_HOLD_PLUGINS = utils::getConfigurationParameterBool("OPENCV_DNN_INFERENCE_ENGINE_HOLD_PLUGINS", true);
    static bool init_IE_plugins_ = param_DNN_INFERENCE_ENGINE_HOLD_PLUGINS && init_IE_plugins(); CV_UNUSED(init_IE_plugins_);

    static bool param_DNN_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND =
            utils::getConfigurationParameterBool("OPENCV_DNN_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND",
#ifdef _WIN32
                true
#else
                false
#endif
            );

    InferenceEngine::Core& core = param_DNN_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND
            ? create_IE_Core_pointer(id)
            : create_IE_Core_instance(id);
    return core;
}
#endif

static bool detectArmPlugin_()
{
    InferenceEngine::Core& ie = getCore("CPU");
    const std::vector<std::string> devices = ie.GetAvailableDevices();
    for (std::vector<std::string>::const_iterator i = devices.begin(); i != devices.end(); ++i)
    {
        if (i->find("CPU") != std::string::npos)
        {
            const std::string name = ie.GetMetric(*i, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
            CV_LOG_INFO(NULL, "CPU plugin: " << name);
            return name.find("arm_compute::NEON") != std::string::npos;
        }
    }
    return false;
}

#if !defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)
static bool detectMyriadX_(std::string device)
{
    AutoLock lock(getInitializationMutex());
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2019R3)
    // Lightweight detection
    InferenceEngine::Core& ie = getCore(device);
    const std::vector<std::string> devices = ie.GetAvailableDevices();
    for (std::vector<std::string>::const_iterator i = devices.begin(); i != devices.end(); ++i)
    {
        if (i->find(device) != std::string::npos)
        {
            const std::string name = ie.GetMetric(*i, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
            CV_LOG_INFO(NULL, "Myriad device: " << name);
            return name.find("MyriadX") != std::string::npos || name.find("Myriad X") != std::string::npos || name.find("HDDL") != std::string::npos;
        }
    }
    return false;
#else
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

#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R1)
    InferenceEngine::InferenceEnginePluginPtr enginePtr;
    {
        auto& sharedPlugins = getSharedPlugins();
        auto pluginIt = sharedPlugins.find(device);
        if (pluginIt != sharedPlugins.end()) {
            enginePtr = pluginIt->second;
        } else {
            auto dispatcher = InferenceEngine::PluginDispatcher({""});
            enginePtr = dispatcher.getPluginByDevice(device);
            sharedPlugins[device] = enginePtr;
        }
    }
    auto plugin = InferenceEngine::InferencePlugin(enginePtr);
    try
    {
        auto netExec = plugin.LoadNetwork(cnn, {{"VPU_PLATFORM", "VPU_2480"}});
#else
    try
    {
#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R3)
        auto netExec = getCore(device).LoadNetwork(cnn, device, {{"VPU_PLATFORM", "VPU_2480"}});
#else
        auto netExec = getCore(device).LoadNetwork(cnn, device, {{"VPU_MYRIAD_PLATFORM", "VPU_MYRIAD_2480"}});
#endif
#endif
        auto infRequest = netExec.CreateInferRequest();
    } catch(...) {
        return false;
    }
    return true;
#endif
}
#endif  // !defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)


#ifdef HAVE_DNN_IE_NN_BUILDER_2019

void InfEngineBackendNet::initPlugin(InferenceEngine::CNNNetwork& net)
{
    CV_Assert(!isInitialized());

    try
    {
        AutoLock lock(getInitializationMutex());
#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R1)
        auto& sharedPlugins = getSharedPlugins();
        auto pluginIt = sharedPlugins.find(device_name);
        if (pluginIt != sharedPlugins.end())
        {
            enginePtr = pluginIt->second;
        }
        else
#else
        InferenceEngine::Core& ie = getCore(device_name);
#endif
        {
#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R1)
            auto dispatcher = InferenceEngine::PluginDispatcher({""});
            if (device_name == "FPGA")
                enginePtr = dispatcher.getPluginByDevice("HETERO:FPGA,CPU");
            else
                enginePtr = dispatcher.getPluginByDevice(device_name);
            sharedPlugins[device_name] = enginePtr;
#else
            isInit = true;
#endif
            std::vector<std::string> candidates;
            std::string param_pluginPath = utils::getConfigurationParameterString("OPENCV_DNN_IE_EXTRA_PLUGIN_PATH", "");
            if (!param_pluginPath.empty())
            {
                candidates.push_back(param_pluginPath);
            }
#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R3)
            if (device_name == "CPU" || device_name == "FPGA")
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
#endif
            bool found = false;
            for (size_t i = 0; i != candidates.size(); ++i)
            {
                const std::string& libName = candidates[i];
                try
                {
                    InferenceEngine::IExtensionPtr extension =
                        InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(libName);

#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R1)
                    enginePtr->AddExtension(extension, 0);
#else
                    ie.AddExtension(extension, "CPU");
#endif
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
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2019R1)
            // OpenCV fallbacks as extensions.
            try
            {
                ie.AddExtension(std::make_shared<InfEngineExtension>(), "CPU");
            }
            catch(const std::exception& e)
            {
                CV_LOG_INFO(NULL, "DNN-IE: Can't register OpenCV custom layers extension: " << e.what());
            }
#endif
            // Limit the number of CPU threads.
#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R1)
#ifndef _WIN32
            enginePtr->SetConfig({{
                InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, format("%d", getNumThreads()),
            }}, 0);
#endif  // _WIN32
#else
            if (device_name == "CPU")
                ie.SetConfig({{
                    InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, format("%d", getNumThreads()),
                }}, device_name);
#endif
        }
#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R1)
        plugin = InferenceEngine::InferencePlugin(enginePtr);
        netExec = plugin.LoadNetwork(net, {});
#else
        bool isHetero = false;
        if (device_name != "CPU")
        {
            isHetero = device_name == "FPGA";
            for (auto& layer : net)
            {
                if (layer->type == kOpenCVLayersType)
                {
                    isHetero = true;
#if INF_ENGINE_VER_MAJOR_LT(INF_ENGINE_RELEASE_2019R3)
                    // Not sure about lower versions but in 2019R3 we do not need this
                    layer->affinity = "CPU";
                }
                else
                {
                    layer->affinity = device_name;
#endif
                }
            }
        }
        if (isHetero)
            netExec = ie.LoadNetwork(net, "HETERO:" + device_name + ",CPU");
        else
            netExec = ie.LoadNetwork(net, device_name);
#endif
    }
    catch (const std::exception& ex)
    {
        CV_Error(Error::StsError, format("Failed to initialize Inference Engine backend (device = %s): %s", device_name.c_str(), ex.what()));
    }
}

bool InfEngineBackendNet::isInitialized()
{
#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R1)
    return (bool)enginePtr;
#else
    return isInit;
#endif
}

void InfEngineBackendNet::reset()
{
    allBlobs.clear();
    infRequests.clear();
    isInit = false;
}

void InfEngineBackendNet::addBlobs(const std::vector<cv::Ptr<BackendWrapper> >& ptrs)
{
    auto wrappers = infEngineWrappers(ptrs);
    for (const auto& wrapper : wrappers)
    {
        std::string name = wrapper->dataPtr->getName();
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
        outsNames[i] = outs[i]->dataPtr->getName();
    }
}

void InfEngineBackendNet::forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                                  bool isAsync)
{
    CV_LOG_DEBUG(NULL, "InfEngineBackendNet::forward(" << (isAsync ? "async" : "sync") << ")");
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
                CV_LOG_DEBUG(NULL, "DNN(IE): completionCallback(" << (int)status << ")");

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
                                CV_LOG_ERROR(NULL, "DNN: Exception occurred during async inference exception propagation");
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
                            CV_LOG_ERROR(NULL, "DNN: Exception occurred during async inference exception propagation");
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
    CV_LOG_DEBUG(NULL, "InfEngineBackendLayer::supportBackend(" << backendId << ")");
    return backendId == DNN_BACKEND_DEFAULT ||
           (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019);
}

void InfEngineBackendLayer::forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs,
                                    OutputArrayOfArrays internals)
{
    CV_Error(Error::StsInternal, "Choose Inference Engine as a preferable backend.");
}

InferenceEngine::Blob::Ptr convertFp16(const InferenceEngine::Blob::Ptr& blob)
{
    auto halfs = InferenceEngine::make_shared_blob<int16_t>({
                     InferenceEngine::Precision::FP16, blob->getTensorDesc().getDims(),
                     blob->getTensorDesc().getLayout()
                 });
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

#endif // HAVE_DNN_IE_NN_BUILDER_2019

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
#ifdef HAVE_DNN_IE_NN_BUILDER_2019
    CV_Assert(!node.empty());
    Ptr<InfEngineBackendNode> ieNode = node.dynamicCast<InfEngineBackendNode>();
    CV_Assert(!ieNode.empty());
    ieNode->net->forward(outBlobsWrappers, isAsync);
#else
    CV_Error(Error::StsNotImplemented, "This OpenCV version is built without Inference Engine NN Builder API support");
#endif  // HAVE_INF_ENGINE
}

CV__DNN_INLINE_NS_BEGIN

void resetMyriadDevice()
{
#ifdef HAVE_INF_ENGINE
    AutoLock lock(getInitializationMutex());
#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R1)
    getSharedPlugins().erase("MYRIAD");
#else
    // Unregister both "MYRIAD" and "HETERO:MYRIAD,CPU" plugins
    InferenceEngine::Core& ie = getCore("MYRIAD");
    try
    {
        ie.UnregisterPlugin("MYRIAD");
        ie.UnregisterPlugin("HETERO");
    }
    catch (...) {}
#endif
#endif  // HAVE_INF_ENGINE
}

void releaseHDDLPlugin()
{
#ifdef HAVE_INF_ENGINE
    AutoLock lock(getInitializationMutex());
#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2019R1)
    getSharedPlugins().erase("HDDL");
#else
    // Unregister both "HDDL" and "HETERO:HDDL,CPU" plugins
    InferenceEngine::Core& ie = getCore("HDDL");
    try
    {
        ie.UnregisterPlugin("HDDL");
        ie.UnregisterPlugin("HETERO");
    }
    catch (...) {}
#endif
#endif  // HAVE_INF_ENGINE
}

#ifdef HAVE_INF_ENGINE
bool isMyriadX()
{
    static bool myriadX = getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X;
    return myriadX;
}

bool isArmComputePlugin()
{
    static bool armPlugin = getInferenceEngineCPUType() == CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE;
    return armPlugin;
}

static std::string getInferenceEngineVPUType_()
{
    static std::string param_vpu_type = utils::getConfigurationParameterString("OPENCV_DNN_IE_VPU_TYPE", "");
    if (param_vpu_type == "")
    {
#if defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)
        param_vpu_type = OPENCV_DNN_IE_VPU_TYPE_DEFAULT;
#else
        CV_LOG_INFO(NULL, "OpenCV-DNN: running Inference Engine VPU autodetection: Myriad2/X or HDDL. In case of other accelerator types specify 'OPENCV_DNN_IE_VPU_TYPE' parameter");
        try {
            bool isMyriadX_ = detectMyriadX_("MYRIAD");
            bool isHDDL_ = detectMyriadX_("HDDL");
            if (isMyriadX_ || isHDDL_)
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

cv::String getInferenceEngineCPUType()
{
    static cv::String cpu_type = detectArmPlugin_() ?
                                 CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE :
                                 CV_DNN_INFERENCE_ENGINE_CPU_TYPE_X86;
    return cpu_type;
}

#else  // HAVE_INF_ENGINE

cv::String getInferenceEngineBackendType()
{
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}
cv::String setInferenceEngineBackendType(const cv::String& newBackendType)
{
    CV_UNUSED(newBackendType);
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}
cv::String getInferenceEngineVPUType()
{
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}

cv::String getInferenceEngineCPUType()
{
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}
#endif  // HAVE_INF_ENGINE


CV__DNN_INLINE_NS_END
}}  // namespace dnn, namespace cv
