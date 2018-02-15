// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "op_inf_engine.hpp"
#include <opencv2/dnn/shape_utils.hpp>

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

static InferenceEngine::DataPtr wrapToInfEngineDataNode(const Mat& m, const std::string& name = "")
{
    std::vector<size_t> reversedShape(&m.size[0], &m.size[0] + m.dims);
    std::reverse(reversedShape.begin(), reversedShape.end());
    return InferenceEngine::DataPtr(
      new InferenceEngine::Data(name, reversedShape, InferenceEngine::Precision::FP32,
                                InferenceEngine::Layout::ANY)
    );
}

InferenceEngine::TBlob<float>::Ptr wrapToInfEngineBlob(const Mat& m, const std::vector<size_t>& shape)
{
    return InferenceEngine::make_shared_blob<float>(InferenceEngine::Precision::FP32,
                                                    shape, (float*)m.data);
}

InferenceEngine::TBlob<float>::Ptr wrapToInfEngineBlob(const Mat& m)
{
    std::vector<size_t> reversedShape(&m.size[0], &m.size[0] + m.dims);
    std::reverse(reversedShape.begin(), reversedShape.end());
    return wrapToInfEngineBlob(m, reversedShape);
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
    blob = wrapToInfEngineBlob(m);
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

void InfEngineBackendNet::Release() noexcept
{
    layers.clear();
    inputs.clear();
    outputs.clear();
}

InferenceEngine::Precision InfEngineBackendNet::getPrecision() noexcept
{
    return InferenceEngine::Precision::FP32;
}

// Assume that outputs of network is unconnected blobs.
void InfEngineBackendNet::getOutputsInfo(InferenceEngine::OutputsDataMap &outputs_) noexcept
{
    outputs_ = outputs;
}

// Returns input references that aren't connected to internal outputs.
void InfEngineBackendNet::getInputsInfo(InferenceEngine::InputsDataMap &inputs_) noexcept
{
    inputs_ = inputs;
}

// Returns input references that aren't connected to internal outputs.
void InfEngineBackendNet::getInputsInfo(InferenceEngine::InputsDataMap &inputs_) const noexcept
{
    inputs_ = inputs;
}

InferenceEngine::InputInfo::Ptr InfEngineBackendNet::getInput(const std::string &inputName) noexcept
{
    getInputsInfo(inputs);
    const auto& it = inputs.find(inputName);
    CV_Assert(it != inputs.end());
    return it->second;
}

void InfEngineBackendNet::getName(char *pName, size_t len) noexcept
{
    CV_Error(Error::StsNotImplemented, "");
}

size_t InfEngineBackendNet::layerCount() noexcept
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
    CV_Error(Error::StsNotImplemented, "");
    return InferenceEngine::StatusCode::OK;
}

void InfEngineBackendNet::setTargetDevice(InferenceEngine::TargetDevice device) noexcept
{
    if (device != InferenceEngine::TargetDevice::eCPU)
        CV_Error(Error::StsNotImplemented, "");
}

InferenceEngine::TargetDevice InfEngineBackendNet::getTargetDevice() noexcept
{
    return InferenceEngine::TargetDevice::eCPU;
}

InferenceEngine::StatusCode InfEngineBackendNet::setBatchSize(const size_t size) noexcept
{
    CV_Error(Error::StsNotImplemented, "");
    return InferenceEngine::StatusCode::OK;
}

size_t InfEngineBackendNet::getBatchSize() const noexcept
{
    CV_Error(Error::StsNotImplemented, "");
    return 0;
}

void InfEngineBackendNet::initEngine()
{
    CV_Assert(!isInitialized(), !layers.empty());

    // Collect all external input blobs.
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
            // TODO: Replace to uniquness assertion.
            if (internalOutputs.find(out->name) == internalOutputs.end())
                internalOutputs[out->name] = out;
        }
    }
    CV_Assert(!inputs.empty());

    // Add all unconnected blobs to output blobs.
    InferenceEngine::OutputsDataMap unconnectedOuts;
    for (const auto& l : layers)
    {
        // Add all outputs.
        for (const InferenceEngine::DataPtr& out : l->outData)
        {
            // TODO: Replace to uniquness assertion.
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

    // Set up input blobs.
    inpBlobs.clear();
    for (const auto& it : inputs)
    {
        CV_Assert(allBlobs.find(it.first) != allBlobs.end());
        inpBlobs[it.first] = allBlobs[it.first];
    }

    // Set up output blobs.
    outBlobs.clear();
    for (const auto& it : outputs)
    {
        CV_Assert(allBlobs.find(it.first) != allBlobs.end());
        outBlobs[it.first] = allBlobs[it.first];
    }

#ifdef _WIN32
    engine = InferenceEngine::InferenceEnginePluginPtr("MKLDNNPlugin.dll");
#else
    engine = InferenceEngine::InferenceEnginePluginPtr("libMKLDNNPlugin.so");
#endif  // _WIN32
    InferenceEngine::ResponseDesc resp;
    InferenceEngine::StatusCode status = engine->LoadNetwork(*this, &resp);
    if (status != InferenceEngine::StatusCode::OK)
        CV_Error(Error::StsAssert, resp.msg);
}

bool InfEngineBackendNet::isInitialized()
{
    return (bool)engine;
}

void InfEngineBackendNet::addBlobs(const std::vector<Ptr<BackendWrapper> >& ptrs)
{
    auto wrappers = infEngineWrappers(ptrs);
    for (const auto& wrapper : wrappers)
    {
        allBlobs[wrapper->dataPtr->name] = wrapper->blob;
    }
}

void InfEngineBackendNet::forward()
{
    InferenceEngine::ResponseDesc resp;
    InferenceEngine::StatusCode status = engine->Infer(inpBlobs, outBlobs, &resp);
    if (status != InferenceEngine::StatusCode::OK)
        CV_Error(Error::StsAssert, resp.msg);
}

static inline Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob)
{
    // NOTE: Inference Engine sizes are reversed.
    std::vector<size_t> dims = blob->dims();
    std::vector<int> size(dims.begin(), dims.end());
    std::reverse(size.begin(), size.end());
    return Mat(size, CV_32F, (void*)blob->buffer());
}

void fuseConvWeights(const std::shared_ptr<InferenceEngine::ConvolutionLayer>& conv,
                     const Mat& w, const Mat& b)
{
    CV_Assert(!w.empty() || !b.empty());
    if (!w.empty())
    {
        // Get convolution's weights. Clone the data because Inference Engine can host it
        // and conv->_weights->allocate() below will deallocate it.
        Mat originWeights = infEngineBlobToMat(conv->_weights).clone();

        // Create new weights blob.
        conv->_weights = InferenceEngine::make_shared_blob<float>(
                            InferenceEngine::Precision::FP32, conv->_weights->dims());
        conv->_weights->allocate();

        // Convolution weights have OIHW data layout.
        // (conv(I) + b1 ) * w + b2
        // w*conv(I) + b1 * w + b2
        Mat fusedWeights = infEngineBlobToMat(conv->_weights);

        const int numChannels = fusedWeights.size[0];
        // Mat weights = blobs[0].reshape(1, 1);
        // Mat bias = hasBias ? blobs[1].reshape(1, 1) : Mat();
        CV_Assert(numChannels == w.total());
        CV_Assert(b.empty() || numChannels == b.total());
        for (int i = 0; i < numChannels; ++i)
        {
            cv::multiply(slice(originWeights, i), w.at<float>(i), slice(fusedWeights, i));
        }
    }
    if (conv->_biases)
    {
        // The same for biases.
        Mat originBiases = infEngineBlobToMat(conv->_biases).clone();

        conv->_biases = InferenceEngine::make_shared_blob<float>(
                            InferenceEngine::Precision::FP32, conv->_biases->dims());
        conv->_biases->allocate();
        Mat fusedBiases = infEngineBlobToMat(conv->_biases);
        originBiases.copyTo(fusedBiases);

        if (!w.empty())
            cv::multiply(w.reshape(1, fusedBiases.dims, &fusedBiases.size[0]), fusedBiases, fusedBiases);
        if (!b.empty())
            cv::add(fusedBiases, b.reshape(1, fusedBiases.dims, &fusedBiases.size[0]), fusedBiases);
    }
    else
        conv->_biases = wrapToInfEngineBlob(b);
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

}}  // namespace dnn, namespace cv
