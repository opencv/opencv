// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_OP_INF_ENGINE_HPP__
#define __OPENCV_DNN_OP_INF_ENGINE_HPP__

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/dnn.hpp"

#ifdef HAVE_INF_ENGINE
#if defined(__GNUC__) && __GNUC__ >= 5
//#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include <inference_engine.hpp>
#if defined(__GNUC__) && __GNUC__ >= 5
//#pragma GCC diagnostic pop
#endif

#define INF_ENGINE_RELEASE_2018R3 2018030000
#define INF_ENGINE_RELEASE_2018R4 2018040000
#define INF_ENGINE_RELEASE_2018R5 2018050000

#ifndef INF_ENGINE_RELEASE
#warning("IE version have not been provided via command-line. Using 2018R5 by default")
#define INF_ENGINE_RELEASE INF_ENGINE_RELEASE_2018R5
#endif

#define INF_ENGINE_VER_MAJOR_GT(ver) (((INF_ENGINE_RELEASE) / 10000) > ((ver) / 10000))
#define INF_ENGINE_VER_MAJOR_GE(ver) (((INF_ENGINE_RELEASE) / 10000) >= ((ver) / 10000))
#define INF_ENGINE_VER_MAJOR_LT(ver) (((INF_ENGINE_RELEASE) / 10000) < ((ver) / 10000))
#define INF_ENGINE_VER_MAJOR_EQ(ver) (((INF_ENGINE_RELEASE) / 10000) == ((ver) / 10000))

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2018R5)
#include <ie_builders.hpp>
#endif

#endif  // HAVE_INF_ENGINE

namespace cv { namespace dnn {

#ifdef HAVE_INF_ENGINE

#if INF_ENGINE_VER_MAJOR_LT(INF_ENGINE_RELEASE_2018R5)
class InfEngineBackendNet : public InferenceEngine::ICNNNetwork
{
public:
    InfEngineBackendNet();

    InfEngineBackendNet(InferenceEngine::CNNNetwork& net);

    virtual void Release() CV_NOEXCEPT CV_OVERRIDE;

    void setPrecision(InferenceEngine::Precision p) CV_NOEXCEPT;

    virtual InferenceEngine::Precision getPrecision() CV_NOEXCEPT;

    virtual InferenceEngine::Precision getPrecision() const CV_NOEXCEPT;

    virtual void getOutputsInfo(InferenceEngine::OutputsDataMap &out) CV_NOEXCEPT /*CV_OVERRIDE*/;

    virtual void getOutputsInfo(InferenceEngine::OutputsDataMap &out) const CV_NOEXCEPT /*CV_OVERRIDE*/;

    virtual void getInputsInfo(InferenceEngine::InputsDataMap &inputs) CV_NOEXCEPT /*CV_OVERRIDE*/;

    virtual void getInputsInfo(InferenceEngine::InputsDataMap &inputs) const CV_NOEXCEPT /*CV_OVERRIDE*/;

    virtual InferenceEngine::InputInfo::Ptr getInput(const std::string &inputName) CV_NOEXCEPT;

    virtual InferenceEngine::InputInfo::Ptr getInput(const std::string &inputName) const CV_NOEXCEPT;

    virtual InferenceEngine::StatusCode serialize(const std::string &xmlPath, const std::string &binPath, InferenceEngine::ResponseDesc* resp) const CV_NOEXCEPT;

    virtual void getName(char *pName, size_t len) CV_NOEXCEPT;

    virtual void getName(char *pName, size_t len) const CV_NOEXCEPT;

    virtual const std::string& getName() const CV_NOEXCEPT;

    virtual size_t layerCount() CV_NOEXCEPT;

    virtual size_t layerCount() const CV_NOEXCEPT;

    virtual InferenceEngine::DataPtr& getData(const char *dname) CV_NOEXCEPT CV_OVERRIDE;

    virtual void addLayer(const InferenceEngine::CNNLayerPtr &layer) CV_NOEXCEPT CV_OVERRIDE;

    virtual InferenceEngine::StatusCode addOutput(const std::string &layerName,
                                                  size_t outputIndex = 0,
                                                  InferenceEngine::ResponseDesc *resp = nullptr) CV_NOEXCEPT;

    virtual InferenceEngine::StatusCode getLayerByName(const char *layerName,
                                                       InferenceEngine::CNNLayerPtr &out,
                                                       InferenceEngine::ResponseDesc *resp) CV_NOEXCEPT;

    virtual InferenceEngine::StatusCode getLayerByName(const char *layerName,
                                                       InferenceEngine::CNNLayerPtr &out,
                                                       InferenceEngine::ResponseDesc *resp) const CV_NOEXCEPT;

    virtual void setTargetDevice(InferenceEngine::TargetDevice device) CV_NOEXCEPT CV_OVERRIDE;

    virtual InferenceEngine::TargetDevice getTargetDevice() CV_NOEXCEPT;

    virtual InferenceEngine::TargetDevice getTargetDevice() const CV_NOEXCEPT;

    virtual InferenceEngine::StatusCode setBatchSize(const size_t size) CV_NOEXCEPT CV_OVERRIDE;

    virtual InferenceEngine::StatusCode setBatchSize(size_t size, InferenceEngine::ResponseDesc* responseDesc) CV_NOEXCEPT;

    virtual size_t getBatchSize() const CV_NOEXCEPT CV_OVERRIDE;

#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2018R2)
    virtual InferenceEngine::StatusCode AddExtension(const InferenceEngine::IShapeInferExtensionPtr& extension, InferenceEngine::ResponseDesc* resp) CV_NOEXCEPT;
    virtual InferenceEngine::StatusCode reshape(const InputShapes& inputShapes, InferenceEngine::ResponseDesc* resp) CV_NOEXCEPT;
#endif

    void init(int targetId);

    void addBlobs(const std::vector<Ptr<BackendWrapper> >& wrappers);

    void forward();

    bool isInitialized();

private:
    std::vector<InferenceEngine::CNNLayerPtr> layers;
    InferenceEngine::InputsDataMap inputs;
    InferenceEngine::OutputsDataMap outputs;
    InferenceEngine::BlobMap inpBlobs;
    InferenceEngine::BlobMap outBlobs;
    InferenceEngine::BlobMap allBlobs;
    InferenceEngine::TargetDevice targetDevice;
    InferenceEngine::Precision precision;
    InferenceEngine::InferenceEnginePluginPtr enginePtr;
    InferenceEngine::InferencePlugin plugin;
    InferenceEngine::ExecutableNetwork netExec;
    InferenceEngine::InferRequest infRequest;
    // In case of models from Model Optimizer we need to manage their lifetime.
    InferenceEngine::CNNNetwork netOwner;
    // There is no way to check if netOwner is initialized or not so we use
    // a separate flag to determine if the model has been loaded from IR.
    bool hasNetOwner;

    std::string name;

    void initPlugin(InferenceEngine::ICNNNetwork& net);
};

#else  // IE < R5

class InfEngineBackendNet
{
public:
    InfEngineBackendNet();

    InfEngineBackendNet(InferenceEngine::CNNNetwork& net);

    void addLayer(InferenceEngine::Builder::Layer& layer);

    void addOutput(const std::string& name);

    void connect(const std::vector<Ptr<BackendWrapper> >& inputs,
                 const std::vector<Ptr<BackendWrapper> >& outputs,
                 const std::string& layerName);

    bool isInitialized();

    void init(int targetId);

    void forward();

    void initPlugin(InferenceEngine::ICNNNetwork& net);

    void addBlobs(const std::vector<Ptr<BackendWrapper> >& ptrs);

private:
    InferenceEngine::Builder::Network netBuilder;

    InferenceEngine::InferenceEnginePluginPtr enginePtr;
    InferenceEngine::InferencePlugin plugin;
    InferenceEngine::ExecutableNetwork netExec;
    InferenceEngine::InferRequest infRequest;
    InferenceEngine::BlobMap allBlobs;
    InferenceEngine::BlobMap inpBlobs;
    InferenceEngine::BlobMap outBlobs;
    InferenceEngine::TargetDevice targetDevice;

    InferenceEngine::CNNNetwork cnn;
    bool hasNetOwner;

    std::map<std::string, int> layers;
    std::vector<std::string> requestedOutputs;

    std::set<int> unconnectedLayersIds;
};
#endif  // IE < R5

class InfEngineBackendNode : public BackendNode
{
public:
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2018R5)
    InfEngineBackendNode(const InferenceEngine::Builder::Layer& layer);
#else
    InfEngineBackendNode(const InferenceEngine::CNNLayerPtr& layer);
#endif

    void connect(std::vector<Ptr<BackendWrapper> >& inputs,
                 std::vector<Ptr<BackendWrapper> >& outputs);

    // Inference Engine network object that allows to obtain the outputs of this layer.
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2018R5)
    InferenceEngine::Builder::Layer layer;
    Ptr<InfEngineBackendNet> net;
#else
    InferenceEngine::CNNLayerPtr layer;
    Ptr<InfEngineBackendNet> net;
#endif
};

class InfEngineBackendWrapper : public BackendWrapper
{
public:
    InfEngineBackendWrapper(int targetId, const Mat& m);

    InfEngineBackendWrapper(Ptr<BackendWrapper> wrapper);

    ~InfEngineBackendWrapper();

    static Ptr<BackendWrapper> create(Ptr<BackendWrapper> wrapper);

    virtual void copyToHost() CV_OVERRIDE;

    virtual void setHostDirty() CV_OVERRIDE;

    InferenceEngine::DataPtr dataPtr;
    InferenceEngine::Blob::Ptr blob;
};

InferenceEngine::Blob::Ptr wrapToInfEngineBlob(const Mat& m, InferenceEngine::Layout layout = InferenceEngine::Layout::ANY);

InferenceEngine::Blob::Ptr wrapToInfEngineBlob(const Mat& m, const std::vector<size_t>& shape, InferenceEngine::Layout layout);

InferenceEngine::DataPtr infEngineDataNode(const Ptr<BackendWrapper>& ptr);

Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob);

// Convert Inference Engine blob with FP32 precision to FP16 precision.
// Allocates memory for a new blob.
InferenceEngine::Blob::Ptr convertFp16(const InferenceEngine::Blob::Ptr& blob);

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2018R5)
void addConstantData(const std::string& name, InferenceEngine::Blob::Ptr data, InferenceEngine::Builder::Layer& l);
#endif

// This is a fake class to run networks from Model Optimizer. Objects of that
// class simulate responses of layers are imported by OpenCV and supported by
// Inference Engine. The main difference is that they do not perform forward pass.
class InfEngineBackendLayer : public Layer
{
public:
    InfEngineBackendLayer(const InferenceEngine::CNNNetwork &t_net_) : t_net(t_net_) {};

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE;

    virtual void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs,
                         OutputArrayOfArrays internals) CV_OVERRIDE;

    virtual bool supportBackend(int backendId) CV_OVERRIDE;

private:
    InferenceEngine::CNNNetwork t_net;
};

#endif  // HAVE_INF_ENGINE

bool haveInfEngine();

void forwardInfEngine(Ptr<BackendNode>& node);

}}  // namespace dnn, namespace cv

#endif  // __OPENCV_DNN_OP_INF_ENGINE_HPP__
