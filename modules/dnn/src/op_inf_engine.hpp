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

#define INF_ENGINE_RELEASE_2018R1 2018010000
#define INF_ENGINE_RELEASE_2018R2 2018020000
#define INF_ENGINE_RELEASE_2018R3 2018030000
#define INF_ENGINE_RELEASE_2018R4 2018040000
#define INF_ENGINE_RELEASE_2018R5 2018050000

#ifndef INF_ENGINE_RELEASE
#warning("IE version have not been provided via command-line. Using 2018R5 by default")
#define INF_ENGINE_RELEASE INF_ENGINE_RELEASE_2018R5
#endif

#define INF_ENGINE_VER_MAJOR_GT(ver) (((INF_ENGINE_RELEASE) / 10000) > ((ver) / 10000))
#define INF_ENGINE_VER_MAJOR_GE(ver) (((INF_ENGINE_RELEASE) / 10000) >= ((ver) / 10000))

#endif  // HAVE_INF_ENGINE

namespace cv { namespace dnn {

#ifdef HAVE_INF_ENGINE

class InfEngineBackendNet : public InferenceEngine::ICNNNetwork
{
public:
    InfEngineBackendNet();

    InfEngineBackendNet(InferenceEngine::CNNNetwork& net);

    virtual void Release() noexcept CV_OVERRIDE;

    void setPrecision(InferenceEngine::Precision p) noexcept;

    virtual InferenceEngine::Precision getPrecision() noexcept;

    virtual InferenceEngine::Precision getPrecision() const noexcept;

    virtual void getOutputsInfo(InferenceEngine::OutputsDataMap &out) noexcept /*CV_OVERRIDE*/;

    virtual void getOutputsInfo(InferenceEngine::OutputsDataMap &out) const noexcept /*CV_OVERRIDE*/;

    virtual void getInputsInfo(InferenceEngine::InputsDataMap &inputs) noexcept /*CV_OVERRIDE*/;

    virtual void getInputsInfo(InferenceEngine::InputsDataMap &inputs) const noexcept /*CV_OVERRIDE*/;

    virtual InferenceEngine::InputInfo::Ptr getInput(const std::string &inputName) noexcept;

    virtual InferenceEngine::InputInfo::Ptr getInput(const std::string &inputName) const noexcept;

    virtual InferenceEngine::StatusCode serialize(const std::string &xmlPath, const std::string &binPath, InferenceEngine::ResponseDesc* resp) const noexcept;

    virtual void getName(char *pName, size_t len) noexcept;

    virtual void getName(char *pName, size_t len) const noexcept;

    virtual const std::string& getName() const noexcept;

    virtual size_t layerCount() noexcept;

    virtual size_t layerCount() const noexcept;

    virtual InferenceEngine::DataPtr& getData(const char *dname) noexcept CV_OVERRIDE;

    virtual void addLayer(const InferenceEngine::CNNLayerPtr &layer) noexcept CV_OVERRIDE;

    virtual InferenceEngine::StatusCode addOutput(const std::string &layerName,
                                                  size_t outputIndex = 0,
                                                  InferenceEngine::ResponseDesc *resp = nullptr) noexcept;

    virtual InferenceEngine::StatusCode getLayerByName(const char *layerName,
                                                       InferenceEngine::CNNLayerPtr &out,
                                                       InferenceEngine::ResponseDesc *resp) noexcept;

    virtual InferenceEngine::StatusCode getLayerByName(const char *layerName,
                                                       InferenceEngine::CNNLayerPtr &out,
                                                       InferenceEngine::ResponseDesc *resp) const noexcept;

    virtual void setTargetDevice(InferenceEngine::TargetDevice device) noexcept CV_OVERRIDE;

    virtual InferenceEngine::TargetDevice getTargetDevice() noexcept;

    virtual InferenceEngine::TargetDevice getTargetDevice() const noexcept;

    virtual InferenceEngine::StatusCode setBatchSize(const size_t size) noexcept CV_OVERRIDE;

    virtual InferenceEngine::StatusCode setBatchSize(size_t size, InferenceEngine::ResponseDesc* responseDesc) noexcept;

    virtual size_t getBatchSize() const noexcept CV_OVERRIDE;

#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2018R2)
    virtual InferenceEngine::StatusCode AddExtension(const InferenceEngine::IShapeInferExtensionPtr& extension, InferenceEngine::ResponseDesc* resp) noexcept;
    virtual InferenceEngine::StatusCode reshape(const InputShapes& inputShapes, InferenceEngine::ResponseDesc* resp) noexcept;
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

class InfEngineBackendNode : public BackendNode
{
public:
    InfEngineBackendNode(const InferenceEngine::CNNLayerPtr& layer);

    void connect(std::vector<Ptr<BackendWrapper> >& inputs,
                 std::vector<Ptr<BackendWrapper> >& outputs);

    InferenceEngine::CNNLayerPtr layer;
    // Inference Engine network object that allows to obtain the outputs of this layer.
    Ptr<InfEngineBackendNet> net;
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
InferenceEngine::TBlob<int16_t>::Ptr convertFp16(const InferenceEngine::Blob::Ptr& blob);

// This is a fake class to run networks from Model Optimizer. Objects of that
// class simulate responses of layers are imported by OpenCV and supported by
// Inference Engine. The main difference is that they do not perform forward pass.
class InfEngineBackendLayer : public Layer
{
public:
    InfEngineBackendLayer(const InferenceEngine::DataPtr& output);

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE;

    virtual void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs,
                         OutputArrayOfArrays internals) CV_OVERRIDE;

    virtual bool supportBackend(int backendId) CV_OVERRIDE;

private:
    InferenceEngine::DataPtr output;
};

#endif  // HAVE_INF_ENGINE

bool haveInfEngine();

void forwardInfEngine(Ptr<BackendNode>& node);

}}  // namespace dnn, namespace cv

#endif  // __OPENCV_DNN_OP_INF_ENGINE_HPP__
