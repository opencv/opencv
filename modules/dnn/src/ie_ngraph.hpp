// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_IE_NGRAPH_HPP__
#define __OPENCV_DNN_IE_NGRAPH_HPP__

#include "op_inf_engine.hpp"

#ifdef HAVE_DNN_NGRAPH

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4245)
#pragma warning(disable : 4268)
#endif
#include <ngraph/ngraph.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif  // HAVE_DNN_NGRAPH

namespace cv { namespace dnn {

#ifdef HAVE_DNN_NGRAPH

class InfEngineNgraphNode;


class InfEngineNgraphNet
{
public:
    InfEngineNgraphNet(detail::NetImplBase& netImpl);
    InfEngineNgraphNet(detail::NetImplBase& netImpl, InferenceEngine::CNNNetwork& net);

    void addOutput(const Ptr<InfEngineNgraphNode>& node);

    bool isInitialized();
    void init(Target targetId);

    void forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers, bool isAsync);

    void initPlugin(InferenceEngine::CNNNetwork& net);
    ngraph::ParameterVector setInputs(const std::vector<cv::Mat>& inputs, const std::vector<std::string>& names);

    void addBlobs(const std::vector<cv::Ptr<BackendWrapper> >& ptrs);

    void createNet(Target targetId);
    void setNodePtr(std::shared_ptr<ngraph::Node>* ptr);

    void reset();

//private:
    detail::NetImplBase& netImpl_;

    void release();
    int getNumComponents();
    void dfs(std::shared_ptr<ngraph::Node>& node, std::vector<std::shared_ptr<ngraph::Node>>& comp,
             std::unordered_map<std::string, bool>& used);

    ngraph::ParameterVector inputs_vec;
    std::shared_ptr<ngraph::Function> ngraph_function;
    std::vector<std::vector<std::shared_ptr<ngraph::Node>>> components;
    std::unordered_map<std::string, std::shared_ptr<ngraph::Node>* > all_nodes;

    InferenceEngine::ExecutableNetwork netExec;
    InferenceEngine::BlobMap allBlobs;
    std::string device_name;
    bool isInit = false;

    struct NgraphReqWrapper
    {
        NgraphReqWrapper() : isReady(true) {}

        void makePromises(const std::vector<Ptr<BackendWrapper> >& outs);

        InferenceEngine::InferRequest req;
        std::vector<cv::AsyncPromise> outProms;
        std::vector<std::string> outsNames;
        bool isReady;
    };
    std::vector<Ptr<NgraphReqWrapper> > infRequests;

    InferenceEngine::CNNNetwork cnn;
    bool hasNetOwner;
    std::unordered_map<std::string, Ptr<InfEngineNgraphNode> > requestedOutputs;

    std::map<std::string, InferenceEngine::TensorDesc> outputsDesc;
};

class InfEngineNgraphNode : public BackendNode
{
public:
    InfEngineNgraphNode(const std::vector<Ptr<BackendNode> >& nodes, Ptr<Layer>& layer,
                        std::vector<Mat*>& inputs, std::vector<Mat>& outputs,
                        std::vector<Mat>& internals);

    InfEngineNgraphNode(std::shared_ptr<ngraph::Node>&& _node);
    InfEngineNgraphNode(const std::shared_ptr<ngraph::Node>& _node);

    void setName(const std::string& name);

    // Inference Engine network object that allows to obtain the outputs of this layer.
    std::shared_ptr<ngraph::Node> node;
    Ptr<InfEngineNgraphNet> net;
    Ptr<dnn::Layer> cvLayer;
};

class NgraphBackendWrapper : public BackendWrapper
{
public:
    NgraphBackendWrapper(int targetId, const Mat& m);
    NgraphBackendWrapper(Ptr<BackendWrapper> wrapper);
    ~NgraphBackendWrapper();

    static Ptr<BackendWrapper> create(Ptr<BackendWrapper> wrapper);

    virtual void copyToHost() CV_OVERRIDE;
    virtual void setHostDirty() CV_OVERRIDE;

    Mat* host;
    InferenceEngine::DataPtr dataPtr;
    InferenceEngine::Blob::Ptr blob;
    AsyncArray futureMat;
};

InferenceEngine::DataPtr ngraphDataNode(const Ptr<BackendWrapper>& ptr);
InferenceEngine::DataPtr ngraphDataOutputNode(
        const Ptr<BackendWrapper>& ptr,
        const InferenceEngine::TensorDesc& description,
        const std::string name);

// This is a fake class to run networks from Model Optimizer. Objects of that
// class simulate responses of layers are imported by OpenCV and supported by
// Inference Engine. The main difference is that they do not perform forward pass.
class NgraphBackendLayer : public Layer
{
public:
    NgraphBackendLayer(const InferenceEngine::CNNNetwork &t_net_) : t_net(t_net_) {};

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

#endif  // HAVE_DNN_NGRAPH

void forwardNgraph(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                   Ptr<BackendNode>& node, bool isAsync);

}}  // namespace cv::dnn


#endif  // __OPENCV_DNN_IE_NGRAPH_HPP__
