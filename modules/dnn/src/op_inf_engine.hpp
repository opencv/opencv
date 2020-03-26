// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_OP_INF_ENGINE_HPP__
#define __OPENCV_DNN_OP_INF_ENGINE_HPP__

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/dnn.hpp"

#include "opencv2/core/async.hpp"
#include "opencv2/core/detail/async_promise.hpp"

#include "opencv2/dnn/utils/inference_engine.hpp"

#ifdef HAVE_INF_ENGINE

#define INF_ENGINE_RELEASE_2020_1 2020010000

#ifndef INF_ENGINE_RELEASE
#warning("IE version have not been provided via command-line. Using 2020.1 by default")
#define INF_ENGINE_RELEASE INF_ENGINE_RELEASE_2020_1
#endif

#define INF_ENGINE_VER_MAJOR_GT(ver) (((INF_ENGINE_RELEASE) / 10000) > ((ver) / 10000))
#define INF_ENGINE_VER_MAJOR_GE(ver) (((INF_ENGINE_RELEASE) / 10000) >= ((ver) / 10000))
#define INF_ENGINE_VER_MAJOR_LT(ver) (((INF_ENGINE_RELEASE) / 10000) < ((ver) / 10000))
#define INF_ENGINE_VER_MAJOR_LE(ver) (((INF_ENGINE_RELEASE) / 10000) <= ((ver) / 10000))
#define INF_ENGINE_VER_MAJOR_EQ(ver) (((INF_ENGINE_RELEASE) / 10000) == ((ver) / 10000))

#if defined(__GNUC__) && __GNUC__ >= 5
//#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif

#include <inference_engine.hpp>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4245)
#pragma warning(disable : 4268)
#endif
#include <ngraph/ngraph.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#if defined(__GNUC__) && __GNUC__ >= 5
//#pragma GCC diagnostic pop
#endif

#endif  // HAVE_INF_ENGINE

namespace cv { namespace dnn {

#ifdef HAVE_INF_ENGINE


Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob);

void infEngineBlobsToMats(const std::vector<InferenceEngine::Blob::Ptr>& blobs,
                          std::vector<Mat>& mats);


class InfEngineNgraphNode;

class InfEngineNgraphNet
{
public:
    InfEngineNgraphNet();
    InfEngineNgraphNet(InferenceEngine::CNNNetwork& net);

    void addOutput(const std::string& name);

    bool isInitialized();
    void init(Target targetId);

    void forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers, bool isAsync);

    void initPlugin(InferenceEngine::CNNNetwork& net);
    ngraph::ParameterVector setInputs(const std::vector<cv::Mat>& inputs, const std::vector<std::string>& names);

    void setUnconnectedNodes(Ptr<InfEngineNgraphNode>& node);
    void addBlobs(const std::vector<cv::Ptr<BackendWrapper> >& ptrs);

    void createNet(Target targetId);
    void setNodePtr(std::shared_ptr<ngraph::Node>* ptr);
private:
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
    std::vector<std::string> requestedOutputs;
    std::unordered_set<std::shared_ptr<ngraph::Node>> unconnectedNodes;
};

class InfEngineNgraphNode : public BackendNode
{
public:
    InfEngineNgraphNode(const std::vector<Ptr<BackendNode> >& nodes, Ptr<Layer>& layer,
                        std::vector<Mat*>& inputs, std::vector<Mat>& outputs,
                        std::vector<Mat>& internals);

    InfEngineNgraphNode(std::shared_ptr<ngraph::Node>&& _node);
    InfEngineNgraphNode(std::shared_ptr<ngraph::Node>& _node);

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

    InferenceEngine::DataPtr dataPtr;
    InferenceEngine::Blob::Ptr blob;
    AsyncArray futureMat;
};

InferenceEngine::DataPtr ngraphDataNode(const Ptr<BackendWrapper>& ptr);

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

CV__DNN_EXPERIMENTAL_NS_BEGIN

bool isMyriadX();

CV__DNN_EXPERIMENTAL_NS_END

InferenceEngine::Core& getCore(const std::string& id);

template<typename T = size_t>
static inline std::vector<T> getShape(const Mat& mat)
{
    std::vector<T> result(mat.dims);
    for (int i = 0; i < mat.dims; i++)
        result[i] = (T)mat.size[i];
    return result;
}


#endif  // HAVE_INF_ENGINE

void forwardNgraph(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                   Ptr<BackendNode>& node, bool isAsync);


}}  // namespace dnn, namespace cv

#endif  // __OPENCV_DNN_OP_INF_ENGINE_HPP__
