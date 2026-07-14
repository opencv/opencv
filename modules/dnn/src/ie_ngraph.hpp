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
#include <openvino/openvino.hpp>
#include <openvino/op/ops.hpp>
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
    InfEngineNgraphNet(detail::NetImplBase& netImpl, std::shared_ptr<ov::Model>& net);

    void addOutput(const Ptr<InfEngineNgraphNode>& node);

    bool isInitialized();
    void init(Target targetId);

    void forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers, bool isAsync);

    void initPlugin(std::shared_ptr<ov::Model>& net);
    ov::ParameterVector setInputs(const std::vector<cv::Mat>& inputs, const std::vector<std::string>& names);

    void addBlobs(const std::vector<cv::Ptr<BackendWrapper> >& ptrs);

    void createNet(Target targetId);

    void reset();

//private:
    detail::NetImplBase& netImpl_;

    ov::ParameterVector inputs_vec;
    std::shared_ptr<ov::Model> ngraph_function;

    ov::CompiledModel netExec;
    std::map<std::string, ov::Tensor> allBlobs;
    std::string device_name;
    bool isInit = false;

    struct NgraphReqWrapper
    {
        NgraphReqWrapper() : isReady(true) {}

        void makePromises(const std::vector<Ptr<BackendWrapper> >& outs);

        ov::InferRequest req;
        std::vector<cv::AsyncPromise> outProms;
        std::vector<std::string> outsNames;
        bool isReady;
    };
    std::vector<Ptr<NgraphReqWrapper> > infRequests;

    std::shared_ptr<ov::Model> cnn;
    bool hasNetOwner;
    std::unordered_map<std::string, InfEngineNgraphNode*> requestedOutputs;
};

class InfEngineNgraphNode : public BackendNode
{
public:
    InfEngineNgraphNode(const std::vector<Ptr<BackendNode> >& nodes, Ptr<Layer>& layer,
                        std::vector<Mat*>& inputs, std::vector<Mat>& outputs,
                        std::vector<Mat>& internals);

    InfEngineNgraphNode(ov::Output<ov::Node>&& _node);
    InfEngineNgraphNode(const ov::Output<ov::Node>& _node);

    void setName(const std::string& name);

    // Inference Engine network object that allows to obtain the outputs of this layer.
    ov::Output<ov::Node> node;
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
    std::string name;
    ov::Tensor blob;
    AsyncArray futureMat;
};

// This is a fake class to run networks from Model Optimizer. Objects of that
// class simulate responses of layers are imported by OpenCV and supported by
// Inference Engine. The main difference is that they do not perform forward pass.
class NgraphBackendLayer : public Layer
{
public:
    NgraphBackendLayer(const std::shared_ptr<ov::Model> &t_net_) : t_net(t_net_) {};

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE;

    virtual void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs,
                         OutputArrayOfArrays internals) CV_OVERRIDE;

    virtual bool supportBackend(int backendId) CV_OVERRIDE;

private:
    std::shared_ptr<ov::Model> t_net;
};

ov::Output<ov::Node> ngraphQuantize(ov::Output<ov::Node> input, float output_sc, float output_zp);
ov::Output<ov::Node> ngraphDequantize(ov::Output<ov::Node> input, float input_sc, float input_zp);

#endif  // HAVE_DNN_NGRAPH

}}  // namespace cv::dnn


#endif  // __OPENCV_DNN_IE_NGRAPH_HPP__
