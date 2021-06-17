// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_OP_WEBNN_HPP__
#define __OPENCV_DNN_OP_WEBNN_HPP__

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/dnn.hpp"

#ifdef HAVE_WEBNN

#include <webnn/webnn_cpp.h>
#include <webnn/webnn.h>
#include <webnn/webnn_proc.h>
#include <webnn_native/WebnnNative.h>

#include <unordered_map>
#include <unordered_set>

#endif  // HAVE_WEBNN

namespace cv { namespace dnn {

constexpr bool haveWebnn() {
#ifdef HAVE_WEBNN
        return true;
#else
        return false;
#endif
}

#ifdef HAVE_WEBNN

class WebnnBackendNode;
class WebnnBackendWrapper;



class WebnnNet
{
public:
    WebnnNet();

    void addOutput(const std::string& name);

    bool isInitialized();
    void init(Target targetId);

    void forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers, bool isAsync);

    std::vector<ml::Operand> setInputs(const std::vector<cv::Mat>& inputs, const std::vector<std::string>& names);

    void setUnconnectedNodes(Ptr<WebnnBackendNode>& node);
    void addBlobs(const std::vector<cv::Ptr<BackendWrapper> >& ptrs);

    void createNet(Target targetId);
    // void setNodePtr(std::shared_ptr<ngraph::Node>* ptr);

    void reset();

    ml::GraphBuilder builder;
    ml::Context context;
    ml::Graph graph;

    std::unordered_map<std::string, cv::Ptr<WebnnBackendWrapper>> allBlobs;

    bool hasNetOwner;
    std::string device_name;
    bool isInit = false;

    std::vector<std::string> requestedOutputs;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    ml::NamedOperands namedOperands;
};

class WebnnBackendNode : public BackendNode
{
public:
    WebnnBackendNode(ml::Operand&& operand);
    WebnnBackendNode(ml::Operand& operand);

    std::string name;
    ml::Operand operand;
    Ptr<WebnnNet> net;
};

class WebnnBackendWrapper : public BackendWrapper
{
public:
    WebnnBackendWrapper(int targetId, Mat& m);
    ~WebnnBackendWrapper();

    virtual void copyToHost() CV_OVERRIDE;
    virtual void setHostDirty() CV_OVERRIDE;

    std::string name;
    Mat* host;
    std::unique_ptr<char> buffer;
    size_t size;
    std::vector<int32_t> dimensions;
    ml::OperandDescriptor descriptor;
};

#endif  // HAVE_WebNN

void forwardWebnn(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                   Ptr<BackendNode>& node, bool isAsync);

}}  // namespace cv::dnn


#endif  // __OPENCV_DNN_OP_WEBNN_HPP__
