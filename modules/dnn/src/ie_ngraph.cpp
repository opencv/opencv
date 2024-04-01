// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

#include <fstream>

#include "ie_ngraph.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_DNN_NGRAPH
#include <openvino/core/extension.hpp>
#endif  // HAVE_DNN_NGRAPH

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/core/utils/filesystem.private.hpp"

namespace cv { namespace dnn {

#ifdef HAVE_DNN_NGRAPH

static bool DNN_IE_SERIALIZE = utils::getConfigurationParameterBool("OPENCV_DNN_IE_SERIALIZE", false);

// For networks with input layer which has an empty name, IE generates a name id[some_number].
// OpenCV lets users use an empty input name and to prevent unexpected naming,
// we can use some predefined name.
static std::string kDefaultInpLayerName = "opencv_ngraph_empty_inp_layer_name";
static constexpr const char* kOpenCVLayersType = "opencv_ngraph_layer";

static std::vector<Ptr<NgraphBackendWrapper> >
ngraphWrappers(const std::vector<Ptr<BackendWrapper> >& ptrs)
{
    std::vector<Ptr<NgraphBackendWrapper> > wrappers(ptrs.size());
    for (int i = 0; i < ptrs.size(); ++i)
    {
        CV_Assert(!ptrs[i].empty());
        wrappers[i] = ptrs[i].dynamicCast<NgraphBackendWrapper>();
        CV_Assert(!wrappers[i].empty());
    }
    return wrappers;
}

class NgraphCustomOp: public ov::op::Op {
public:
    OPENVINO_OP(kOpenCVLayersType);

    NgraphCustomOp(const ov::OutputVector& inputs, Ptr<Layer>& cvLayer, const std::vector<Mat>& outputs, const std::vector<Mat>& internals):
        Op(inputs), cvLayer(cvLayer), outputs(outputs), internals(internals)
    {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override
    {
        set_output_size(outputs.size());
        for (int i = 0; i < outputs.size(); ++i)
        {
            ov::PartialShape shape;
            for (int j = 0; j < outputs[i].dims; ++j) {
                shape.push_back(outputs[i].size[j]);
            }
            set_output_type(i, get_input_element_type(0), shape);
        }
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override
    {
        return std::make_shared<NgraphCustomOp>(new_args, cvLayer, outputs, internals);
    }

    bool has_evaluate() const {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        std::vector<Mat> inpMats, outMats;
        infEngineBlobsToMats(inputs, inpMats);
        infEngineBlobsToMats(outputs, outMats);
        try
        {
            cvLayer->forward(inpMats, outMats, internals);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    Ptr<Layer>& cvLayer;
    std::vector<Mat> outputs, internals;
};

InfEngineNgraphNode::InfEngineNgraphNode(ov::Output<ov::Node>&& _node)
    : BackendNode(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH), node(std::move(_node)) {
    CV_Assert(node.get_node());
    CV_Assert(node.get_node_shared_ptr());
}

InfEngineNgraphNode::InfEngineNgraphNode(const ov::Output<ov::Node>& _node)
    : BackendNode(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH), node(_node) {
    CV_Assert(node.get_node());
    CV_Assert(node.get_node_shared_ptr());
}

InfEngineNgraphNode::InfEngineNgraphNode(const std::vector<Ptr<BackendNode> >& nodes,
                                         Ptr<Layer>& cvLayer_, std::vector<Mat*>& inputs,
                                         std::vector<Mat>& outputs, std::vector<Mat>& internals)
    : BackendNode(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH), cvLayer(cvLayer_)
{
    ov::OutputVector inp_nodes;
    for (const auto& node : nodes)
        inp_nodes.emplace_back(node.dynamicCast<InfEngineNgraphNode>()->node);

    node = std::make_shared<NgraphCustomOp>(inp_nodes, cvLayer, outputs, internals);
    CV_Assert(!cvLayer->name.empty());
    setName(cvLayer->name);
}

void InfEngineNgraphNode::setName(const std::string& name) {
    node.get_node()->set_friendly_name(name);
}

InfEngineNgraphNet::InfEngineNgraphNet(detail::NetImplBase& netImpl)
    : netImpl_(netImpl)
{
    hasNetOwner = false;
    device_name = "CPU";
}

InfEngineNgraphNet::InfEngineNgraphNet(detail::NetImplBase& netImpl, std::shared_ptr<ov::Model>& net)
    : netImpl_(netImpl)
    , cnn(net)
{
    hasNetOwner = true;
    device_name = "CPU";
}

void InfEngineNgraphNet::addOutput(const Ptr<InfEngineNgraphNode>& node)
{
    CV_Assert(node);
    const std::string& name = node->node.get_node()->get_friendly_name();
    requestedOutputs.insert({name, node.get()});
}

void InfEngineNgraphNet::createNet(Target targetId) {
    if (!hasNetOwner)
    {
        CV_Assert(!requestedOutputs.empty());
        ov::ResultVector outs;

        for (auto output_node_it = requestedOutputs.begin(); output_node_it != requestedOutputs.end(); ++output_node_it)
        {
            CV_LOG_DEBUG(NULL, "DNN/NGRAPH: Add 'Result' output: " << output_node_it->first);
            CV_Assert(output_node_it->second);
            auto out = std::make_shared<ov::op::v0::Result>(output_node_it->second->node);
            out->set_friendly_name(output_node_it->first + (output_node_it->second->node.get_node()->get_output_size() == 1 ? "" : ".0"));
            outs.push_back(out);
        }
        CV_Assert_N(!inputs_vec.empty(), !outs.empty());
        ngraph_function = std::make_shared<ov::Model>(outs, inputs_vec);
        init(targetId);
    }
}

void InfEngineNgraphNet::init(Target targetId)
{
    if (!hasNetOwner)
    {
        if (targetId == DNN_TARGET_OPENCL_FP16)
        {
            ov::pass::ConvertFP32ToFP16().run_on_model(ngraph_function);
        }
        cnn = ngraph_function;

        if (DNN_IE_SERIALIZE)
        {
#ifndef OPENCV_DNN_DISABLE_NETWORK_AUTO_DUMP
            std::string dumpFileNameBase = netImpl_.getDumpFileNameBase();
            try
            {
                ov::pass::Serialize(dumpFileNameBase + "_ngraph.xml", dumpFileNameBase + "_ngraph.bin").run_on_model(cnn);
            }
            catch (const std::exception& e)
            {
                std::ofstream out((dumpFileNameBase + "_ngraph.error").c_str(), std::ios::out);
                out << "Exception: " << e.what() << std::endl;
            }
            catch (...)
            {
                std::ofstream out((dumpFileNameBase + "_ngraph.error").c_str(), std::ios::out);
                out << "Can't dump: unknown exception" << std::endl;
            }
#endif
        }
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

    ov::preprocess::PrePostProcessor ppp(cnn);
    int i = 0;
    for (const auto& inp : cnn->inputs()) {  // TODO: not sure why but ngraph_function->inputs() here causes segfault.
        const std::string& name = inp.get_node()->get_friendly_name();
        auto blobIt = allBlobs.find(name);
        CV_Assert(blobIt != allBlobs.end());

        auto srcT = blobIt->second.get_element_type();
        if (srcT != inp.get_node()->get_element_type()) {
            ppp.input(i++).tensor().set_element_type(srcT);
        }
    }

    i = 0;
    for (const auto& it : cnn->outputs())
    {
        const std::string& name = it.get_node()->get_friendly_name();
        auto blobIt = allBlobs.find(name);
        CV_Assert(blobIt != allBlobs.end());
        const auto& src = blobIt->second;

        // A workaround for single dimension output for which OpenCV allocates 2d Mat.
        // For example, face-detection-0105 with Result of shape {200} while output blob is {200, 1}
        auto outShape = it.get_partial_shape().get_max_shape();
        if (outShape != src.get_shape()) {
            size_t sz = std::accumulate(outShape.begin(), outShape.end(), 1, std::multiplies<size_t>());
            CV_Assert(sz == src.get_size());
            allBlobs[name] = ov::Tensor(src.get_element_type(), outShape, src.data());
        }

        ppp.output(i++).tensor().set_element_type(src.get_element_type());
    }

    ppp.build();

    initPlugin(cnn);
}

ov::ParameterVector InfEngineNgraphNet::setInputs(const std::vector<cv::Mat>& inputs,
                                   const std::vector<std::string>& names) {
    CV_Assert_N(inputs.size() == names.size());
    ov::ParameterVector current_inp;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        std::vector<size_t> shape = getShape<size_t>(inputs[i]);
        auto inp = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(shape));
        inp->set_friendly_name(names[i]);

        auto it = std::find_if(inputs_vec.begin(), inputs_vec.end(),
                                [&inp](const std::shared_ptr<ov::op::v0::Parameter>& a) {
                                return a->get_friendly_name() == inp->get_friendly_name();
                  });
        if (it == inputs_vec.end()) {
            inputs_vec.push_back(inp);
            current_inp.push_back(inp);
        } else {
            current_inp.push_back(*it);
        }
    }
    return current_inp;
}


void InfEngineNgraphNet::initPlugin(std::shared_ptr<ov::Model>& net)
{
    CV_Assert(!isInitialized());

    try
    {
        AutoLock lock(getInitializationMutex());
        ov::Core& ie = getCore(device_name);
        {
            isInit = true;
            std::vector<std::string> candidates;
            std::string param_pluginPath = utils::getConfigurationParameterString("OPENCV_DNN_IE_EXTRA_PLUGIN_PATH", "");
            if (!param_pluginPath.empty())
            {
                candidates.push_back(param_pluginPath);
            }
            bool found = false;
            for (size_t i = 0; i != candidates.size(); ++i)
            {
                const std::string& libName = candidates[i];
                try
                {
                    ie.add_extension(libName);
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
#ifndef _WIN32
            // Limit the number of CPU threads.
            if (device_name == "CPU")
                ie.set_property(device_name, ov::inference_num_threads(getNumThreads()));
#endif
            if (device_name.find("GPU") == 0)
            {
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
                std::string cache_path = utils::fs::getCacheDirectory((std::string("dnn_ie_cache_") + device_name).c_str(), "OPENCV_DNN_IE_GPU_CACHE_DIR");
#else
                std::string cache_path = utils::getConfigurationParameterString("OPENCV_DNN_IE_GPU_CACHE_DIR", "");
#endif
                if (!cache_path.empty() && cache_path != "disabled")
                {
                    CV_LOG_INFO(NULL, "OpenCV/nGraph: using GPU kernels cache: " << cache_path);
                    ie.set_property(device_name, ov::cache_dir(cache_path));
                }
            }
        }
        ov::AnyMap config;
        if (device_name == "MYRIAD" || device_name == "HDDL") {
            config.emplace("MYRIAD_DETECT_NETWORK_BATCH", "NO");
        }

        bool isHetero = device_name == "FPGA";
        // It is actual only for non-CPU targets and networks built in runtime using nGraph.
        // We do not check IR models because they can be with version less than IRv10
        if (!isHetero && device_name != "CPU" && !hasNetOwner)
        {
            for (auto& node : net->get_ops())
            {
                if (node->description() == kOpenCVLayersType)
                {
                    isHetero = true;
                    break;
                }
            }
        }

        std::string ieDevice = isHetero ? ("HETERO:" + device_name + ",CPU") : device_name;
        CV_LOG_INFO(NULL, "DNN/IE: Calling LoadNetwork(device=" << ieDevice << ")...");
        netExec = ie.compile_model(net, ieDevice, config);
    }
    catch (const std::exception& ex)
    {
        CV_Error(Error::StsError, format("Failed to initialize Inference Engine backend (device = %s): %s", device_name.c_str(), ex.what()));
    }
}

bool InfEngineNgraphNet::isInitialized()
{
    return isInit;
}

bool NgraphBackendLayer::getMemoryShapes(const std::vector<MatShape> &inputs,
                                            const int requiredOutputs,
                                            std::vector<MatShape> &outputs,
                                            std::vector<MatShape> &internals) const
{
    bool equal_flag = true;
    std::map<std::string, ov::PartialShape> inShapes;
    int i = 0;
    for (const auto& inp : t_net->get_parameters())
    {
        ov::Shape oldShape = inp->get_shape();
        ov::Shape newShape(inputs[i].begin(), inputs[i].end());
        inShapes.insert({inp->get_friendly_name(), newShape});
        if (oldShape != newShape)
        {
            equal_flag = false;
        }
        i++;
    }

    if (!equal_flag)
    {
        std::shared_ptr<ov::Model> curr_t_net(t_net);
        curr_t_net->reshape(inShapes);
    }
    std::vector<size_t> dims;
    for (const auto& it : t_net->outputs()) {
        if (it.get_node()->get_friendly_name() == name) {
            dims = it.get_partial_shape().get_max_shape();
        }
    }
    if (dims.empty())
        CV_Error(Error::StsError, format("Unable find result with name %s", name.c_str()));
    outputs.push_back(MatShape(dims.begin(), dims.end()));
    return false;
}

bool NgraphBackendLayer::supportBackend(int backendId)
{
    CV_LOG_DEBUG(NULL, "NgraphBackendLayer::supportBackend(" << backendId << ")");
    return backendId == DNN_BACKEND_DEFAULT ||
           (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH);
}

void NgraphBackendLayer::forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs,
                                    OutputArrayOfArrays internals)
{
    CV_Error(Error::StsInternal, "Choose Inference Engine as a preferable backend.");
}

ov::Tensor wrapToNgraphBlob(const Mat& m) {
    std::vector<size_t> shape = getShape<size_t>(m);
    if (m.type() == CV_32F)
        return ov::Tensor(ov::element::f32, shape, m.data);
    else if (m.type() == CV_8U)
        return ov::Tensor(ov::element::u8, shape, m.data);
    else if (m.type() == CV_8SC1)
        return ov::Tensor(ov::element::i8, shape, m.data);
    else if (m.type() == CV_32SC1)
        return ov::Tensor(ov::element::i32, shape, m.data);
    else
        CV_Error(Error::StsNotImplemented, format("Unsupported data type %s", typeToString(m.type()).c_str()));
}


NgraphBackendWrapper::NgraphBackendWrapper(int targetId, const cv::Mat& m)
    : BackendWrapper(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, targetId)
    , host((Mat*)&m)
{
    blob = wrapToNgraphBlob(m);
}

NgraphBackendWrapper::NgraphBackendWrapper(Ptr<BackendWrapper> wrapper)
    : BackendWrapper(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, wrapper->targetId)
{
    Ptr<NgraphBackendWrapper> ieWrapper = wrapper.dynamicCast<NgraphBackendWrapper>();
    CV_Assert(!ieWrapper.empty());
    name = ieWrapper->name;
    blob = ieWrapper->blob;
}

Ptr<BackendWrapper> NgraphBackendWrapper::create(Ptr<BackendWrapper> wrapper)
{
    return Ptr<BackendWrapper>(new NgraphBackendWrapper(wrapper));
}

NgraphBackendWrapper::~NgraphBackendWrapper()
{
    // nothing
}

void NgraphBackendWrapper::copyToHost()
{
    CV_LOG_DEBUG(NULL, "NgraphBackendWrapper::copyToHost()");
    //CV_Error(Error::StsNotImplemented, "");
}

void NgraphBackendWrapper::setHostDirty()
{
    CV_LOG_DEBUG(NULL, "NgraphBackendWrapper::setHostDirty()");
    //CV_Error(Error::StsNotImplemented, "");
}

ov::Tensor copyBlob(const ov::Tensor& blob)
{
    return ov::Tensor(blob.get_element_type(), blob.get_shape());
}

void InfEngineNgraphNet::reset()
{
    allBlobs.clear();
    infRequests.clear();
    isInit = false;
}

void InfEngineNgraphNet::addBlobs(const std::vector<cv::Ptr<BackendWrapper> >& ptrs)
{
    auto wrappers = ngraphWrappers(ptrs);
    for (const auto& wrapper : wrappers)
    {
        std::string name = wrapper->name;
        name = name.empty() ? kDefaultInpLayerName : name;
        allBlobs.insert({name, wrapper->blob});
    }
}

void InfEngineNgraphNet::NgraphReqWrapper::makePromises(const std::vector<Ptr<BackendWrapper> >& outsWrappers)
{
    auto outs = ngraphWrappers(outsWrappers);
    outProms.clear();
    outProms.resize(outs.size());
    outsNames.resize(outs.size());
    for (int i = 0; i < outs.size(); ++i)
    {
        outs[i]->futureMat = outProms[i].getArrayResult();
        outsNames[i] = outs[i]->name;
    }
}

void InfEngineNgraphNet::forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers, bool isAsync)
{
    CV_LOG_DEBUG(NULL, "InfEngineNgraphNet::forward(" << (isAsync ? "async" : "sync") << ")");

    // Look for finished requests.
    Ptr<NgraphReqWrapper> reqWrapper;
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
        reqWrapper = Ptr<NgraphReqWrapper>(new NgraphReqWrapper());
        try
        {
            reqWrapper->req = netExec.create_infer_request();
        }
        catch (const std::exception& ex)
        {
            CV_Error(Error::StsAssert, format("Failed to initialize Inference Engine backend: %s", ex.what()));
        }
        infRequests.push_back(reqWrapper);

        int i = 0;
        for (const auto& it : netExec.inputs())
        {
            const std::string& name = it.get_node()->get_friendly_name();
            auto blobIt = allBlobs.find(name);
            CV_Assert(blobIt != allBlobs.end());
            reqWrapper->req.set_input_tensor(i++, isAsync ? copyBlob(blobIt->second) : blobIt->second);
        }

        i = 0;
        for (const auto& it : netExec.outputs())
        {
            const std::string& name = it.get_node()->get_friendly_name();
            auto blobIt = allBlobs.find(name);
            CV_Assert(blobIt != allBlobs.end());
            reqWrapper->req.set_output_tensor(i++, isAsync ? copyBlob(blobIt->second) : blobIt->second);
        }

    if (isAsync) {
        bool* isReady = &reqWrapper->isReady;
        auto* promises = &reqWrapper->outProms;
        auto* req = &reqWrapper->req;
        reqWrapper->req.set_callback([isReady, promises, req](std::exception_ptr ex) {
            CV_LOG_DEBUG(NULL, "DNN(nGraph): completionCallback()");

            size_t processedOutputs = 0;
            try
            {
                for (; processedOutputs < promises->size(); ++processedOutputs)
                {
                    Mat m = infEngineBlobToMat(req->get_output_tensor(processedOutputs));

                    try
                    {
                        (*promises)[processedOutputs].setValue(m.clone());
                    }
                    catch (...)
                    {
                        try {
                            (*promises)[processedOutputs].setException(std::current_exception());
                        } catch(...) {
                            CV_LOG_ERROR(NULL, "DNN: Exception occurred during async inference exception propagation");
                        }
                    }
                }
            }
            catch (...)
            {
                std::exception_ptr e = std::current_exception();
                for (; processedOutputs < promises->size(); ++processedOutputs)
                {
                    try {
                        (*promises)[processedOutputs].setException(e);
                    } catch(...) {
                        CV_LOG_ERROR(NULL, "DNN: Exception occurred during async inference exception propagation");
                    }
                }
            }
            *isReady = true;
        });
    }
    }

    if (isAsync)
    {
        // Copy actual data to infer request's input blobs.
        int i = 0;
        for (const auto& it : cnn->get_parameters())
        {
            const std::string& name = it->get_friendly_name();
            auto blobIt = allBlobs.find(name);
            Mat srcMat = infEngineBlobToMat(blobIt->second);
            Mat dstMat = infEngineBlobToMat(reqWrapper->req.get_input_tensor(i++));
            srcMat.copyTo(dstMat);
        }

        // Set promises to output blobs wrappers.
        reqWrapper->makePromises(outBlobsWrappers);

        reqWrapper->isReady = false;
        reqWrapper->req.start_async();
    }
    else
    {
        reqWrapper->req.infer();
    }
}

ov::Output<ov::Node> ngraphQuantize(ov::Output<ov::Node> input, float output_sc, float output_zp) {
    float outLow = -128, outHigh = 127;
    float inpLow = output_sc * (outLow - output_zp);
    float inpHigh = output_sc * (outHigh - output_zp);
    return std::make_shared<ov::op::v0::FakeQuantize>(input,
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &inpLow),
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &inpHigh),
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &outLow),
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &outHigh),
        256 // levels
    );
}

ov::Output<ov::Node> ngraphDequantize(ov::Output<ov::Node> input, float input_sc, float input_zp) {
    float inpLow = -128, inpHigh = 127;
    float outLow = input_sc * (inpLow - input_zp);
    float outHigh = input_sc * (inpHigh - input_zp);
    return std::make_shared<ov::op::v0::FakeQuantize>(input,
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &inpLow),
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &inpHigh),
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &outLow),
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &outHigh),
        256 // levels
    );
}

#endif

}}
