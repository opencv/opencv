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
#include <ie_extension.h>
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

class NgraphCustomOp: public ngraph::op::Op {
public:
    const ngraph::NodeTypeInfo& get_type_info() const override
    {
        static constexpr ngraph::NodeTypeInfo type_info{kOpenCVLayersType, 0};
        return type_info;
    }

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2020_3)
    NgraphCustomOp(const ngraph::OutputVector& inputs,
#else
    NgraphCustomOp(const ngraph::NodeVector& inputs,
#endif
                   const std::map<std::string, InferenceEngine::Parameter>& params = {}):
        Op(inputs), params(params)
    {
        constructor_validate_and_infer_types();
    }

    ~NgraphCustomOp()
    {
        // nothing
    }

    void validate_and_infer_types() override
    {
        std::vector<std::vector<size_t> > shapes;
        strToShapes(params["outputs"], shapes);
        set_output_size(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i)
        {
            ngraph::Shape output_shape(shapes[i]);
            set_output_type(i, get_input_element_type(0), output_shape);
        }
    }

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2020_4)
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override
    {
        return std::make_shared<NgraphCustomOp>(new_args, params);
    }
#else
    std::shared_ptr<ngraph::Node> copy_with_new_args(const ngraph::NodeVector& new_args) const override
    {
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2020_3)
        return std::make_shared<NgraphCustomOp>(ngraph::as_output_vector(new_args), params);
#else
        return std::make_shared<NgraphCustomOp>(new_args, params);
#endif
    }
#endif

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override
    {
        for (auto& attr : params)
        {
            if (attr.second.is<std::string>())
                visitor.on_attribute(attr.first, attr.second.as<std::string>());
        }
        return true;
    }

    std::map<std::string, InferenceEngine::Parameter> params;
};


class InfEngineNgraphCustomLayer : public InferenceEngine::ILayerExecImpl
{
public:
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2020_2)
    explicit InfEngineNgraphCustomLayer(const std::shared_ptr<ngraph::Node>& _node)
    {
        node = std::dynamic_pointer_cast<NgraphCustomOp>(_node);
        CV_Assert(node);
        std::string implStr = node->params["impl"];
        std::istringstream iss(implStr);
#else
    explicit InfEngineNgraphCustomLayer(const InferenceEngine::CNNLayer& layer) : cnnLayer(layer)
    {
        std::istringstream iss(layer.GetParamAsString("impl"));
#endif
        size_t ptr;
        iss >> ptr;
        cvLayer = (Layer*)ptr;

        std::vector<std::vector<size_t> > shapes;
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2020_2)
        strToShapes(node->params["internals"], shapes);
#else
        strToShapes(layer.GetParamAsString("internals"), shapes);
#endif
        internals.resize(shapes.size());
        for (int i = 0; i < shapes.size(); ++i)
            internals[i].create(std::vector<int>(shapes[i].begin(), shapes[i].end()), CV_32F);
    }

    ~InfEngineNgraphCustomLayer()
    {
        // nothing
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
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2020_2)
        InferenceEngine::SizeVector order;
        size_t offset = std::numeric_limits<size_t>::max();
        for (int i = 0; i < node->get_input_size(); ++i)
        {
            InferenceEngine::DataConfig conf;
            auto shape = node->input_value(i).get_shape();
            order.resize(shape.size());
            std::iota(order.begin(), order.end(), 0);
            conf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, shape, {shape, order, offset});
            inDataConfig.push_back(conf);
        }

        for (int i = 0; i < node->get_output_size(); ++i)
        {
            InferenceEngine::DataConfig conf;
            auto shape = node->output(i).get_shape();
            order.resize(shape.size());
            std::iota(order.begin(), order.end(), 0);
            conf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, shape, {shape, order, offset});
            outDataConfig.push_back(conf);
        }
#else
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
#endif

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
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2020_2)
    std::shared_ptr<NgraphCustomOp> node;
#else
    InferenceEngine::CNNLayer cnnLayer;
#endif
    dnn::Layer* cvLayer;
    std::vector<Mat> internals;
};

#if INF_ENGINE_VER_MAJOR_LT(INF_ENGINE_RELEASE_2020_2)
class InfEngineNgraphCustomLayerFactory : public InferenceEngine::ILayerImplFactory {
public:
    explicit InfEngineNgraphCustomLayerFactory(const InferenceEngine::CNNLayer* layer) : cnnLayer(*layer)
    {
        // nothing
    }

    InferenceEngine::StatusCode
    getImplementations(std::vector<InferenceEngine::ILayerImpl::Ptr>& impls,
                       InferenceEngine::ResponseDesc* resp) noexcept override
    {
        impls.push_back(std::make_shared<InfEngineNgraphCustomLayer>(cnnLayer));
        return InferenceEngine::StatusCode::OK;
    }

private:
    InferenceEngine::CNNLayer cnnLayer;
};
#endif


class InfEngineNgraphExtension : public InferenceEngine::IExtension
{
public:
    void Unload() noexcept override {}
    void Release() noexcept override { delete this; }
    void GetVersion(const InferenceEngine::Version*&) const noexcept override {}

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2020_2)
    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override {
        return {"CPU"};
    }

    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override {
        if (std::dynamic_pointer_cast<NgraphCustomOp>(node) && implType == "CPU") {
            return std::make_shared<InfEngineNgraphCustomLayer>(node);
        }
        return nullptr;
    }
#else
    virtual void SetLogCallback(InferenceEngine::IErrorListener&) noexcept {}

    virtual InferenceEngine::StatusCode getPrimitiveTypes(char**&, unsigned int&,
                                                          InferenceEngine::ResponseDesc*) noexcept
    {
        return InferenceEngine::StatusCode::OK;
    }

    InferenceEngine::StatusCode getFactoryFor(InferenceEngine::ILayerImplFactory*& factory,
                                              const InferenceEngine::CNNLayer* cnnLayer,
                                              InferenceEngine::ResponseDesc* resp) noexcept
    {
        if (cnnLayer->type != kOpenCVLayersType)
            return InferenceEngine::StatusCode::NOT_IMPLEMENTED;
        factory = new InfEngineNgraphCustomLayerFactory(cnnLayer);
        return InferenceEngine::StatusCode::OK;
    }
#endif
};



InfEngineNgraphNode::InfEngineNgraphNode(std::shared_ptr<ngraph::Node>&& _node)
    : BackendNode(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH), node(std::move(_node)) {}

InfEngineNgraphNode::InfEngineNgraphNode(const std::shared_ptr<ngraph::Node>& _node)
    : BackendNode(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH), node(_node) {}

InfEngineNgraphNode::InfEngineNgraphNode(const std::vector<Ptr<BackendNode> >& nodes,
                                         Ptr<Layer>& cvLayer_, std::vector<Mat*>& inputs,
                                         std::vector<Mat>& outputs, std::vector<Mat>& internals)
    : BackendNode(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH), cvLayer(cvLayer_)
{
    std::ostringstream oss;
    oss << (size_t)cvLayer.get();

    std::map<std::string, InferenceEngine::Parameter> params = {
        {"impl", oss.str()},
        {"outputs", shapesToStr(outputs)},
        {"internals", shapesToStr(internals)}
    };

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2020_3)
    ngraph::OutputVector inp_nodes;
#else
    ngraph::NodeVector inp_nodes;
#endif
    for (const auto& node : nodes)
        inp_nodes.emplace_back(node.dynamicCast<InfEngineNgraphNode>()->node);
    node = std::make_shared<NgraphCustomOp>(inp_nodes, params);

    CV_Assert(!cvLayer->name.empty());
    setName(cvLayer->name);
}

void InfEngineNgraphNode::setName(const std::string& name) {
    node->set_friendly_name(name);
}

InfEngineNgraphNet::InfEngineNgraphNet(detail::NetImplBase& netImpl)
    : netImpl_(netImpl)
{
    hasNetOwner = false;
    device_name = "CPU";
}

InfEngineNgraphNet::InfEngineNgraphNet(detail::NetImplBase& netImpl, InferenceEngine::CNNNetwork& net)
    : netImpl_(netImpl)
    , cnn(net)
{
    hasNetOwner = true;
    device_name = "CPU";
}

void InfEngineNgraphNet::addOutput(const Ptr<InfEngineNgraphNode>& node)
{
    CV_Assert(node);
    CV_Assert(node->node);
    const std::string& name = node->node->get_friendly_name();
    requestedOutputs.insert({name, node});
}

void InfEngineNgraphNet::setNodePtr(std::shared_ptr<ngraph::Node>* ptr) {
    all_nodes.emplace((*ptr)->get_friendly_name(), ptr);
}

 void InfEngineNgraphNet::release()
 {
     // FIXIT release should not be conditional, release ALL
     for (auto& node : components.back()) {
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_4)
         if (!(ngraph::op::is_parameter(node) || ngraph::op::is_output(node) || ngraph::op::is_constant(node)) ) {
#else
         if (!(node->is_parameter() || node->is_output() || node->is_constant()) ) {
#endif
             auto it = all_nodes.find(node->get_friendly_name());
             if (it != all_nodes.end()) {
                 it->second->reset();
                 all_nodes.erase(it);
             }
         }
     }
 }

void InfEngineNgraphNet::dfs(std::shared_ptr<ngraph::Node>& node,
                             std::vector<std::shared_ptr<ngraph::Node>>& comp,
                             std::unordered_map<std::string, bool>& used) {
    used[node->get_friendly_name()] = true;
    comp.push_back(node);
    auto inputs = node->get_users();
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        inputs.push_back(node->input_value(i).get_node()->shared_from_this());
    }

    for (auto& to : inputs) {
        if (!used[to->get_friendly_name()]) {
            dfs(to, comp, used);
        }
    }
}

int InfEngineNgraphNet::getNumComponents()
{
    if (!components.empty()) {
        return components.size();
    }
    std::unordered_map<std::string, bool> used;
    auto inputs = ngraph_function->get_ordered_ops();
    for (auto& node : inputs) {
        used.emplace(node->get_friendly_name(), false);
    }

    for (auto& node : inputs) {
        if (!used[node->get_friendly_name()]) {
            std::vector<std::shared_ptr<ngraph::Node>> current_comp;
            dfs(node, current_comp, used);
            components.push_back(current_comp);
        }
    }
    return components.size();
}

void InfEngineNgraphNet::createNet(Target targetId) {
    if (!hasNetOwner)
    {
        CV_Assert(!requestedOutputs.empty());
        ngraph::ResultVector outs;

        for (auto output_node_it = requestedOutputs.begin(); output_node_it != requestedOutputs.end(); ++output_node_it)
        {
            CV_LOG_DEBUG(NULL, "DNN/NGRAPH: Add 'Result' output: " << output_node_it->first);
            CV_Assert(output_node_it->second);
            auto out = std::make_shared<ngraph::op::Result>(output_node_it->second->node);
            outs.push_back(out);
        }
        CV_Assert_N(!inputs_vec.empty(), !outs.empty());
        ngraph_function = std::make_shared<ngraph::Function>(outs, inputs_vec);

        int num_comp = getNumComponents();
        CV_LOG_DEBUG(NULL, "DNN/IE: number of subgraphs: " << num_comp);
        if (num_comp > 1) {
            for (int i = num_comp - 1; i >= 0; --i) {
                ngraph::ResultVector outputs;
                ngraph::ParameterVector inps;
                for (auto& node : components.back()) {
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_4)
                    if (ngraph::op::is_parameter(node)) {
#else
                    if (node->is_parameter()) {
#endif
                        CV_LOG_DEBUG(NULL, "DNN/IE: subgraph[" << i << "]: +input[" << inps.size() << "] = '" << node->get_friendly_name() << "'");
                        auto parameter = std::dynamic_pointer_cast<ngraph::op::Parameter>(node);
                        inps.push_back(parameter);
                    }
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_4)
                    else if (ngraph::op::is_output(node)) {
#else
                    else if (node->is_output()) {
#endif
                        CV_LOG_DEBUG(NULL, "DNN/IE: subgraph[" << i << "]: +output[" << outputs.size() << "] = '" << node->get_friendly_name() << "'");
                        auto result = std::dynamic_pointer_cast<ngraph::op::Result>(node);
                        outputs.push_back(result);
                    }
                }
                CV_LOG_DEBUG(NULL, "DNN/IE: subgraph[" << i << ": nodes=" << components.back().size() << " inputs=" << inps.size() << " outputs=" << outputs.size());
                isInit = false;
                CV_Assert_N(!inps.empty(), !outputs.empty());
                ngraph_function = std::make_shared<ngraph::Function>(outputs, inps);
                release();
                components.pop_back();
                init(targetId);
            }
        } else {
            release();
            components.clear();
            init(targetId);
        }
    }
}

void InfEngineNgraphNet::init(Target targetId)
{
    if (!hasNetOwner)
    {
        if (targetId == DNN_TARGET_OPENCL_FP16)
        {
            auto nodes = ngraph_function->get_ordered_ops();
            for (auto& node : nodes)
            {
                auto parameter = std::dynamic_pointer_cast<ngraph::op::Parameter>(node);
                if (parameter && parameter->get_element_type() == ngraph::element::f32)
                {
                    parameter->set_element_type(ngraph::element::f16);
                }
                auto constant = std::dynamic_pointer_cast<ngraph::op::Constant>(node);
                if (constant && constant->get_element_type() == ngraph::element::f32)
                {
                    const float* floatsData = constant->get_data_ptr<float>();
                    size_t total = ngraph::shape_size(constant->get_shape());
                    Mat floats(1, total, CV_32F, (void*)floatsData);
                    Mat halfs;
                    cv::convertFp16(floats, halfs);

                    auto new_const = std::make_shared<ngraph::op::Constant>(ngraph::element::f16, constant->get_shape(), halfs.data);
                    new_const->set_friendly_name(constant->get_friendly_name());
                    ngraph::replace_node(constant, new_const);
                }
            }
            ngraph_function->validate_nodes_and_infer_types();
        }
        cnn = InferenceEngine::CNNNetwork(ngraph_function);

        if (DNN_IE_SERIALIZE)
        {
#ifndef OPENCV_DNN_DISABLE_NETWORK_AUTO_DUMP
            std::string dumpFileNameBase = netImpl_.getDumpFileNameBase();
            try
            {
                cnn.serialize(dumpFileNameBase + "_ngraph.xml", dumpFileNameBase + "_ngraph.bin");
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

    if (!hasNetOwner) {
        for (size_t i = 0; i < ngraph_function->get_output_size(); ++i) {
            auto node = ngraph_function->output(i).get_node();
            for (size_t j = 0; j < node->get_input_size(); ++j) {
                std::string name = node->input_value(j).get_node()->get_friendly_name();
                auto iter = requestedOutputs.find(name);
                if (iter != requestedOutputs.end()) {
                    requestedOutputs.erase(iter);
                    cnn.addOutput(name);
                }
            }
        }
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

ngraph::ParameterVector InfEngineNgraphNet::setInputs(const std::vector<cv::Mat>& inputs,
                                   const std::vector<std::string>& names) {
    CV_Assert_N(inputs.size() == names.size());
    ngraph::ParameterVector current_inp;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        std::vector<size_t> shape = getShape<size_t>(inputs[i]);
        auto inp = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape(shape));
        inp->set_friendly_name(names[i]);

        auto it = std::find_if(inputs_vec.begin(), inputs_vec.end(),
                                [&inp](const std::shared_ptr<ngraph::op::Parameter>& a) {
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


void InfEngineNgraphNet::initPlugin(InferenceEngine::CNNNetwork& net)
{
    CV_Assert(!isInitialized());

    try
    {
        AutoLock lock(getInitializationMutex());
        InferenceEngine::Core& ie = getCore(device_name);
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
                    InferenceEngine::IExtensionPtr extension =
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2021_4)
                        std::make_shared<InferenceEngine::Extension>(libName);
#else
                        InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(libName);
#endif

                    ie.AddExtension(extension, "CPU");
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
            // OpenCV fallbacks as extensions.
            try
            {
                ie.AddExtension(std::make_shared<InfEngineNgraphExtension>(), "CPU");
            }
            catch(const std::exception& e)
            {
                CV_LOG_INFO(NULL, "DNN-IE: Can't register OpenCV custom layers nGraph extension: " << e.what());
            }
#ifndef _WIN32
            // Limit the number of CPU threads.
            if (device_name == "CPU")
                ie.SetConfig({{
                    InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, format("%d", getNumThreads()),
                }}, device_name);
#endif
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2021_2)
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
                    ie.SetConfig({{
                        InferenceEngine::PluginConfigParams::KEY_CACHE_DIR, cache_path,
                    }}, device_name);
                }
            }
#endif
        }
        std::map<std::string, std::string> config;
        if (device_name == "MYRIAD" || device_name == "HDDL") {
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_4)
            config.emplace("MYRIAD_DETECT_NETWORK_BATCH", CONFIG_VALUE(NO));
#else
            config.emplace("VPU_DETECT_NETWORK_BATCH", CONFIG_VALUE(NO));
#endif
        }

        bool isHetero = device_name == "FPGA";
        // It is actual only for non-CPU targets and networks built in runtime using nGraph.
        // We do not check IR models because they can be with version less than IRv10
        if (!isHetero && device_name != "CPU" && !hasNetOwner)
        {
            for (auto& node : net.getFunction()->get_ops())
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
        netExec = ie.LoadNetwork(net, ieDevice, config);
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


static InferenceEngine::Layout estimateLayout(int dims)
{
    if (dims == 4)
        return InferenceEngine::Layout::NCHW;
    else if (dims == 3)
        return InferenceEngine::Layout::CHW;
    else if (dims == 2)
        return InferenceEngine::Layout::NC;
    else if (dims == 1)
        return InferenceEngine::Layout::C;
    else if (dims == 5)
        return InferenceEngine::Layout::NCDHW;
    else
        return InferenceEngine::Layout::ANY;
}
static inline
InferenceEngine::Layout estimateLayout(size_t dims)
{
    return estimateLayout((int)dims);
}

static inline
InferenceEngine::Layout estimateLayout(const Mat& m)
{
    return estimateLayout(m.dims);
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
        CV_Error(Error::StsNotImplemented, format("Unsupported data type %s", typeToString(m.type()).c_str()));
}

InferenceEngine::Blob::Ptr wrapToNgraphBlob(const Mat& m, const std::vector<size_t>& shape,
                                               InferenceEngine::Layout layout)
{
    if (m.type() == CV_32F)
        return InferenceEngine::make_shared_blob<float>(
               {InferenceEngine::Precision::FP32, shape, layout}, (float*)m.data);
    else if (m.type() == CV_8U)
        return InferenceEngine::make_shared_blob<uint8_t>(
               {InferenceEngine::Precision::U8, shape, layout}, (uint8_t*)m.data);
    else
        CV_Error(Error::StsNotImplemented, format("Unsupported data type %s", typeToString(m.type()).c_str()));
}

InferenceEngine::Blob::Ptr wrapToNgraphBlob(const Mat& m, InferenceEngine::Layout layout)
{
    std::vector<size_t> shape = getShape<size_t>(m);
    return wrapToNgraphBlob(m, shape, layout);
}

NgraphBackendWrapper::NgraphBackendWrapper(int targetId, const cv::Mat& m)
    : BackendWrapper(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, targetId)
    , host((Mat*)&m)
{
    dataPtr = wrapToInfEngineDataNode(m);
    blob = wrapToNgraphBlob(m, estimateLayout(m));
}

NgraphBackendWrapper::NgraphBackendWrapper(Ptr<BackendWrapper> wrapper)
    : BackendWrapper(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, wrapper->targetId)
{
    Ptr<NgraphBackendWrapper> ieWrapper = wrapper.dynamicCast<NgraphBackendWrapper>();
    CV_Assert(!ieWrapper.empty());
    InferenceEngine::DataPtr srcData = ieWrapper->dataPtr;
    dataPtr = InferenceEngine::DataPtr(new InferenceEngine::Data(srcData->getName(), srcData->getTensorDesc()));
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

InferenceEngine::Blob::Ptr copyBlob(const InferenceEngine::Blob::Ptr& blob)
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
    {
        std::ostringstream msg;
        msg << precision;
        CV_Error_(Error::StsNotImplemented, ("Unsupported blob precision: %s", msg.str().c_str()));
    }
    copy->allocate();
    return copy;
}

InferenceEngine::DataPtr ngraphDataNode(const Ptr<BackendWrapper>& ptr)
{
    CV_Assert(!ptr.empty());
    Ptr<NgraphBackendWrapper> p = ptr.dynamicCast<NgraphBackendWrapper>();
    CV_Assert(!p.empty());
    return p->dataPtr;
}

static
InferenceEngine::Blob::Ptr reallocateBlob(Mat &m, const InferenceEngine::TensorDesc& description)
{
    auto dims = description.getDims();
    auto layout = estimateLayout(dims.size());
    MatShape matShape(dims.begin(), dims.end());
    if (description.getPrecision() == InferenceEngine::Precision::FP32)
    {
        m.create(matShape, CV_32FC1);
        return InferenceEngine::make_shared_blob<float>(
                {description.getPrecision(), dims, layout}, (float*)m.data);
    }
    else if (description.getPrecision() == InferenceEngine::Precision::I32)
    {
        m.create(matShape, CV_32SC1);
        return InferenceEngine::make_shared_blob<int>(
                {description.getPrecision(), dims, layout}, (int*)m.data);
    }
    else if (description.getPrecision() == InferenceEngine::Precision::U8)
    {
        m.create(matShape, CV_8UC1);
        return InferenceEngine::make_shared_blob<uchar>(
                {description.getPrecision(), dims, layout}, (uchar*)m.data);
    }
    std::ostringstream msg;
    msg << "Unsupported IE precision: " << description.getPrecision();
    CV_Error(Error::StsNotImplemented, msg.str());
}

InferenceEngine::DataPtr ngraphDataOutputNode(
        const Ptr<BackendWrapper>& ptr,
        const InferenceEngine::TensorDesc& description,
        const std::string name)
{
    CV_Assert(!ptr.empty());
    Ptr<NgraphBackendWrapper> p = ptr.dynamicCast<NgraphBackendWrapper>();
    CV_Assert(!p.empty());
    NgraphBackendWrapper& w = *p;
    const InferenceEngine::TensorDesc& blobDesc = w.blob.get()->getTensorDesc();
    auto dims = description.getDims();
    bool reallocate = false;
    if (blobDesc.getPrecision() != description.getPrecision())
    {
        reallocate = true;
        CV_LOG_WARNING(NULL, "Reallocate output '" << name << "' blob due to wrong precision: " << blobDesc.getPrecision() << " => " << description.getPrecision() << "  ndims=" << dims.size());
    }
    if (dims.size() != blobDesc.getDims().size())
    {
        reallocate = true;
        CV_LOG_WARNING(NULL, "Reallocate output '" << name << "' blob due to wrong dims: " << blobDesc.getDims().size() << " => " << dims.size());
    }
    if (reallocate)
    {
        auto layout = estimateLayout(dims.size());
        w.dataPtr = InferenceEngine::DataPtr(new InferenceEngine::Data(name,
               {description.getPrecision(), dims, layout}));
        w.blob = reallocateBlob(*w.host, description);
    }
    return w.dataPtr;
}

void forwardNgraph(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                      Ptr<BackendNode>& node, bool isAsync)
{
    CV_Assert(!node.empty());
    Ptr<InfEngineNgraphNode> ieNode = node.dynamicCast<InfEngineNgraphNode>();
    CV_Assert(!ieNode.empty());
    ieNode->net->forward(outBlobsWrappers, isAsync);
}

void InfEngineNgraphNet::reset()
{
    allBlobs.clear();
    infRequests.clear();
    isInit = false;

    outputsDesc.clear();
    for (const auto& it : cnn.getOutputsInfo())
    {
        const std::string& name = it.first;
        outputsDesc.insert({name, it.second->getTensorDesc()});
    }
}

void InfEngineNgraphNet::addBlobs(const std::vector<cv::Ptr<BackendWrapper> >& ptrs)
{
    auto wrappers = ngraphWrappers(ptrs);
    for (const auto& wrapper : wrappers)
    {
        std::string name = wrapper->dataPtr->getName();
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
        outsNames[i] = outs[i]->dataPtr->getName();
    }
}

Mat ngraphBlobToMat(const InferenceEngine::Blob::Ptr& blob)
{
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
            inpBlobs[name] = isAsync ? copyBlob(blobIt->second) : blobIt->second;
        }
        for (const auto& it : cnn.getOutputsInfo())
        {
            const std::string& name = it.first;
            auto blobIt = allBlobs.find(name);
            CV_Assert(blobIt != allBlobs.end());
            outBlobs[name] = isAsync ? copyBlob(blobIt->second) : blobIt->second;
        }
        reqWrapper->req.SetInput(inpBlobs);
        reqWrapper->req.SetOutput(outBlobs);

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2021_4)
        InferenceEngine::InferRequest infRequest = reqWrapper->req;
        NgraphReqWrapper* wrapperPtr = reqWrapper.get();
        CV_Assert(wrapperPtr && "Internal error");
#else
        InferenceEngine::IInferRequest::Ptr infRequestPtr = reqWrapper->req;
        CV_Assert(infRequestPtr);
        InferenceEngine::IInferRequest& infRequest = *infRequestPtr.get();
        infRequest.SetUserData(reqWrapper.get(), 0);
#endif

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2021_4)
        // do NOT capture 'reqWrapper' (smart ptr) in the lambda callback
        infRequest.SetCompletionCallback<std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>>(
            [wrapperPtr](InferenceEngine::InferRequest /*request*/, InferenceEngine::StatusCode status)
#else
        infRequest.SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr requestPtr, InferenceEngine::StatusCode status)
#endif
            {
                CV_LOG_DEBUG(NULL, "DNN(nGraph): completionCallback(" << (int)status << ")");
#if !INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2021_4)
                CV_Assert(requestPtr);
                InferenceEngine::IInferRequest& request = *requestPtr.get();

                NgraphReqWrapper* wrapperPtr;
                request.GetUserData((void**)&wrapperPtr, 0);
                CV_Assert(wrapperPtr && "Internal error");
#endif
                NgraphReqWrapper& wrapper = *wrapperPtr;

                size_t processedOutputs = 0;
                try
                {
                    for (; processedOutputs < wrapper.outProms.size(); ++processedOutputs)
                    {
                        const std::string& name = wrapper.outsNames[processedOutputs];
                        Mat m = ngraphBlobToMat(wrapper.req.GetBlob(name));

                        try
                        {
                            CV_Assert(status == InferenceEngine::StatusCode::OK);
                            wrapper.outProms[processedOutputs].setValue(m.clone());
                        }
                        catch (...)
                        {
                            try {
                                wrapper.outProms[processedOutputs].setException(std::current_exception());
                            } catch(...) {
                                CV_LOG_ERROR(NULL, "DNN: Exception occurred during async inference exception propagation");
                            }
                        }
                    }
                }
                catch (...)
                {
                    std::exception_ptr e = std::current_exception();
                    for (; processedOutputs < wrapper.outProms.size(); ++processedOutputs)
                    {
                        try {
                            wrapper.outProms[processedOutputs].setException(e);
                        } catch(...) {
                            CV_LOG_ERROR(NULL, "DNN: Exception occurred during async inference exception propagation");
                        }
                    }
                }
                wrapper.isReady = true;
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
            Mat srcMat = ngraphBlobToMat(blobIt->second);
            Mat dstMat = ngraphBlobToMat(reqWrapper->req.GetBlob(name));
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

#else
void forwardNgraph(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                   Ptr<BackendNode>& node, bool isAsync)
{
    CV_Assert(false && "nGraph is not enabled in this OpenCV build");
}
#endif

}}
