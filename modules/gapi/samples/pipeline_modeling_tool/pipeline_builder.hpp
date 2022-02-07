#ifndef OPENCV_GAPI_PIPELINE_MODELING_TOOL_PIPELINE_BUILDER_HPP
#define OPENCV_GAPI_PIPELINE_MODELING_TOOL_PIPELINE_BUILDER_HPP

#include <map>

#include <opencv2/gapi/infer.hpp> // cv::gapi::GNetPackage
#include <opencv2/gapi/streaming/cap.hpp> // cv::gapi::wip::IStreamSource
#include <opencv2/gapi/infer/ie.hpp> // cv::gapi::ie::Params
#include <opencv2/gapi/gcommon.hpp> // cv::gapi::GCompileArgs
#include <opencv2/gapi/cpu/gcpukernel.hpp> // GAPI_OCV_KERNEL
#include <opencv2/gapi/gkernel.hpp> // G_API_OP

#include "pipeline.hpp"
#include "utils.hpp"

struct Edge {
    struct P {
        std::string name;
        size_t      port;
    };

    P src;
    P dst;
};

struct CallNode {
    using F = std::function<void(const cv::GProtoArgs&, cv::GProtoArgs&)>;

    std::string name;
    F           run;
};

struct DataNode {
    cv::optional<cv::GProtoArg> arg;
};

struct Node {
    using Ptr  = std::shared_ptr<Node>;
    using WPtr = std::weak_ptr<Node>;
    using Kind = cv::util::variant<CallNode, DataNode>;

    std::vector<Node::WPtr> in_nodes;
    std::vector<Node::Ptr>  out_nodes;
    Kind kind;
};

struct DummyCall {
    G_API_OP(GDummy,
             <cv::GMat(cv::GMat, double, OutputDescr)>,
             "custom.dummy") {
        static cv::GMatDesc outMeta(const cv::GMatDesc& /* in */,
                                    double              /* time */,
                                    const OutputDescr& output) {
            if (output.dims.size() == 2) {
                return cv::GMatDesc(output.precision,
                                    1,
                                    cv::Size(output.dims[0], output.dims[1]));
            }
            return cv::GMatDesc(output.precision, output.dims);
        }
    };

    struct DummyState {
        cv::Mat mat;
    };

    // NB: Generate random mat once and then
    // copy to dst buffer on every iteration.
    GAPI_OCV_KERNEL_ST(GCPUDummy, GDummy, DummyState) {
            static void setup(const cv::GMatDesc&          /*in*/,
                              double                       /*time*/,
                              const OutputDescr&           output,
                              std::shared_ptr<DummyState>& state,
                              const cv::GCompileArgs&      /*args*/) {
            state.reset(new DummyState{});
            utils::createNDMat(state->mat, output.dims, output.precision);
            utils::generateRandom(state->mat);
        }

        static void run(const cv::Mat&     /*in_mat*/,
                        double             time,
                        const OutputDescr& /*output*/,
                        cv::Mat&           out_mat,
                        DummyState&        state) {
            using namespace std::chrono;
            double total = 0;
            auto   start = high_resolution_clock::now();
            state.mat.copyTo(out_mat);
            while (total < time) {
                total = duration_cast<duration<double, std::milli>>(
                            high_resolution_clock::now() - start).count();
            }
        }
    };

    void operator()(const cv::GProtoArgs& inputs, cv::GProtoArgs& outputs);

    size_t numInputs()  const { return 1; }
    size_t numOutputs() const { return 1; }

    double      time;
    OutputDescr output;
};

void DummyCall::operator()(const cv::GProtoArgs& inputs,
                                 cv::GProtoArgs& outputs) {
    GAPI_Assert(inputs.size() == 1u);
    GAPI_Assert(cv::util::holds_alternative<cv::GMat>(inputs[0]));
    GAPI_Assert(outputs.empty());
    auto in = cv::util::get<cv::GMat>(inputs[0]);
    outputs.emplace_back(GDummy::on(in, time, output));
}

struct InferCall {
    void operator()(const cv::GProtoArgs& inputs, cv::GProtoArgs& outputs);
    size_t numInputs()  const { return input_layers.size();  }
    size_t numOutputs() const { return output_layers.size(); }

    std::string               tag;
    std::vector<std::string>  input_layers;
    std::vector<std::string>  output_layers;
};

void InferCall::operator()(const cv::GProtoArgs& inputs,
                                 cv::GProtoArgs& outputs) {
    GAPI_Assert(inputs.size() == input_layers.size());
    GAPI_Assert(outputs.empty());

    cv::GInferInputs g_inputs;
    // TODO: Add an opportunity not specify input/output layers in case
    // there is only single layer.
    for (size_t i = 0; i < inputs.size(); ++i) {
        // TODO: Support GFrame as well.
        GAPI_Assert(cv::util::holds_alternative<cv::GMat>(inputs[i]));
        auto in = cv::util::get<cv::GMat>(inputs[i]);
        g_inputs[input_layers[i]] = in;
    }
    auto g_outputs = cv::gapi::infer<cv::gapi::Generic>(tag, g_inputs);
    for (size_t i = 0; i < output_layers.size(); ++i) {
        outputs.emplace_back(g_outputs.at(output_layers[i]));
    }
}

struct SourceCall {
    void operator()(const cv::GProtoArgs& inputs, cv::GProtoArgs& outputs);
    size_t numInputs()  const { return 0; }
    size_t numOutputs() const { return 1; }
};

void SourceCall::operator()(const cv::GProtoArgs& inputs,
                                  cv::GProtoArgs& outputs) {
    GAPI_Assert(inputs.empty());
    GAPI_Assert(outputs.empty());
    // NB: Since NV12 isn't exposed source always produce GMat.
    outputs.emplace_back(cv::GMat());
}

struct LoadPath {
    std::string xml;
    std::string bin;
};

struct ImportPath {
    std::string blob;
};

using ModelPath = cv::util::variant<ImportPath, LoadPath>;

struct InferParams {
    std::string name;
    ModelPath   path;
    std::string device;
    std::vector<std::string> input_layers;
    std::vector<std::string> output_layers;
    std::map<std::string, std::string> config;
};

class PipelineBuilder {
public:
    PipelineBuilder();
    void addDummy(const std::string& name,
                  const double       time,
                  const OutputDescr& output);

    void addInfer(const std::string& name, const InferParams& params);

    void setSource(const std::string& name,
                   double latency,
                   const OutputDescr& output);

    void addEdge(const Edge& edge);
    void setMode(PLMode mode);
    void setDumpFilePath(const std::string& dump);
    void setQueueCapacity(const size_t qc);
    void setName(const std::string& name);

    Pipeline::Ptr build();

private:
    template <typename CallT>
    void addCall(const std::string& name,
                 CallT&&            call);

    Pipeline::Ptr construct();

    template <typename K, typename V>
    using M = std::unordered_map<K, V>;
    struct State {
        struct NodeEdges {
            std::vector<Edge> input_edges;
            std::vector<Edge> output_edges;
        };

        M<std::string, Node::Ptr>         calls_map;
        std::vector<Node::Ptr>            all_calls;

        cv::gapi::GNetPackage             networks;
        cv::gapi::GKernelPackage          kernels;
        cv::GCompileArgs                  compile_args;
        cv::gapi::wip::IStreamSource::Ptr src;
        PLMode                            mode = PLMode::STREAMING;
        std::string                       name;
    };

    std::unique_ptr<State> m_state;
};

PipelineBuilder::PipelineBuilder() : m_state(new State{}) { };

void PipelineBuilder::addDummy(const std::string&  name,
                               const double        time,
                               const OutputDescr&  output) {
    m_state->kernels.include<DummyCall::GCPUDummy>();
    addCall(name, DummyCall{time, output});
}

template <typename CallT>
void PipelineBuilder::addCall(const std::string& name,
                              CallT&&            call) {

    size_t num_inputs  = call.numInputs();
    size_t num_outputs = call.numOutputs();
    Node::Ptr call_node(new Node{{},{},Node::Kind{CallNode{name, std::move(call)}}});
    // NB: Create placeholders for inputs.
    call_node->in_nodes.resize(num_inputs);
    // NB: Create outputs with empty data.
    for (size_t i = 0; i < num_outputs; ++i) {
        call_node->out_nodes.emplace_back(new Node{{call_node},
                                                   {},
                                                   Node::Kind{DataNode{}}});
    }

    auto it = m_state->calls_map.find(name);
    if (it != m_state->calls_map.end()) {
        throw std::logic_error("Node: " + name + " already exists!");
    }
    m_state->calls_map.emplace(name, call_node);
    m_state->all_calls.emplace_back(call_node);
}

void PipelineBuilder::addInfer(const std::string& name,
                               const InferParams& params) {
    // NB: No default ctor for Params.
    std::unique_ptr<cv::gapi::ie::Params<cv::gapi::Generic>> pp;
    if (cv::util::holds_alternative<LoadPath>(params.path)) {
       auto load_path = cv::util::get<LoadPath>(params.path);
       pp.reset(new cv::gapi::ie::Params<cv::gapi::Generic>(name,
                                                            load_path.xml,
                                                            load_path.bin,
                                                            params.device));
    } else {
        GAPI_Assert(cv::util::holds_alternative<ImportPath>(params.path));
        auto import_path = cv::util::get<ImportPath>(params.path);
        pp.reset(new cv::gapi::ie::Params<cv::gapi::Generic>(name,
                                                             import_path.blob,
                                                             params.device));
    }

    pp->pluginConfig(params.config);
    m_state->networks += cv::gapi::networks(*pp);

    addCall(name, InferCall{name, params.input_layers, params.output_layers});
}

void PipelineBuilder::addEdge(const Edge& edge) {
    const auto& src_it = m_state->calls_map.find(edge.src.name);
    if (src_it == m_state->calls_map.end()) {
        throw std::logic_error("Failed to find node: " + edge.src.name);
    }
    auto src_node = src_it->second;
    if (src_node->out_nodes.size() <= edge.src.port) {
        throw std::logic_error("Failed to access node: " + edge.src.name +
                               " by out port: " + std::to_string(edge.src.port));
    }

    auto dst_it = m_state->calls_map.find(edge.dst.name);
    if (dst_it == m_state->calls_map.end()) {
        throw std::logic_error("Failed to find node: " + edge.dst.name);
    }
    auto dst_node = dst_it->second;
    if (dst_node->in_nodes.size() <= edge.dst.port) {
        throw std::logic_error("Failed to access node: " + edge.dst.name +
                               " by in port: " + std::to_string(edge.dst.port));
    }

    auto  out_data = src_node->out_nodes[edge.src.port];
    auto& in_data  = dst_node->in_nodes[edge.dst.port];
    // NB: in_data != nullptr.
    if (!in_data.expired()) {
        throw std::logic_error("Node: " + edge.dst.name +
                               " already connected by in port: " +
                               std::to_string(edge.dst.port));
    }
    dst_node->in_nodes[edge.dst.port] = out_data;
    out_data->out_nodes.push_back(dst_node);
}

void PipelineBuilder::setSource(const std::string& name,
                                double latency,
                                const OutputDescr& output) {
    GAPI_Assert(!m_state->src);
    m_state->src = std::make_shared<DummySource>(latency, output);
    addCall(name, SourceCall{});
}

void PipelineBuilder::setMode(PLMode mode) {
    m_state->mode = mode;
}

void PipelineBuilder::setDumpFilePath(const std::string& dump) {
    m_state->compile_args.emplace_back(cv::graph_dump_path{dump});
}

void PipelineBuilder::setQueueCapacity(const size_t qc) {
    m_state->compile_args.emplace_back(cv::gapi::streaming::queue_capacity{qc});
}

void PipelineBuilder::setName(const std::string& name) {
    m_state->name = name;
}

static bool visit(Node::Ptr node,
                  std::vector<Node::Ptr>& sorted,
                  std::unordered_map<Node::Ptr, int>& visited) {
    if (!node) {
        throw std::logic_error("Found null node");
    }

    visited[node] = 1;
    for (auto in : node->in_nodes) {
        auto in_node = in.lock();
        if (visited[in_node] == 0) {
            if (visit(in_node, sorted, visited)) {
                return true;
            }
        } else if (visited[in_node] == 1) {
            return true;
        }
    }
    visited[node] = 2;
    sorted.push_back(node);
    return false;
}

static cv::optional<std::vector<Node::Ptr>>
toposort(const std::vector<Node::Ptr> nodes) {
    std::vector<Node::Ptr> sorted;
    std::unordered_map<Node::Ptr, int> visited;
    for (auto n : nodes) {
        if (visit(n, sorted, visited)) {
            return cv::optional<std::vector<Node::Ptr>>{};
        }
    }
    return cv::util::make_optional(sorted);
}

Pipeline::Ptr PipelineBuilder::construct() {
    // NB: Unlike G-API, pipeline_builder_tool graph always starts with CALL node
    // (not data) that produce datas, so the call node which doesn't have
    // inputs is considered as "producer" node.
    //
    // Graph always starts with CALL node and ends with DATA node.
    // Graph example: [source] -> (source:0) -> [PP] -> (PP:0)
    //
    // The algorithm is quite simple:
    // 0. Verify that every call input node exists (connected).
    // 1. Sort all nodes by visiting only call nodes,
    // since there is no data nodes that's not connected with any call node,
    // it's guarantee that every node will be visited.
    // 2. Fillter call nodes.
    // 3. Go through every call node.
    // FIXME: Add toposort in case user passed nodes
    // in arbitrary order which is unlikely happened.
    // 4. Extract proto input from every input node
    // 5. Run call and get outputs
    // 6. If call node doesn't have inputs it means that it's "producer" node,
    // so collect all outputs to graph_inputs vector.
    // 7. Assign proto outputs to output data nodes,
    // so the next calls can use them as inputs.
    cv::GProtoArgs graph_inputs;
    cv::GProtoArgs graph_outputs;
    // 0. Verify that every call input node exists (connected).
    for (auto call_node : m_state->all_calls) {
        for (size_t i = 0; i < call_node->in_nodes.size(); ++i) {
            const auto& in_data_node = call_node->in_nodes[i];
            // NB: in_data_node == nullptr.
            if (in_data_node.expired()) {
                const auto& call = cv::util::get<CallNode>(call_node->kind);
                throw std::logic_error(
                        "Node: " + call.name + " in Pipeline: " + m_state->name +
                        " has dangling input by in port: " + std::to_string(i));
            }
        }
    }
    // (0) Sort all nodes;
    auto has_sorted = toposort(m_state->all_calls);
    if (!has_sorted) {
       throw std::logic_error(
               "Pipeline: " + m_state->name + " has cyclic dependencies") ;
    }
    auto& sorted = has_sorted.value();
    // (1). Fillter call nodes.
    std::vector<Node::Ptr> sorted_calls;
    for (auto n : sorted) {
        if (cv::util::holds_alternative<CallNode>(n->kind)) {
            sorted_calls.push_back(n);
        }
    }
    // (2). Go through every call node.
    for (auto call_node : sorted_calls) {
        cv::GProtoArgs outputs;
        cv::GProtoArgs inputs;
        for (size_t i = 0; i < call_node->in_nodes.size(); ++i) {
            auto in_node = call_node->in_nodes.at(i);
            auto in_data = cv::util::get<DataNode>(in_node.lock()->kind);
            if (!in_data.arg.has_value()) {
                throw std::logic_error("data hasn't been provided");
            }
            // (3). Extract proto input from every input node.
            inputs.push_back(in_data.arg.value());
        }
        // (4). Run call and get outputs.
        auto call = cv::util::get<CallNode>(call_node->kind);
        call.run(inputs, outputs);
        // (5) If call node doesn't have inputs
        // it means that it's input producer node (Source).
        if (call_node->in_nodes.empty()) {
            for (auto out : outputs) {
                graph_inputs.push_back(out);
            }
        }
        // (6). Assign proto outputs to output data nodes,
        // so the next calls can use them as inputs.
        GAPI_Assert(outputs.size() == call_node->out_nodes.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto out_node = call_node->out_nodes[i];
            auto& out_data = cv::util::get<DataNode>(out_node->kind);
            out_data.arg = cv::util::make_optional(outputs[i]);
            if (out_node->out_nodes.empty()) {
                graph_outputs.push_back(out_data.arg.value());
            }
        }
    }

    m_state->compile_args.emplace_back(m_state->networks);
    m_state->compile_args.emplace_back(m_state->kernels);

    if (m_state->mode == PLMode::STREAMING) {
        GAPI_Assert(graph_inputs.size() == 1);
        GAPI_Assert(cv::util::holds_alternative<cv::GMat>(graph_inputs[0]));
        // FIXME: Handle GFrame when NV12 comes.
        const auto& graph_input = cv::util::get<cv::GMat>(graph_inputs[0]);
        // NB: In case streaming mode need to expose timestamp in order to
        // calculate performance metrics.
        graph_outputs.emplace_back(
                cv::gapi::streaming::timestamp(graph_input).strip());

        return std::make_shared<StreamingPipeline>(std::move(m_state->name),
                                                   cv::GComputation(
                                                       cv::GProtoInputArgs{graph_inputs},
                                                       cv::GProtoOutputArgs{graph_outputs}),
                                                   std::move(m_state->src),
                                                   std::move(m_state->compile_args),
                                                   graph_outputs.size());
    }
    GAPI_Assert(m_state->mode == PLMode::REGULAR);
    return std::make_shared<RegularPipeline>(std::move(m_state->name),
                                             cv::GComputation(
                                                 cv::GProtoInputArgs{graph_inputs},
                                                 cv::GProtoOutputArgs{graph_outputs}),
                                             std::move(m_state->src),
                                             std::move(m_state->compile_args),
                                             graph_outputs.size());
}

Pipeline::Ptr PipelineBuilder::build() {
    auto pipeline = construct();
    m_state.reset(new State{});
    return pipeline;
}

#endif // OPENCV_GAPI_PIPELINE_MODELING_TOOL_PIPELINE_BUILDER_HPP
