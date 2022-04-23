// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021-2022 Intel Corporation

#include <opencv2/gapi/gkernel.hpp> // GKernelPackage
#include <opencv2/gapi/oak/oak.hpp> // kernels()

#ifdef HAVE_OAK

#include <cstring>
#include <unordered_set>
#include <algorithm> // any_of
#include <functional> // reference_wrapper

#include <ade/util/zip_range.hpp>

#include <api/gbackend_priv.hpp>
#include <backends/common/gbackend.hpp>

#include <opencv2/gapi/infer.hpp> // GInferBase
#include <opencv2/gapi/streaming/meta.hpp> // streaming::meta_tag

#include "depthai/depthai.hpp"

#include "oak_memory_adapters.hpp"

#include <opencv2/gapi/oak/infer.hpp> // infer params

namespace cv { namespace gimpl {

// Forward declaration
class GOAKContext;
class OAKKernelParams;

class GOAKExecutable final: public GIslandExecutable {
    friend class GOAKContext;
    friend class OAKKernelParams;
    virtual void run(std::vector<InObj>&&,
                     std::vector<OutObj>&&) override {
        GAPI_Assert(false && "Not implemented");
    }

    virtual void run(GIslandExecutable::IInput &in,
                     GIslandExecutable::IOutput &out) override;

    void linkToParent(ade::NodeHandle handle);
    void linkCopy(ade::NodeHandle handle);

    class ExtractTypeHelper : protected dai::Node {
    public:
        using Input = dai::Node::Input;
        using Output = dai::Node::Output;
        using InputPtr = dai::Node::Input*;
        using OutputPtr = dai::Node::Output*;
    };

    struct OAKNodeInfo {
        std::shared_ptr<dai::Node> node = nullptr;
        std::vector<ExtractTypeHelper::InputPtr> inputs = {};
        std::vector<ExtractTypeHelper::OutputPtr> outputs = {};
    };

    struct OAKOutQueueInfo {
        std::shared_ptr<dai::node::XLinkOut> xlink_output;
        std::shared_ptr<dai::DataOutputQueue> out_queue;
        std::string out_queue_name;
        size_t gapi_out_data_index;
    };

    cv::GArg packInArg(const GArg &arg, std::vector<ExtractTypeHelper::InputPtr>& oak_ins);
    void packOutArg(const RcDesc &rc, std::vector<ExtractTypeHelper::OutputPtr>& oak_outs);

    const ade::Graph& m_g;
    GModel::ConstGraph m_gm;
    cv::GCompileArgs m_args;

    std::unordered_map<ade::NodeHandle,
                       OAKNodeInfo,
                       ade::HandleHasher<ade::Node>> m_oak_nodes;

    // Will be reworked later when XLinkIn will be introduced as input
    std::shared_ptr<dai::node::ColorCamera> m_camera_input;
    cv::Size m_camera_size;

    // Backend outputs
    std::unordered_map<ade::NodeHandle,
                       OAKOutQueueInfo,
                       ade::HandleHasher<ade::Node>> m_out_queues;

    // Backend inputs
    std::vector<std::pair<std::string, dai::Buffer>> m_in_queues;

    std::unordered_set<ade::NodeHandle,
                       ade::HandleHasher<ade::Node>> m_passthrough_copy_nodes;

    // Note: dai::Pipeline should be the only one for the whole pipeline,
    // so there is no way to insert any non-OAK node in graph between other OAK nodes.
    // The only heterogeneous case possible is if we insert other backends after or before
    // OAK island.
    std::unique_ptr<dai::Device> m_device;
    std::unique_ptr<dai::Pipeline> m_pipeline;

    // Camera config
    cv::gapi::oak::ColorCameraParams m_ccp;

    // Infer info
    std::unordered_map<ade::NodeHandle,
                       cv::gapi::oak::detail::ParamDesc,
                       ade::HandleHasher<ade::Node>> m_oak_infer_info;

public:
    GOAKExecutable(const ade::Graph& g,
                   const cv::GCompileArgs& args,
                   const std::vector<ade::NodeHandle>& nodes,
                   const std::vector<cv::gimpl::Data>& ins_data,
                   const std::vector<cv::gimpl::Data>& outs_data);
    ~GOAKExecutable() = default;

    // FIXME: could it reshape?
    virtual bool canReshape() const override { return false; }
    virtual void reshape(ade::Graph&, const GCompileArgs&) override {
        GAPI_Assert(false && "GOAKExecutable::reshape() is not supported");
    }

    virtual void handleNewStream() override;
    virtual void handleStopStream() override;
};

class GOAKContext {
public:
    // FIXME: make private?
    using Input = GOAKExecutable::ExtractTypeHelper::Input;
    using Output = GOAKExecutable::ExtractTypeHelper::Output;
    using InputPtr = GOAKExecutable::ExtractTypeHelper::Input*;
    using OutputPtr = GOAKExecutable::ExtractTypeHelper::Output*;

    GOAKContext(const std::unique_ptr<dai::Pipeline>& pipeline,
                const cv::Size& camera_size,
                std::vector<cv::GArg>& args,
                std::vector<OutputPtr>& results);

    GOAKContext(const std::unique_ptr<dai::Pipeline>& pipeline,
                const cv::Size& camera_size,
                const cv::gapi::oak::detail::ParamDesc& infer_info,
                std::vector<cv::GArg>& args,
                std::vector<OutputPtr>& results);

    // Generic accessor API
    template<typename T>
    T& inArg(int input) { return m_args.at(input).get<T>(); }

    // FIXME: consider not using raw pointers
    InputPtr& in(int input);
    OutputPtr& out(int output);

    const std::unique_ptr<dai::Pipeline>& pipeline() const;
    const cv::Size& camera_size() const;
    const cv::gapi::oak::detail::ParamDesc& ii() const;

private:
    const std::unique_ptr<dai::Pipeline>& m_pipeline;
    const cv::Size m_camera_size;
    const cv::gapi::oak::detail::ParamDesc m_infer_info;
    std::vector<cv::GArg>& m_args;
    std::vector<OutputPtr>& m_outputs;
};

GOAKContext::GOAKContext(const std::unique_ptr<dai::Pipeline>& pipeline,
                         const cv::Size& camera_size,
                         std::vector<cv::GArg>& args,
                         std::vector<OutputPtr>& results)
    : m_pipeline(pipeline), m_camera_size(camera_size),
      m_args(args), m_outputs(results) {}

GOAKContext::GOAKContext(const std::unique_ptr<dai::Pipeline>& pipeline,
                         const cv::Size& camera_size,
                         const cv::gapi::oak::detail::ParamDesc& infer_info,
                         std::vector<cv::GArg>& args,
                         std::vector<OutputPtr>& results)
    : m_pipeline(pipeline), m_camera_size(camera_size),
      m_infer_info(infer_info), m_args(args), m_outputs(results) {}

const std::unique_ptr<dai::Pipeline>& GOAKContext::pipeline() const {
    return m_pipeline;
}

const cv::Size& GOAKContext::camera_size() const {
    return m_camera_size;
}

const cv::gapi::oak::detail::ParamDesc& GOAKContext::ii() const {
    return m_infer_info;
}

GOAKContext::InputPtr& GOAKContext::in(int input) {
    return inArg<std::reference_wrapper<GOAKContext::InputPtr>>(input).get();
}

GOAKContext::OutputPtr& GOAKContext::out(int output) {
    return m_outputs.at(output);
}

class OAKKernelParams {
public:
    const std::unique_ptr<dai::Pipeline>& pipeline;
    const cv::Size& camera_size;
    const cv::gapi::oak::detail::ParamDesc& infer_info;
    std::vector<std::pair<std::string, dai::Buffer>>& in_queues;
};

namespace detail {
template<class T> struct get_in;
template<> struct get_in<cv::GFrame> {
    static GOAKContext::InputPtr& get(GOAKContext &ctx, int idx) { return ctx.in(idx); }
};
template<class T> struct get_in {
    static T get(GOAKContext &ctx, int idx) { return ctx.inArg<T>(idx); }
};
// FIXME: add support of other types

template<class T> struct get_out;
template<> struct get_out<cv::GFrame> {
    static GOAKContext::OutputPtr& get(GOAKContext &ctx, int idx) { return ctx.out(idx); }
};
template<typename U> struct get_out<cv::GArray<U>> {
    static GOAKContext::OutputPtr& get(GOAKContext &ctx, int idx) { return ctx.out(idx); }
};
template<> struct get_out<cv::GMat> {
    static GOAKContext::OutputPtr& get(GOAKContext &ctx, int idx) { return ctx.out(idx); }
};
// FIXME: add support of other types

template<typename, typename, typename>
struct OAKCallHelper;

template<typename Impl, typename... Ins, typename... Outs>
struct OAKCallHelper<Impl, std::tuple<Ins...>, std::tuple<Outs...> > {
    template<int... IIs, int... OIs>
    static std::shared_ptr<dai::Node> construct_impl(  GOAKContext &ctx
                                                     , std::vector<std::pair<std::string,
                                                                             dai::Buffer>>& in_queues_params
                                                     , cv::detail::Seq<IIs...>
                                                     , cv::detail::Seq<OIs...>) {
        return Impl::put(OAKKernelParams{ctx.pipeline(),
                                         ctx.camera_size(),
                                         ctx.ii(),
                                         in_queues_params},
                         get_in<Ins>::get(ctx, IIs)...,
                         get_out<Outs>::get(ctx, OIs)...);
    }

    static std::shared_ptr<dai::Node> construct(GOAKContext &ctx,
                                                std::vector<std::pair<std::string,
                                                                      dai::Buffer>>& in_queues_params) {
        return construct_impl(ctx,
                              in_queues_params,
                              typename cv::detail::MkSeq<sizeof...(Ins)>::type(),
                              typename cv::detail::MkSeq<sizeof...(Outs)>::type());
    }
};

} // namespace detail

struct GOAKKernel {
    using F = std::function<std::shared_ptr<dai::Node>(GOAKContext&,
                                                       std::vector<std::pair<std::string, dai::Buffer>>&)>;
    explicit GOAKKernel(const F& f) : m_put_f(f) {}
    const F m_put_f;
};

struct OAKComponent
{
    static const char *name() { return "OAK Component"; }
    GOAKKernel k;
};
} // namespace gimpl
} // namespace cv

using OAKGraph = ade::TypedGraph
    < cv::gimpl::Protocol
    , cv::gimpl::Op
    , cv::gimpl::NetworkParams
    , cv::gimpl::CustomMetaFunction
    // OAK specific
    , cv::gimpl::OAKComponent
    >;

using ConstOAKGraph = ade::ConstTypedGraph
    < cv::gimpl::Protocol
    , cv::gimpl::Op
    , cv::gimpl::NetworkParams
    , cv::gimpl::CustomMetaFunction
    // OAK specific
    , cv::gimpl::OAKComponent
    >;

namespace
{
std::pair<dai::TensorInfo, dai::TensorInfo>
parseDaiInferMeta(const cv::gapi::oak::detail::ParamDesc& pd) {
    dai::OpenVINO::Blob blob(pd.blob_file);

    GAPI_Assert(blob.networkInputs.size() == 1);
    GAPI_Assert(blob.networkOutputs.size() == 1);

    return {blob.networkInputs.begin()->second,
            blob.networkOutputs.begin()->second};
}

std::string
getDaiInferOutLayerName(const cv::gapi::oak::detail::ParamDesc& pd) {
    dai::OpenVINO::Blob blob(pd.blob_file);

    GAPI_Assert(blob.networkInputs.size() == 1);
    GAPI_Assert(blob.networkOutputs.size() == 1);

    return blob.networkOutputs.begin()->first;
}
} // anonymous namespace

// Custom meta function for OAK backend for infer
static cv::GMetaArgs customOutMeta(const ade::Graph      &gr,
                                   const ade::NodeHandle &nh,
                                   const cv::GMetaArgs   &/*in_metas*/,
                                   const cv::GArgs       &/*in_args*/) {
    cv::GMetaArgs result;
    const auto &np = ConstOAKGraph(gr).metadata(nh).get<cv::gimpl::NetworkParams>();
    const auto &pd = cv::util::any_cast<cv::gapi::oak::detail::ParamDesc>(np.opaque);

    // FIXME: Infer kernel and backend does rather the same
    auto in_out_tensor_info = parseDaiInferMeta(pd);

    GAPI_Assert(in_out_tensor_info.second.dataType ==
                dai::TensorInfo::DataType::FP16);

    // FIXME: add proper layout converter here
    GAPI_Assert(in_out_tensor_info.second.order ==
                dai::TensorInfo::StorageOrder::NCHW);

    // FIXME: DAI returns vector<unsigned>, remove workaround
    std::vector<int> wrapped_dims;
    for (const auto& d : in_out_tensor_info.second.dims) {
        wrapped_dims.push_back(d);
    }
    result = {cv::GMetaArg{cv::GMatDesc(CV_16F, 1, cv::Size(wrapped_dims[1], wrapped_dims[0]), false)}};

    return result;
}

// This function links DAI operation nodes - parent's output to child's input.
// It utilizes G-API graph to search for operation's node it's previous operation in graph
// when links them in DAI graph.
void cv::gimpl::GOAKExecutable::linkToParent(ade::NodeHandle handle)
{
    ade::NodeHandle parent;
    for (const auto& data_nh : handle.get()->inNodes()) {
        // Data node has only 1 input
        GAPI_Assert(data_nh.get()->inNodes().size() == 1);
        parent = data_nh.get()->inNodes().front();

        // Don't link if parent is copy - the case is handled differently
        // in linkCopy
        const auto& op = m_gm.metadata(parent).get<Op>();
        if (op.k.name == "org.opencv.oak.copy") {
            continue;
        }

        // Assuming that OAK nodes are aligned for linking.
        // FIXME: potential rework might be needed then
        //        counterexample is found.
        GAPI_Assert(m_oak_nodes.at(handle).inputs.size() ==
                    m_oak_nodes.at(parent).outputs.size() &&
                    "Internal OAK nodes are not aligned for linking");
        for (auto && it : ade::util::zip(ade::util::toRange(m_oak_nodes.at(parent).outputs),
                                         ade::util::toRange(m_oak_nodes.at(handle).inputs)))
        {
            auto &out = std::get<0>(it);
            auto &in = std::get<1>(it);
            out->link(*in);
        }
    }
}

// This function links DAI operations for Copy OP in G-API graph
void cv::gimpl::GOAKExecutable::linkCopy(ade::NodeHandle handle) {
    // 1. Check that there are no back-to-back Copy OPs in graph
    auto copy_out = handle.get()->outNodes();
    GAPI_Assert(copy_out.size() == 1);
    for (const auto& copy_next_op : copy_out.front().get()->outNodes()) {
        const auto& op = m_gm.metadata(copy_next_op).get<Op>();
        if (op.k.name == "org.opencv.oak.copy") {
            GAPI_Assert(false && "Back-to-back Copy operations are not supported in graph");
        }
    }

    // 2. Link passthrough case
    if (m_passthrough_copy_nodes.find(handle) != m_passthrough_copy_nodes.end()) {
        ExtractTypeHelper::OutputPtr parent;
        bool parent_is_camera = false;
        // Copy has only 1 input data
        GAPI_Assert(handle.get()->inNodes().size() == 1);
        auto in_ops = handle.get()->inNodes().front().get()->inNodes();
        if (in_ops.size() == 0) {
            // No parent nodes - parent = camera
            parent = &m_camera_input->video;
            parent_is_camera = true;
        } else {
            // Data has only 1 input
            GAPI_Assert(in_ops.size() == 1);
            auto node = m_oak_nodes.at(in_ops.front());
            // Should only have 1 output
            GAPI_Assert(node.outputs.size() == 1);
            parent = node.outputs[0];
        }

        // Now link DAI parent output to Copy's child's inputs ignoring the Copy operation
        // FIXME: simplify this loop
        auto copy_out_data = handle.get()->outNodes();
        // Copy has only 1 output
        GAPI_Assert(copy_out_data.size() == 1);
        for (const auto& copy_next_op : copy_out_data.front().get()->outNodes()) {
            if (m_oak_nodes.find(copy_next_op) != m_oak_nodes.end()) {
                // FIXME: consider a better approach
                if (parent_is_camera) {
                    if (m_oak_infer_info.find(copy_next_op) != m_oak_infer_info.end()) {
                        parent = &m_camera_input->preview;
                    } else {
                        parent = &m_camera_input->video;
                    }
                }
                // Found next Copy OP which needs to be linked to Copy's parent
                GAPI_Assert(m_oak_nodes.at(copy_next_op).inputs.size() == 1 &&
                            "Internal OAK nodes are not aligned for linking (Copy operation)");
                parent->link(*(m_oak_nodes.at(copy_next_op).inputs.front()));
            }
        }
    }

    // 3. Link output Copy case
    if (m_out_queues.find(handle) != m_out_queues.end()) {
        // DAI XLinkOutput node
        auto xout = m_out_queues[handle].xlink_output->input;

        // Find parent node
        // FIXME: copypasted from case 2 above
        ExtractTypeHelper::OutputPtr parent;
        // Copy has only 1 input data
        GAPI_Assert(handle.get()->inNodes().size() == 1);
        auto in_ops = handle.get()->inNodes().front().get()->inNodes();
        if (in_ops.size() == 0) {
            // No parent nodes - parent = camera
            parent = &m_camera_input->video;
        } else {
            // Data has only 1 input
            GAPI_Assert(in_ops.size() == 1);
            auto node = m_oak_nodes.at(in_ops.front());
            // Should only have 1 output
            GAPI_Assert(node.outputs.size() == 1);
            parent = node.outputs[0];
        }

        // Link parent to xout
        parent->link(xout);
    }
}

cv::GArg
cv::gimpl::GOAKExecutable::packInArg(const GArg &arg,
                                     std::vector<ExtractTypeHelper::InputPtr>& oak_ins) {
    if (arg.kind != cv::detail::ArgKind::GOBJREF) {
        GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
                    && arg.kind != cv::detail::ArgKind::GSCALAR
                    && arg.kind != cv::detail::ArgKind::GARRAY
                    && arg.kind != cv::detail::ArgKind::GOPAQUE
                    && arg.kind != cv::detail::ArgKind::GFRAME);
        // All other cases - pass as-is, with no transformations to
        // GArg contents.
        return const_cast<cv::GArg&>(arg);
    }
    const cv::gimpl::RcDesc &ref = arg.get<cv::gimpl::RcDesc>();
    switch (ref.shape) {
    case GShape::GFRAME:
        oak_ins.push_back(nullptr);
        return GArg(std::reference_wrapper<ExtractTypeHelper::InputPtr>(oak_ins.back()));
        break;
    default:
        util::throw_error(std::logic_error("Unsupported GShape type in OAK backend"));
        break;
    }
}

void cv::gimpl::GOAKExecutable::packOutArg(const RcDesc &rc,
                                           std::vector<ExtractTypeHelper::OutputPtr>& oak_outs) {
    switch (rc.shape) {
    case GShape::GFRAME:
    case GShape::GARRAY:
    case GShape::GMAT:
        oak_outs.push_back(nullptr);
        break;
    default:
        util::throw_error(std::logic_error("Unsupported GShape type in OAK backend"));
        break;
    }
}

namespace {
static dai::CameraBoardSocket extractCameraBoardSocket(cv::gapi::oak::ColorCameraParams ccp) {
    switch (ccp.board_socket) {
        case cv::gapi::oak::ColorCameraParams::BoardSocket::RGB:
            return dai::CameraBoardSocket::RGB;
        // FIXME: extend
        default:
            // basically unreachable
            GAPI_Assert("Unsupported camera board socket");
            return {};
    }
}

static dai::ColorCameraProperties::SensorResolution
extractCameraResolution(cv::gapi::oak::ColorCameraParams ccp) {
    switch (ccp.resolution) {
        case cv::gapi::oak::ColorCameraParams::Resolution::THE_1080_P:
            return dai::ColorCameraProperties::SensorResolution::THE_1080_P;
        // FIXME: extend
        default:
            // basically unreachable
            GAPI_Assert("Unsupported camera board socket");
            return {};
    }
}
} // anonymous namespace

cv::gimpl::GOAKExecutable::GOAKExecutable(const ade::Graph& g,
                                          const cv::GCompileArgs &args,
                                          const std::vector<ade::NodeHandle>& nodes,
                                          const std::vector<cv::gimpl::Data>& ins_data,
                                          const std::vector<cv::gimpl::Data>& outs_data)
    : m_g(g), m_gm(m_g), m_args(args),
      m_device(nullptr), m_pipeline(new dai::Pipeline)
    {
        // FIXME: currently OAK backend only works with camera as input,
        //        so it must be a single object
        GAPI_Assert(ins_data.size() == 1);

        // Check that there is only one OAK island in graph since there
        // can only be one instance of dai::Pipeline in the application
        auto isl_graph = m_gm.metadata().get<IslandModel>().model;
        GIslandModel::Graph gim(*isl_graph);
        size_t oak_islands = 0;

        for (const auto& nh : gim.nodes())
        {
            if (gim.metadata(nh).get<NodeKind>().k == NodeKind::ISLAND)
            {
                const auto isl = gim.metadata(nh).get<FusedIsland>().object;
                if (isl->backend() == cv::gapi::oak::backend())
                {
                    ++oak_islands;
                }
                if (oak_islands > 1) {
                    util::throw_error
                        (std::logic_error
                            ("There can only be one OAK island in graph"));
                }
            }
        }

        m_ccp = cv::gimpl::getCompileArg<cv::gapi::oak::ColorCameraParams>(args)
                    .value_or(cv::gapi::oak::ColorCameraParams{});

        // FIXME: change the hard-coded behavior (XLinkIn path)
        auto camRgb = m_pipeline->create<dai::node::ColorCamera>();
        // FIXME: extract camera compile arguments here and properly convert them for dai
        camRgb->setBoardSocket(extractCameraBoardSocket(m_ccp));
        camRgb->setResolution(extractCameraResolution(m_ccp));
        camRgb->setInterleaved(m_ccp.interleaved);

        // Extract infer params
        for (const auto& nh : nodes) {
            if (m_gm.metadata(nh).get<NodeType>().t == NodeType::OP) {
                if (ConstOAKGraph(m_g).metadata(nh).contains<cv::gimpl::NetworkParams>()) {
                    const auto &np = ConstOAKGraph(m_g).metadata(nh).get<cv::gimpl::NetworkParams>();
                    const auto &pp = cv::util::any_cast<cv::gapi::oak::detail::ParamDesc>(np.opaque);
                    m_oak_infer_info[nh] = pp;
                    break;
                }
            }
        }

        // FIXME: handle multiple infers
        if (!m_oak_infer_info.empty()) {
            GAPI_Assert(m_oak_infer_info.size() == 1);
            // FIXME: move to infer node?
            auto in_out_tensor_info = parseDaiInferMeta(m_oak_infer_info.begin()->second);

            if (in_out_tensor_info.first.dataType ==
                dai::TensorInfo::DataType::FP16 ||
                in_out_tensor_info.first.dataType ==
                dai::TensorInfo::DataType::FP32) {
                camRgb->setFp16(true);
            } else {
                camRgb->setFp16(false);
            }

            // FIXME: add proper layout converter here
            GAPI_Assert(in_out_tensor_info.first.order ==
                        dai::TensorInfo::StorageOrder::NCHW);
            camRgb->setPreviewSize(in_out_tensor_info.first.dims[0], in_out_tensor_info.first.dims[1]);
        }

        m_camera_input = camRgb;
        // FIXME: change when other camera censors are introduced
        std::tuple<int, int> video_size = m_camera_input->getVideoSize();
        m_camera_size = cv::Size{std::get<0>(video_size), std::get<1>(video_size)};

        // Prepare XLinkOut nodes for each output object in graph
        for (size_t i = 0; i < outs_data.size(); ++i) {
            auto xout = m_pipeline->create<dai::node::XLinkOut>();
            std::string xout_name = "xout" + std::to_string(i);
            xout->setStreamName(xout_name);

            // Find parent OP's nh
            ade::NodeHandle parent_op_nh;
            for (const auto& nh : nodes) {
                for (const auto& outdata : nh.get()->outNodes()) {
                    if (m_gm.metadata(outdata).get<NodeType>().t == NodeType::DATA) {
                        auto rc = m_gm.metadata(outdata).get<cv::gimpl::Data>().rc;
                        auto shape = m_gm.metadata(outdata).get<cv::gimpl::Data>().shape;
                        // Match outs_data with the actual operation
                        if (rc == outs_data[i].rc && shape == outs_data[i].shape) {
                            parent_op_nh = nh;
                        }
                    }
                }
            }

            m_out_queues[parent_op_nh] = {xout, nullptr, xout_name, i};
        }

        // Create OAK node for each node in this backend
        for (const auto& nh : nodes) {
            if (m_gm.metadata(nh).get<NodeType>().t == NodeType::OP) {
                const auto& op = m_gm.metadata(nh).get<Op>();
                const auto &u = ConstOAKGraph(m_g).metadata(nh).get<OAKComponent>();
                // pass kernel input args and compile args to prepare OAK node and
                // store it to link later
                m_oak_nodes[nh] = {};
                m_oak_nodes.at(nh).inputs.reserve(op.args.size());
                m_oak_nodes.at(nh).outputs.reserve(op.outs.size());

                // Copy operation in graph can fall into 3 cases:
                // 1) Copy is an output of the island -
                //    in that case we link it to XLinkOut node from m_out_queues
                // 2) Copy is between other two operations in the same OAK island -
                //    in that case we link its parent operation (could be camera) to
                //    the child one (those copy operations are placed in m_passthrough_copy_nodes)
                // 3) Copy can fall into cases 1) and 2) at the same time

                // Prepare passthrough Copy operations
                if (op.k.name == "org.opencv.oak.copy") {
                    // Copy has only 1 output
                    auto copy_out = nh.get()->outNodes();
                    GAPI_Assert(copy_out.size() == 1);
                    for (const auto& copy_next_op : copy_out.front().get()->outNodes()) {
                        // Check that copy is a passthrough OP
                        if (std::find(nodes.begin(), nodes.end(), copy_next_op) != nodes.end()) {
                            m_passthrough_copy_nodes.insert(nh);
                            break;
                        }
                    }
                }

                std::vector<cv::GArg> in_ctx_args;
                in_ctx_args.reserve(op.args.size());
                for (auto &op_arg : op.args) in_ctx_args.push_back(packInArg(op_arg,
                                                                             m_oak_nodes.at(nh).inputs));
                for (auto &&op_out : op.outs) packOutArg(op_out, m_oak_nodes.at(nh).outputs);
                GAPI_Assert(!m_oak_nodes.at(nh).inputs.empty());
                GAPI_Assert(!m_oak_nodes.at(nh).outputs.empty());

                if (ConstOAKGraph(m_g).metadata(nh).contains<cv::gimpl::NetworkParams>()) {
                    GOAKContext ctx(m_pipeline, m_camera_size, m_oak_infer_info[nh],
                                    in_ctx_args, m_oak_nodes.at(nh).outputs);
                    m_oak_nodes.at(nh).node = u.k.m_put_f(ctx, m_in_queues);
                } else {
                    GOAKContext ctx(m_pipeline, m_camera_size,
                                    in_ctx_args, m_oak_nodes.at(nh).outputs);
                    m_oak_nodes.at(nh).node = u.k.m_put_f(ctx, m_in_queues);
                }

                // Check that all inputs and outputs are properly filled after constructing kernels
                // to then link it together
                // FIXME: add more logging
                const auto& node_info = m_oak_nodes.at(nh);
                // Copy operations don't set their inputs/outputs properly
                if (op.k.name != "org.opencv.oak.copy") {
                    GAPI_Assert(node_info.node != nullptr);
                    if (std::any_of(node_info.inputs.cbegin(), node_info.inputs.cend(),
                                    [](ExtractTypeHelper::InputPtr ptr) {
                            return ptr == nullptr;
                        })) {
                        GAPI_Assert(false && "DAI input are not set");
                    }

                    if (std::any_of(node_info.outputs.cbegin(), node_info.outputs.cend(),
                                    [](ExtractTypeHelper::OutputPtr ptr) {
                            return ptr == nullptr;
                        })) {
                        GAPI_Assert(false && "DAI outputs are not set");
                    }
                }
            }
        }

        // Prepare nodes for linking
        std::unordered_set<ade::NodeHandle,
                           ade::HandleHasher<ade::Node>> in_nodes;
        std::unordered_set<ade::NodeHandle,
                           ade::HandleHasher<ade::Node>> out_nodes;
        std::unordered_set<ade::NodeHandle,
                           ade::HandleHasher<ade::Node>> inter_nodes;
        std::unordered_set<ade::NodeHandle,
                           ade::HandleHasher<ade::Node>> copy_nodes;

        // TODO: optimize this loop
        for (const auto& node : m_oak_nodes) {
            auto nh = node.first;
            // Check if it's a Copy OP - will be handled differently when linking
            GAPI_Assert(m_gm.metadata(nh).get<NodeType>().t == NodeType::OP);
            const auto& op = m_gm.metadata(nh).get<Op>();
            if (op.k.name == "org.opencv.oak.copy") {
                copy_nodes.insert(nh);
                continue;
            }

            // Fill input op nodes
            for (const auto& d : ins_data) {
                for (const auto& indata : nh.get()->inNodes()) {
                    auto rc = m_gm.metadata(indata).get<cv::gimpl::Data>().rc;
                    auto shape = m_gm.metadata(indata).get<cv::gimpl::Data>().shape;
                    if (rc == d.rc && shape == d.shape) {
                        in_nodes.insert(nh);
                    }
                }
            }
            // Fill output op nodes
            for (const auto& d : outs_data) {
                for (const auto& outdata : nh.get()->outNodes()) {
                    auto rc = m_gm.metadata(outdata).get<cv::gimpl::Data>().rc;
                    auto shape = m_gm.metadata(outdata).get<cv::gimpl::Data>().shape;
                    if (rc == d.rc && shape == d.shape) {
                        out_nodes.insert(nh);
                    }
                }
            }
            // Fill internal op nodes
            if (in_nodes.find(nh) == in_nodes.end() &&
                out_nodes.find(nh) == in_nodes.end()) {
                inter_nodes.insert(nh);
            }
        }

        // Properly link all nodes
        // 1. Link input nodes to camera
        for (const auto& nh : in_nodes) {
            GAPI_Assert(m_oak_nodes.at(nh).inputs.size() == 1);
            // FIXME: convert other camera outputs
            // Link preview to infer, video to all other nodes
            if (m_oak_infer_info.find(nh) == m_oak_infer_info.end()) {
                m_camera_input->video.link(*(m_oak_nodes.at(nh).inputs[0]));
            } else {
                m_camera_input->preview.link(*(m_oak_nodes.at(nh).inputs[0]));
            }
        }

        // 2. Link output nodes to XLinkOut nodes
        for (const auto& nh : out_nodes) {
            for (const auto& out : m_oak_nodes.at(nh).outputs) {
                out->link(m_out_queues[nh].xlink_output->input);
            }
            // Input nodes in OAK doesn't have parent operation - just camera (for now)
            if (in_nodes.find(nh) == in_nodes.end()) {
                linkToParent(nh);
            }
        }

        // 3. Link internal nodes to their parents
        for (const auto& nh : inter_nodes) {
            linkToParent(nh);
        }

        // 4. Link copy nodes
        for (const auto& nh : copy_nodes) {
            linkCopy(nh);
        }

        m_device = std::unique_ptr<dai::Device>(new dai::Device(*m_pipeline));

        // Prepare OAK output queues
        GAPI_Assert(m_out_queues.size() == outs_data.size());
        for (const auto out_it : ade::util::indexed(m_out_queues))
        {
            auto& q = ade::util::value(out_it).second;
            GAPI_Assert(q.out_queue == nullptr); // shouldn't be not filled till this point
            // FIXME: add queue parameters
            // Currently: 4 - max DAI queue capacity, true - blocking queue
            q.out_queue = m_device->getOutputQueue(q.out_queue_name, 4, true);
        }
    }

void cv::gimpl::GOAKExecutable::handleNewStream() {
    // do nothing
}

void cv::gimpl::GOAKExecutable::handleStopStream() {
    // do nothing
}

void cv::gimpl::GOAKExecutable::run(GIslandExecutable::IInput  &in,
                                    GIslandExecutable::IOutput &out) {
    const auto in_msg = in.get();

    if (cv::util::holds_alternative<cv::gimpl::EndOfStream>(in_msg)) {
        out.post(cv::gimpl::EndOfStream{});
        return;
    }

    for (const auto& in_q : m_in_queues) {
        auto q = m_device->getInputQueue(in_q.first);
        q->send(in_q.second);
    }

    for (size_t i = 0; i < m_in_queues.size(); ++i) {
        auto q = m_device->getInputQueue(m_in_queues[i].first);
        q->send(m_in_queues[i].second);
    }

    for (const auto el : m_out_queues) {
        const auto out_q = el.second;
        auto& q = out_q.out_queue;

        auto out_arg = out.get(out_q.gapi_out_data_index);

        // FIXME: misc info to be utilized in switch below
        cv::GRunArg::Meta meta;
        std::shared_ptr<dai::ImgFrame> oak_frame;

        switch(out_arg.index()) {
        case cv::GRunArgP::index_of<cv::MediaFrame*>():
        {
            oak_frame = q->get<dai::ImgFrame>();
            // FIXME: hard-coded NV12
            *cv::util::get<cv::MediaFrame*>(out_arg) =
                    cv::MediaFrame::Create<cv::gapi::oak::OAKMediaAdapter>(
                            cv::Size(static_cast<int>(oak_frame->getWidth()),
                                     static_cast<int>(oak_frame->getHeight())),
                            cv::MediaFormat::NV12,
                            std::move(oak_frame->getData()));

            using namespace cv::gapi::streaming::meta_tag;
            meta[timestamp] = oak_frame->getTimestamp();
            meta[seq_id]    = oak_frame->getSequenceNum();

            break;
        }
        case cv::GRunArgP::index_of<cv::detail::VectorRef>():
        {
            oak_frame = q->get<dai::ImgFrame>();
            cv::util::get<cv::detail::VectorRef>(out_arg).wref<uint8_t>() = std::move(oak_frame->getData());

            using namespace cv::gapi::streaming::meta_tag;
            meta[timestamp] = oak_frame->getTimestamp();
            meta[seq_id]    = oak_frame->getSequenceNum();

            break;
        }
        case cv::GRunArgP::index_of<cv::RMat*>(): // only supported for infer
        {
            auto nn_data = q->get<dai::NNData>();

            auto out_layer_name = getDaiInferOutLayerName(m_oak_infer_info.begin()->second);
            auto in_out_tensor_info = parseDaiInferMeta(m_oak_infer_info.begin()->second);

            auto layer = std::move(nn_data->getLayerFp16(out_layer_name));

            // FIXME: add proper layout converter here
            GAPI_Assert(in_out_tensor_info.second.order ==
                        dai::TensorInfo::StorageOrder::NCHW);
            // FIMXE: only 1-channel data is supported for now
            GAPI_Assert(in_out_tensor_info.second.dims[2] == 1);

            *cv::util::get<cv::RMat*>(out_arg) =
                    cv::make_rmat<cv::gapi::oak::OAKRMatAdapter>(
                        cv::Size(in_out_tensor_info.second.dims[1],
                                 in_out_tensor_info.second.dims[0]),
                        CV_16F, // FIXME: cover other precisions
                        std::move(layer)
                    );

            using namespace cv::gapi::streaming::meta_tag;
            meta[timestamp] = nn_data->getTimestamp();
            meta[seq_id]    = nn_data->getSequenceNum();

            break;
        }
        // FIXME: Add support for remaining types
        default:
            GAPI_Assert(false && "Unsupported type in OAK backend");
        }

        out.meta(out_arg, meta);
        out.post(std::move(out_arg));
    }
}

namespace cv {
namespace gimpl {
namespace oak {

namespace {
static dai::VideoEncoderProperties::Profile convertEncProfile(cv::gapi::oak::EncoderConfig::Profile pf) {
    switch (pf) {
        case cv::gapi::oak::EncoderConfig::Profile::H264_BASELINE:
            return dai::VideoEncoderProperties::Profile::H264_BASELINE;
        case cv::gapi::oak::EncoderConfig::Profile::H264_HIGH:
            return dai::VideoEncoderProperties::Profile::H264_HIGH;
        case cv::gapi::oak::EncoderConfig::Profile::H264_MAIN:
            return dai::VideoEncoderProperties::Profile::H264_MAIN;
        case cv::gapi::oak::EncoderConfig::Profile::H265_MAIN:
            return dai::VideoEncoderProperties::Profile::H265_MAIN;
        case cv::gapi::oak::EncoderConfig::Profile::MJPEG:
            return dai::VideoEncoderProperties::Profile::MJPEG;
        default:
            // basically unreachable
            GAPI_Assert("Unsupported encoder profile");
            return {};
    }
}
} // anonymous namespace

// Kernels ///////////////////////////////////////////////////////////////

// FIXME: consider a better solution - hard-coded API
//        Is there a way to extract API from somewhereelse/utilize structs
//        like in streaming/infer backends (mainly infer and copy operations)
template<class Impl, class K, class InArgs = typename K::InArgs, class OutArgs = typename K::OutArgs>
class GOAKKernelImpl: public detail::OAKCallHelper<Impl, InArgs, OutArgs>
                    , public cv::detail::KernelTag {
    using P = detail::OAKCallHelper<Impl, InArgs, OutArgs>;
public:
    using API = K;
    static cv::gapi::GBackend   backend() { return cv::gapi::oak::backend();  }
    static GOAKKernel kernel()  { return GOAKKernel(&P::construct); }
};

#define GAPI_OAK_KERNEL(Name, API) \
    struct Name: public cv::gimpl::oak::GOAKKernelImpl<Name, API>

#define GAPI_OAK_FIXED_API_KERNEL(Name, API, InArgs, OutArgs) \
    struct Name: public cv::gimpl::oak::GOAKKernelImpl<Name, API, InArgs, OutArgs>

namespace {
GAPI_OAK_FIXED_API_KERNEL(GOAKInfer, cv::GInferBase, std::tuple<cv::GFrame>, std::tuple<cv::GMat>) {
    static std::shared_ptr<dai::Node> put(const cv::gimpl::OAKKernelParams& params,
                                          GOAKContext::InputPtr& in,
                                          GOAKContext::OutputPtr& out) {
        auto nn = params.pipeline->create<dai::node::NeuralNetwork>();

        nn->input.setBlocking(true);
        nn->input.setQueueSize(1);

        // FIXME: add G-API built-in preproc here (currently it's only setPreviewSize() on the camera node)
        // Note: for some reason currently it leads to:
        // "Fatal error. Please report to developers. Log: 'ImageManipHelper' '61'"

        nn->setBlobPath(params.infer_info.blob_file);

        in = &(nn->input);
        out = &(nn->out);

        return nn;
    }
};

GAPI_OAK_KERNEL(GOAKCopy, cv::gapi::oak::GCopy) {
    static std::shared_ptr<dai::Node> put(const cv::gimpl::OAKKernelParams&,
                                          GOAKContext::InputPtr&,
                                          GOAKContext::OutputPtr&) {
        // Do nothing in Copy OP since it's either already represented
        // by XLinkOut node (bonded to output queues) or it's a passthrough OP
        return nullptr;
    }
};

GAPI_OAK_KERNEL(GOAKEncFrame, cv::gapi::oak::GEncFrame) {
    static std::shared_ptr<dai::Node> put(const cv::gimpl::OAKKernelParams& params,
                                          GOAKContext::InputPtr& in,
                                          const cv::gapi::oak::EncoderConfig& cfg,
                                          GOAKContext::OutputPtr& out) {
        auto videoEnc = params.pipeline->create<dai::node::VideoEncoder>();

        // FIXME: convert all the parameters to dai
        videoEnc->setDefaultProfilePreset(cfg.frameRate,
                                          convertEncProfile(cfg.profile));

        in = &(videoEnc->input);
        out = &(videoEnc->bitstream);

        return videoEnc;
    }
};

GAPI_OAK_KERNEL(GOAKSobelXY, cv::gapi::oak::GSobelXY) {
    static std::shared_ptr<dai::Node> put(const cv::gimpl::OAKKernelParams& params,
                                          GOAKContext::InputPtr& in,
                                          const cv::Mat& hk,
                                          const cv::Mat& vk,
                                          GOAKContext::OutputPtr& out) {
        auto edgeDetector = params.pipeline->create<dai::node::EdgeDetector>();

        edgeDetector->setMaxOutputFrameSize(params.camera_size.width * params.camera_size.height);

        auto xinEdgeCfg = params.pipeline->create<dai::node::XLinkIn>();
        xinEdgeCfg->setStreamName("sobel_cfg");

        auto mat2vec = [&](cv::Mat m) {
            std::vector<std::vector<int>> v(m.rows);
            for (int i = 0; i < m.rows; ++i)
            {
                m.row(i).reshape(1,1).copyTo(v[i]);
            }
            return v;
        };

        dai::EdgeDetectorConfig cfg;
        cfg.setSobelFilterKernels(mat2vec(hk), mat2vec(vk));

        xinEdgeCfg->out.link(edgeDetector->inputConfig);

        params.in_queues.push_back({"sobel_cfg", cfg});

        in = &(edgeDetector->inputImage);
        out = &(edgeDetector->outputImage);

        return edgeDetector;
    }
};

} // anonymous namespace
} // namespace oak
} // namespace gimpl
} // namespace cv

class GOAKBackendImpl final : public cv::gapi::GBackend::Priv {
    virtual void unpackKernel(ade::Graph            &graph,
                              const ade::NodeHandle &op_node,
                              const cv::GKernelImpl &impl) override {
        using namespace cv::gimpl;

        OAKGraph gm(graph);

        const auto &kimpl  = cv::util::any_cast<GOAKKernel>(impl.opaque);
        gm.metadata(op_node).set(OAKComponent{kimpl});

        // Set custom meta for infer
        if (gm.metadata(op_node).contains<cv::gimpl::NetworkParams>()) {
            gm.metadata(op_node).set(CustomMetaFunction{customOutMeta});
        }
    }

    virtual EPtr compile(const ade::Graph &graph,
                         const cv::GCompileArgs &args,
                         const std::vector<ade::NodeHandle> &nodes,
                         const std::vector<cv::gimpl::Data>& ins_data,
                         const std::vector<cv::gimpl::Data>& outs_data) const override {
        cv::gimpl::GModel::ConstGraph gm(graph);
        // FIXME: pass streaming/non-streaming option to support non-camera case
        // NB: how could we have non-OAK source in streaming mode, then OAK backend in
        //     streaming mode but without camera input?
        if (!gm.metadata().contains<cv::gimpl::Streaming>()) {
            GAPI_Assert(false && "OAK backend only supports Streaming mode for now");
        }
        return EPtr{new cv::gimpl::GOAKExecutable(graph, args, nodes, ins_data, outs_data)};
    }

    virtual cv::GKernelPackage auxiliaryKernels() const override {
        return cv::gapi::kernels< cv::gimpl::oak::GOAKInfer
                                >();
    }
};

cv::gapi::GBackend cv::gapi::oak::backend() {
    static cv::gapi::GBackend this_backend(std::make_shared<GOAKBackendImpl>());
    return this_backend;
}

namespace cv {
namespace gapi {
namespace oak {

cv::gapi::GKernelPackage kernels() {
    return cv::gapi::kernels< cv::gimpl::oak::GOAKEncFrame
                            , cv::gimpl::oak::GOAKSobelXY
                            , cv::gimpl::oak::GOAKCopy
                            >();
}

} // namespace oak
} // namespace gapi
} // namespace cv

#else

namespace cv {
namespace gapi {
namespace oak {

cv::gapi::GKernelPackage kernels() {
    GAPI_Assert(false && "Built without OAK support");
    return {};
}

cv::gapi::GBackend backend() {
    GAPI_Assert(false && "Built without OAK support");
    static cv::gapi::GBackend this_backend(nullptr);
    return this_backend;
}

} // namespace oak
} // namespace gapi
} // namespace cv

#endif // HAVE_OAK
