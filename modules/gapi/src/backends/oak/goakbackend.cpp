// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

#ifdef WITH_OAK_BACKEND

#include <cstring>
#include <unordered_set>

#include <api/gbackend_priv.hpp>
#include <backends/common/gbackend.hpp>

#include "depthai/depthai.hpp"

#include <opencv2/gapi/oak/oak.hpp>
#include "oak_media_adapter.hpp"

namespace cv { namespace gimpl {

// Forward declaration
class GOAKContext;

class GOAKExecutable final: public GIslandExecutable {
    friend class GOAKContext;
    virtual void run(std::vector<InObj>&&,
                     std::vector<OutObj>&&) override {
        GAPI_Assert(false && "Not implemented");
    }

    virtual void run(GIslandExecutable::IInput &in,
                     GIslandExecutable::IOutput &out) override;

    void LinkToParentHelper(ade::NodeHandle handle,
                            const std::vector<ade::NodeHandle>& nodes);

    class ExtractTypeHelper : protected dai::Node {
    public:
        using Input = dai::Node::Input;
        using Output = dai::Node::Output;
        using InputPtr = dai::Node::Input*;
        using OutputPtr = dai::Node::Output*;
    };

    cv::GArg packArg(const GArg &arg, std::vector<ExtractTypeHelper::InputPtr>& oak_ins,
                     std::vector<cv::GArg>& oak_wrapped_args);
    void outArg(const RcDesc &rc, std::vector<ExtractTypeHelper::OutputPtr>& oak_outs);

    const ade::Graph& m_g;
    GModel::ConstGraph m_gm;
    cv::GCompileArgs m_args;

    std::unordered_map<ade::NodeHandle,
                       std::shared_ptr<dai::Node>,
                       ade::HandleHasher<ade::Node>> m_oak_nodes;
    std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>> m_processed_nodes;
    // Will be reworked later when XLinkIn will be introduced as input
    std::shared_ptr<dai::node::ColorCamera> m_camera_input;
    std::tuple<int, int> m_camera_wh;

    std::unordered_map<ade::NodeHandle,
                       std::vector<ExtractTypeHelper::InputPtr>,
                       ade::HandleHasher<ade::Node>> m_oak_node_inputs;
    std::unordered_map<ade::NodeHandle,
                       std::vector<ExtractTypeHelper::OutputPtr>,
                       ade::HandleHasher<ade::Node>> m_oak_node_outputs;

    // Backend outputs
    std::vector<std::shared_ptr<dai::node::XLinkOut>> m_xlink_outputs;
    std::vector<std::shared_ptr<dai::DataOutputQueue>> m_out_queues;
    std::vector<std::string> m_out_queue_names;

    // Backend inputs
    std::vector<std::pair<std::string, dai::Buffer>> m_in_queues;

    // Note: dai::Pipeline should be the only one for the whole pipeline,
    // so there is no way to insert any non-OAK node in graph between other OAK nodes.
    // The only heterogeneous case possible is if we insert other backends after or before
    // OAK island.
    std::unique_ptr<dai::Device> m_device;
    std::unique_ptr<dai::Pipeline> m_pipeline;

    std::vector<cv::GArg> m_oak_wrapped_args;

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
        util::throw_error
            (std::logic_error
             ("GOAKExecutable::reshape() is not supported"));
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
                const std::tuple<int, int>& camera_size,
                std::vector<cv::GArg>& args,
                std::vector<OutputPtr>& results);

    // Generic accessor API
    template<typename T>
    T& inArg(int input) { return m_args.at(input).get<T>(); }

    // FIXME: consider not using raw pointers
    InputPtr* in(int input);
    OutputPtr* out(int output);

    const std::unique_ptr<dai::Pipeline>& pipeline();
    const std::tuple<int, int>& camera_size() const;

private:
    const std::unique_ptr<dai::Pipeline>& m_pipeline;
    const std::tuple<int, int>& m_camera_size;
    std::vector<cv::GArg>& m_args;
    std::vector<OutputPtr>& m_outputs;
};

GOAKContext::GOAKContext(const std::unique_ptr<dai::Pipeline>& pipeline,
                         const std::tuple<int, int>& camera_size,
                         std::vector<cv::GArg>& args,
                         std::vector<OutputPtr>& results)
    : m_pipeline(pipeline), m_camera_size(camera_size), m_args(args), m_outputs(results) {}

const std::unique_ptr<dai::Pipeline>& GOAKContext::pipeline() {
    return m_pipeline;
}

const std::tuple<int, int>& GOAKContext::camera_size() const {
    return m_camera_size;
}

GOAKContext::InputPtr* GOAKContext::in(int input) {
    return inArg<GOAKContext::InputPtr*>(input);
}

GOAKContext::OutputPtr* GOAKContext::out(int output) {
    return &(m_outputs.at(output));
}

namespace detail {
template<class T> struct get_in;
template<> struct get_in<cv::GFrame> {
    static GOAKContext::InputPtr* get(GOAKContext &ctx, int idx) { return ctx.in(idx); }
};
template<class T> struct get_in {
    static T get(GOAKContext &ctx, int idx) { return ctx.inArg<T>(idx); }
};
// FIXME: add support of other types

template<class T> struct get_out;
template<> struct get_out<cv::GFrame> {
    static GOAKContext::OutputPtr* get(GOAKContext &ctx, int idx) { return ctx.out(idx); }
};
template<typename U> struct get_out<cv::GArray<U>> {
    static GOAKContext::OutputPtr* get(GOAKContext &ctx, int idx) { return ctx.out(idx); }
};
// FIXME: add support of other types

template<typename, typename, typename>
struct OAKCallHelper;

template<typename Impl, typename... Ins, typename... Outs>
struct OAKCallHelper<Impl, std::tuple<Ins...>, std::tuple<Outs...> > {
    template<int... IIs, int... OIs>
    static void construct_impl(  GOAKContext &ctx
                               , std::shared_ptr<dai::Node>& node
                               , std::vector<std::pair<std::string, dai::Buffer>>& in_queues_params
                               , cv::detail::Seq<IIs...>
                               , cv::detail::Seq<OIs...>) {
        Impl::put(ctx.pipeline(),
                  ctx.camera_size(),
                  node,
                  in_queues_params,
                  get_in<Ins>::get(ctx, IIs)...,
                  get_out<Outs>::get(ctx, OIs)...);
    }

    static void construct(GOAKContext &ctx,
                          std::shared_ptr<dai::Node>& node,
                          std::vector<std::pair<std::string, dai::Buffer>>& in_queues_params) {
        construct_impl(ctx,
                       node,
                       in_queues_params,
                       typename cv::detail::MkSeq<sizeof...(Ins)>::type(),
                       typename cv::detail::MkSeq<sizeof...(Outs)>::type());
    }
};

} // namespace detail

struct GOAKKernel {
    using F = std::function<void(GOAKContext&,
                                 std::shared_ptr<dai::Node>&,
                                 std::vector<std::pair<std::string, dai::Buffer>>&)>;
    explicit GOAKKernel(const F& f) : m_f(f) {}
    const F m_f;
};

struct OAKComponent
{
    static const char *name() { return "OAK Component"; }
    GOAKKernel k;
};

}} // namespace gimpl // namespace cv

using OAKGraph = ade::TypedGraph
    < cv::gimpl::OAKComponent
    // FIXME: extend
    >;

using ConstOAKGraph = ade::ConstTypedGraph
    < cv::gimpl::OAKComponent
    // FIXME: extend
    >;

void cv::gimpl::GOAKExecutable::LinkToParentHelper(ade::NodeHandle handle,
                                                   const std::vector<ade::NodeHandle>& nodes)
{
    ade::NodeHandle parent;
    for (const auto& indatah : handle.get()->inNodes()) {
        // indatah - node's input data
        // need to find which other node produces that data
        for (const auto& nhp : nodes) {
            if (m_gm.metadata(nhp).get<NodeType>().t == NodeType::OP) {
                for (const auto& outdatah : nhp.get()->outNodes()) {
                    if (indatah == outdatah) {
                        parent = nhp;
                    }
                }
            }
        }
    }
    GAPI_Assert(m_oak_nodes.find(parent) != m_oak_nodes.end());
    GAPI_Assert(m_oak_node_inputs[handle].size() ==
                m_oak_node_outputs[parent].size());
    for (size_t i = 0; i < m_oak_node_inputs[handle].size(); ++i) {
        m_oak_node_outputs[parent][i]->link(*(m_oak_node_inputs[handle][i]));
    }
}

cv::GArg
cv::gimpl::GOAKExecutable::packArg(const GArg &arg,
                                   std::vector<ExtractTypeHelper::InputPtr>& oak_ins,
                                   std::vector<cv::GArg>& oak_wrapped_args) {
    if (arg.kind != cv::detail::ArgKind::GOBJREF) {
        GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
                    && arg.kind != cv::detail::ArgKind::GSCALAR
                    && arg.kind != cv::detail::ArgKind::GARRAY
                    && arg.kind != cv::detail::ArgKind::GOPAQUE
                    && arg.kind != cv::detail::ArgKind::GFRAME);
        // All other cases - pass as-is, with no transformations to
        // GArg contents.
        return arg;
    }
    const cv::gimpl::RcDesc &ref = arg.get<cv::gimpl::RcDesc>();
    switch (ref.shape) {
    case GShape::GFRAME:
        oak_ins.push_back(nullptr);
        oak_wrapped_args.push_back(GArg(&(oak_ins.back())));
        return oak_wrapped_args.back();
        break;
    default:
        util::throw_error(std::logic_error("Unsupported GShape type in OAK backend"));
        break;
    }
}

void cv::gimpl::GOAKExecutable::outArg(const RcDesc &rc,
                                       std::vector<ExtractTypeHelper::OutputPtr>& oak_outs) {
    switch (rc.shape) {
    case GShape::GFRAME:
        oak_outs.push_back(nullptr);
        break;
    case GShape::GARRAY:
        oak_outs.push_back(nullptr);
        break;
    default:
        util::throw_error(std::logic_error("Unsupported GShape type in OAK backend"));
        break;
    }
}

cv::gimpl::GOAKExecutable::GOAKExecutable(const ade::Graph& g,
                                          const cv::GCompileArgs &args,
                                          const std::vector<ade::NodeHandle>& nodes,
                                          const std::vector<cv::gimpl::Data>& ins_data,
                                          const std::vector<cv::gimpl::Data>& outs_data)
    : m_g(g), m_gm(m_g), m_args(args),
      m_device(nullptr), m_pipeline(new dai::Pipeline)
    {
        // FIXME: change the hard-coded behavior (XLinkIn path)
        auto camRgb = m_pipeline->create<dai::node::ColorCamera>();
        // FIXME: extract camera compile arguments here and properly convert them for dai
        camRgb->setBoardSocket(dai::CameraBoardSocket::RGB);
        camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);

        // Set camera output. Fixme: consider working with other camera outputs
        m_camera_input = camRgb;
        // FIXME: change when other camera censors are introduced
        m_camera_wh = camRgb->getVideoSize();

        // Prepare XLinkOut nodes for each output object in graph
        for (size_t i = 0; i < outs_data.size(); ++i) {
            auto xout = m_pipeline->create<dai::node::XLinkOut>();
            std::string xout_name = "xout" + std::to_string(i);
            m_out_queue_names.push_back(xout_name);
            xout->setStreamName(xout_name);
            m_xlink_outputs.push_back(xout);
        }

        // Create OAK node for each node in this backend
        for (const auto& nh : nodes) {
            if (m_gm.metadata(nh).get<NodeType>().t == NodeType::OP) {
                const auto& op = m_gm.metadata(nh).get<Op>();
                const auto &u = ConstOAKGraph(m_g).metadata(nh).get<OAKComponent>();
                // pass kernel input args and compile args to prepare OAK node and
                // store it to link later
                m_oak_nodes[nh] = nullptr;
                m_oak_node_inputs[nh] = {};
                m_oak_node_outputs[nh] = {};

                std::vector<cv::GArg> in_ctx_args;
                in_ctx_args.reserve(op.args.size());
                for (auto &op_arg : op.args) in_ctx_args.push_back(packArg(op_arg,
                                                                           m_oak_node_inputs[nh],
                                                                           m_oak_wrapped_args));
                for (auto &&op_out : op.outs) outArg(op_out, m_oak_node_outputs[nh]);
                GAPI_Assert(!m_oak_node_inputs[nh].empty());
                GAPI_Assert(!m_oak_node_outputs[nh].empty());

                GOAKContext ctx(m_pipeline, m_camera_wh, in_ctx_args, m_oak_node_outputs[nh]);
                u.k.m_f(ctx, m_oak_nodes[nh], m_in_queues);
                GAPI_Assert(m_oak_nodes[nh] != nullptr);
            }
        }

        for (const auto& el : m_oak_node_inputs) {
            for (const auto& in : el.second) {
                GAPI_Assert(in != nullptr);
            }
        }

        for (const auto& el : m_oak_node_outputs) {
            for (const auto& out : el.second) {
                GAPI_Assert(out != nullptr);
            }
        }

        // Properly link all nodes
        // 1. Link input nodes to camera
        for (const auto& d : ins_data)
        {
            for (const auto& nh : nodes)
            {
                if (m_gm.metadata(nh).contains<cv::gimpl::Op>())
                {
                    for (const auto& indata : nh.get()->inNodes()) {
                        auto rc = m_gm.metadata(indata).get<cv::gimpl::Data>().rc;
                        if (rc == d.rc)
                        {
                            GAPI_Assert(m_oak_nodes.find(nh) != m_oak_nodes.end());
                            GAPI_Assert(m_oak_node_inputs[nh].size() == 1);
                            // FIXME: covert other camera outputs
                            m_camera_input->video.link(*(m_oak_node_inputs[nh][0]));
                            m_processed_nodes.insert(nh);
                        }
                    }
                }
            }
        }

        // 2. Link output nodes to XLinkOut nodes
        size_t out_counter = 0;
        for (const auto& d : outs_data)
        {
            for (const auto& nh : nodes)
            {
                if (m_gm.metadata(nh).contains<cv::gimpl::Op>())
                {
                    for (const auto& outdata : nh.get()->outNodes()) {
                        auto rc = m_gm.metadata(outdata).get<cv::gimpl::Data>().rc;
                        if (rc == d.rc)
                        {
                            GAPI_Assert(m_oak_nodes.find(nh) != m_oak_nodes.end());
                            GAPI_Assert(m_oak_node_outputs[nh].size() == 1);
                            GAPI_Assert(out_counter < m_xlink_outputs.size());
                            m_oak_node_outputs[nh][0]->link(m_xlink_outputs[out_counter++]->input);

                            if (m_processed_nodes.find(nh) == m_processed_nodes.end()) {
                                LinkToParentHelper(nh, nodes);
                                m_processed_nodes.insert(nh);
                            }
                        }
                    }
                }
            }
        }

        // 3. Link internal nodes to their parents
        for (const auto& nh : nodes) {
            if (m_gm.metadata(nh).get<NodeType>().t == NodeType::OP) {
                GAPI_Assert(m_oak_nodes.find(nh) != m_oak_nodes.end());
                if (m_processed_nodes.find(nh) == m_processed_nodes.end()) {
                    LinkToParentHelper(nh, nodes);
                    m_processed_nodes.insert(nh);
                }
            }
        }

        m_device = std::unique_ptr<dai::Device>(new dai::Device(*m_pipeline));

        // Prepare all output queues
        for (size_t i = 0; i < outs_data.size(); ++i) {
            // FIXME: add queue parameters
            m_out_queues.push_back(m_device->getOutputQueue(m_out_queue_names[i], 30, true));
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

    for (size_t i = 0; i < m_in_queues.size(); ++i) {
        auto q = m_device->getInputQueue(m_in_queues[i].first);
        q->send(m_in_queues[i].second);
    }

    for (size_t i = 0; i < m_out_queues.size(); ++i) {
        auto q = m_out_queues[i];
        auto oak_frame = q->get<dai::ImgFrame>();

        auto out_arg = out.get(i);

        if (util::holds_alternative<cv::detail::VectorRef>(out_arg)) {
            cv::util::get<cv::detail::VectorRef>(out_arg).wref<uint8_t>() = oak_frame->getData();
            // FIXME: should we copy data instead?
        } else if (util::holds_alternative<cv::MediaFrame*>(out_arg)) {
            // FIXME: hard-coded NV12
            *cv::util::get<cv::MediaFrame*>(out_arg) =
                    cv::MediaFrame::Create<cv::gapi::oak::OAKMediaAdapter>(
                            cv::Size(static_cast<int>(oak_frame->getWidth()),
                                     static_cast<int>(oak_frame->getHeight())),
                            cv::gapi::oak::OAKFrameFormat::NV12,
                            oak_frame->getData().data(),
                            oak_frame->getData().data() + static_cast<long>(oak_frame->getData().size() / 3 * 2));
            // FIXME: should we copy data instead?
        } else {
            GAPI_Assert(false && "Unsupported output type");
        }

        out.meta(out_arg, cv::util::get<cv::GRunArgs>(in_msg)[i].meta);
        out.post(std::move(out_arg));
    }
}

// Built-in kernels for OAK /////////////////////////////////////////////////////

class GOAKBackendImpl final : public cv::gapi::GBackend::Priv {
    virtual void unpackKernel(ade::Graph            &graph,
                              const ade::NodeHandle &op_node,
                              const cv::GKernelImpl &impl) override {
        OAKGraph gm(graph);

        const auto &kimpl  = cv::util::any_cast<cv::gimpl::GOAKKernel>(impl.opaque);
        gm.metadata(op_node).set(cv::gimpl::OAKComponent{kimpl});
    }

    virtual EPtr compile(const ade::Graph &graph,
                         const cv::GCompileArgs &args,
                         const std::vector<ade::NodeHandle> &nodes,
                         const std::vector<cv::gimpl::Data>& ins_data,
                         const std::vector<cv::gimpl::Data>& outs_data) const override {
        return EPtr{new cv::gimpl::GOAKExecutable(graph, args, nodes, ins_data, outs_data)};
    }
};

cv::gapi::GBackend cv::gapi::oak::backend() {
    static cv::gapi::GBackend this_backend(std::make_shared<GOAKBackendImpl>());
    return this_backend;
}

namespace cv {
namespace gimpl {
namespace oak {

// Kernels ///////////////////////////////////////////////////////////////

template<class Impl, class K>
class GOAKKernelImpl: public detail::OAKCallHelper<Impl, typename K::InArgs, typename K::OutArgs>
                    , public cv::detail::KernelTag {
    using P = detail::OAKCallHelper<Impl, typename K::InArgs, typename K::OutArgs>;
public:
    using API = K;
    static cv::gapi::GBackend   backend() { return cv::gapi::oak::backend();  }
    static GOAKKernel kernel()  { return GOAKKernel(&P::construct); }
};

#define GAPI_OAK_KERNEL(Name, API) \
    struct Name: public cv::gimpl::oak::GOAKKernelImpl<Name, API>

namespace {
GAPI_OAK_KERNEL(GOAKEncFrame, cv::gapi::oak::GEncFrame) {
    static void put(const std::unique_ptr<dai::Pipeline>& pipeline,
                    const std::tuple<int, int>&,
                    std::shared_ptr<dai::Node>& node,
                    std::vector<std::pair<std::string, dai::Buffer>>&,
                    GOAKContext::InputPtr* in,
                    const cv::gapi::oak::EncoderConfig& cfg,
                    GOAKContext::OutputPtr* out) {
        auto videoEnc = pipeline->create<dai::node::VideoEncoder>();

        // FIXME: convert all the parameters to dai
        videoEnc->setDefaultProfilePreset(cfg.width, cfg.height,
                                          cfg.frameRate,
                                          dai::VideoEncoderProperties::Profile::H265_MAIN);
        node = videoEnc;
        *in = &(videoEnc->input);
        *out = &(videoEnc->bitstream);
    }
};

GAPI_OAK_KERNEL(GOAKSobelXY, cv::gapi::oak::GSobelXY) {
    static void put(const std::unique_ptr<dai::Pipeline>& pipeline,
                    const std::tuple<int, int>& camera_wh,
                    std::shared_ptr<dai::Node>& node,
                    std::vector<std::pair<std::string, dai::Buffer>>& m_in_queues,
                    GOAKContext::InputPtr* in,
                    const std::vector<std::vector<int>>& hk,
                    const std::vector<std::vector<int>>& vk,
                    GOAKContext::OutputPtr* out) {
        auto edgeDetector = pipeline->create<dai::node::EdgeDetector>();

        edgeDetector->setMaxOutputFrameSize(std::get<0>(camera_wh) * std::get<1>(camera_wh));

        auto xinEdgeCfg = pipeline->create<dai::node::XLinkIn>();
        xinEdgeCfg->setStreamName("sobel_cfg");

        dai::EdgeDetectorConfig cfg;
        cfg.setSobelFilterKernels(hk, vk);

        xinEdgeCfg->out.link(edgeDetector->inputConfig);

        m_in_queues.push_back({"sobel_cfg", cfg});

        node = edgeDetector;
        *in = &(edgeDetector->inputImage);
        *out = &(edgeDetector->outputImage);
    }
};
} // anonymous namespace

cv::gapi::GKernelPackage kernels();

cv::gapi::GKernelPackage kernels() {
    return cv::gapi::kernels< GOAKEncFrame
                            , GOAKSobelXY
                            >();
}

} // namespace oak
} // namespace gimpl
} // namespace cv

#else

namespace cv {
namespace gimpl {
namespace oak {

cv::gapi::GKernelPackage kernels();

cv::gapi::GKernelPackage kernels() {
    GAPI_Assert(false && "Built without OAK support");
    return {};
}

} // namespace oak
} // namespace gimpl
} // namespace cv

#endif // WITH_OAK_BACKEND
