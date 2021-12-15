// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef WITH_OAK_BACKEND

#include <cstring>
#include <unordered_set>

#include <api/gbackend_priv.hpp>
#include <backends/common/gbackend.hpp>

#include "depthai/depthai.hpp"

#include <opencv2/gapi/oak/oak.hpp>
#include "oak_media_adapter.hpp"

namespace cv { namespace gimpl {

class GOAKExecutable final: public GIslandExecutable {
    virtual void run(std::vector<InObj>&&,
                     std::vector<OutObj>&&) override {
        GAPI_Assert(false && "Not implemented");
    }

    virtual void run(GIslandExecutable::IInput &in,
                     GIslandExecutable::IOutput &out) override;

    void LinkToParentHelper(ade::NodeHandle handle,
                            const std::vector<ade::NodeHandle>& nodes);

    const ade::Graph& m_g;
    GModel::ConstGraph m_gm;
    cv::GCompileArgs m_args;

    std::unordered_map<ade::NodeHandle,
                       std::shared_ptr<dai::Node>,
                       ade::HandleHasher<ade::Node>> m_oak_nodes;
    std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>> m_processed_nodes;
    // Will be reworked later when XLinkIn will be introduced as input
    std::shared_ptr<dai::node::ColorCamera> m_camera_input;

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

struct GOAKKernel {
    using F = std::function<void(const std::unique_ptr<dai::Pipeline>&,
                                 const GArgs&,
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
    GAPI_Assert(m_oak_nodes[handle]->getInputs().size() ==
                m_oak_nodes[parent]->getOutputs().size());
    for (size_t i = 0; i < m_oak_nodes[handle]->getInputs().size(); ++i) {
        m_oak_nodes[parent]->getOutputs()[i].link(m_oak_nodes[handle]->getInputs()[i]);
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
                u.k.m_f(m_pipeline, op.args, m_oak_nodes[nh], m_in_queues);
                GAPI_Assert(m_oak_nodes[nh] != nullptr);
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
                            GAPI_Assert(m_oak_nodes[nh]->getInputs().size() == 1);
                            m_camera_input->video.link(m_oak_nodes[nh]->getInputs()[0]);
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
                            GAPI_Assert(m_oak_nodes[nh]->getOutputs().size() == 1);
                            GAPI_Assert(out_counter < m_xlink_outputs.size());
                            m_oak_nodes[nh]->getOutputs()[0].link(m_xlink_outputs[out_counter++]->input);

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
    const auto  in_msg = in.get();

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
        } else if (util::holds_alternative<cv::MediaFrame*>(out_arg)) {
            *cv::util::get<cv::MediaFrame*>(out_arg) =
                    cv::MediaFrame::Create<cv::gapi::oak::OAKMediaAdapter>(
                            cv::Size(static_cast<int>(oak_frame->getWidth()),
                                     static_cast<int>(oak_frame->getHeight())),
                            cv::gapi::oak::OAKFrameFormat::BGR,
                            oak_frame->getData().data());
        } else {
            GAPI_Assert(false && "Unsupported output type");
        }

        // FIXME: do we need to pass meta here?
        //out.meta(out_arg, {});
        //out.post(std::move(out_arg));

        out.meta(out_arg, cv::util::get<cv::GRunArgs>(in_msg)[0].meta);
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

namespace detail {

template<typename>
struct OAKCallHelper;

template<typename Impl>
struct OAKCallHelper {
    static void construct(const std::unique_ptr<dai::Pipeline>& pipeline,
                          const GArgs& in_args,
                          std::shared_ptr<dai::Node>& node,
                          std::vector<std::pair<std::string, dai::Buffer>>& m_in_queues) {
        Impl::put(pipeline, in_args, node, m_in_queues);
    }
};

} // namespace detail

// Kernels ///////////////////////////////////////////////////////////////

template<class Impl, class K>
class GOAKKernelImpl: public detail::OAKCallHelper<Impl>
                    , public cv::detail::KernelTag {
    using P = detail::OAKCallHelper<Impl>;
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
                    const GArgs& in_args,
                    std::shared_ptr<dai::Node>& node,
                    std::vector<std::pair<std::string, dai::Buffer>>&) {
        auto videoEnc = pipeline->create<dai::node::VideoEncoder>();
        // FIXME: encoder params is the 2nd arg - consider a better approach here
        auto m_enc_config = in_args[1].get<cv::gapi::oak::EncoderConfig>();
        // FIXME: convert all the parameters to dai
        videoEnc->setDefaultProfilePreset(m_enc_config.width, m_enc_config.height,
                                          m_enc_config.frameRate,
                                          dai::VideoEncoderProperties::Profile::H265_MAIN);
        node = videoEnc;
    }
};

GAPI_OAK_KERNEL(GOAKSobelXY, cv::gapi::oak::GSobelXY) {
    static void put(const std::unique_ptr<dai::Pipeline>& pipeline,
                    const GArgs& in_args,
                    std::shared_ptr<dai::Node>& node,
                    std::vector<std::pair<std::string, dai::Buffer>>& m_in_queues) {
        auto edgeDetector = pipeline->create<dai::node::EdgeDetector>();

        auto xinEdgeCfg = pipeline->create<dai::node::XLinkIn>();
        xinEdgeCfg->setStreamName("sobel_cfg");

        dai::EdgeDetectorConfig cfg;
        // FIXME: sobel params is the 2nd and 3rd args - consider a better approach here
        auto shk = in_args[1].get<std::vector<std::vector<int>>>();
        auto svk = in_args[2].get<std::vector<std::vector<int>>>();

        cfg.setSobelFilterKernels(shk, svk);

        xinEdgeCfg->out.link(edgeDetector->inputConfig);

        node = edgeDetector;
        m_in_queues.push_back({"sobel_cfg", cfg});
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

#endif // WITH_OAK_BACKEND
