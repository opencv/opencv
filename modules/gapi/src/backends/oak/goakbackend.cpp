// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef WITH_OAK_BACKEND

#include <cstring>

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

    const ade::Graph& m_g;
    GModel::ConstGraph m_gm;
    cv::GCompileArgs m_args;

    std::unordered_map<ade::NodeHandle,
                       std::shared_ptr<dai::Node>,
                       ade::HandleHasher<ade::Node>> m_oak_nodes;
    // Will be reworked later when XLinkIn will be introduced as input
    std::shared_ptr<dai::Node> m_camera_input;

    // Backend outputs
    std::vector<std::shared_ptr<dai::Node>> m_xlink_outputs;
    std::vector<std::shared_ptr<dai::DataOutputQueue>> m_out_queues;
    std::vector<std::string> m_out_queue_names;

    // Note: dai::Pipeline should be the only one for the whole pipeline,
    // so there is no way to insert any non-OAK node in graph between other OAK nodes.
    // The only heterogeneous case possible is if we insert other backends after or before
    // OAK island.
    std::unique_ptr<dai::Device> m_device;
    std::unique_ptr<dai::Pipeline> m_pipeline;

    cv::gapi::oak::EncoderConfig m_enc_config;

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
    using F = std::function<void(const std::unique_ptr<dai::Pipeline>&, const GArgs&, const GCompileArgs&)>;
    explicit GOAKKernel(const F& f) : m_f(f) {}
    const F m_f;
    std::shared_ptr<dai::Node> m_oak_node;
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

cv::gimpl::GOAKExecutable::GOAKExecutable(const ade::Graph& g,
                                          const cv::GCompileArgs &args,
                                          const std::vector<ade::NodeHandle>& nodes,
                                          const std::vector<cv::gimpl::Data>& /*ins_data*/,
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
            // FIXME: consider a better solution
            if (m_gm.metadata(nh).get<NodeType>().t == NodeType::OP) {
                const auto& op = m_gm.metadata(nh).get<Op>();

                    //auto videoEnc = m_pipeline->create<dai::node::VideoEncoder>();
                    // FIXME: encoder params is the 2nd arg - consider a better approach here
                    //m_enc_config = op.args[1].get<cv::gapi::oak::EncoderConfig>();
                    // FIXME: convert all the parameters to dai
                    //videoEnc->setDefaultProfilePreset(m_enc_config.width, m_enc_config.height,
                    //                                  m_enc_config.frameRate,
                    //                                  dai::VideoEncoderProperties::Profile::H264_MAIN);
                    // FIXME: think about proper linking:
                    // probably, firts need to link in nodes to camera, then
                    // for each non-input node check their in edges, get dai elements and link it there
                    //camRgb->video.link(videoEnc->input);

                    // FIXME: think about proper linking:
                    //videoEnc->bitstream.link(xout->input);

                    const auto &u = ConstOAKGraph(m_g).metadata(nh).get<OAKComponent>();
                    u.k.m_f(m_pipeline, op.args, args); // pass kernel input args and compile args to prepare OAK node
                    m_oak_nodes[nh] = u.k.m_oak_node;   // store OAK node to link it later
            }
        }

        // Properly link all nodes
        // TODO!

        m_device = std::unique_ptr<dai::Device>(new dai::Device(*m_pipeline));

        // Prepare all output queues
        for (size_t i = 0; i < outs_data.size(); ++i) {
            // FIXME: add queue parameters
            m_out_queues.push_back(m_device->getOutputQueue(m_out_queue_names[i], 30, true));
        }
    }

void cv::gimpl::GOAKExecutable::handleNewStream() {
    // FIXME: implement
    GAPI_Assert(false && "Not implemented");
}

void cv::gimpl::GOAKExecutable::handleStopStream() {
    // FIXME: implement
    GAPI_Assert(false && "Not implemented");
}

void cv::gimpl::GOAKExecutable::run(GIslandExecutable::IInput  &in,
                                    GIslandExecutable::IOutput &out) {
    if (cv::util::holds_alternative<cv::gimpl::EndOfStream>(in.get())) {
        out.post(cv::gimpl::EndOfStream{});
    }

    // FIXME: consider a better solution
    /*auto packet = m_out_queue->get<dai::ImgFrame>();

    // FIXME: cover all outputs
    auto out_arg = out.get(0);

    // Encoder case
    if (util::holds_alternative<cv::detail::VectorRef>(out_arg)) {
        /*
        *cv::util::get<cv::MediaFrame*>(out_arg) = cv::MediaFrame::Create<cv::gapi::oak::OAKMediaAdapter>();
        auto frame = cv::util::get<MediaFrame*>(out_arg);
        auto adapter = frame->get<cv::gapi::oak::OAKMediaAdapter>();
        adapter->setParams(cv::Size(static_cast<int>(packet->getWidth()),
                                    static_cast<int>(packet->getHeight())),
                            cv::gapi::oak::OAKFrameFormat::BGR,
                            packet->getData().data(),
                            packet->getData().size());

        // FIXME: do we need to pass meta here?
        out.meta(out_arg, {});
        out.post(std::move(out_arg));

    } else {
        util::throw_error(std::logic_error("Expected GArray at the end of the OAK pipeline"));
    }*/
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
                          const GCompileArgs& comp_args) {
        Impl::put(pipeline, in_args, comp_args);
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
    static GBackend   backend() { return cv::gapi::oak::backend(); }
    static GOAKKernel kernel()  { return GOAKKernel(&P::construct);     }
};

#define GAPI_OAK_KERNEL(Name, API) \
    struct Name: public cv::gimpl::oak::GOAKKernelImpl<Name, API>

/*
struct Encode: public cv::detail::KernelTag {
    using API = cv::gapi::oak::GEnc;
    static cv::gapi::GBackend backend() { return cv::gapi::oak::backend(); }
    static GOAKKernel          kernel() { return GOAKKernel{}; }
};*/

cv::gapi::GKernelPackage kernels();
cv::gapi::GKernelPackage kernels() {
    return cv::gapi::kernels< //cv::gimpl::oak::Encode
                            >();
}

} // namespace oak
} // namespace gimpl
} // namespace cv

#endif // WITH_OAK_BACKEND
