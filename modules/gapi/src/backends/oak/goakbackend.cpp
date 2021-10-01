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
#include <opencv2/gapi/oak/oak_media_adapter.hpp>

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

    dai::Pipeline m_pipeline;
    dai::Device m_device;

    std::string m_out_queue_name;
    std::shared_ptr<dai::DataOutputQueue> m_out_queue;

    cv::gapi::oak::EncoderConfig m_enc_config;

public:
    GOAKExecutable(const ade::Graph& g,
                   const cv::GCompileArgs &args,
                   const std::vector<ade::NodeHandle> &nodes,
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

struct GOAKKernel{
    // FIXME: extend
    GOAKKernel() = default;
};

struct OAKComponent
{
    static const char *name() { return "OAK Component"; }
    GOAKKernel k;
};

}} // namespace gimpl // namespace cv

using OAKGraph = ade::TypedGraph
    < cv::gimpl::NetworkParams       // opaque structure, assigned by G-API
    , cv::gimpl::Op
    , cv::gimpl::CustomMetaFunction  // custom meta function expected by G-API
    , cv::gimpl::OAKComponent
    // FIXME: extend
    >;

using ConstOAKGraph = ade::ConstTypedGraph
    < cv::gimpl::NetworkParams
    , cv::gimpl::Op
    , cv::gimpl::CustomMetaFunction
    , cv::gimpl::OAKComponent
    // FIXME: extend
    >;

cv::gimpl::GOAKExecutable::GOAKExecutable(const ade::Graph& g,
                                          const cv::GCompileArgs &args,
                                          const std::vector<ade::NodeHandle>& nodes,
                                          const std::vector<cv::gimpl::Data>& /*ins_data*/,
                                          const std::vector<cv::gimpl::Data>& /*outs_data*/)
    : m_g(g), m_gm(m_g), m_args(args),
    // FIXME: consider a better solution
    // We need to get dai::Device first to set dai::DataOutputQueue
    m_device([this, nodes]() {
        // FIXME: change the hard-coded behavior (XLinkIn path)
        auto camRgb = m_pipeline.create<dai::node::ColorCamera>();
        // FIXME: extract camera compile arguments here and properly convert them for dai
        camRgb->setBoardSocket(dai::CameraBoardSocket::RGB);
        camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_4_K);

        // FIXME: change the hard-coded behavior
        auto xout = m_pipeline.create<dai::node::XLinkOut>();
        m_out_queue_name = "xout";
        xout->setStreamName(m_out_queue_name);

        for (const auto& nh : nodes) {
            // FIXME: consider a better solution
            if (m_gm.metadata(nh).get<NodeType>().t == NodeType::OP) {
                auto op = m_gm.metadata(nh).get<Op>();
                // FIXME: op name or kernel name?
                if (std::strcmp(op.name(), "org.opencv.oak.enc") == 0) {
                    auto videoEnc = m_pipeline.create<dai::node::VideoEncoder>();
                    // FIXME: encoder params is the 2nd arg - consider a better approach here
                    m_enc_config = op.args[1].get<cv::gapi::oak::EncoderConfig>();
                    // FIXME: convert all the parameters to dai
                    videoEnc->setDefaultProfilePreset(m_enc_config.width, m_enc_config.height,
                                                      m_enc_config.frameRate,
                                                      dai::VideoEncoderProperties::Profile::H265_MAIN);
                    // FIXME: think about proper linking:
                    // probably, firts need to link in nodes to camera, then
                    // for each non-input node check their in edges, get dai elements and link it there
                    camRgb->video.link(videoEnc->input);

                    // FIXME: think about proper linking:
                    videoEnc->bitstream.link(xout->input);
                } else {
                    GAPI_Assert("Unsupported operation in OAK backend");
                }
            }
        }
        return m_pipeline;
    }()),
    // FIXME: add queue parameters
    m_out_queue(m_device.getOutputQueue(m_out_queue_name, 30, true)) {}

void cv::gimpl::GOAKExecutable::handleNewStream() {
    // FIXME: implement
}

void cv::gimpl::GOAKExecutable::handleStopStream() {
    // FIXME: implement
}

void cv::gimpl::GOAKExecutable::run(GIslandExecutable::IInput  &in,
                                    GIslandExecutable::IOutput &out) {
    if (cv::util::holds_alternative<cv::gimpl::EndOfStream>(in.get())) {
        out.post(cv::gimpl::EndOfStream{});
    }

    auto packet = m_out_queue->get<dai::ImgFrame>();

    // FIXME: consider a better solution
    // FIXME: cover all outputs
    auto adapter = cv::util::get<MediaFrame*>(out.get(0))->get<cv::gapi::oak::OAKMediaAdapter>();
    adapter->setParams({m_enc_config.width, m_enc_config.height},
                        cv::gapi::oak::OAKFrameFormat::BGR, packet->getData().data());
}

// Built-in kernels for OAK /////////////////////////////////////////////////////

namespace cv {
namespace gimpl {
namespace oak {
namespace {

// Encode kernel ///////////////////////////////////////////////////////////////

struct Encode: public cv::detail::KernelTag {
    using API = cv::gapi::oak::GEnc;
    static cv::gapi::GBackend backend() { return cv::gapi::oak::backend(); }
    static GOAKKernel          kernel() { return GOAKKernel{}; }
};
} // anonymous namespace

cv::gapi::GKernelPackage kernels();
cv::gapi::GKernelPackage kernels() {
    return cv::gapi::kernels< cv::gimpl::oak::Encode
                            >();
}

} // namespace oak
} // namespace gimpl
} // namespace cv

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

// FIXME: handle #else?

#endif // WITH_OAK_BACKEND
