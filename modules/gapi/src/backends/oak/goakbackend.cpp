// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <api/gbackend_priv.hpp>
#include <backends/common/gbackend.hpp>

#include "depthai/depthai.hpp"

#include <opencv2/gapi/oak/oak.hpp>

namespace cv { namespace gimpl {
namespace oak {

} // namespace oak

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
                                        const std::vector<cv::gimpl::Data>& ins_data,
                                        const std::vector<cv::gimpl::Data>& outs_data)
    : m_g(g), m_gm(m_g), m_args(args) {

    // FIXME: change the hard-coded behavior (XLinkIn path)
    auto camRgb = m_pipeline.create<dai::node::ColorCamera>();
    // FIXME: extract camera compile arguments here and properly convert them for dai
    camRgb->setBoardSocket(dai::CameraBoardSocket::RGB);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);

    // FIXME: change the hard-coded behavior
    auto xout = m_pipeline.create<dai::node::XLinkOut>();
    xout->setStreamName("h265");

    for (const auto& nh : nodes) {
        if (m_gm.metadata(nh).get<NodeType>().t == NodeType::OP) {
            auto op = m_gm.metadata(nh).get<Op>();
            if (op.name() == "org.opencv.oak.enc") {
                auto videoEnc = m_pipeline.create<dai::node::VideoEncoder>();
                // FIXME: extract kernel arguments here and properly convert them for dai
                videoEnc->setDefaultProfilePreset(1920, 1080, 30,
                                                  dai::VideoEncoderProperties::Profile::H265_MAIN);
                // FIXME: think about proper linking:
                // probably, firts need to link in nodes to camera, then
                // for each non-input node check their in edges, get dai elements and link it there
                camRgb->video.link(videoEnc->input);

                // FIXME: think about proper linking:
                videoEnc->bitstream.link(xout->input);
            } else {
                GAPI_Assert("Unsupported operation name in OAK backend");
            }
        }
    }
}

void cv::gimpl::GOAKExecutable::handleNewStream() {
    // FIXME: extend
}

void cv::gimpl::GOAKExecutable::handleStopStream() {
    // FIXME: extend
}

void cv::gimpl::GOAKExecutable::run(GIslandExecutable::IInput  &in,
                                    GIslandExecutable::IOutput &out) {
    // FIXME: extend
    dai::Device device(m_pipeline);
    auto q = device.getOutputQueue("h265", 30, true); // change hard-coded params
    auto h265Packet = q->get<dai::ImgFrame>();
    //videoFile.write((char*)(h265Packet->getData().data()), h265Packet->getData().size());
    // somehow put out data to the appropriate mediaframe
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
    static GOAKKernel          kernel()  { return GOAKKernel{}; }
};

G_API_OP(GWriteToHost, <GFrame(GFrame)>, "org.opencv.oak.writeToHost") {
    static GFrameDesc outMeta(const GFrameDesc& in) {
        return in;
    }
};

struct WriteToHost: public cv::detail::KernelTag {
    using API = GWriteToHost;
    static cv::gapi::GBackend backend() { return cv::gapi::oak::backend(); }
    static GOAKKernel kernel() { return GOAKKernel{}; }
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

    /*virtual cv::gapi::GKernelPackage auxiliaryKernels() const override {
        return cv::gapi::combine(cv::gimpl::oak::kernels(),
                                 cv::gapi::kernels<
                                                  cv::gimpl::oak::WriteToHost
                                                  >());
    }*/

    //virtual void addBackendPasses(ade::ExecutionEngineSetupContext &ectx) override;
};
/*
void GOAKBackendImpl::addBackendPasses(ade::ExecutionEngineSetupContext &ectx) {
    using namespace cv::gimpl;

    ectx.addPass("kernels", "oak_insert_write_to_host", [&](ade::passes::PassContext &ctx) {
        GModel::Graph gm(ctx.graph);
        if (!GModel::isActive(gm, cv::gapi::oak::backend()))  // FIXME: Rearchitect this!
            return;

        const auto outputs_in_graph = [&gm] (const ade::NodeHandle &nh) {
            return nh->outEdges().empty();
        };

        std::unordered_set< ade::NodeHandle
                          , ade::HandleHasher<ade::Node>
                          > data_nodes;
        for (const auto& data_nh : ade::util::filter(ctx.graph.nodes(), outputs_in_graph)) {
            data_nodes.insert(data_nh);
        }

        // Create new writeToHost op nodes, connect their inputs with according old graph outputs
        cv::GKernel writeToHostKernel{ cv::gimpl::oak::GWriteToHost::id(), {}
                                    , cv::gimpl::oak::GWriteToHost::getOutMeta
                                    , {cv::GShape::GFRAME}, {}, {} };
        cv::GKernelImpl writeToHostKernelImpl{ cv::gimpl::oak::WriteToHost::kernel(), {} };

        for (const auto& data_nh : data_nodes) {
            auto w_op = GModel::mkOpNode(gm, writeToHostKernel, {}, {}, {});
            auto w_data = GModel::mkDataNode(gm, cv::GShape::GFRAME);
            GModel::linkIn(gm, data_nh, w_op, 0u);
            GModel::linkOut(gm, w_data, w_op, 0u);
            // FIXME: looks like these three lines can be reused by the engine/backends
            // so it's likely to worth to put them to some common place
            auto& op = gm.metadata(w_op).get<Op>();
            op.backend = cv::gapi::oak::backend();
            op.backend.priv().unpackKernel(ctx.graph, w_op, writeToHostKernelImpl);
        }
    });
    // Add appropriate producer node at the beginning of the island
    ectx.addPass("kernels", "oak_setup_input", [&](ade::passes::PassContext &ctx) {
        });
    // Check that all data nodes in graph are gframe only ???
    ectx.addPass("kernels", "oak_check_gframe", [&](ade::passes::PassContext &ctx) {
        });
    // Add topo sort since we added new nodes to the graph
    ectx.addPass("kernels", "topo_sort", ade::passes::TopologicalSort());
}*/

cv::gapi::GBackend cv::gapi::oak::backend() {
    static cv::gapi::GBackend this_backend(std::make_shared<GOAKBackendImpl>());
    return this_backend;
}
