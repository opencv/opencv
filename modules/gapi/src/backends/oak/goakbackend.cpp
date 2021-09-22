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

    ade::NodeHandle m_op_nh;
public:
    GOAKExecutable(const ade::Graph& g,
                   const cv::GCompileArgs &args,
                   const std::vector<ade::NodeHandle> &nodes);
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

}} // namespace gimpl // namespace cv

using OAKGraph = ade::TypedGraph
    < cv::gimpl::NetworkParams       // opaque structure, assigned by G-API
    , cv::gimpl::Op
    , cv::gimpl::CustomMetaFunction  // custom meta function expected by G-API
    // FIXME: extend
    >;

using ConstOAKGraph = ade::ConstTypedGraph
    < cv::gimpl::NetworkParams
    , cv::gimpl::Op
    , cv::gimpl::CustomMetaFunction
    // FIXME: extend
    >;

cv::gimpl::GOAKExecutable::GOAKExecutable(const ade::Graph& g,
                                        const cv::GCompileArgs &args,
                                        const std::vector<ade::NodeHandle>& nodes)
    : m_g(g), m_gm(m_g), m_args(args) {
}

void cv::gimpl::GOAKExecutable::handleNewStream() {
}

void cv::gimpl::GOAKExecutable::handleStopStream() {
}

void cv::gimpl::GOAKExecutable::run(GIslandExecutable::IInput  &in,
                                   GIslandExecutable::IOutput &out) {
}

// Built-in kernels for OAK /////////////////////////////////////////////////////

namespace cv {
namespace gimpl {
namespace oak {
namespace {

// Encode kernel ///////////////////////////////////////////////////////////////

struct GOAKKernel{};

struct Encode: public cv::detail::KernelTag {
    using API = cv::gapi::oak::GEnc;
    static cv::gapi::GBackend backend() { return cv::gapi::oak::backend(); }
    static GOAKKernel          kernel()  { return GOAKKernel{}; }
};

G_API_OP(GWriteToHost, <GMat(GMat)>, "org.opencv.oak.writeToHost") {
    static GMatDesc outMeta(const GMatDesc& in) {
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
    }

    virtual EPtr compile(const ade::Graph &graph,
                         const cv::GCompileArgs &args,
                         const std::vector<ade::NodeHandle> &nodes) const override {
        return EPtr{new cv::gimpl::GOAKExecutable(graph, args, nodes)};
    }

    virtual cv::gapi::GKernelPackage auxiliaryKernels() const override {
        return cv::gapi::combine(cv::gimpl::oak::kernels(),
                                 cv::gapi::kernels<
                                                  cv::gimpl::oak::WriteToHost
                                                  >());
    }

    virtual void addBackendPasses(ade::ExecutionEngineSetupContext &ectx) override;
};

void GOAKBackendImpl::addBackendPasses(ade::ExecutionEngineSetupContext &ectx) {
    using namespace cv::gimpl;

    ectx.addPass("kernels", "oak_insert_write_to_host", [&](ade::passes::PassContext &ctx) {
    });
    // Add topo sort since we added new nodes to the graph
    ectx.addPass("kernels", "topo_sort", ade::passes::TopologicalSort());
    // Add appropriate producer node at the beginning of the island
    ectx.addPass("kernels", "oak_setup_input", [&](ade::passes::PassContext &ctx) {
        });
}

cv::gapi::GBackend cv::gapi::oak::backend() {
    static cv::gapi::GBackend this_backend(std::make_shared<GOAKBackendImpl>());
    return this_backend;
}
