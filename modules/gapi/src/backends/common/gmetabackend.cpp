// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "precomp.hpp"

#include <opencv2/gapi/gcommon.hpp>        // compile args
#include <opencv2/gapi/util/any.hpp>       // any
#include <opencv2/gapi/streaming/meta.hpp> // GMeta

#include "compiler/gobjref.hpp"            // RcDesc
#include "compiler/gmodel.hpp"             // GModel, Op
#include "backends/common/gbackend.hpp"
#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

#include "backends/common/gmetabackend.hpp"

namespace {

class GraphMetaExecutable final: public cv::gimpl::GIslandExecutable {
    std::string m_meta_tag;

public:
    GraphMetaExecutable(const ade::Graph& g,
                        const std::vector<ade::NodeHandle>& nodes);
    bool canReshape() const override;
    void reshape(ade::Graph&, const cv::GCompileArgs&) override;

    void run(std::vector<InObj> &&input_objs,
             std::vector<OutObj> &&output_objs) override;
};

bool GraphMetaExecutable::canReshape() const {
    return true;
}
void GraphMetaExecutable::reshape(ade::Graph&, const cv::GCompileArgs&) {
    // do nothing here
}

GraphMetaExecutable::GraphMetaExecutable(const ade::Graph& g,
                                         const std::vector<ade::NodeHandle>& nodes) {
    // There may be only one node in the graph
    GAPI_Assert(nodes.size() == 1u);

    cv::gimpl::GModel::ConstGraph cg(g);
    const auto &op = cg.metadata(nodes[0]).get<cv::gimpl::Op>();
    GAPI_Assert(op.k.name == cv::gapi::streaming::detail::GMeta::id());
    m_meta_tag = op.k.tag;
}

void GraphMetaExecutable::run(std::vector<InObj>  &&input_objs,
                              std::vector<OutObj> &&output_objs) {
    GAPI_Assert(input_objs.size() == 1u);
    GAPI_Assert(output_objs.size() == 1u);

    const cv::GRunArg in_arg = input_objs[0].second;
    cv::GRunArgP out_arg = output_objs[0].second;

    auto it = in_arg.meta.find(m_meta_tag);
    if (it == in_arg.meta.end()) {
        cv::util::throw_error
            (std::logic_error("Run-time meta "
                              + m_meta_tag
                              + " is not found in object "
                              + std::to_string(static_cast<int>(input_objs[0].first.shape))
                              + "/"
                              + std::to_string(input_objs[0].first.id)));
    }
    cv::util::get<cv::detail::OpaqueRef>(out_arg) = it->second;
}

class GGraphMetaBackendImpl final: public cv::gapi::GBackend::Priv {
    virtual void unpackKernel(ade::Graph            &,
                              const ade::NodeHandle &,
                              const cv::GKernelImpl &) override {
        // Do nothing here
    }

    virtual EPtr compile(const ade::Graph& graph,
                         const cv::GCompileArgs&,
                         const std::vector<ade::NodeHandle>& nodes,
                         const std::vector<cv::gimpl::Data>&,
                         const std::vector<cv::gimpl::Data>&) const override {
        return EPtr{new GraphMetaExecutable(graph, nodes)};
    }
};

cv::gapi::GBackend graph_meta_backend() {
    static cv::gapi::GBackend this_backend(std::make_shared<GGraphMetaBackendImpl>());
    return this_backend;
}

struct InGraphMetaKernel final: public cv::detail::KernelTag {
    using API = cv::gapi::streaming::detail::GMeta;
    static cv::gapi::GBackend backend() { return graph_meta_backend(); }
    static int                kernel()  { return 42; }
};

} // anonymous namespace

cv::gapi::GKernelPackage cv::gimpl::meta::kernels() {
    return cv::gapi::kernels<InGraphMetaKernel>();
}
