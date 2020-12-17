// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/util/throw.hpp> // throw_error
#include <opencv2/gapi/streaming/format.hpp> // kernels

#include "api/gbackend_priv.hpp"
#include "backends/common/gbackend.hpp"

#include "gstreamingbackend.hpp"
#include "gstreamingkernel.hpp"

namespace {

struct StreamingCreateFunction
{
    static const char *name() { return "StreamingCreateFunction";  }
    cv::gapi::streaming::CreateActorFunction createActorFunction;
};

using StreamingGraph = ade::TypedGraph
    < cv::gimpl::Op
    , StreamingCreateFunction
    >;

using ConstStreamingGraph = ade::ConstTypedGraph
    < cv::gimpl::Op
    , StreamingCreateFunction
    >;


class GStreamingIntrinExecutable final: public cv::gimpl::GIslandExecutable
{
    virtual void run(std::vector<InObj>  &&,
                     std::vector<OutObj> &&) override {
        GAPI_Assert(false && "Not implemented");
    }

    virtual void run(GIslandExecutable::IInput &in,
                     GIslandExecutable::IOutput &out) override;

    virtual bool allocatesOutputs() const override { return true; }
    // Return an empty RMat since we will reuse the input.
    // There is no need to allocate and copy 4k image here.
    virtual cv::RMat allocate(const cv::GMatDesc&) const override { return {}; }

    virtual bool canReshape() const override { return true; }
    virtual void reshape(ade::Graph&, const cv::GCompileArgs&) override {
        // Do nothing here
    }

public:
    GStreamingIntrinExecutable(const ade::Graph                   &,
                               const cv::GCompileArgs             &,
                               const std::vector<ade::NodeHandle> &);

    const ade::Graph& m_g;
    cv::gimpl::GModel::ConstGraph m_gm;
    cv::gapi::streaming::IActor::Ptr m_actor;
};

void GStreamingIntrinExecutable::run(GIslandExecutable::IInput  &in,
                                     GIslandExecutable::IOutput &out)
{
    m_actor->run(in, out);
}

class GStreamingBackendImpl final: public cv::gapi::GBackend::Priv
{
    virtual void unpackKernel(ade::Graph            &graph,
                              const ade::NodeHandle &op_node,
                              const cv::GKernelImpl &impl) override
    {
        StreamingGraph gm(graph);
        const auto &kimpl  = cv::util::any_cast<cv::gapi::streaming::GStreamingKernel>(impl.opaque);
        gm.metadata(op_node).set(StreamingCreateFunction{kimpl.createActorFunction});
    }

    virtual EPtr compile(const ade::Graph &graph,
                         const cv::GCompileArgs &args,
                         const std::vector<ade::NodeHandle> &nodes) const override
    {
        return EPtr{new GStreamingIntrinExecutable(graph, args, nodes)};
    }

    virtual bool controlsMerge() const override
    {
        return true;
    }

    virtual bool allowsMerge(const cv::gimpl::GIslandModel::Graph &,
                             const ade::NodeHandle &,
                             const ade::NodeHandle &,
                             const ade::NodeHandle &) const override
    {
        return false;
    }
};

GStreamingIntrinExecutable::GStreamingIntrinExecutable(const ade::Graph& g,
                                                       const cv::GCompileArgs& args,
                                                       const std::vector<ade::NodeHandle>& nodes)
    : m_g(g), m_gm(m_g)
{
    using namespace cv::gimpl;
    const auto is_op = [this](const ade::NodeHandle &nh)
    {
        return m_gm.metadata(nh).get<NodeType>().t == NodeType::OP;
    };

    auto it = std::find_if(nodes.begin(), nodes.end(), is_op);
    GAPI_Assert(it != nodes.end() && "No operators found for this island?!");

    ConstStreamingGraph cag(m_g);
    m_actor = cag.metadata(*it).get<StreamingCreateFunction>().createActorFunction(args);

    // Ensure this the only op in the graph
    if (std::any_of(it+1, nodes.end(), is_op))
    {
        cv::util::throw_error
            (std::logic_error
             ("Internal error: Streaming subgraph has multiple operations"));
    }
}

} // anonymous namespace

cv::gapi::GBackend cv::gapi::streaming::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GStreamingBackendImpl>());
    return this_backend;
}

cv::gapi::GKernelPackage cv::gapi::streaming::kernels()
{
    return cv::gapi::kernels<cv::gimpl::BGR>();
}

cv::gapi::GKernelPackage cv::gimpl::streaming::kernels()
{
    return cv::gapi::kernels<cv::gimpl::Copy>();
}

void cv::gimpl::Copy::Actor::run(cv::gimpl::GIslandExecutable::IInput  &in,
                                 cv::gimpl::GIslandExecutable::IOutput &out)
{
    const auto in_msg = in.get();
    if (cv::util::holds_alternative<cv::gimpl::EndOfStream>(in_msg))
    {
        out.post(cv::gimpl::EndOfStream{});
        return;
    }

    const cv::GRunArgs &in_args = cv::util::get<cv::GRunArgs>(in_msg);
    GAPI_Assert(in_args.size() == 1u);

    const auto& in_arg = in_args[0];
    auto out_arg = out.get(0);
    using cv::util::get;
    switch (in_arg.index()) {
    case cv::GRunArg::index_of<cv::RMat>():
        *get<cv::RMat*>(out_arg) = get<cv::RMat>(in_arg);
        break;
    case cv::GRunArg::index_of<cv::MediaFrame>():
        *get<cv::MediaFrame*>(out_arg) = get<cv::MediaFrame>(in_arg);
        break;
    // FIXME: Add support for remaining types
    default:
        GAPI_Assert(false && "Copy: unsupported data type");
    }
    out.meta(out_arg, in_arg.meta);
    out.post(std::move(out_arg));
}

void cv::gimpl::BGR::Actor::run(cv::gimpl::GIslandExecutable::IInput  &in,
                                cv::gimpl::GIslandExecutable::IOutput &out)
{
    const auto in_msg = in.get();
    if (cv::util::holds_alternative<cv::gimpl::EndOfStream>(in_msg))
    {
        out.post(cv::gimpl::EndOfStream{});
        return;
    }

    const cv::GRunArgs &in_args = cv::util::get<cv::GRunArgs>(in_msg);
    GAPI_Assert(in_args.size() == 1u);

    cv::GRunArgP out_arg = out.get(0);
    auto frame = cv::util::get<cv::MediaFrame>(in_args[0]);
    const auto& desc = frame.desc();

    auto& rmat = *cv::util::get<cv::RMat*>(out_arg);
    switch (desc.fmt)
    {
        case cv::MediaFormat::BGR:
            rmat = cv::make_rmat<cv::gimpl::RMatMediaAdapterBGR>(frame);
            break;
        case cv::MediaFormat::NV12:
            {
                cv::Mat bgr;
                auto view = frame.access(cv::MediaFrame::Access::R);
                cv::Mat y_plane (desc.size,     CV_8UC1, view.ptr[0], view.stride[0]);
                cv::Mat uv_plane(desc.size / 2, CV_8UC2, view.ptr[1], view.stride[1]);
                cv::cvtColorTwoPlane(y_plane, uv_plane, bgr, cv::COLOR_YUV2BGR_NV12);
                rmat = cv::make_rmat<cv::gimpl::RMatAdapter>(bgr);
                break;
            }
        default:
            cv::util::throw_error(
                    std::logic_error("Unsupported MediaFormat for cv::gapi::streaming::BGR"));
    }
    out.post(std::move(out_arg));
}

cv::GMat cv::gapi::copy(const cv::GMat& in) {
    return cv::gimpl::streaming::GCopy::on<cv::GMat>(in);
}

cv::GFrame cv::gapi::copy(const cv::GFrame& in) {
    return cv::gimpl::streaming::GCopy::on<cv::GFrame>(in);
}
