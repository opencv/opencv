// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2020 Intel Corporation

#include "precomp.hpp"

#include "backends/ov/ovdef.hpp"
#ifdef HAVE_OPENVINO_2_0

#include <ade/util/algorithm.hpp>
#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>
#include <ade/typed_graph.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/gtype_traits.hpp>

#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "backends/ov/govcvbackend.hpp"
#include "backends/ov/util.hpp"  // wrap::getCore()

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

using GOVCVModel = ade::TypedGraph
    < cv::gimpl::OVCVUnit
    , cv::gimpl::Protocol
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGOVCVModel = ade::ConstTypedGraph
    < cv::gimpl::OVCVUnit
    , cv::gimpl::Protocol
    >;

namespace
{
class GOVCVBackendImpl final: public cv::gapi::GBackend::Priv
{
    virtual void unpackKernel(ade::Graph            &graph,
                              const ade::NodeHandle &op_node,
                              const cv::GKernelImpl &impl) override
    {
        GOVCVModel gm(graph);
        auto ovcv_impl = cv::util::any_cast<cv::GOVCVKernel>(impl.opaque);
        gm.metadata(op_node).set(cv::gimpl::OVCVUnit{ovcv_impl});
    }

    virtual EPtr compile(const ade::Graph& graph,
                         const cv::GCompileArgs& /*args*/,
                         const std::vector<ade::NodeHandle>& nodes,
                         const std::vector<cv::gimpl::Data>& ins_data,
                         const std::vector<cv::gimpl::Data>& outs_data) const override
    {
        return EPtr{new cv::gimpl::GOVCVExecutable(graph, nodes, ins_data, outs_data)};
    }
};
}

cv::gapi::GBackend cv::gimpl::ovcv::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GOVCVBackendImpl>());
    return this_backend;
}

cv::gimpl::GOVCVExecutable::GOVCVExecutable(const ade::Graph& g,
                                            const std::vector<ade::NodeHandle>& nodes,
                                            const std::vector<cv::gimpl::Data>& ins_data,
                                            const std::vector<cv::gimpl::Data>& outs_data)
    : m_g(g), m_gm(m_g)
{
    auto is_op = [&](ade::NodeHandle nh) {
        return m_gm.metadata(nh).get<NodeType>().t == NodeType::OP;
    };
    std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(m_all_ops), is_op);
    compile(ins_data, outs_data);
}

namespace {
ov::element::Type type_to_ov(const cv::GMatDesc &desc)
{
    switch (desc.depth) {
    case CV_8U: return ov::element::u8;
    default:
        GAPI_Assert(false && "Other types are not supported right now.");
    }
}

ov::PartialShape shape_to_ov(const cv::GMatDesc &desc) {
    GAPI_Assert(!desc.isND() && "So far only expect image data");
    return ov::Shape{
        1u,
        static_cast<std::size_t>(desc.size.height),
        static_cast<std::size_t>(desc.size.width),
        static_cast<std::size_t>(desc.chan)
    };
}
} // FIXME: move to util

std::shared_ptr<ov::op::v0::Parameter> mkParam(const cv::gimpl::Data &data)
{
    GAPI_Assert(data.shape == cv::GShape::GMAT);
    const cv::GMatDesc& desc = cv::util::get<cv::GMatDesc>(data.meta);
    return std::make_shared<ov::op::v0::Parameter>
        (type_to_ov(desc), shape_to_ov(desc));
}

void cv::gimpl::GOVCVExecutable::compile(const std::vector<cv::gimpl::Data>& ins_data,
                                         const std::vector<cv::gimpl::Data>& outs_data)
{
    // Translate a G-API subgraph to OpenVINO graph
    // Use kernels to put respective OpenVINO layers ("Operators") in the graph.

    // Prepare ov::Parameters as inputs to the model first: make it
    // available in the internal magazie.
    ov::ParameterVector ov_params;
    ov_params.reserve(ins_data.size());
    for (auto &&it : ade::util::indexed(ins_data)) {
        auto &data = ade::util::value(it);
        auto param = mkParam(data);
        m_res.slot<ov::Output<ov::Node> >()[data.rc] = param;
        ov_params.push_back(param);
        m_param_remap[data.rc] = ade::util::index(it);
    }

    GConstGOVCVModel gcm(m_g);

    // Walk all operators in the graph (in a topological order),
    // instantiate the respective layers and fill-in the magazine with
    // OpenVINO's "Outputs" (their name for graph-time tensors)
    for (const auto& nh : m_all_ops)
    {
        const auto&  k = gcm.metadata(nh).get<OVCVUnit>().k;
        const auto& op = m_gm.metadata(nh).get<Op>();

        // Initialize Operation parameters:
        GOVCVContext ctx;

        // ..bind Inputs
        ctx.m_args.reserve(op.args.size());
        using namespace std::placeholders;
        ade::util::transform(op.args,
            std::back_inserter(ctx.m_args),
            std::bind(&GOVCVExecutable::packArg, this, _1));

        // ..bind Outputs
        for (const auto &out_it : ade::util::indexed(op.outs))
        {
            const auto out_port  = ade::util::index(out_it);
            const auto out_desc  = ade::util::value(out_it);
            auto& out_output     = m_res.slot<ov::Output<ov::Node> >()[out_desc.id];
            ctx.m_results[out_port] = GArg(&out_output);
        }

        // Construct the OV opeartion
        k.apply(ctx);
    } // for(m_all_ops)

    // Construct the OpenVINO model:
    // ..Collect the Result vector
    ov::ResultVector ov_results;
    ov_results.reserve(outs_data.size());
    for (auto &&it : ade::util::indexed(outs_data)) {
        auto& data = ade::util::value(it);
        auto  out  = m_res.slot<ov::Output<ov::Node> >().at(data.rc);
        ov_results.push_back(std::make_shared<ov::op::v0::Result>(out));
        m_result_remap[data.rc] = ade::util::index(it);
    }

    // FIXME: Make device configurable
    const char *ov_device    = std::getenv("GAPI_OVCV_DEVICE");
    const char *ov_dump_path = std::getenv("GAPI_OVCV_DUMP_PATH");

    m_ov_model    = std::make_shared<ov::Model>(ov_results, ov_params);

    if (ov_dump_path) {
        ov::save_model(m_ov_model, ov_dump_path);
    }

    m_ov_compiled = cv::gapi::ov::wrap::getCore()
        .compile_model(m_ov_model, ov_device ? ov_device : "CPU");
    m_ov_req      = m_ov_compiled.create_infer_request();
}

cv::GArg cv::gimpl::GOVCVExecutable::packArg(const GArg &arg)
{
    GAPI_Assert( arg.kind != cv::detail::ArgKind::GMAT
              && arg.kind != cv::detail::ArgKind::GSCALAR
              && arg.kind != cv::detail::ArgKind::GARRAY
              && arg.kind != cv::detail::ArgKind::GOPAQUE);
    if (arg.kind != cv::detail::ArgKind::GOBJREF)
    {
        // All other cases - pass as-is, with no transformations to GArg contents.
        return arg;
    }
    GAPI_Assert(arg.kind == cv::detail::ArgKind::GOBJREF);

    const cv::gimpl::RcDesc &ref = arg.get<cv::gimpl::RcDesc>();
    switch (ref.shape)
    {
    case GShape::GMAT:
    {
        return GArg(m_res.slot<ov::Output<ov::Node> >().at(ref.id));
    }
    break;
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void cv::gimpl::GOVCVExecutable::run(std::vector<InObj>  &&input_objs,
                                     std::vector<OutObj> &&output_objs)
{
    for (auto& it : input_objs) bindInArg (it.first, it.second);
    m_ov_req.infer();
    for (auto& it : output_objs) bindOutArg(it.first, it.second);
}

void cv::gimpl::GOVCVExecutable::bindInArg(const RcDesc &rc, const GRunArg  &arg)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        auto param_id = m_param_remap.at(rc.id);
        auto param_tr = m_ov_req.get_input_tensor(param_id);

        switch (arg.index())
        {
        case GRunArg::index_of<cv::RMat>():
        {
            // The input (G-API) buffer we're reading from
            auto& rmat = cv::util::get<cv::RMat>(arg);
            auto  view = rmat.access(cv::RMat::Access::R);
            auto  omat = cv::gimpl::asMat(view);

            // The original (OpenVINO) buffer we're writing to
            auto steps = param_tr.get_strides();
            auto ovmat = cv::Mat(view.rows(), view.cols(), view.type(),
                                 param_tr.data<std::uint8_t>(),
                                 steps[1]); // H stride as in HWC (WHY not NHWC??)
            omat.copyTo(ovmat);
        }
        break;
        default: util::throw_error(std::logic_error("No hope"));
        }
    }
    break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

void cv::gimpl::GOVCVExecutable::bindOutArg(const RcDesc &rc, const GRunArgP  &arg)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        auto result_id = m_result_remap.at(rc.id);
        auto result_tr = m_ov_req.get_output_tensor(result_id);

        switch (arg.index())
        {
        case GRunArgP::index_of<cv::RMat*>() :
        {
            // The output (G-API) buffer we're writing into
            auto& rmat = *cv::util::get<cv::RMat*>(arg);
            auto  view = rmat.access(cv::RMat::Access::W);
            auto  omat = cv::gimpl::asMat(view);

            // The original (OpenVINO) buffer we're reading from
            auto steps = result_tr.get_strides();
            auto ovmat = cv::Mat(view.rows(), view.cols(), view.type(),
                                 result_tr.data<std::uint8_t>(),
                                 steps[1]); // H stride as in NCHW
            ovmat.copyTo(omat);
        }
        break;
        default:
            util::throw_error(std::logic_error("NO Hope"));
        }
    }
    break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

#endif // HAVE_OPENVINO_2_0
