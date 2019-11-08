// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "precomp.hpp"

#include <ade/util/algorithm.hpp>
#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>
#include <ade/typed_graph.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/gtype_traits.hpp>

#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "backends/plaidml/gplaidmlbackend.hpp"
#include "backends/plaidml/plaidml_util.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

using GPlaidMLModel = ade::TypedGraph
    < cv::gimpl::PlaidMLUnit
    , cv::gimpl::Protocol
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGPlaidMLModel = ade::ConstTypedGraph
    < cv::gimpl::PlaidMLUnit
    , cv::gimpl::Protocol
    >;

namespace
{
    class GPlaidMLBackendImpl final: public cv::gapi::GBackend::Priv
    {
        virtual void unpackKernel(ade::Graph            &graph,
                                  const ade::NodeHandle &op_node,
                                  const cv::GKernelImpl &impl) override
        {
            GPlaidMLModel gm(graph);
            auto plaidml_impl = cv::util::any_cast<cv::GPlaidMLKernel>(impl.opaque);
            gm.metadata(op_node).set(cv::gimpl::PlaidMLUnit{plaidml_impl});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs &,
                             const std::vector<ade::NodeHandle> &nodes) const override
        {
            return EPtr{new cv::gimpl::GPlaidMLExecutable(graph, nodes)};
        }
   };
}

cv::gapi::GBackend cv::gapi::plaidml::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GPlaidMLBackendImpl>());
    return this_backend;
}

void cv::gimpl::GPlaidMLExecutable::initBuffers(const std::vector<cv::gimpl::Data>& data,
                                                std::vector<plaidml::exec::Binding>& bindings)
{

    // NB: This is necessary because we keep a pointer to bindings elements to buffer_map
    // In order to them to remain valid it's required to prevant reallocation
    bindings.reserve(data.size());
    for (const auto& d : data)
    {
        GAPI_Assert(d.meta.index() == cv::GMetaArg::index_of<cv::GMatDesc>() &&
                    "Now PlaidML backend supported only cv::GMat's");

        const auto& desc = cv::util::get<cv::GMatDesc>(d.meta);

        auto placeholder = plaidml::edsl::Placeholder(
                           cv::util::plaidml::depth_from_ocv(desc.depth),
                           {desc.size.width, desc.size.height, desc.chan});

        const auto& shape = placeholder.shape();
        plaidml::TensorShape tshape(shape.dtype(), shape.int_dims());
        plaidml::Buffer buffer(device_id_, tshape);

        bindings.push_back(plaidml::exec::Binding{std::move(placeholder),
                                                  std::move(buffer)});

        auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();
        // FIXME Avoid Copy here !!!
        tensor_map.emplace(d.rc, placeholder);

        auto& buffer_map = m_res.slot<plaidml::Buffer*>();
        buffer_map.emplace(d.rc, &(bindings.back().buffer));
    }
}

void cv::gimpl::GPlaidMLExecutable::compile(const std::vector<cv::gimpl::Data>& ins_data,
                                            const std::vector<cv::gimpl::Data>& outs_data)
{
    // FIXME Move this hardcode configuration
    device_id_ = "opencl_intel_gen9_hd_graphics_neo.0";
    target_id_ = "intel_gen9_opencl";

    initBuffers(ins_data,  input_bindings_);
    initBuffers(outs_data, output_bindings_);

    ade::util::transform(outs_data, std::back_inserter(output_ids_),
                   [](const cv::gimpl::Data& d) { return d.rc; });

    GConstGPlaidMLModel gcm(m_g);
    for (const auto& nh : m_all_ops)
    {
        GPlaidMLKernel k = gcm.metadata(nh).get<PlaidMLUnit>().k;
        GPlaidMLContext ctx;

        const auto &op = m_gm.metadata(nh).get<Op>();
        ctx.m_args.reserve(op.args.size());

        using namespace std::placeholders;
        ade::util::transform(op.args,
                std::back_inserter(ctx.m_args),
                std::bind(&GPlaidMLExecutable::packArg, this, _1));

        for (const auto &out_it : ade::util::indexed(op.outs))
        {
            const auto out_port  = ade::util::index(out_it);
            const auto out_desc  = ade::util::value(out_it);

            auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();

            // NB: Create tensor if need
            auto& tensor = tensor_map[out_desc.id];
            ctx.m_results[out_port] = GArg(&(tensor));
        }

        k.apply(ctx);
    }

    std::vector<plaidml::edsl::Tensor> output_tensors;
    for (const auto& out_id : output_ids_)
    {
        auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();
        // FIXME Avoid copy here !!!
        output_tensors.emplace_back(tensor_map[out_id]);
    }

    program_ = std::unique_ptr<plaidml::edsl::Program>(new plaidml::edsl::Program("Program", output_tensors));

    // FIXME Need to update tensors here
    for (int i = 0; i < output_tensors.size(); ++i)
    {
        output_bindings_[i].tensor = program_->outputs()[i];
    }

    exec_ = std::make_shared<plaidml::exec::Executable>(*program_, device_id_, target_id_, input_bindings_, output_bindings_);
}

cv::gimpl::GPlaidMLExecutable::GPlaidMLExecutable(const ade::Graph &g,
                                                  const std::vector<ade::NodeHandle> &nodes)
    : m_g(g), m_gm(m_g)
{
    auto is_op = [&](ade::NodeHandle nh) {
        return m_gm.metadata(nh).get<NodeType>().t == NodeType::OP;
    };

    std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(m_all_ops), is_op);
}

void cv::gimpl::GPlaidMLExecutable::run(std::vector<InObj>  &&input_objs,
                                        std::vector<OutObj> &&output_objs)
{
    for (auto& it : input_objs) bindInArg (it.first, it.second);

    exec_->run();

    for (auto& it : output_objs) bindOutArg(it.first, it.second);
}

void cv::gimpl::GPlaidMLExecutable::bindInArg(const RcDesc &rc, const GRunArg  &arg)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        auto& buffer_map = m_res.slot<plaidml::Buffer*>();
        auto it = buffer_map.find(rc.id);
        GAPI_Assert(it != buffer_map.end());

        switch (arg.index())
        {
        case GRunArg::index_of<cv::gapi::own::Mat>():
        {
            auto& arg_mat = util::get<cv::gapi::own::Mat>(arg);
            it->second->copy_from(arg_mat.data);
        }
        break;
#if !defined(GAPI_STANDALONE)
        case GRunArg::index_of<cv::Mat>() :
        {
            auto& arg_mat = util::get<cv::Mat>(arg);
            it->second->copy_from(arg_mat.data);
        }
        break;
#endif //  !defined(GAPI_STANDALONE)
        default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
    }
    break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

void cv::gimpl::GPlaidMLExecutable::bindOutArg(const RcDesc &rc, const GRunArgP  &arg)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        auto& buffer_map = m_res.slot<plaidml::Buffer*>();
        auto it = buffer_map.find(rc.id);
        GAPI_Assert(it != buffer_map.end());

        switch (arg.index())
        {
        case GRunArgP::index_of<cv::gapi::own::Mat*>():
        {
            auto& arg_mat = *util::get<cv::gapi::own::Mat*>(arg);
            it->second->copy_into(arg_mat.data);
        }
        break;
#if !defined(GAPI_STANDALONE)
        case GRunArgP::index_of<cv::Mat*>() :
        {
            auto& arg_mat = *util::get<cv::Mat*>(arg);
            it->second->copy_into(arg_mat.data);
        }
        break;
#endif //  !defined(GAPI_STANDALONE)
        default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
    }
    break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

cv::GArg cv::gimpl::GPlaidMLExecutable::packArg(const GArg &arg)
{
    GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
              && arg.kind != cv::detail::ArgKind::GSCALAR
              && arg.kind != cv::detail::ArgKind::GARRAY);

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
        auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();
        auto it = tensor_map.find(ref.id);
        GAPI_Assert(it != tensor_map.end());
        return GArg(it->second);
    }
    break;
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}
