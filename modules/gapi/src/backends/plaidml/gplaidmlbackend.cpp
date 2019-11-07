// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "precomp.hpp"

#include <ade/util/algorithm.hpp>

#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>
#include <ade/util/chain_range.hpp>

#include <ade/typed_graph.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/gtype_traits.hpp>

#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "backends/plaidml/gplaidmlbackend.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

// FIXME: Is there a way to take a typed graph (our GModel),
// and create a new typed graph _ATOP_ of that (by extending with a couple of
// new types?).
// Alternatively, is there a way to compose types graphs?
//
// If not, we need to introduce that!
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
            std::cout << "plaidml backend unpackKernel" << std::endl;
            GPlaidMLModel gm(graph);
            auto plaidml_impl = cv::util::any_cast<cv::GPlaidMLKernel>(impl.opaque);
            gm.metadata(op_node).set(cv::gimpl::PlaidMLUnit{plaidml_impl});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs &,
                             const std::vector<ade::NodeHandle> &nodes) const override
        {
            std::cout << "plaidml backend compile " << std::endl;
            return EPtr{new cv::gimpl::GPlaidMLExecutable(graph, nodes)};
        }
   };
}

cv::gapi::GBackend cv::gapi::plaidml::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GPlaidMLBackendImpl>());
    return this_backend;
}

void cv::gimpl::GPlaidMLExecutable::initInputs(const std::vector<ade::NodeHandle> &nodes)
{

    auto is_op = [&](ade::NodeHandle nh) {
        return m_gm.metadata(nh).get<NodeType>().t == NodeType::OP;
    };

    auto first_op_it = ade::util::find_if(nodes, is_op);
    GAPI_Assert(first_op_it != nodes.end());

    const auto &op = m_gm.metadata(*first_op_it).get<Op>();
    auto ins_meta = GModel::collectInputMeta(m_gm, *first_op_it);

    // FIXME We have to reserve memory that avoid reallocation
    // because we use pointer on vector elements
    input_bindings_.reserve(op.args.size());
    GAPI_Assert(op.args.size() == ins_meta.size());
    for (const auto& it : ade::util::zip(ins_meta, op.args))
    {
        const auto& in_meta = std::get<0>(it);
        const auto& in_arg  = std::get<1>(it);

        // FIXME now supported only Tensors and corresponding Mat
        if (in_meta.index() == cv::GMetaArg::index_of<cv::GMatDesc>())
        {
            const auto& ref  = in_arg.get<cv::gimpl::RcDesc>();
            const auto& desc = cv::util::get<cv::GMatDesc>(in_meta);

            // FIXME Remove hardcode PLAIDML_DATA_UINT8
            auto placeholder = plaidml::edsl::Placeholder(PLAIDML_DATA_UINT8,
                               {desc.size.width, desc.size.height, desc.chan});

            auto shape = placeholder.shape();
            plaidml::TensorShape tensor_shape(shape.dtype(), shape.int_dims());
            plaidml::Buffer buffer(device_id_, tensor_shape);

            auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();
            // FIXME piecewise construct
            input_bindings_.emplace_back(plaidml::exec::Binding{placeholder, buffer});
            // FIXME Copy here !!!;
            tensor_map.emplace(ref.id, placeholder);
            //tensor_map.emplace(ref.id, plaidml::exec::Binding{placeholder, buffer});

            auto& buffer_map = m_res.slot<plaidml::Buffer*>();
            std::cout << "ref.id = " << ref.id << std::endl;
            std::cout << "addr = " << &(input_bindings_.back().buffer) << std::endl;
            buffer_map.emplace(ref.id, &(input_bindings_.back().buffer));
        }
    }
}

void cv::gimpl::GPlaidMLExecutable::initOutputs(const std::vector<ade::NodeHandle> &nodes)
{
    auto is_op = [&](ade::NodeHandle nh) {
        return m_gm.metadata(nh).get<NodeType>().t == NodeType::OP;
    };

    auto last_op_it = std::find_if(nodes.rbegin(), nodes.rend(), is_op);
    GAPI_Assert(last_op_it != nodes.rend());

    const auto &op = m_gm.metadata(*last_op_it).get<Op>();
    auto outs_meta = GModel::collectOutputMeta(m_gm, *last_op_it);

    output_bindings_.reserve(op.args.size());
    // FIXME add assert for compare sizes
    for (const auto& it : ade::util::zip(outs_meta, op.outs))
    {
        const auto& out_meta = std::get<0>(it);
        const auto& out_rc   = std::get<1>(it);

        // FIXME now supported only Tensors and corresponding Mat
        GAPI_Assert(out_meta.index() == cv::GMetaArg::index_of<cv::GMatDesc>());

        const auto& out_desc = cv::util::get<cv::GMatDesc>(out_meta);
        std::cout << "out_desc id = " << out_rc.id << std::endl;
        std::cout << "out_desc = " << out_desc << std::endl;

        // FIXME Remove hardcode PLAIDML_DATA_UINT8
        auto placeholder = plaidml::edsl::Placeholder(PLAIDML_DATA_UINT8,
                           {out_desc.size.width, out_desc.size.height, out_desc.chan});

        auto shape = placeholder.shape();
        plaidml::TensorShape tensor_shape(shape.dtype(), shape.int_dims());
        plaidml::Buffer buffer(device_id_, tensor_shape);

        auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();
        // FIXME piecewise construct
        //tensor_map.emplace(out_rc.id, plaidml::exec::Binding{placeholder, buffer});
        output_bindings_.emplace_back(plaidml::exec::Binding{placeholder, buffer});
        // FIXME Copy here !!!
        tensor_map.emplace(out_rc.id, placeholder);

        output_ids_.push_back(out_rc.id);

        auto& buffer_map = m_res.slot<plaidml::Buffer*>();
        buffer_map.emplace(out_rc.id, &output_bindings_.back().buffer);
    }
}

cv::gimpl::GPlaidMLExecutable::GPlaidMLExecutable(const ade::Graph &g,
                                                  const std::vector<ade::NodeHandle> &nodes)
    : m_g(g), m_gm(m_g)
{
    device_id_ = "opencl_intel_gen9_hd_graphics_neo.0";
    target_id_ = "intel_gen9_opencl";

    initInputs(nodes);
    initOutputs(nodes);

    GConstGPlaidMLModel gcm(m_g);
    for (const auto& nh : nodes)
    {
        if (m_gm.metadata(nh).get<NodeType>().t != NodeType::OP)
            continue;

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
    int i = 0;
    for (const auto& out_id : output_ids_)
    {
        auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();
        // FIXME Copy here !!!
        output_tensors.emplace_back(tensor_map[out_id]);
    }

    program_ = std::unique_ptr<plaidml::edsl::Program>(new plaidml::edsl::Program("name", output_tensors));

    for (int i = 0; i < output_tensors.size(); ++i)
    {
        output_bindings_[i].tensor = program_->outputs()[i];
    }

    exec_    = std::make_shared<plaidml::exec::Executable>(*program_, device_id_, target_id_, input_bindings_, output_bindings_);
    std::cout << program_->str() << std::endl;
}

void cv::gimpl::GPlaidMLExecutable::run(std::vector<InObj>  &&input_objs,
                                        std::vector<OutObj> &&output_objs)
{
    std::cout << "GPlaidMLBackend run" << std::endl;

    for (auto& it : input_objs) bindInArg (it.first, it.second);

    exec_->run();

    for (auto& it : output_objs)  bindOutArg(it.first, it.second);
}

void cv::gimpl::GPlaidMLExecutable::bindInArg(const RcDesc &rc, const GRunArg  &arg)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        switch (arg.index())
        {
        case GRunArg::index_of<cv::gapi::own::Mat>():
        {
            std::cout << "bind inputs here = " << rc.id << std::endl;
            auto& buffer_map = m_res.slot<plaidml::Buffer*>();
            auto it = buffer_map.find(rc.id);
            GAPI_Assert(it != buffer_map.end());

            GAPI_Assert(it->second != nullptr);
            std::cout << "rc.id = " << rc.id << std::endl;
            std::cout << "it->second = " << it->second << std::endl;
            //auto& mag_buffer = *(it->second);
            std::cout << "before" << std::endl;
            auto& arg_mat = util::get<cv::gapi::own::Mat>(arg);
            std::cout << "arg_mat " << to_ocv(arg_mat) << std::endl;
            //input_bindings_[0].buffer.copy_from(arg_mat.data);
            it->second->copy_from(arg_mat.data);
            //mag_buffer.copy_from(arg_mat.data);
            std::cout << "after" << std::endl;
        }
        break;
#if !defined(GAPI_STANDALONE)
        case GRunArg::index_of<cv::Mat>() :
        {
            GAPI_Assert(false);
            //std::cout << "bind inputs here cv::Mat" << std::endl;
            //auto& mag_buffer = m_res.template slot<plaidml::exec::Binding>()[rc.id].buffer;
            //mag_buffer.copy_from(util::get<cv::Mat>(arg).data);
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
        switch (arg.index())
        {
        case GRunArgP::index_of<cv::gapi::own::Mat*>():
        {
            auto& buffer_map = m_res.slot<plaidml::Buffer*>();
            auto it = buffer_map.find(rc.id);
            GAPI_Assert(it != buffer_map.end());

            GAPI_Assert(it->second != nullptr);
            //auto& mag_buffer = *(it->second);
            auto& arg_mat = *util::get<cv::gapi::own::Mat*>(arg);
            it->second->copy_into(arg_mat.data);
        }
        break;
#if !defined(GAPI_STANDALONE)
        case GRunArgP::index_of<cv::Mat*>() :
        {
            GAPI_Assert(false);
            //std::cout << "bind inputs here cv::Mat" << std::endl;
            //auto& mag_buffer = m_res.template slot<plaidml::exec::Binding>()[rc.id].buffer;
            //mag_buffer.copy_from(util::get<cv::Mat>(arg).data);
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
    //// No API placeholders allowed at this point
    //// FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
              && arg.kind != cv::detail::ArgKind::GSCALAR
              && arg.kind != cv::detail::ArgKind::GARRAY);

    if (arg.kind != cv::detail::ArgKind::GOBJREF)
    {
        // All other cases - pass as-is, with no transformations to GArg contents.
        return arg;
    }
    GAPI_Assert(arg.kind == cv::detail::ArgKind::GOBJREF);
    std::cout << "packArg " << std::endl;

    const cv::gimpl::RcDesc &ref = arg.get<cv::gimpl::RcDesc>();
    std::cout << "ref.id = " << ref.id << std::endl;
    switch (ref.shape)
    {
    case GShape::GMAT:
    {
        std::cout << "GMAT" << std::endl;
        auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();
        auto it = tensor_map.find(ref.id);
        if (it != tensor_map.end()) {
            std::cout << "FOUND!!!" << std::endl;
        } else {
            std::cout << "NOT found" << std::endl;
        }
        return GArg(it->second);
    }
    break;
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}
