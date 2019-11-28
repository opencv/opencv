// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


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

#include "backends/cpu/gcpubackend.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

// FIXME: Is there a way to take a typed graph (our GModel),
// and create a new typed graph _ATOP_ of that (by extending with a couple of
// new types?).
// Alternatively, is there a way to compose types graphs?
//
// If not, we need to introduce that!
using GCPUModel = ade::TypedGraph
    < cv::gimpl::Unit
    , cv::gimpl::Protocol
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGCPUModel = ade::ConstTypedGraph
    < cv::gimpl::Unit
    , cv::gimpl::Protocol
    >;

namespace
{
    class GCPUBackendImpl final: public cv::gapi::GBackend::Priv
    {
        virtual void unpackKernel(ade::Graph            &graph,
                                  const ade::NodeHandle &op_node,
                                  const cv::GKernelImpl &impl) override
        {
            GCPUModel gm(graph);
            auto cpu_impl = cv::util::any_cast<cv::GCPUKernel>(impl.opaque);
            gm.metadata(op_node).set(cv::gimpl::Unit{cpu_impl});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs &,
                             const std::vector<ade::NodeHandle> &nodes) const override
        {
            return EPtr{new cv::gimpl::GCPUExecutable(graph, nodes)};
        }
   };
}

cv::gapi::GBackend cv::gapi::cpu::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GCPUBackendImpl>());
    return this_backend;
}

// GCPUExecutable implementation //////////////////////////////////////////////
cv::gimpl::GCPUExecutable::GCPUExecutable(const ade::Graph &g,
                                          const std::vector<ade::NodeHandle> &nodes)
    : m_g(g), m_gm(m_g)
{
    // Convert list of operations (which is topologically sorted already)
    // into an execution script.
    for (auto &nh : nodes)
    {
        std::cout << m_gm.metadata(nh).get<NodeType>().name() << " " <<  (int)m_gm.metadata(nh).get<NodeType>().t << std::endl;
        std::cout << m_gm.metadata(nh).get<Island>().name() << " " << m_gm.metadata(nh).get<Island>().island << std::endl;
        //std::cout << m_gm.metadata(nh).get<OutputMeta>().name() << " size " << m_gm.metadata(nh).get<OutputMeta>().outMeta.size() << std::endl;
        //std::cout << m_gm.metadata(nh).get<Protocol>().name() << " " << m_gm.metadata(nh).get<Protocol>().inputs.size() << std::endl;
        //std::cout << m_gm.metadata(nh).get<ActiveBackends>().name() << " " << m_gm.metadata(nh).get<ActiveBackends>().backends.size() << std::endl;
        //std::cout << m_gm.metadata(nh).get<Journal>().name() << " " << m_gm.metadata(nh).get<Journal>().messages[0] << std::endl;
        if(m_gm.metadata(nh).get<NodeType>().t == NodeType::OP)
        {
            std::cout << m_gm.metadata(nh).get<Op>().name() << " op kernel name" << m_gm.metadata(nh).get<Op>().k.name << std::endl;
            std::cout << m_gm.metadata(nh).get<Op>().name() << " op kernel tag" << m_gm.metadata(nh).get<Op>().k.tag << std::endl;
            std::cout << m_gm.metadata(nh).get<Op>().name() << " op kernel outShapes.size() " << m_gm.metadata(nh).get<Op>().k.outShapes.size() << std::endl;
            for(size_t i=0; i < m_gm.metadata(nh).get<Op>().k.outShapes.size(); ++i)
            {
                std::cout << m_gm.metadata(nh).get<Op>().name() << " op kernel outShapes[" << i << "] " << (int)m_gm.metadata(nh).get<Op>().k.outShapes[i] << std::endl;
            }

            std::cout << "args size = " << m_gm.metadata(nh).get<Op>().args.size() << "  outs size = " << m_gm.metadata(nh).get<Op>().outs.size() << std::endl;

            for(size_t i=0; i < m_gm.metadata(nh).get<Op>().args.size(); ++i)
            {
                std::cout << " arg kind " << (int)m_gm.metadata(nh).get<Op>().args[i].kind << std::endl;
                if(m_gm.metadata(nh).get<Op>().args[i].kind == cv::detail::ArgKind::GOBJREF)
                {
                    std::cout << "kind GOBJREF!!!" << std::endl;
                    const cv::gimpl::RcDesc &ref = m_gm.metadata(nh).get<Op>().args[i].get<cv::gimpl::RcDesc>();
                    std::cout << "ref.id " << ref.id << std::endl;
                    switch (ref.shape)
                    {
                        case GShape::GMAT:    std::cout << "GMAT" << std::endl; break;
                        case GShape::GSCALAR: std::cout << "GSCALAR" << std::endl; break;
                        case GShape::GARRAY:  std::cout << "GARRAY" << std::endl; break;
                        default:
                           util::throw_error(std::logic_error("Unsupported GShape type"));
                        break;
                    }
                }
                else if(m_gm.metadata(nh).get<Op>().args[i].kind == cv::detail::ArgKind::GMAT)
                {
                    std::cout << "kind GMAT!!!" << std::endl;
                }
                else if(m_gm.metadata(nh).get<Op>().args[i].kind == cv::detail::ArgKind::GMATP)
                {
                    std::cout << "kind GMATP!!!" << std::endl;
                }
                else if(m_gm.metadata(nh).get<Op>().args[i].kind == cv::detail::ArgKind::GSCALAR)
                {
                    std::cout << "kind GSCALAR!!!" << std::endl;
                }
                else if(m_gm.metadata(nh).get<Op>().args[i].kind == cv::detail::ArgKind::GARRAY)
                {
                    std::cout << "kind GSCALAR!!!" << std::endl;
                }
                else if(m_gm.metadata(nh).get<Op>().args[i].kind == cv::detail::ArgKind::OPAQUE)
                {
                    std::cout << "kind OPAQUE!!!" << std::endl;
                    //try to get int or double
                    //std::cout << m_gm.metadata(nh).get<Op>().args[i].get<int>() << std::endl;
                    if(m_gm.metadata(nh).get<Op>().args[i].opaque_kind == cv::detail::OpaqueKind::OPAQUE_INT)
                    {
                        std::cout << m_gm.metadata(nh).get<Op>().args[i].get<int>() << std::endl;
                    }
                    else if(m_gm.metadata(nh).get<Op>().args[i].opaque_kind == cv::detail::OpaqueKind::OPAQUE_DOUBLE)
                    {
                        std::cout << m_gm.metadata(nh).get<Op>().args[i].get<double>() << std::endl;
                    }
                    else if(m_gm.metadata(nh).get<Op>().args[i].opaque_kind == cv::detail::OpaqueKind::OPAQUE_CV_SIZE)
                    {
                        auto tmp_size =  m_gm.metadata(nh).get<Op>().args[i].get<cv::Size>();
                        std::cout << tmp_size.width << " " << tmp_size.height << std::endl;
                    }
                    else
                    {
                        std::cout << "UNSUPPORTED OPAQUE!!!" << std::endl;
                    }
                    //std::cout << m_gm.metadata(nh).get<Op>().args[i].unsafe_get<int>() << std::endl;
                }
                else
                {
                    util::throw_error(std::logic_error("Unknown arg type"));
                }
            }

            for (const auto &eh : nh->inEdges())
            {
                std::cout << m_gm.metadata(eh).get<Input>().name() << " port " << m_gm.metadata(eh).get<Input>().port << std::endl;
                auto in_edge = GModel::getInEdgeByPort(m_gm, nh, m_gm.metadata(eh).get<Input>().port);
                GAPI_Assert(in_edge == eh);
            }

            for (const auto &eh : nh->outEdges())
            {
                std::cout << m_gm.metadata(eh).get<Output>().name() << " port " << m_gm.metadata(eh).get<Output>().port << std::endl;
            }


            GMetaArgs tmp_in_meta = GModel::collectInputMeta(m_gm, nh);
            std::cout << " tmp_in_meta size " << tmp_in_meta.size() << std::endl;
            for(auto &meta : tmp_in_meta)
            {
                std::cout << meta << std::endl;
            }
            GMetaArgs tmp_out_meta = GModel::collectOutputMeta(m_gm, nh);
            std::cout << " tmp_out_meta size " << tmp_out_meta.size() << std::endl;
            for(auto &meta : tmp_out_meta)
            {
                std::cout << meta << std::endl;
            }
            auto inputs_nh = GModel::orderedInputs(m_gm, nh);
            std::cout << " inputs_nh size " << inputs_nh.size() << std::endl;
            for (auto &in_nh : inputs_nh)
            {
                std::cout << m_gm.metadata(in_nh).get<Data>().name() << " in data shape " << (int)m_gm.metadata(in_nh).get<Data>().shape << std::endl;
            }
            auto outputs_nh = GModel::orderedOutputs(m_gm, nh);
            std::cout << " outputs_nh size " << outputs_nh.size() << std::endl;
            for (auto &out_nh : outputs_nh)
            {
                std::cout << m_gm.metadata(out_nh).get<Data>().name() << " out data shape " << (int)m_gm.metadata(out_nh).get<Data>().shape << std::endl;
            }
        }
        else if(m_gm.metadata(nh).get<NodeType>().t == NodeType::DATA)
        {
            std::cout << m_gm.metadata(nh).get<Data>().name() << " data shape " << (int)m_gm.metadata(nh).get<Data>().shape << std::endl;
        }
        else
        {
            util::throw_error(std::logic_error("Unsupported NodeType type"));
        }
        std::cout << "*** NEXT NODE ***" << std::endl;
    }
    for (auto &nh : nodes)
    {
        switch (m_gm.metadata(nh).get<NodeType>().t)
        {
        case NodeType::OP: m_script.push_back({nh, GModel::collectOutputMeta(m_gm, nh)}); break;
        case NodeType::DATA:
        {
            m_dataNodes.push_back(nh);
            const auto &desc = m_gm.metadata(nh).get<Data>();
            if (desc.storage == Data::Storage::CONST_VAL)
            {
                auto rc = RcDesc{desc.rc, desc.shape, desc.ctor};
                magazine::bindInArg(m_res, rc, m_gm.metadata(nh).get<ConstValue>().arg);
            }
            //preallocate internal Mats in advance
            if (desc.storage == Data::Storage::INTERNAL && desc.shape == GShape::GMAT)
            {
                const auto mat_desc = util::get<cv::GMatDesc>(desc.meta);
                auto& mat = m_res.slot<cv::gapi::own::Mat>()[desc.rc];
                createMat(mat_desc, mat);
            }
            break;
        }
        default: util::throw_error(std::logic_error("Unsupported NodeType type"));
        }
    }
}

// FIXME: Document what it does
cv::GArg cv::gimpl::GCPUExecutable::packArg(const GArg &arg)
{
    // No API placeholders allowed at this point
    // FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
              && arg.kind != cv::detail::ArgKind::GSCALAR
              && arg.kind != cv::detail::ArgKind::GARRAY);

    if (arg.kind != cv::detail::ArgKind::GOBJREF)
    {
        // All other cases - pass as-is, with no transformations to GArg contents.
        return arg;
    }
    GAPI_Assert(arg.kind == cv::detail::ArgKind::GOBJREF);

    // Wrap associated CPU object (either host or an internal one)
    // FIXME: object can be moved out!!! GExecutor faced that.
    const cv::gimpl::RcDesc &ref = arg.get<cv::gimpl::RcDesc>();
    switch (ref.shape)
    {
    case GShape::GMAT:    return GArg(m_res.slot<cv::gapi::own::Mat>()   [ref.id]);
    case GShape::GSCALAR: return GArg(m_res.slot<cv::gapi::own::Scalar>()[ref.id]);
    // Note: .at() is intentional for GArray as object MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GARRAY:  return GArg(m_res.slot<cv::detail::VectorRef>().at(ref.id));
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void cv::gimpl::GCPUExecutable::run(std::vector<InObj>  &&input_objs,
                                    std::vector<OutObj> &&output_objs)
{
    // Update resources with run-time information - what this Island
    // has received from user (or from another Island, or mix...)
    // FIXME: Check input/output objects against GIsland protocol

    for (auto& it : input_objs)   magazine::bindInArg (m_res, it.first, it.second);
    for (auto& it : output_objs)  magazine::bindOutArg(m_res, it.first, it.second);

    // Initialize (reset) internal data nodes with user structures
    // before processing a frame (no need to do it for external data structures)
    GModel::ConstGraph gm(m_g);
    for (auto nh : m_dataNodes)
    {
        const auto &desc = gm.metadata(nh).get<Data>();

        if (   desc.storage == Data::Storage::INTERNAL
            && !util::holds_alternative<util::monostate>(desc.ctor))
        {
            // FIXME: Note that compile-time constant data objects (like
            // a value-initialized GArray<T>) also satisfy this condition
            // and should be excluded, but now we just don't support it
            magazine::resetInternalData(m_res, desc);
        }
    }

    // OpenCV backend execution is not a rocket science at all.
    // Simply invoke our kernels in the proper order.
    GConstGCPUModel gcm(m_g);
    for (auto &op_info : m_script)
    {
        const auto &op = m_gm.metadata(op_info.nh).get<Op>();

        // Obtain our real execution unit
        // TODO: Should kernels be copyable?
        GCPUKernel k = gcm.metadata(op_info.nh).get<Unit>().k;

        // Initialize kernel's execution context:
        // - Input parameters
        GCPUContext context;
        context.m_args.reserve(op.args.size());

        using namespace std::placeholders;
        ade::util::transform(op.args,
                          std::back_inserter(context.m_args),
                          std::bind(&GCPUExecutable::packArg, this, _1));

        // - Output parameters.
        // FIXME: pre-allocate internal Mats, etc, according to the known meta
        for (const auto &out_it : ade::util::indexed(op.outs))
        {
            // FIXME: Can the same GArg type resolution mechanism be reused here?
            const auto out_port  = ade::util::index(out_it);
            const auto out_desc  = ade::util::value(out_it);
            context.m_results[out_port] = magazine::getObjPtr(m_res, out_desc);
        }

        // Now trigger the executable unit
        k.apply(context);

        //As Kernels are forbidden to allocate memory for (Mat) outputs,
        //this code seems redundant, at least for Mats
        //FIXME: unify with cv::detail::ensure_out_mats_not_reallocated
        //FIXME: when it's done, remove can_describe(const GMetaArg&, const GRunArgP&)
        //and descr_of(const cv::GRunArgP &argp)
        for (const auto &out_it : ade::util::indexed(op_info.expected_out_metas))
        {
            const auto out_index      = ade::util::index(out_it);
            const auto expected_meta  = ade::util::value(out_it);

            if (!can_describe(expected_meta, context.m_results[out_index]))
            {
                const auto out_meta = descr_of(context.m_results[out_index]);
                util::throw_error
                    (std::logic_error
                     ("Output meta doesn't "
                      "coincide with the generated meta\n"
                      "Expected: " + ade::util::to_string(expected_meta) + "\n"
                      "Actual  : " + ade::util::to_string(out_meta)));
            }
        }
    } // for(m_script)

    for (auto &it : output_objs) magazine::writeBack(m_res, it.first, it.second);
}
