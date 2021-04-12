// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


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

#include "utils/itt.hpp"

// FIXME: Is there a way to take a typed graph (our GModel),
// and create a new typed graph _ATOP_ of that (by extending with a couple of
// new types?).
// Alternatively, is there a way to compose types graphs?
//
// If not, we need to introduce that!
using GCPUModel = ade::TypedGraph
    < cv::gimpl::CPUUnit
    , cv::gimpl::Protocol
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGCPUModel = ade::ConstTypedGraph
    < cv::gimpl::CPUUnit
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
            gm.metadata(op_node).set(cv::gimpl::CPUUnit{cpu_impl});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs &compileArgs,
                             const std::vector<ade::NodeHandle> &nodes) const override
        {
            return EPtr{new cv::gimpl::GCPUExecutable(graph, compileArgs, nodes)};
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
                                          const cv::GCompileArgs &compileArgs,
                                          const std::vector<ade::NodeHandle> &nodes)
    : m_g(g), m_gm(m_g), m_compileArgs(compileArgs)
{
    // Convert list of operations (which is topologically sorted already)
    // into an execution script.
    GConstGCPUModel gcm(m_g);
    for (auto &nh : nodes)
    {
        switch (m_gm.metadata(nh).get<NodeType>().t)
        {
        case NodeType::OP:
        {
            m_script.push_back({nh, GModel::collectOutputMeta(m_gm, nh)});

            // If kernel is stateful then prepare storage for its state.
            GCPUKernel k = gcm.metadata(nh).get<CPUUnit>().k;
            if (k.m_isStateful)
            {
                m_nodesToStates[nh] = GArg{ };
            }
            break;
        }
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
                auto& mat = m_res.slot<cv::Mat>()[desc.rc];
                createMat(mat_desc, mat);
            }
            break;
        }
        default: util::throw_error(std::logic_error("Unsupported NodeType type"));
        }
    }

    // For each stateful kernel call 'setup' user callback to initialize state.
    setupKernelStates();
}

// FIXME: Document what it does
cv::GArg cv::gimpl::GCPUExecutable::packArg(const GArg &arg)
{
    // No API placeholders allowed at this point
    // FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
                && arg.kind != cv::detail::ArgKind::GSCALAR
                && arg.kind != cv::detail::ArgKind::GARRAY
                && arg.kind != cv::detail::ArgKind::GOPAQUE
                && arg.kind != cv::detail::ArgKind::GFRAME);

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
    case GShape::GMAT:    return GArg(m_res.slot<cv::Mat>()   [ref.id]);
    case GShape::GSCALAR: return GArg(m_res.slot<cv::Scalar>()[ref.id]);
    // Note: .at() is intentional for GArray and GOpaque as objects MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GARRAY:  return GArg(m_res.slot<cv::detail::VectorRef>().at(ref.id));
    case GShape::GOPAQUE: return GArg(m_res.slot<cv::detail::OpaqueRef>().at(ref.id));
    case GShape::GFRAME:  return GArg(m_res.slot<cv::MediaFrame>().at(ref.id));
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void cv::gimpl::GCPUExecutable::setupKernelStates()
{
    GConstGCPUModel gcm(m_g);
    for (auto& nodeToState : m_nodesToStates)
    {
        auto& kernelNode = nodeToState.first;
        auto& kernelState = nodeToState.second;

        const GCPUKernel& kernel = gcm.metadata(kernelNode).get<CPUUnit>().k;
        kernel.m_setupF(GModel::collectInputMeta(m_gm, kernelNode),
                        m_gm.metadata(kernelNode).get<Op>().args,
                        kernelState,
                        m_compileArgs);
    }
}

void cv::gimpl::GCPUExecutable::handleNewStream()
{
    m_newStreamStarted = true;
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

        if (   desc.storage == Data::Storage::INTERNAL               // FIXME: to reconsider
            && !util::holds_alternative<util::monostate>(desc.ctor))
        {
            // FIXME: Note that compile-time constant data objects (like
            // a value-initialized GArray<T>) also satisfy this condition
            // and should be excluded, but now we just don't support it
            magazine::resetInternalData(m_res, desc);
        }
    }

    // In case if new video-stream happens - for each stateful kernel
    // call 'setup' user callback to re-initialize state.
    if (m_newStreamStarted)
    {
        setupKernelStates();
        m_newStreamStarted = false;
    }

    // OpenCV backend execution is not a rocket science at all.
    // Simply invoke our kernels in the proper order.
    GConstGCPUModel gcm(m_g);
    for (auto &op_info : m_script)
    {
        const auto &op = m_gm.metadata(op_info.nh).get<Op>();

        // Obtain our real execution unit
        // TODO: Should kernels be copyable?
        GCPUKernel k = gcm.metadata(op_info.nh).get<CPUUnit>().k;

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
        for (const auto out_it : ade::util::indexed(op.outs))
        {
            // FIXME: Can the same GArg type resolution mechanism be reused here?
            const auto  out_port  = ade::util::index(out_it);
            const auto& out_desc  = ade::util::value(out_it);
            context.m_results[out_port] = magazine::getObjPtr(m_res, out_desc);
        }

        // For stateful kernel add state to its execution context
        if (k.m_isStateful)
        {
            context.m_state = m_nodesToStates.at(op_info.nh);
        }

        {
            GAPI_ITT_DYNAMIC_LOCAL_HANDLE(op_hndl, op.k.name.c_str());
            GAPI_ITT_AUTO_TRACE_GUARD(op_hndl);

            // Now trigger the executable unit
            k.m_runF(context);
        }

        //As Kernels are forbidden to allocate memory for (Mat) outputs,
        //this code seems redundant, at least for Mats
        //FIXME: unify with cv::detail::ensure_out_mats_not_reallocated
        //FIXME: when it's done, remove can_describe(const GMetaArg&, const GRunArgP&)
        //and descr_of(const cv::GRunArgP &argp)
        for (const auto out_it : ade::util::indexed(op_info.expected_out_metas))
        {
            const auto  out_index      = ade::util::index(out_it);
            const auto& expected_meta  = ade::util::value(out_it);

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

    // In/Out args clean-up is mandatory now with RMat
    for (auto &it : input_objs) magazine::unbind(m_res, it.first);
    for (auto &it : output_objs) magazine::unbind(m_res, it.first);
}
