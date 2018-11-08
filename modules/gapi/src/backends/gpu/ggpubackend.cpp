// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <functional>
#include <unordered_set>

#include <ade/util/algorithm.hpp>

#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>
#include <ade/util/chain_range.hpp>

#include <ade/typed_graph.hpp>

#include "opencv2/gapi/gcommon.hpp"
#include "opencv2/gapi/util/any.hpp"
#include "opencv2/gapi/gtype_traits.hpp"

#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "backends/gpu/ggpubackend.hpp"
#include "backends/gpu/ggpuimgproc.hpp"
#include "backends/gpu/ggpucore.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

// FIXME: Is there a way to take a typed graph (our GModel),
// and create a new typed graph _ATOP_ of that (by extending with a couple of
// new types?).
// Alternatively, is there a way to compose types graphs?
//
// If not, we need to introduce that!
using GGPUModel = ade::TypedGraph
    < cv::gimpl::Unit
    , cv::gimpl::Protocol
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGGPUModel = ade::ConstTypedGraph
    < cv::gimpl::Unit
    , cv::gimpl::Protocol
    >;

namespace
{
    class GGPUBackendImpl final: public cv::gapi::GBackend::Priv
    {
        virtual void unpackKernel(ade::Graph            &graph,
                                  const ade::NodeHandle &op_node,
                                  const cv::GKernelImpl &impl) override
        {
            GGPUModel gm(graph);
            auto gpu_impl = cv::util::any_cast<cv::GGPUKernel>(impl.opaque);
            gm.metadata(op_node).set(cv::gimpl::Unit{gpu_impl});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs &,
                             const std::vector<ade::NodeHandle> &nodes) const override
        {
            return EPtr{new cv::gimpl::GGPUExecutable(graph, nodes)};
        }
   };
}

cv::gapi::GBackend cv::gapi::gpu::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GGPUBackendImpl>());
    return this_backend;
}

// GGPUExcecutable implementation //////////////////////////////////////////////
cv::gimpl::GGPUExecutable::GGPUExecutable(const ade::Graph &g,
                                          const std::vector<ade::NodeHandle> &nodes)
    : m_g(g), m_gm(m_g)
{
    // Convert list of operations (which is topologically sorted already)
    // into an execution script.
    for (auto &nh : nodes)
    {
        switch (m_gm.metadata(nh).get<NodeType>().t)
        {
        case NodeType::OP: m_script.push_back({nh, GModel::collectOutputMeta(m_gm, nh)}); break;
        case NodeType::DATA:
        {
            m_dataNodes.push_back(nh);
            const auto &desc = m_gm.metadata(nh).get<Data>();
            if (desc.storage == Data::Storage::CONST)
            {
                auto rc = RcDesc{desc.rc, desc.shape, desc.ctor};
                magazine::bindInArg(m_res, rc, m_gm.metadata(nh).get<ConstValue>().arg);
            }
            //preallocate internal Mats in advance
            if (desc.storage == Data::Storage::INTERNAL && desc.shape == GShape::GMAT)
            {
                const auto mat_desc = util::get<cv::GMatDesc>(desc.meta);
                const auto type = CV_MAKETYPE(mat_desc.depth, mat_desc.chan);
                m_res.slot<cv::UMat>()[desc.rc].create(mat_desc.size.width, mat_desc.size.height, type);
            }
            break;
        }
        default: util::throw_error(std::logic_error("Unsupported NodeType type"));
        }
    }
}

// FIXME: Document what it does
cv::GArg cv::gimpl::GGPUExecutable::packArg(const GArg &arg)
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
    case GShape::GMAT:    return GArg(m_res.slot<cv::UMat>()[ref.id]);
    case GShape::GSCALAR: return GArg(m_res.slot<cv::gapi::own::Scalar>()[ref.id]);
        // Note: .at() is intentional for GArray as object MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GARRAY:  return GArg(m_res.slot<cv::detail::VectorRef>().at(ref.id));
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void cv::gimpl::GGPUExecutable::run(std::vector<InObj>  &&input_objs,
                                    std::vector<OutObj> &&output_objs)
{
    // Update resources with run-time information - what this Island
    // has received from user (or from another Island, or mix...)
    // FIXME: Check input/output objects against GIsland protocol

    for (auto& it : input_objs)   magazine::bindInArg (m_res, it.first, it.second, true);
    for (auto& it : output_objs)  magazine::bindOutArg(m_res, it.first, it.second, true);

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
    GConstGGPUModel gcm(m_g);
    for (auto &op_info : m_script)
    {
        const auto &op = m_gm.metadata(op_info.nh).get<Op>();

        // Obtain our real execution unit
        // TODO: Should kernels be copyable?
        GGPUKernel k = gcm.metadata(op_info.nh).get<Unit>().k;

        // Initialize kernel's execution context:
        // - Input parameters
        GGPUContext context;
        context.m_args.reserve(op.args.size());

        using namespace std::placeholders;
        ade::util::transform(op.args,
                          std::back_inserter(context.m_args),
                          std::bind(&GGPUExecutable::packArg, this, _1));

        // - Output parameters.
        // FIXME: pre-allocate internal Mats, etc, according to the known meta
        for (const auto &out_it : ade::util::indexed(op.outs))
        {
            // FIXME: Can the same GArg type resolution mechanism be reused here?
            const auto out_port  = ade::util::index(out_it);
            const auto out_desc  = ade::util::value(out_it);
            context.m_results[out_port] = magazine::getObjPtr(m_res, out_desc, true);
        }

        // Now trigger the executable unit
        k.apply(context);

        for (const auto &out_it : ade::util::indexed(op_info.expected_out_metas))
        {
            const auto out_index      = ade::util::index(out_it);
            const auto expected_meta  = ade::util::value(out_it);
            const auto out_meta       = descr_of(context.m_results[out_index]);

            if (expected_meta != out_meta)
            {
                util::throw_error
                    (std::logic_error
                     ("Output meta doesn't "
                      "coincide with the generated meta\n"
                      "Expected: " + ade::util::to_string(expected_meta) + "\n"
                      "Actual  : " + ade::util::to_string(out_meta)));
            }
        }
    } // for(m_script)

    for (auto &it : output_objs) magazine::writeBack(m_res, it.first, it.second, true);
}
