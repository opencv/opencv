// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


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

#include "backends/ocl/goclbackend.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

// FIXME: Is there a way to take a typed graph (our GModel),
// and create a new typed graph _ATOP_ of that (by extending with a couple of
// new types?).
// Alternatively, is there a way to compose types graphs?
//
// If not, we need to introduce that!
using GOCLModel = ade::TypedGraph
    < cv::gimpl::OCLUnit
    , cv::gimpl::Protocol
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGOCLModel = ade::ConstTypedGraph
    < cv::gimpl::OCLUnit
    , cv::gimpl::Protocol
    >;

namespace
{
    class GOCLBackendImpl final: public cv::gapi::GBackend::Priv
    {
        virtual void unpackKernel(ade::Graph            &graph,
                                  const ade::NodeHandle &op_node,
                                  const cv::GKernelImpl &impl) override
        {
            GOCLModel gm(graph);
            auto ocl_impl = cv::util::any_cast<cv::GOCLKernel>(impl.opaque);
            gm.metadata(op_node).set(cv::gimpl::OCLUnit{ocl_impl});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs &,
                             const std::vector<ade::NodeHandle> &nodes) const override
        {
            return EPtr{new cv::gimpl::GOCLExecutable(graph, nodes)};
        }
   };
}

cv::gapi::GBackend cv::gapi::ocl::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GOCLBackendImpl>());
    return this_backend;
}

// GOCLExcecutable implementation //////////////////////////////////////////////
cv::gimpl::GOCLExecutable::GOCLExecutable(const ade::Graph &g,
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
}

// FIXME: Document what it does
cv::GArg cv::gimpl::GOCLExecutable::packArg(const GArg &arg)
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
    case GShape::GMAT:    return GArg(m_res.slot<cv::UMat>()[ref.id]);
    case GShape::GSCALAR: return GArg(m_res.slot<cv::Scalar>()[ref.id]);
    // Note: .at() is intentional for GArray as object MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GARRAY:  return GArg(m_res.slot<cv::detail::VectorRef>().at(ref.id));
    // Note: .at() is intentional for GOpaque as object MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GOPAQUE:  return GArg(m_res.slot<cv::detail::OpaqueRef>().at(ref.id));
    case GShape::GFRAME: return GArg(m_res.slot<cv::MediaFrame>().at(ref.id));
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void cv::gimpl::GOCLExecutable::run(std::vector<InObj>  &&input_objs,
                                    std::vector<OutObj> &&output_objs)
{
    // Update resources with run-time information - what this Island
    // has received from user (or from another Island, or mix...)
    // FIXME: Check input/output objects against GIsland protocol

    // NB: We must clean-up m_res before this function returns because internally (bindInArg,
    //     bindOutArg) we work with cv::UMats, not cv::Mats that were originally placed into the
    //     input/output objects. If this is not done and cv::UMat "leaves" the local function scope,
    //     certain problems may occur.
    //
    //     For example, if the original output (cv::Mat) is re-initialized by the user but we still
    //     hold cv::UMat -> we get cv::UMat that has a parent that was already destroyed. Also,
    //     since we don't own the data (the user does), there's no point holding it after we're done
    const auto clean_up = [&input_objs, &output_objs] (cv::gimpl::Mag* p)
    {
        // Only clean-up UMat entries from current scope, we know that inputs and outputs are stored
        // as UMats from the context below, so the following procedure is safe
        auto& umats = p->slot<cv::UMat>();
        // NB: avoid clearing the whole magazine, there's also pre-allocated internal data
        for (auto& it : input_objs)  umats.erase(it.first.id);
        for (auto& it : output_objs) umats.erase(it.first.id);

        // In/Out args clean-up is mandatory now with RMat
        for (auto &it : input_objs)  magazine::unbind(*p, it.first);
        for (auto &it : output_objs) magazine::unbind(*p, it.first);
    };
    // RAII wrapper to clean-up m_res
    std::unique_ptr<cv::gimpl::Mag, decltype(clean_up)> cleaner(&m_res, clean_up);

    const auto bindUMat = [this](const RcDesc& rc) {
            auto& mag_umat = m_res.template slot<cv::UMat>()[rc.id];
            mag_umat = m_res.template slot<cv::Mat>()[rc.id].getUMat(ACCESS_READ);
    };

    for (auto& it : input_objs) {
        const auto& rc = it.first;
        magazine::bindInArg (m_res, rc, it.second);
        // There is already cv::Mat in the magazine after bindInArg call,
        // extract UMat from it, put into the magazine
        if (rc.shape == GShape::GMAT) bindUMat(rc);
    }
    for (auto& it : output_objs) {
        const auto& rc = it.first;
        magazine::bindOutArg(m_res, rc, it.second);
        if (rc.shape == GShape::GMAT) bindUMat(rc);
    }

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
    GConstGOCLModel gcm(m_g);
    for (auto &op_info : m_script)
    {
        const auto &op = m_gm.metadata(op_info.nh).get<Op>();

        // Obtain our real execution unit
        // TODO: Should kernels be copyable?
        GOCLKernel k = gcm.metadata(op_info.nh).get<OCLUnit>().k;

        // Initialize kernel's execution context:
        // - Input parameters
        GOCLContext context;
        context.m_args.reserve(op.args.size());

        using namespace std::placeholders;
        ade::util::transform(op.args,
                          std::back_inserter(context.m_args),
                          std::bind(&GOCLExecutable::packArg, this, _1));

        // - Output parameters.
        // FIXME: pre-allocate internal Mats, etc, according to the known meta
        for (const auto out_it : ade::util::indexed(op.outs))
        {
            // FIXME: Can the same GArg type resolution mechanism be reused here?
            const auto  out_port  = ade::util::index(out_it);
            const auto& out_desc  = ade::util::value(out_it);
            context.m_results[out_port] = magazine::getObjPtr(m_res, out_desc, true);
        }

        // Now trigger the executable unit
        k.apply(context);

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

    for (auto &it : output_objs)
    {
        const auto& rc    = it.first;
              auto& g_arg = it.second;
        magazine::writeBack(m_res, rc, g_arg);
        if (rc.shape == GShape::GMAT)
        {
            uchar* out_arg_data = m_res.template slot<cv::Mat>()[rc.id].data;
            auto& mag_mat = m_res.template slot<cv::UMat>().at(rc.id);
            GAPI_Assert((out_arg_data == (mag_mat.getMat(ACCESS_RW).data)) && " data for output parameters was reallocated ?");
        }
    }

    // In/Out args clean-up is mandatory now with RMat
    for (auto &it : input_objs) magazine::unbind(m_res, it.first);
    for (auto &it : output_objs) magazine::unbind(m_res, it.first);
}
