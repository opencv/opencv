// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <ade/util/zip_range.hpp>

#include <opencv2/gapi/opencv_includes.hpp>

#include "api/gproto_priv.hpp" // ptr(GRunArgP)
#include "executor/gexecutor.hpp"
#include "compiler/passes/passes.hpp"

cv::gimpl::GExecutor::GExecutor(std::unique_ptr<ade::Graph> &&g_model)
    : m_orig_graph(std::move(g_model))
    , m_island_graph(GModel::Graph(*m_orig_graph).metadata()
                     .get<IslandModel>().model)
    , m_gm(*m_orig_graph)
    , m_gim(*m_island_graph)
{
    // NB: Right now GIslandModel is acyclic, so for a naive execution,
    // simple unrolling to a list of triggers is enough

    // Naive execution model is similar to current CPU (OpenCV) plugin
    // execution model:
    // 1. Allocate all internal resources first (NB - CPU plugin doesn't do it)
    // 2. Put input/output GComputation arguments to the storage
    // 3. For every Island, prepare vectors of input/output parameter descs
    // 4. Ask every GIslandExecutable to prepare its internal states for a new stream
    // 5. Iterate over a list of operations (sorted in the topological order)
    // 6. For every operation, form a list of input/output data objects
    // 7. Run GIslandExecutable
    // 8. writeBack

    auto sorted = m_gim.metadata().get<ade::passes::TopologicalSortData>();
    for (auto nh : sorted.nodes())
    {
        switch (m_gim.metadata(nh).get<NodeKind>().k)
        {
        case NodeKind::ISLAND:
            {
                std::vector<RcDesc> input_rcs;
                std::vector<RcDesc> output_rcs;
                input_rcs.reserve(nh->inNodes().size());
                output_rcs.reserve(nh->outNodes().size());

                auto xtract = [&](ade::NodeHandle slot_nh, std::vector<RcDesc> &vec) {
                    const auto orig_data_nh
                        = m_gim.metadata(slot_nh).get<DataSlot>().original_data_node;
                    const auto &orig_data_info
                        = m_gm.metadata(orig_data_nh).get<Data>();
                    vec.emplace_back(RcDesc{ orig_data_info.rc
                                           , orig_data_info.shape
                                           , orig_data_info.ctor});
                };
                // (3)
                for (auto in_slot_nh  : nh->inNodes())  xtract(in_slot_nh,  input_rcs);
                for (auto out_slot_nh : nh->outNodes()) xtract(out_slot_nh, output_rcs);

                m_ops.emplace_back(OpDesc{ std::move(input_rcs)
                                         , std::move(output_rcs)
                                         , m_gim.metadata(nh).get<IslandExec>().object
                                         });
            }
            break;

        case NodeKind::SLOT:
            {
                const auto orig_data_nh
                    = m_gim.metadata(nh).get<DataSlot>().original_data_node;
                // (1)
                initResource(nh, orig_data_nh);
                m_slots.emplace_back(DataDesc{nh, orig_data_nh});
            }
            break;

        default:
            GAPI_Assert(false);
            break;
        } // switch(kind)
    } // for(gim nodes)

    // (4)
    prepareForNewStream();
}

namespace cv {
namespace gimpl {
namespace magazine {
namespace {

void bindInArgExec(Mag& mag, const RcDesc &rc, const GRunArg &arg)
{
    if (rc.shape != GShape::GMAT)
    {
        bindInArg(mag, rc, arg);
        return;
    }
    auto& mag_rmat = mag.template slot<cv::RMat>()[rc.id];
    switch (arg.index())
    {
    case GRunArg::index_of<Mat>() :
        mag_rmat = make_rmat<RMatOnMat>(util::get<Mat>(arg)); break;
    case GRunArg::index_of<cv::RMat>() :
        mag_rmat = util::get<cv::RMat>(arg); break;
    default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
    }
    // FIXME: has to take extra care about meta here for this particuluar
    // case, just because this function exists at all
    mag.meta<cv::RMat>()[rc.id] = arg.meta;
}

void bindOutArgExec(Mag& mag, const RcDesc &rc, const GRunArgP &arg)
{
    if (rc.shape != GShape::GMAT)
    {
        bindOutArg(mag, rc, arg);
        return;
    }
    auto& mag_rmat = mag.template slot<cv::RMat>()[rc.id];
    switch (arg.index())
    {
    case GRunArgP::index_of<Mat*>() :
        mag_rmat = make_rmat<RMatOnMat>(*util::get<Mat*>(arg)); break;
    case GRunArgP::index_of<cv::RMat*>() :
        mag_rmat = *util::get<cv::RMat*>(arg); break;
    default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
    }
}

cv::GRunArgP getObjPtrExec(Mag& mag, const RcDesc &rc)
{
    if (rc.shape != GShape::GMAT)
    {
        return getObjPtr(mag, rc);
    }
    return GRunArgP(&mag.slot<cv::RMat>()[rc.id]);
}

void writeBackExec(const Mag& mag, const RcDesc &rc, GRunArgP &g_arg)
{
    if (rc.shape != GShape::GMAT)
    {
        writeBack(mag, rc, g_arg);
        return;
    }

    switch (g_arg.index())
    {
    case GRunArgP::index_of<cv::Mat*>() : {
        // If there is a copy intrinsic at the end of the graph
        // we need to actually copy the data to the user buffer
        // since output runarg was optimized to simply point
        // to the input of the copy kernel
        // FIXME:
        // Rework, find a better way to check if there should be
        // a real copy (add a pass to StreamingBackend?)
        // NB: In case RMat adapter not equal to "RMatOnMat" need to
        // copy data back to the host as well.
        auto& out_mat = *util::get<cv::Mat*>(g_arg);
        const auto& rmat = mag.template slot<cv::RMat>().at(rc.id);
        auto* adapter = rmat.get<RMatOnMat>();
        if ((adapter != nullptr && out_mat.data != adapter->data()) ||
            (adapter == nullptr)) {
            auto view = rmat.access(RMat::Access::R);
            asMat(view).copyTo(out_mat);
        }
        break;
    }
    case GRunArgP::index_of<cv::RMat*>() : /* do nothing */ break;
    default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
    }
}

void assignMetaStubExec(Mag& mag, const RcDesc &rc, const cv::GRunArg::Meta &meta) {
    switch (rc.shape)
    {
    case GShape::GARRAY:  mag.meta<cv::detail::VectorRef>()[rc.id] = meta; break;
    case GShape::GOPAQUE: mag.meta<cv::detail::OpaqueRef>()[rc.id] = meta; break;
    case GShape::GSCALAR: mag.meta<cv::Scalar>()[rc.id]            = meta; break;
    case GShape::GFRAME:  mag.meta<cv::MediaFrame>()[rc.id]        = meta; break;
    case GShape::GMAT:
        mag.meta<cv::Mat>() [rc.id] = meta;
        mag.meta<cv::RMat>()[rc.id] = meta;
#if !defined(GAPI_STANDALONE)
        mag.meta<cv::UMat>()[rc.id] = meta;
#endif
        break;
    default: util::throw_error(std::logic_error("Unsupported GShape type")); break;
    }
}

} // anonymous namespace
}}} // namespace cv::gimpl::magazine


void cv::gimpl::GExecutor::initResource(const ade::NodeHandle & nh, const ade::NodeHandle &orig_nh)
{
    const Data &d = m_gm.metadata(orig_nh).get<Data>();

    if (   d.storage != Data::Storage::INTERNAL
        && d.storage != Data::Storage::CONST_VAL)
        return;

    // INTERNALS+CONST only! no need to allocate/reset output objects
    // to as it is bound externally (e.g. already in the m_res)

    switch (d.shape)
    {
    case GShape::GMAT:
        {
            // Let island allocate it's outputs if it can,
            // allocate cv::Mat and wrap it with RMat otherwise
            GAPI_Assert(!nh->inNodes().empty());
            const auto desc = util::get<cv::GMatDesc>(d.meta);
            auto& exec = m_gim.metadata(nh->inNodes().front()).get<IslandExec>().object;
            auto& rmat = m_res.slot<cv::RMat>()[d.rc];
            if (exec->allocatesOutputs()) {
                rmat = exec->allocate(desc);
            } else {
                Mat mat;
                createMat(desc, mat);
                rmat = make_rmat<RMatOnMat>(mat);
            }
        }
        break;

    case GShape::GSCALAR:
        if (d.storage == Data::Storage::CONST_VAL)
        {
            auto rc = RcDesc{d.rc, d.shape, d.ctor};
            magazine::bindInArg(m_res, rc, m_gm.metadata(orig_nh).get<ConstValue>().arg);
        }
        break;

    case GShape::GARRAY:
        if (d.storage == Data::Storage::CONST_VAL)
        {
            auto rc = RcDesc{d.rc, d.shape, d.ctor};
            magazine::bindInArg(m_res, rc, m_gm.metadata(orig_nh).get<ConstValue>().arg);
        }
        break;
    case GShape::GOPAQUE:
        // Constructed on Reset, do nothing here
        break;
    case GShape::GFRAME: {
        // Should be defined by backend, do nothing here
        break;
    }
    default:
        GAPI_Assert(false);
    }
}

class cv::gimpl::GExecutor::Input final: public cv::gimpl::GIslandExecutable::IInput
{
    cv::gimpl::Mag &mag;
    virtual StreamMsg get() override
    {
        cv::GRunArgs res;
        for (const auto &rc : desc()) { res.emplace_back(magazine::getArg(mag, rc)); }
        return StreamMsg{std::move(res)};
    }
    virtual StreamMsg try_get() override { return get(); }
public:
    Input(cv::gimpl::Mag &m, const std::vector<RcDesc> &rcs) : mag(m) { set(rcs); }
};

class cv::gimpl::GExecutor::Output final: public cv::gimpl::GIslandExecutable::IOutput
{
    cv::gimpl::Mag &mag;
    std::unordered_map<const void*, int> out_idx;
    std::exception_ptr eptr;

    GRunArgP get(int idx) override
    {
        auto r = magazine::getObjPtrExec(mag, desc()[idx]);
        // Remember the output port for this output object
        out_idx[cv::gimpl::proto::ptr(r)] = idx;
        return r;
    }
    void post(GRunArgP&&, const std::exception_ptr& e) override
    {
        if (e)
        {
            eptr = e;
        }
    }
    void post(EndOfStream&&) override {} // Do nothing here too
    void post(Exception&& ex) override
    {
        eptr = std::move(ex.eptr);
    }
    void meta(const GRunArgP &out, const GRunArg::Meta &m) override
    {
        const auto idx = out_idx.at(cv::gimpl::proto::ptr(out));
        magazine::assignMetaStubExec(mag, desc()[idx], m);
    }
public:
    Output(cv::gimpl::Mag &m, const std::vector<RcDesc> &rcs)
        : mag(m)
    {
        set(rcs);
    }

    void verify()
    {
        if (eptr)
        {
            std::rethrow_exception(eptr);
        }
    }
};

void cv::gimpl::GExecutor::run(cv::gimpl::GRuntimeArgs &&args)
{
    // (2)
    const auto proto = m_gm.metadata().get<Protocol>();

    // Basic check if input/output arguments are correct
    // FIXME: Move to GCompiled (do once for all GExecutors)
    if (proto.inputs.size() != args.inObjs.size()) // TODO: Also check types
    {
        util::throw_error(std::logic_error
                          ("Computation's input protocol doesn\'t "
                           "match actual arguments!"));
    }
    if (proto.outputs.size() != args.outObjs.size()) // TODO: Also check types
    {
        util::throw_error(std::logic_error
                          ("Computation's output protocol doesn\'t "
                           "match actual arguments!"));
    }

    namespace util = ade::util;

    // ensure that output Mat parameters are correctly allocated
    // FIXME: avoid copy of NodeHandle and GRunRsltComp ?
    for (auto index : util::iota(proto.out_nhs.size()))
    {
        auto& nh = proto.out_nhs.at(index);
        const Data &d = m_gm.metadata(nh).get<Data>();
        if (d.shape == GShape::GMAT)
        {
            using cv::util::get;
            const auto desc = get<cv::GMatDesc>(d.meta);

            auto check_rmat = [&desc, &args, &index]()
            {
                auto& out_mat = *get<cv::RMat*>(args.outObjs.at(index));
                GAPI_Assert(desc.canDescribe(out_mat));
            };

#if !defined(GAPI_STANDALONE)
            // Building as part of OpenCV - follow OpenCV behavior In
            // the case of cv::Mat if output buffer is not enough to
            // hold the result, reallocate it
            if (cv::util::holds_alternative<cv::Mat*>(args.outObjs.at(index)))
            {
                auto& out_mat = *get<cv::Mat*>(args.outObjs.at(index));
                createMat(desc, out_mat);
            }
            // In the case of RMat check to fit required meta
            else
            {
                check_rmat();
            }
#else
            // Building standalone - output buffer should always exist,
            // and _exact_ match our inferred metadata
            if (cv::util::holds_alternative<cv::Mat*>(args.outObjs.at(index)))
            {
                auto& out_mat = *get<cv::Mat*>(args.outObjs.at(index));
                GAPI_Assert(out_mat.data != nullptr &&
                        desc.canDescribe(out_mat));
            }
            // In the case of RMat check to fit required meta
            else
            {
                check_rmat();
            }
#endif // !defined(GAPI_STANDALONE)
        }
    }
    // Update storage with user-passed objects
    for (auto it : ade::util::zip(ade::util::toRange(proto.inputs),
                                  ade::util::toRange(args.inObjs)))
    {
        magazine::bindInArgExec(m_res, std::get<0>(it), std::get<1>(it));
    }
    for (auto it : ade::util::zip(ade::util::toRange(proto.outputs),
                                  ade::util::toRange(args.outObjs)))
    {
        magazine::bindOutArgExec(m_res, std::get<0>(it), std::get<1>(it));
    }

    // Reset internal data
    for (auto &sd : m_slots)
    {
        const auto& data = m_gm.metadata(sd.data_nh).get<Data>();
        magazine::resetInternalData(m_res, data);
    }

    // Run the script (5)
    for (auto &op : m_ops)
    {
        // (6), (7)
        Input i{m_res, op.in_objects};
        Output o{m_res, op.out_objects};
        op.isl_exec->run(i, o);
        // NB: Check if execution finished without exception.
        o.verify();
    }

    // (8)
    for (auto it : ade::util::zip(ade::util::toRange(proto.outputs),
                                  ade::util::toRange(args.outObjs)))
    {
        magazine::writeBackExec(m_res, std::get<0>(it), std::get<1>(it));
    }
}

const cv::gimpl::GModel::Graph& cv::gimpl::GExecutor::model() const
{
    return m_gm;
}

bool cv::gimpl::GExecutor::canReshape() const
{
    // FIXME: Introduce proper reshaping support on GExecutor level
    // for all cases!
    return std::all_of(m_ops.begin(), m_ops.end(),
                       [](const OpDesc& op) { return op.isl_exec->canReshape(); });
}

void cv::gimpl::GExecutor::reshape(const GMetaArgs& inMetas, const GCompileArgs& args)
{
    GAPI_Assert(canReshape());
    auto& g = *m_orig_graph.get();
    ade::passes::PassContext ctx{g};
    passes::initMeta(ctx, inMetas);
    passes::inferMeta(ctx, true);

    // NB: Before reshape islands need to re-init resources for every slot.
    for (auto slot : m_slots)
    {
        initResource(slot.slot_nh, slot.data_nh);
    }

    for (auto& op : m_ops)
    {
        op.isl_exec->reshape(g, args);
    }
}

void cv::gimpl::GExecutor::prepareForNewStream()
{
    for (auto &op : m_ops)
    {
        op.isl_exec->handleNewStream();
    }
}
