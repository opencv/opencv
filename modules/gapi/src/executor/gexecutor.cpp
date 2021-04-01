// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <iostream>

#include <ade/util/zip_range.hpp>
#include <ade/util/filter_range.hpp>

#include <opencv2/gapi/opencv_includes.hpp>

#include "api/gproto_priv.hpp" // ptr(GRunArgP)
#include "executor/gexecutor.hpp"
#include "compiler/passes/passes.hpp"

#include "logger.hpp" // GAPI_LOG

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
    // 4. Iterate over a list of operations (sorted in the topological order)
    // 5. For every operation, form a list of input/output data objects
    // 6. Run GIslandExecutable
    // 7. writeBack

    auto is_slot = [&](ade::NodeHandle const& nh){
        return m_gim.metadata(nh).get<NodeKind>().k == NodeKind::SLOT;
    };

    auto is_island = [&](ade::NodeHandle const& nh){
        return m_gim.metadata(nh).get<NodeKind>().k == NodeKind::ISLAND;
    };

    namespace r = ade::util::Range;

    auto sorted = m_gim.metadata().get<ade::passes::TopologicalSortData>();

    for (auto&& nh : r::filter(sorted.nodes(), is_slot)){
        const auto orig_data_nh
            = m_gim.metadata(nh).get<DataSlot>().original_data_node;
        // (1)
        initResource(nh, orig_data_nh);
        m_slots.emplace_back(DataDesc{nh, orig_data_nh});
    }

    for (auto&& nh : r::filter(sorted.nodes(), is_island)){
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

}

#if defined(USE_GAPI_TBB_EXECUTOR)
cv::gimpl::GTBBExecutor::GTBBExecutor(std::unique_ptr<ade::Graph> &&g_model)
    : GExecutor(std::move(g_model))
{
    using handle_t = ade::NodeHandle;
    using handle_hasher_t = ade::HandleHasher<ade::Node>;

    using node_handles_set_t = std::unordered_set<handle_t, handle_hasher_t>;

    using op_index_t = size_t;

    std::unordered_map<handle_t, op_index_t, handle_hasher_t> op_indexes;
    std::unordered_map<op_index_t, node_handles_set_t> op_dependant_node_handles;

    namespace r = ade::util::Range;

    auto is_island = [&](handle_t const& nh){
        return m_gim.metadata(nh).get<NodeKind>().k == NodeKind::ISLAND;
    };

    auto sorted = m_gim.metadata().get<ade::passes::TopologicalSortData>();

    for (auto&& indexed_nh : r::indexed(r::filter(sorted.nodes(), is_island))){
        auto&& island_nh    = r::value(indexed_nh);
        const auto op_index = r::index(indexed_nh);

        op_indexes[island_nh] = op_index;

        //TODO: use async tasks for non CPU islands
        std::function<void()> body = [this, op_index](){
            GExecutor::run_op(m_ops[op_index], m_res);
        };
        tasks.emplace_back(std::move(body));

        auto island_dependencies_count = [&island_nh]() -> std::size_t{
            //task dependencies count is equal to the number of input slot objects
            //(as there is only one writer to the any slot object by definition)

            //does the slot has producers (or it is passed from outside)?
            auto slot_has_producer =  [](handle_t const& in_slot_nh){
                return ! in_slot_nh->inNodes().empty();
            };

            auto slot_producer = [](handle_t const& in_slot_nh){
                GAPI_Assert(in_slot_nh->inNodes().size() == 1 &&
                        "By definition data slot has a single producer");
                return in_slot_nh->inNodes().front();
            };

            //get producers of input data nodes what are result of other operations
            auto dependencies_range = r::map(
                    r::filter(island_nh->inNodes(), slot_has_producer),
                    slot_producer
            );
            //need to remove duplicates from the range (as island can produce several data slots)
            return node_handles_set_t{
                dependencies_range.begin(), dependencies_range.end()
            }.size();

        };

        auto dependent_islands_handles = [&island_nh]() -> node_handles_set_t {
            //get the dependent node_handles to convert them into operation indexes later
            node_handles_set_t dependent_node_handles;
            for (auto&& out_slot_nh : island_nh->outNodes()){
                auto dependent_islands = out_slot_nh->outNodes();
                dependent_node_handles.insert(dependent_islands.begin(), dependent_islands.end());
            }

            return dependent_node_handles;
        };

        tasks.back().dependencies = island_dependencies_count();
        tasks.back().dependency_count = tasks.back().dependencies;

        op_dependant_node_handles.emplace(op_index ,dependent_islands_handles());
    }

    auto total_order_index = tasks.size();

    //fill the tasks graph : set priority indexes, number of dependencies, and dependent nodes sets
    for (auto i : r::iota(tasks.size())){
        tasks[i].total_order_index = --total_order_index;

        auto dependents_range = r::map(
                r::toRange(op_dependant_node_handles.at(i)),
                [&](handle_t const& nh)-> parallel::tile_node *{
                    return &tasks.at(op_indexes.at(nh));
                }
        );
        tasks[i].dependants.assign(dependents_range.begin(), dependents_range.end());
    }

    using task_t = parallel::tile_node;
    auto start_task_r = r::map(
            r::filter(r::toRange(tasks),[](const task_t& task){
                return task.dependencies == 0;
            }),
            [](task_t& t) { return &t;}
    );
    start_tasks.assign(start_task_r.begin(), start_task_r.end());

    GAPI_LOG_INFO(NULL,
                 "Total tasks in graph :" <<tasks.size() <<"; "
              << "Start tasks count :" << start_tasks.size() << ";"
    );
}

void cv::gimpl::GTBBExecutor::runImpl(){
    parallel::prio_items_queue_t q {start_tasks.begin(), start_tasks.end()};
    parallel::execute(q);
}
#endif //USE_GAPI_TBB_EXECUTOR

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
        mag_rmat = make_rmat<RMatAdapter>(util::get<Mat>(arg)); break;
    case GRunArg::index_of<cv::RMat>() :
        mag_rmat = util::get<cv::RMat>(arg); break;
    default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
    }
    // FIXME: has to take extra care about meta here for this particular
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
        mag_rmat = make_rmat<RMatAdapter>(*util::get<Mat*>(arg)); break;
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
        // we need to actualy copy the data to the user buffer
        // since output runarg was optimized to simply point
        // to the input of the copy kernel
        // FIXME:
        // Rework, find a better way to check if there should be
        // a real copy (add a pass to StreamingBackend?)
        auto& out_mat = *util::get<cv::Mat*>(g_arg);
        const auto& rmat = mag.template slot<cv::RMat>().at(rc.id);
        auto mag_data = rmat.get<RMatAdapter>()->data();
        if (out_mat.data != mag_data) {
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
                rmat = make_rmat<RMatAdapter>(mat);
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

    GRunArgP get(int idx) override
    {
        auto r = magazine::getObjPtrExec(mag, desc()[idx]);
        // Remember the output port for this output object
        out_idx[cv::gimpl::proto::ptr(r)] = idx;
        return r;
    }
    void post(GRunArgP&&) override { } // Do nothing here
    void post(EndOfStream&&) override {} // Do nothing here too
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
};

void cv::gimpl::GExecutor::run_op(OpDesc& op, Mag& m_res)
{
    Input i{m_res, op.in_objects};
    Output o{m_res, op.out_objects};

    op.isl_exec->run(i,o);
}
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

    // (5) and  (6)
    // Run the script
    runImpl();
    // (7)
    for (auto it : ade::util::zip(ade::util::toRange(proto.outputs),
                                  ade::util::toRange(args.outObjs)))
    {
        magazine::writeBackExec(m_res, std::get<0>(it), std::get<1>(it));
    }
}

void cv::gimpl::GExecutor::runImpl() {
    for (auto &op : m_ops)
    {
        run_op(op, m_res);
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
    return (m_ops.size() == 1) && m_ops[0].isl_exec->canReshape();
}

void cv::gimpl::GExecutor::reshape(const GMetaArgs& inMetas, const GCompileArgs& args)
{
    GAPI_Assert(canReshape());
    auto& g = *m_orig_graph.get();
    ade::passes::PassContext ctx{g};
    passes::initMeta(ctx, inMetas);
    passes::inferMeta(ctx, true);
    m_ops[0].isl_exec->reshape(g, args);
}

void cv::gimpl::GExecutor::prepareForNewStream()
{
    for (auto &op : m_ops)
    {
        op.isl_exec->handleNewStream();
    }
}
