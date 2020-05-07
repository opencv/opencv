// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <iostream>

#include <ade/util/zip_range.hpp>
#include <ade/util/filter_range.hpp>

#include <opencv2/gapi/opencv_includes.hpp>
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

    auto sorted = m_gim.metadata().get<ade::passes::TopologicalSortData>();
    for (auto&& nh : ade::util::filter(sorted.nodes(), is_slot)){
        const auto orig_data_nh
            = m_gim.metadata(nh).get<DataSlot>().original_data_node;
        // (1)
        initResource(orig_data_nh);
        m_slots.emplace_back(DataDesc{nh, orig_data_nh});
    }

    using op_index_t = size_t;
    std::unordered_map<ade::NodeHandle, op_index_t, ade::HandleHasher<ade::Node>> op_indexes;
    std::unordered_map<op_index_t, std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>>> op_dependees_node_handles;

    namespace r = ade::util::Range;
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
#if defined(USE_GAPI_TBB_EXECUTOR)
        //TBB executor staff ------
        const auto op_index = m_ops.size() - 1;
        op_indexes[nh] = op_index;

        //TODO: use async tasks for non CPU islands
        std::function<void()> body = [this, op_index](){
            GExecutor::run_op(m_ops[op_index], m_res);
        };
        tasks.emplace_back(std::move(body));

        auto dependees_range = r::map(
                r::filter(nh->inNodes(), [](ade::NodeHandle const& in_slot_nh){
                    //count input data nodes what are result of other operation
                    return ! in_slot_nh->inNodes().empty();
                }),
                [](ade::NodeHandle const& in_slot_nh){
                    return in_slot_nh->inNodes().front();
                }
        );

        //task dependencies is equal to the number of input slot objects (as there is only one writer to the any slot)
        tasks.back().dependencies = std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>>{dependees_range.begin(), dependees_range.end()}.size();
        tasks.back().dependency_count = tasks.back().dependencies;

        //get the dependees node_handles to convert them into op indexes later
        std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>> dependees_node_handles;
        for (auto&& out_slot_nh : nh->outNodes()){
            auto dependee_islands = out_slot_nh->outNodes();
            dependees_node_handles.insert(dependee_islands.begin(), dependee_islands.end());
        }
        op_dependees_node_handles[op_index] = std::move(dependees_node_handles);
#endif //USE_GAPI_TBB_EXECUTOR
    }

#if defined(USE_GAPI_TBB_EXECUTOR)
    auto total_order_index = tasks.size();

    //fill the tasks graph : set priority indexes, number of deps, and dependees

    for (auto i : ade::util::Range::iota(tasks.size())){
        tasks[i].total_order_index = --total_order_index;

        auto dependees_range = ade::util::Range::map(ade::util::Range::toRange(op_dependees_node_handles.at(i)), [&](ade::NodeHandle const& nh)-> parallel::tile_node *{
            return &tasks.at(op_indexes.at(nh));
        });
        tasks[i].dependees.assign(dependees_range.begin(), dependees_range.end());
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
#endif //USE_GAPI_TBB_EXECUTOR

}

void cv::gimpl::GExecutor::initResource(const ade::NodeHandle &orig_nh)
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
            const auto desc = util::get<cv::GMatDesc>(d.meta);
            auto& mat = m_res.slot<cv::Mat>()[d.rc];
            createMat(desc, mat);
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
    case GShape::GOPAQUE:
        // Constructed on Reset, do nothing here
        break;

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
    virtual GRunArgP get(int idx) override { return magazine::getObjPtr(mag, desc()[idx]); }
    virtual void post(GRunArgP&&) override { } // Do nothing here
    virtual void post(EndOfStream&&) override {} // Do nothing here too
public:
    Output(cv::gimpl::Mag &m, const std::vector<RcDesc> &rcs) : mag(m) { set(rcs); }
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

            auto check_own_mat = [&desc, &args, &index]()
            {
                auto& out_mat = *get<cv::Mat*>(args.outObjs.at(index));
                GAPI_Assert(out_mat.data != nullptr &&
                        desc.canDescribe(out_mat));
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
            // In the case of own::Mat never reallocated, checked to perfectly fit required meta
            else
            {
                check_own_mat();
            }
#else
            // Building standalone - output buffer should always exist,
            // and _exact_ match our inferred metadata
            check_own_mat();
#endif // !defined(GAPI_STANDALONE)
        }
    }
    // Update storage with user-passed objects
    for (auto it : ade::util::zip(ade::util::toRange(proto.inputs),
                                  ade::util::toRange(args.inObjs)))
    {
        magazine::bindInArg(m_res, std::get<0>(it), std::get<1>(it));
    }
    for (auto it : ade::util::zip(ade::util::toRange(proto.outputs),
                                  ade::util::toRange(args.outObjs)))
    {
        magazine::bindOutArg(m_res, std::get<0>(it), std::get<1>(it));
    }

    // Reset internal data
    for (auto &sd : m_slots)
    {
        const auto& data = m_gm.metadata(sd.data_nh).get<Data>();
        magazine::resetInternalData(m_res, data);
    }

    // Run the script
#if defined(USE_GAPI_TBB_EXECUTOR)
    tbb::concurrent_priority_queue<parallel::tile_node* , parallel::tile_node_indirect_priority_comparator> q {start_tasks.begin(), start_tasks.end()};
    parallel::execute(q);
#else
    for (auto &op : m_ops)
    {
        // (5) and  (6)
        run_op(op, m_res);
    }
#endif
    // (7)
    for (auto it : ade::util::zip(ade::util::toRange(proto.outputs),
                                  ade::util::toRange(args.outObjs)))
    {
        magazine::writeBack(m_res, std::get<0>(it), std::get<1>(it));
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
