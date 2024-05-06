// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024 Intel Corporation


#include "precomp.hpp"

#include <ade/util/zip_range.hpp>

#include <opencv2/gapi/opencv_includes.hpp>

#include "api/gproto_priv.hpp" // ptr(GRunArgP)
#include "executor/gthreadedexecutor.hpp"
#include "compiler/passes/passes.hpp"

namespace cv {
namespace gimpl {
namespace magazine {
namespace {

void bindInArgExec(Mag& mag, const RcDesc &rc, const GRunArg &arg) {
    if (rc.shape != GShape::GMAT) {
        bindInArg(mag, rc, arg);
        return;
    }
    auto& mag_rmat = mag.template slot<cv::RMat>()[rc.id];
    switch (arg.index()) {
    case GRunArg::index_of<Mat>() :
        mag_rmat = make_rmat<RMatOnMat>(util::get<Mat>(arg));
        break;
    case GRunArg::index_of<cv::RMat>() :
        mag_rmat = util::get<cv::RMat>(arg);
        break;
    default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
    }
    // FIXME: has to take extra care about meta here for this particuluar
    // case, just because this function exists at all
    mag.meta<cv::RMat>()[rc.id] = arg.meta;
}

void bindOutArgExec(Mag& mag, const RcDesc &rc, const GRunArgP &arg) {
    if (rc.shape != GShape::GMAT) {
        bindOutArg(mag, rc, arg);
        return;
    }
    auto& mag_rmat = mag.template slot<cv::RMat>()[rc.id];
    switch (arg.index()) {
    case GRunArgP::index_of<Mat*>() :
        mag_rmat = make_rmat<RMatOnMat>(*util::get<Mat*>(arg)); break;
    case GRunArgP::index_of<cv::RMat*>() :
        mag_rmat = *util::get<cv::RMat*>(arg); break;
    default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
    }
}

cv::GRunArgP getObjPtrExec(Mag& mag, const RcDesc &rc) {
    if (rc.shape != GShape::GMAT) {
        return getObjPtr(mag, rc);
    }
    return GRunArgP(&mag.slot<cv::RMat>()[rc.id]);
}

void writeBackExec(const Mag& mag, const RcDesc &rc, GRunArgP &g_arg) {
    if (rc.shape != GShape::GMAT) {
        writeBack(mag, rc, g_arg);
        return;
    }

    switch (g_arg.index()) {
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
    switch (rc.shape) {
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

cv::gimpl::StreamMsg cv::gimpl::GThreadedExecutor::Input::get() {
    std::lock_guard<std::mutex> lock{m_state.m};
    cv::GRunArgs res;
    for (const auto &rc : desc()) { res.emplace_back(magazine::getArg(m_state.mag, rc)); }
    return cv::gimpl::StreamMsg{std::move(res)};
}

cv::gimpl::GThreadedExecutor::Input::Input(cv::gimpl::GraphState &state,
                                           const std::vector<RcDesc> &rcs)
    : m_state(state) {
    set(rcs);
};

cv::GRunArgP cv::gimpl::GThreadedExecutor::Output::get(int idx) {
    std::lock_guard<std::mutex> lock{m_state.m};
    auto r = magazine::getObjPtrExec(m_state.mag, desc()[idx]);
    // Remember the output port for this output object
    m_out_idx[cv::gimpl::proto::ptr(r)] = idx;
    return r;
}

void cv::gimpl::GThreadedExecutor::Output::post(cv::GRunArgP&&, const std::exception_ptr& e) {
    if (e) {
        m_eptr = e;
    }
}

void cv::gimpl::GThreadedExecutor::Output::post(Exception&& ex) {
    m_eptr = std::move(ex.eptr);
}

void cv::gimpl::GThreadedExecutor::Output::meta(const GRunArgP &out, const GRunArg::Meta &m) {
    const auto idx = m_out_idx.at(cv::gimpl::proto::ptr(out));
    std::lock_guard<std::mutex> lock{m_state.m};
    magazine::assignMetaStubExec(m_state.mag, desc()[idx], m);
}

cv::gimpl::GThreadedExecutor::Output::Output(cv::gimpl::GraphState &state,
                                             const std::vector<RcDesc> &rcs)
    : m_state(state) {
    set(rcs);
}

void cv::gimpl::GThreadedExecutor::Output::verify() {
    if (m_eptr) {
        std::rethrow_exception(m_eptr);
    }
}

void cv::gimpl::GThreadedExecutor::initResource(const ade::NodeHandle &nh, const ade::NodeHandle &orig_nh) {
    const Data &d = m_gm.metadata(orig_nh).get<Data>();

    if (   d.storage != Data::Storage::INTERNAL
        && d.storage != Data::Storage::CONST_VAL) {
        return;
    }

    // INTERNALS+CONST only! no need to allocate/reset output objects
    // to as it is bound externally (e.g. already in the m_state.mag)

    switch (d.shape) {
    case GShape::GMAT:
        if (d.storage == Data::Storage::CONST_VAL) {
            auto rc = RcDesc{d.rc, d.shape, d.ctor};
            magazine::bindInArgExec(m_state.mag, rc, m_gm.metadata(orig_nh).get<ConstValue>().arg);
        } else {
            // Let island allocate it's outputs if it can,
            // allocate cv::Mat and wrap it with RMat otherwise
            GAPI_Assert(!nh->inNodes().empty());
            const auto desc = util::get<cv::GMatDesc>(d.meta);
            auto& exec = m_gim.metadata(nh->inNodes().front()).get<IslandExec>().object;
            auto& rmat = m_state.mag.slot<cv::RMat>()[d.rc];
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
        if (d.storage == Data::Storage::CONST_VAL) {
            auto rc = RcDesc{d.rc, d.shape, d.ctor};
            magazine::bindInArg(m_state.mag, rc, m_gm.metadata(orig_nh).get<ConstValue>().arg);
        }
        break;

    case GShape::GARRAY:
        if (d.storage == Data::Storage::CONST_VAL) {
            auto rc = RcDesc{d.rc, d.shape, d.ctor};
            magazine::bindInArg(m_state.mag, rc, m_gm.metadata(orig_nh).get<ConstValue>().arg);
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
        GAPI_Error("InternalError");
    }
}

cv::gimpl::IslandActor::IslandActor(const std::vector<RcDesc>          &in_objects,
                                    const std::vector<RcDesc>          &out_objects,
                                    std::shared_ptr<GIslandExecutable> isl_exec,
                                    cv::gimpl::GraphState              &state)
    : m_isl_exec(isl_exec),
      m_inputs(state, in_objects),
      m_outputs(state, out_objects) {
}

void cv::gimpl::IslandActor::run() {
    m_isl_exec->run(m_inputs, m_outputs);
}

void cv::gimpl::IslandActor::verify() {
    m_outputs.verify();
};

class cv::gimpl::Task {
    friend class TaskManager;
public:
    using Ptr = std::shared_ptr<Task>;
    Task(TaskManager::F&& f, std::vector<Task::Ptr> &&producers);

    struct ExecutionState {
        cv::gapi::own::ThreadPool& tp;
        cv::gapi::own::Latch& latch;
    };

    void run(ExecutionState& state);
    bool isLast() const { return m_consumers.empty();  }
    void reset()        { m_ready_producers.store(0u); }

private:
    TaskManager::F           m_f;
    const uint32_t           m_num_producers;
    std::atomic<uint32_t>    m_ready_producers;
    std::vector<Task*>       m_consumers;
};

cv::gimpl::Task::Task(TaskManager::F         &&f,
                      std::vector<Task::Ptr> &&producers)
    : m_f(std::move(f)),
      m_num_producers(static_cast<uint32_t>(producers.size())) {
    for (auto producer : producers) {
        producer->m_consumers.push_back(this);
    }
}

void cv::gimpl::Task::run(ExecutionState& state) {
    // Execute the task
    m_f();
    // Notify every consumer about completion one of its dependencies
    for (auto* consumer : m_consumers) {
        const auto num_ready =
            consumer->m_ready_producers.fetch_add(1, std::memory_order_relaxed) + 1;
        // The last completed producer schedule the consumer for execution
        if (num_ready == consumer->m_num_producers) {
            state.tp.schedule([&state, consumer](){
                consumer->run(state);
            });
        }
    }
    // If tasks has no consumers this is the last task
    // Execution lasts until all last tasks are completed
    // Decrement the latch to notify about completion
    if (isLast()) {
        state.latch.count_down();
    }
}

std::shared_ptr<cv::gimpl::Task>
cv::gimpl::TaskManager::createTask(cv::gimpl::TaskManager::F &&f,
                                   std::vector<std::shared_ptr<cv::gimpl::Task>> &&producers) {
    const bool is_initial = producers.empty();
    auto task = std::make_shared<cv::gimpl::Task>(std::move(f),
                                                  std::move(producers));
    m_all_tasks.emplace_back(task);
    if (is_initial) {
        m_initial_tasks.emplace_back(task);
    }
    return task;
}

void cv::gimpl::TaskManager::scheduleAndWait(cv::gapi::own::ThreadPool& tp) {
    // Reset the number of ready dependencies for all tasks
    for (auto& task : m_all_tasks) { task->reset(); }

    // Count the number of last tasks
    auto isLast = [](const std::shared_ptr<Task>& task) { return task->isLast(); };
    const auto kNumLastsTasks =
        std::count_if(m_all_tasks.begin(), m_all_tasks.end(), isLast);

    // Initialize the latch, schedule initial tasks
    // and wait until all lasts tasks are done
    cv::gapi::own::Latch latch(kNumLastsTasks);
    Task::ExecutionState state{tp, latch};
    for (auto task : m_initial_tasks) {
        state.tp.schedule([&state, task](){ task->run(state); });
    }
    latch.wait();
}

cv::gimpl::GThreadedExecutor::GThreadedExecutor(const uint32_t num_threads,
                                                std::unique_ptr<ade::Graph> &&g_model)
    : GAbstractExecutor(std::move(g_model)),
      m_thread_pool(num_threads) {
    auto sorted = m_gim.metadata().get<ade::passes::TopologicalSortData>();

    std::unordered_map< ade::NodeHandle
                       , std::shared_ptr<Task>
                       , ade::HandleHasher<ade::Node>> m_tasks_map;
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
                for (auto in_slot_nh  : nh->inNodes())  xtract(in_slot_nh,  input_rcs);
                for (auto out_slot_nh : nh->outNodes()) xtract(out_slot_nh, output_rcs);

                auto actor = std::make_shared<IslandActor>(std::move(input_rcs),
                                                           std::move(output_rcs),
                                                           m_gim.metadata(nh).get<IslandExec>().object,
                                                           m_state);
                m_actors.push_back(actor);

                std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>> producer_nhs;
                for (auto slot_nh : nh->inNodes()) {
                    for (auto island_nh : slot_nh->inNodes()) {
                        GAPI_Assert(m_gim.metadata(island_nh).get<NodeKind>().k == NodeKind::ISLAND);
                        producer_nhs.emplace(island_nh);
                    }
                }
                std::vector<std::shared_ptr<Task>> producers;
                producers.reserve(producer_nhs.size());
                for (auto producer_nh : producer_nhs) {
                    producers.push_back(m_tasks_map.at(producer_nh));
                }
                auto task = m_task_manager.createTask(
                        [actor](){actor->run();}, std::move(producers));
                m_tasks_map.emplace(nh, task);
            }
            break;

        case NodeKind::SLOT:
            {
                const auto orig_data_nh
                    = m_gim.metadata(nh).get<DataSlot>().original_data_node;
                initResource(nh, orig_data_nh);
                m_slots.emplace_back(DataDesc{nh, orig_data_nh});
            }
            break;

        default:
            GAPI_Error("InternalError");
            break;
        } // switch(kind)
    } // for(gim nodes)

    prepareForNewStream();
}

void cv::gimpl::GThreadedExecutor::run(cv::gimpl::GRuntimeArgs &&args) {
    const auto proto = m_gm.metadata().get<Protocol>();

    // Basic check if input/output arguments are correct
    // FIXME: Move to GCompiled (do once for all GExecutors)
    if (proto.inputs.size() != args.inObjs.size()) { // TODO: Also check types
        util::throw_error(std::logic_error
                          ("Computation's input protocol doesn\'t "
                           "match actual arguments!"));
    }
    if (proto.outputs.size() != args.outObjs.size()) { // TODO: Also check types
        util::throw_error(std::logic_error
                          ("Computation's output protocol doesn\'t "
                           "match actual arguments!"));
    }

    namespace util = ade::util;

    // ensure that output Mat parameters are correctly allocated
    // FIXME: avoid copy of NodeHandle and GRunRsltComp ?
    for (auto index : util::iota(proto.out_nhs.size())) {
        auto& nh = proto.out_nhs.at(index);
        const Data &d = m_gm.metadata(nh).get<Data>();
        if (d.shape == GShape::GMAT) {
            using cv::util::get;
            const auto desc = get<cv::GMatDesc>(d.meta);

            auto check_rmat = [&desc, &args, &index]() {
                auto& out_mat = *get<cv::RMat*>(args.outObjs.at(index));
                GAPI_Assert(desc.canDescribe(out_mat));
            };

#if !defined(GAPI_STANDALONE)
            // Building as part of OpenCV - follow OpenCV behavior In
            // the case of cv::Mat if output buffer is not enough to
            // hold the result, reallocate it
            if (cv::util::holds_alternative<cv::Mat*>(args.outObjs.at(index))) {
                auto& out_mat = *get<cv::Mat*>(args.outObjs.at(index));
                createMat(desc, out_mat);
            }
            // In the case of RMat check to fit required meta
            else {
                check_rmat();
            }
#else
            // Building standalone - output buffer should always exist,
            // and _exact_ match our inferred metadata
            if (cv::util::holds_alternative<cv::Mat*>(args.outObjs.at(index))) {
                auto& out_mat = *get<cv::Mat*>(args.outObjs.at(index));
                GAPI_Assert(out_mat.data != nullptr &&
                        desc.canDescribe(out_mat));
            }
            // In the case of RMat check to fit required meta
            else {
                check_rmat();
            }
#endif // !defined(GAPI_STANDALONE)
        }
    }
    // Update storage with user-passed objects
    for (auto it : ade::util::zip(ade::util::toRange(proto.inputs),
                                  ade::util::toRange(args.inObjs))) {
        magazine::bindInArgExec(m_state.mag, std::get<0>(it), std::get<1>(it));
    }
    for (auto it : ade::util::zip(ade::util::toRange(proto.outputs),
                                  ade::util::toRange(args.outObjs))) {
        magazine::bindOutArgExec(m_state.mag, std::get<0>(it), std::get<1>(it));
    }

    // Reset internal data
    for (auto &sd : m_slots) {
        const auto& data = m_gm.metadata(sd.data_nh).get<Data>();
        magazine::resetInternalData(m_state.mag, data);
    }

    m_task_manager.scheduleAndWait(m_thread_pool);
    for (auto actor : m_actors) {
        actor->verify();
    }
    for (auto it : ade::util::zip(ade::util::toRange(proto.outputs),
                                  ade::util::toRange(args.outObjs))) {
        magazine::writeBackExec(m_state.mag, std::get<0>(it), std::get<1>(it));
    }
}

bool cv::gimpl::GThreadedExecutor::canReshape() const {
    for (auto actor : m_actors) {
        if (actor->exec()->canReshape()) {
            return false;
        }
    }
    return true;
}

void cv::gimpl::GThreadedExecutor::reshape(const GMetaArgs& inMetas, const GCompileArgs& args) {
    GAPI_Assert(canReshape());
    auto& g = *m_orig_graph.get();
    ade::passes::PassContext ctx{g};
    passes::initMeta(ctx, inMetas);
    passes::inferMeta(ctx, true);

    // NB: Before reshape islands need to re-init resources for every slot.
    for (auto slot : m_slots) {
        initResource(slot.slot_nh, slot.data_nh);
    }

    for (auto actor : m_actors) {
        actor->exec()->reshape(g, args);
    }
}

void cv::gimpl::GThreadedExecutor::prepareForNewStream() {
    for (auto actor : m_actors) {
        actor->exec()->handleNewStream();
    }
}
