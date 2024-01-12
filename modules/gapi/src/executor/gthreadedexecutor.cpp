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

class cv::gimpl::GThreadedExecutor::Input final: public cv::gimpl::GIslandExecutable::IInput
{
    cv::gimpl::Mag &mag;
    std::mutex     &m_mutex;

    virtual StreamMsg get() override
    {
        std::lock_guard<std::mutex> lock{m_mutex};
        cv::GRunArgs res;
        for (const auto &rc : desc()) { res.emplace_back(magazine::getArg(mag, rc)); }
        return StreamMsg{std::move(res)};
    }
    virtual StreamMsg try_get() override { return get(); }
public:
    Input(cv::gimpl::Mag &m, const std::vector<RcDesc> &rcs, std::mutex& mutex)
        : mag(m), m_mutex(mutex) { set(rcs); }
};

class cv::gimpl::GThreadedExecutor::Output final: public cv::gimpl::GIslandExecutable::IOutput
{
    cv::gimpl::Mag &mag;
    std::mutex     &m_mutex;

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
        std::lock_guard<std::mutex> lock{m_mutex};
        magazine::assignMetaStubExec(mag, desc()[idx], m);
    }
public:
    Output(cv::gimpl::Mag &m, const std::vector<RcDesc> &rcs, std::mutex &mutex)
        : mag(m), m_mutex(mutex)
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


void cv::gimpl::GThreadedExecutor::initResource(const ade::NodeHandle &nh, const ade::NodeHandle &orig_nh) {
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
        GAPI_Error("InternalError");
    }
}

cv::gimpl::IslandActor::IslandActor(const std::vector<RcDesc>          &in_objects,
                                    const std::vector<RcDesc>          &out_objects,
                                    std::shared_ptr<GIslandExecutable> isl_exec,
                                    Mag                                &res,
                                    std::mutex                         &m)
    : m_in_objs(in_objects),
      m_out_objs(out_objects),
      m_isl_exec(isl_exec),
      m_res(res),
      m_mutex(m) {
}

void cv::gimpl::IslandActor::run() {
    cv::gimpl::GThreadedExecutor::Input  i{m_res,  m_in_objs,  m_mutex};
    cv::gimpl::GThreadedExecutor::Output o{m_res,  m_out_objs, m_mutex};
    m_isl_exec->run(i, o);
    // NB: Can't throw an exception there as it's supposed to be
    // executed in different threads
    try {
        o.verify();
    } catch (...) {
        m_e = std::current_exception();
    }
}

void cv::gimpl::IslandActor::verify() {
    if (m_e) {
        std::rethrow_exception(m_e);
    }
};

class cv::gimpl::Task {
public:
    using F = std::function<void()>;
    using Ptr = std::shared_ptr<Task>;

    Task(F f, std::vector<Task::Ptr> &&deps);

    void run();
    void reset() { m_ready_deps.store(0u); }

private:
    F                     m_f;
    const uint32_t        m_num_deps;
    std::atomic<uint32_t> m_ready_deps;
    std::vector<Task*>    m_dependents;
};

cv::gimpl::Task::Task(F f, std::vector<Task::Ptr> &&deps)
    : m_f(f),
      m_num_deps(static_cast<uint32_t>(deps.size())),
      m_ready_deps(0u) {
    for (auto dep : deps) {
        dep->m_dependents.push_back(this);
    }
}

void cv::gimpl::Task::run() {
    m_f();
    for (auto* dep : m_dependents) {
        if (dep->m_ready_deps.fetch_add(1u) == dep->m_num_deps - 1) {
            cv::gapi::own::ThreadPool::get()->schedule([dep](){
                dep->run();
            });
        }
    }
}

cv::gimpl::GThreadedExecutor::GThreadedExecutor(const uint32_t num_threads,
                                                std::unique_ptr<ade::Graph> &&g_model)
    : GAbstractExecutor(std::move(g_model)),
      m_tp(num_threads) {
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
                for (auto in_slot_nh  : nh->inNodes())  xtract(in_slot_nh,  input_rcs);
                for (auto out_slot_nh : nh->outNodes()) xtract(out_slot_nh, output_rcs);

                auto actor = std::make_shared<IslandActor>(std::move(input_rcs),
                                                           std::move(output_rcs),
                                                           m_gim.metadata(nh).get<IslandExec>().object,
                                                           m_res,
                                                           m_mutex);
                m_actors.push_back(actor);
                std::unordered_set<std::shared_ptr<Task>> deps;
                for (auto slot_nh : nh->inNodes()) {
                    for (auto island_nh : slot_nh->inNodes()) {
                        GAPI_Assert(m_gim.metadata(island_nh).get<NodeKind>().k == NodeKind::ISLAND);
                        deps.emplace(m_tasks.at(island_nh));
                    }
                }

                auto task = std::make_shared<Task>([actor](){actor->run();},
                                                   std::vector<std::shared_ptr<Task>>{deps.begin(), deps.end()});
                m_tasks.emplace(nh, task);
                if (deps.empty()) {
                    m_initial_tasks.push_back(task);
                }
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
    m_tp.start();
}

void cv::gimpl::GThreadedExecutor::run(cv::gimpl::GRuntimeArgs &&args) {
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

    // NB: Reset completion state for tasks and schedule them
    for (auto& it : m_tasks) { it.second->reset(); }
    for (auto task : m_initial_tasks) {
        m_tp.schedule([task](){ task->run(); });
    }
    // NB: Wait for the completion and verify for errors
    m_tp.wait();
    for (auto actor : m_actors) { actor->verify(); }

    for (auto it : ade::util::zip(ade::util::toRange(proto.outputs),
                                  ade::util::toRange(args.outObjs)))
    {
        magazine::writeBackExec(m_res, std::get<0>(it), std::get<1>(it));
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
    for (auto slot : m_slots)
    {
        initResource(slot.slot_nh, slot.data_nh);
    }

    for (auto actor : m_actors) {
        actor->exec()->reshape(g, args);
    }
}

void cv::gimpl::GThreadedExecutor::prepareForNewStream() {
    for (auto actor : m_actors)
    {
        actor->exec()->handleNewStream();
    }
}
