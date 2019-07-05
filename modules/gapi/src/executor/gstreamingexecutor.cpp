// This file is part of OpenCV project.

// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "precomp.hpp"


#include <iostream>

#include <ade/util/zip_range.hpp>

#if !defined(GAPI_STANDALONE)
#include <opencv2/videoio.hpp>   // FIXME: MOVE THIS DEPENDENCY OUT!!1
#endif // GAPI_STANDALONE

#include <opencv2/gapi/opencv_includes.hpp>

#include "executor/gstreamingexecutor.hpp"
#include "compiler/passes/passes.hpp"
#include "backends/common/gbackend.hpp" // createMat

namespace
{
using namespace cv::gimpl::stream;

#if !defined(GAPI_STANDALONE)
class VideoEmitter final: public cv::gimpl::GIslandEmitter {
    // FIXME: This is a huge dependency for core G-API library!
    // It needs to be moved out to some separate module.
    cv::VideoCapture vcap;

    virtual bool pull(cv::GRunArg &arg) override {
        // FIXME: probably we can maintain a pool of (then) pre-allocated
        // buffers to avoid runtime allocations.
        // Pool size can be determined given the internal queue size.
        cv::Mat nextFrame;
        vcap >> nextFrame;
        if (nextFrame.empty()) {
            return false;
        }
        arg = std::move(nextFrame);
        return true;
    }
public:
    explicit VideoEmitter(const cv::GRunArg &arg) {
        const auto &param = cv::util::get<cv::gapi::GVideoCapture>(arg);
        GAPI_Assert(vcap.open(param.path));
    }
};
#endif // GAPI_STANDALONE

class ConstEmitter final: public cv::gimpl::GIslandEmitter {
    cv::GRunArg m_arg;

    virtual bool pull(cv::GRunArg &arg) override {
        arg = const_cast<const cv::GRunArg&>(m_arg); // FIXME: variant workaround
        return true;
    }
public:

    explicit ConstEmitter(const cv::GRunArg &arg) : m_arg(arg) {
    }
};

struct DataQueue {
    static const char *name() { return "StreamingDataQueue"; }

    explicit DataQueue(std::size_t capacity) {
        if (capacity) {
            q.set_capacity(capacity);
        }
    }

    cv::gimpl::stream::Q q;
};

std::vector<cv::gimpl::stream::Q*> reader_queues(      ade::Graph &g,
                                                 const ade::NodeHandle &obj)
{
    ade::TypedGraph<DataQueue> qgr(g);
    std::vector<cv::gimpl::stream::Q*> result;
    for (auto &&out_eh : obj->outEdges())
    {
        result.push_back(&qgr.metadata(out_eh).get<DataQueue>().q);
    }
    return result;
}

std::vector<cv::gimpl::stream::Q*> input_queues(      ade::Graph &g,
                                                const ade::NodeHandle &obj)
{
    ade::TypedGraph<DataQueue> qgr(g);
    std::vector<cv::gimpl::stream::Q*> result;
    for (auto &&in_eh : obj->inEdges())
    {
        result.push_back(&qgr.metadata(in_eh).get<DataQueue>().q);
    }
    return result;
}

void sync_data(cv::GRunArgs &results, cv::GRunArgsP &outputs)
{
    namespace own = cv::gapi::own;

    for (auto && it : ade::util::zip(ade::util::toRange(outputs),
                                     ade::util::toRange(results)))
    {
        auto &out_obj = std::get<0>(it);
        auto &res_obj = std::get<1>(it);

        // FIXME: this conversion should be unified
        switch (out_obj.index())
        {
#if !defined(GAPI_STANDALONE)
        case out_obj.index_of<cv::Mat*>():
            *cv::util::get<cv::Mat*>(out_obj) = std::move(cv::util::get<cv::Mat>(res_obj));
            break;
        case out_obj.index_of<cv::Scalar*>():
            *cv::util::get<cv::Scalar*>(out_obj) = std::move(cv::util::get<cv::Scalar>(res_obj));
            break;
#endif // GAPI_STANDALONE
        case out_obj.index_of<own::Mat*>():
            *cv::util::get<own::Mat*>(out_obj) = std::move(cv::util::get<own::Mat>(res_obj));
            break;
        case out_obj.index_of<own::Scalar*>():
            *cv::util::get<own::Scalar*>(out_obj) = std::move(cv::util::get<own::Scalar>(res_obj));
            break;
        case out_obj.index_of<cv::detail::VectorRef>():
            cv::util::get<cv::detail::VectorRef>(out_obj).mov(cv::util::get<cv::detail::VectorRef>(res_obj));
            break;
        default:
            GAPI_Assert(false && "This value type is not supported!"
#if defined(GAPI_STANDALONE)
                                 " (probably because of STANDALONE mode"
#endif // GAPI_STANDALONE
                        );
            break;
        }
    }
}

// This thread is a plain dump source actor. What it do is just:
// - Check input queue (the only one) for a control command
// - Depending on the state, obtains next data object and pushes it to the
//   pipeline.
void emitterActorThread(std::shared_ptr<cv::gimpl::GIslandEmitter> emitter,
                        Q& in_queue,
                        std::vector<Q*> out_queues)
{
    // Wait for the explicit Start command.
    // ...or Stop command, this also happens.
    Cmd cmd;
    in_queue.pop(cmd);
    GAPI_Assert(   cmd.index() == cmd.index_of<Start>()
              || cmd.index() == cmd.index_of<Stop>());
    if (cmd.index() == cmd.index_of<Stop>()) {
        for (auto &&oq : out_queues) oq->push(cmd);
        return;
    }

    // Now start emitting the data from the source to the pipeline.
    while (true) {
        Cmd cancel;
        if (in_queue.try_pop(cancel)) {
            // if we just popped a cancellation command...
            GAPI_Assert(cancel.index() == cancel.index_of<Stop>());
            // Broadcast it to the readers and quit.
            for (auto &&oq : out_queues) oq->push(cancel);
            return;
        }

        // Try to obrain next data chunk from the source
        cv::GRunArg data;
        if (emitter->pull(data)) {
            // // On success, broadcast it to our readers
            for (auto &&oq : out_queues)
            {
                // FIXME: FOR SOME REASON, oq->push(Cmd{data}) doesn't work!!
                // empty mats are arrived to the receivers!
                // There may be a fatal bug in our variant!
                const auto tmp = data;
                oq->push(Cmd{tmp});
            }
        } else {
            // On failure, broadcast STOP message to our readers and quit.
            for (auto &&oq : out_queues) oq->push(Cmd{Stop{}});
            return;
        }
    }
}

// This thread is a plain dumb processing actor. What it do is just:
// - Reads input from the input queue(s), sleeps if there's nothing to read
// - Once a full input vector is obtained, passes it to the underlying island
//   executable for processing.
// - Pushes processing results down to consumers - to the subsequent queues.
//   Note: Every data object consumer has its own queue.
void islandActorThread(std::vector<cv::gimpl::RcDesc> in_rcs,                // FIXME: this is...
                       std::vector<cv::gimpl::RcDesc> out_rcs,               // FIXME: ...basically just...
                       cv::GMetaArgs out_metas,                              // ...
                       std::shared_ptr<cv::gimpl::GIslandExecutable> island, // FIXME: ...a copy of OpDesc{}.
                       std::vector<Q*> in_queues,
                       std::vector< std::vector<Q*> > out_queues)
{
    GAPI_Assert(in_queues.size() == in_rcs.size());
    GAPI_Assert(out_queues.size() == out_rcs.size());
    GAPI_Assert(out_queues.size() == out_metas.size());
    while (true)
    {
        std::vector<cv::gimpl::GIslandExecutable::InObj> isl_inputs;
        isl_inputs.resize(in_rcs.size());

        // Try to obtain the full input vector.
        // Note this may block us. We also may get Stop signal here
        // and then exit the thread.
        Cmd cmd;
        for (auto &&it : ade::util::indexed(in_queues))
        {
            auto id = ade::util::index(it);
            auto &q = ade::util::value(it);
            q->pop(cmd);
            if (cmd.index() == cmd.index_of<Stop>()) {
                // Broadcast STOP down to the pipeline.
                for (auto &&out_qq : out_queues)
                {
                    for (auto &&out_q : out_qq) out_q->push(cmd);
                }
                return;
            }
            // FIXME: MOVE PROBLEM
            const cv::GRunArg &in_arg = cv::util::get<cv::GRunArg>(cmd);
            isl_inputs[id].first  = in_rcs[id];
            isl_inputs[id].second = in_arg;
        }

        // Once the vector is obtained, prepare data for island execution
        // Note - we first allocate output vector via GRunArg!
        // Then it is converted to a GRunArgP.
        std::vector<cv::gimpl::GIslandExecutable::OutObj> isl_outputs;
        std::vector<cv::GRunArg> out_data;
        isl_outputs.resize(out_rcs.size());
        out_data.resize(out_rcs.size());
        for (auto &&it : ade::util::indexed(out_rcs))
        {
            auto id = ade::util::index(it);
            auto &r = ade::util::value(it);

#if !defined(GAPI_STANDALONE)
            using MatType = cv::Mat;
            using SclType = cv::Scalar;
#else
            using MatType = cv::gapi::own::Mat;
            using SclType = cv::gapi::own::Scalar;
#endif // GAPI_STANDALONE

            switch (r.shape) {
                // Allocate a data object based on its shape & meta, and put it into our vectors.
                // Yes, first we put a cv::Mat GRunArg, and then specify _THAT_
                // pointer as an output parameter - to make sure that after island completes,
                // our GRunArg still has the right (up-to-date) value.
                // Same applies to other types.
                // FIXME: This is absolutely ugly but seem to work perfectly for its purpose.
            case cv::GShape::GMAT:
                {
                    MatType newMat;
                    cv::gimpl::createMat(cv::util::get<cv::GMatDesc>(out_metas[id]), newMat);
                    out_data[id] = cv::GRunArg(std::move(newMat));
                    isl_outputs[id] = { r, cv::GRunArgP(&cv::util::get<MatType>(out_data[id])) };
                }
                break;
            case cv::GShape::GSCALAR:
                {
                    SclType newScl;
                    out_data[id] = cv::GRunArg(std::move(newScl));
                    isl_outputs[id] = { r, cv::GRunArgP(&cv::util::get<SclType>(out_data[id])) };
                }
                break;
            case cv::GShape::GARRAY:
                {
                    cv::detail::VectorRef newVec;
                    cv::util::get<cv::detail::ConstructVec>(r.ctor)(newVec);
                    out_data[id]= cv::GRunArg(std::move(newVec));
                    // VectorRef is implicitly shared so no pointer is taken here
                    const auto &rr = cv::util::get<cv::detail::VectorRef>(out_data[id]); // FIXME: that variant MOVE problem again
                    isl_outputs[id] = { r, cv::GRunArgP(rr) };
                }
                break;
            default:
                cv::util::throw_error(std::logic_error("Unsupported GShape"));
                break;
            }
        }
        // Now ask Island to execute on this data
        island->run(std::move(isl_inputs), std::move(isl_outputs));

        // Once executed, dispatch our results down to the pipeline.
        for (auto &&it : ade::util::zip(ade::util::toRange(out_queues),
                                        ade::util::toRange(out_data)))
        {
            for (auto &&q : std::get<0>(it))
            {
                // FIXME: FATAL VARIANT ISSUE!!
                const auto tmp = std::get<1>(it);
                q->push(Cmd{tmp});
            }
        }
    }
}

void collectorThread(std::vector<Q*> in_queues,
                     Q&              out_queue)
{
    while (true)
    {
        cv::GRunArgs this_result(in_queues.size());
        std::size_t stops = 0u;
        for (auto &&it : ade::util::indexed(in_queues))
        {
            Cmd cmd;
            ade::util::value(it)->pop(cmd);
            if (cmd.index() == cmd.index_of<Stop>()) {
                stops++;
            } else {
                // FIXME: MOVE_PROBLEM
                const cv::GRunArg &in_arg = cv::util::get<cv::GRunArg>(cmd);
                this_result[ade::util::index(it)] = in_arg;
                // FIXME: Check for other message types.
            }
        }
        if (stops > 0)
        {
            GAPI_Assert(stops == in_queues.size());
            out_queue.push(Cmd{Stop{}});
            return;
        }
        out_queue.push(Cmd{this_result});
    }
}
} // anonymous namespace

cv::gimpl::GStreamingExecutor::GStreamingExecutor(std::unique_ptr<ade::Graph> &&g_model)
    : m_orig_graph(std::move(g_model))
    , m_island_graph(GModel::Graph(*m_orig_graph).metadata()
                     .get<IslandModel>().model)
    , m_gm(*m_orig_graph)
    , m_gim(*m_island_graph)
{
    // NB: Right now GIslandModel is acyclic, and all the below code assumes that.
    // NB: This naive execution code is taken from GExecutor nearly "as-is"

    const auto proto = m_gm.metadata().get<Protocol>();
    m_emitters      .resize(proto.in_nhs.size());
    m_emitter_queues.resize(proto.in_nhs.size());
    m_sinks         .resize(proto.out_nhs.size());
    m_sink_queues   .resize(proto.out_nhs.size());

    // Very rough estimation to limit internal queue sizes.
    // Pipeline depth is equal to number of its (pipeline) steps.
    const auto queue_capacity = std::count_if
        (m_gim.nodes().begin(),
         m_gim.nodes().end(),
         [&](ade::NodeHandle nh) {
            return m_gim.metadata(nh).get<NodeKind>().k == NodeKind::ISLAND;
         });

    auto sorted = m_gim.metadata().get<ade::passes::TopologicalSortData>();
    for (auto nh : sorted.nodes())
    {
        switch (m_gim.metadata(nh).get<NodeKind>().k)
        {
        case NodeKind::ISLAND:
            {
                std::vector<RcDesc> input_rcs;
                std::vector<RcDesc> output_rcs;
                cv::GMetaArgs output_metas;
                input_rcs.reserve(nh->inNodes().size());
                output_rcs.reserve(nh->outNodes().size());
                output_metas.reserve(nh->outNodes().size());

                // FIXME: THIS ORDER IS IRRELEVANT TO PROTOCOL OR ANY OTHER ORDER!
                // FIXME: SAME APPLIES TO THE REGULAR GEEXECUTOR!!
                auto xtract_in = [&](ade::NodeHandle slot_nh, std::vector<RcDesc> &vec) {
                    const auto orig_data_nh
                        = m_gim.metadata(slot_nh).get<DataSlot>().original_data_node;
                    const auto &orig_data_info
                        = m_gm.metadata(orig_data_nh).get<Data>();
                    if (orig_data_info.shape == GShape::GARRAY) {
                        GAPI_Assert(orig_data_info.ctor.index() != orig_data_info.ctor.index_of<cv::util::monostate>());
                    }
                    vec.emplace_back(RcDesc{ orig_data_info.rc
                                           , orig_data_info.shape
                                           , orig_data_info.ctor});
                };
                auto xtract_out = [&](ade::NodeHandle slot_nh, std::vector<RcDesc> &vec, cv::GMetaArgs &metas) {
                    const auto orig_data_nh
                        = m_gim.metadata(slot_nh).get<DataSlot>().original_data_node;
                    const auto &orig_data_info
                        = m_gm.metadata(orig_data_nh).get<Data>();
                    if (orig_data_info.shape == GShape::GARRAY) {
                        GAPI_Assert(orig_data_info.ctor.index() != orig_data_info.ctor.index_of<cv::util::monostate>());
                    }
                    vec.emplace_back(RcDesc{ orig_data_info.rc
                                           , orig_data_info.shape
                                           , orig_data_info.ctor});
                    metas.emplace_back(orig_data_info.meta);
                };
                // FIXME: JEZ IT WAS SO AWFUL!!!!
                for (auto in_slot_nh  : nh->inNodes())  xtract_in(in_slot_nh,  input_rcs);
                for (auto out_slot_nh : nh->outNodes()) xtract_out(out_slot_nh, output_rcs, output_metas);

                m_ops.emplace_back(OpDesc{ std::move(input_rcs)
                                         , std::move(output_rcs)
                                         , std::move(output_metas)
                                         , nh
                                         , m_gim.metadata(nh).get<IslandExec>().object});

                // Initialize queues for every operation's input
                ade::TypedGraph<DataQueue> qgr(*m_island_graph);
                for (auto eh : nh->inEdges())
                {
                    qgr.metadata(eh).set(DataQueue(queue_capacity));
                }
            }
            break;
        case NodeKind::SLOT:
            {
                const auto orig_data_nh
                    = m_gim.metadata(nh).get<DataSlot>().original_data_node;
                m_slots.emplace_back(DataDesc{nh, orig_data_nh});
            }
            break;
        case NodeKind::EMIT:
            {
                const auto emitter_idx
                    = m_gim.metadata(nh).get<Emitter>().proto_index;
                GAPI_Assert(emitter_idx < m_emitters.size());
                m_emitters[emitter_idx] = nh;
            }
            break;
        case NodeKind::SINK:
            {
                const auto sink_idx
                    = m_gim.metadata(nh).get<Sink>().proto_index;
                GAPI_Assert(sink_idx < m_sinks.size());
                m_sinks[sink_idx] = nh;

                // Also initialize Sink's input queue
                ade::TypedGraph<DataQueue> qgr(*m_island_graph);
                GAPI_Assert(nh->inEdges().size() == 1u);
                qgr.metadata(nh->inEdges().front()).set(DataQueue(queue_capacity));
                m_sink_queues[sink_idx] = &qgr.metadata(nh->inEdges().front()).get<DataQueue>().q;
            }
            break;
        default:
            GAPI_Assert(false);
            break;
        } // switch(kind)
    } // for(gim nodes)
    m_out_queue.set_capacity(queue_capacity);
}

void cv::gimpl::GStreamingExecutor::setSource(GRunArgs &&ins)
{
    bool has_video = false;

    // Walk through the protocol, set-up emitters appropriately
    // There's a 1:1 mapping between emitters and corresponding data inputs.
    for (auto it : ade::util::zip(ade::util::toRange(m_emitters),
                                  ade::util::toRange(ins)))
    {
        auto  emit_nh  = std::get<0>(it);
        auto& emit_arg = std::get<1>(it);
        auto& emitter  = m_gim.metadata(emit_nh).get<Emitter>().object;

        switch (emit_arg.index())
        {
        // Create a streaming emitter.
        // Produces the next video frame when pulled.
        case emit_arg.index_of<cv::gapi::GVideoCapture>():
            if (has_video)
                util::throw_error(std::logic_error("Only one video source is"
                                                   " currently supported!"));
            has_video = true;
#if !defined(GAPI_STANDALONE)
            emitter.reset(new VideoEmitter{emit_arg});
#else
            util::throw_error(std::logic_error("Video is not supported in the "
                                               "standalone mode"));
#endif
            break;

        default:
            // Create a constant emitter.
            // Produces always the same ("constant") value when pulled.
            emitter.reset(new ConstEmitter{emit_arg});
            break;
        }
    }

    // FIXME: ONLY now, after all executable objects are created,
    // we can set up our execution threads. Let's do it.
    // First create threads for all the emitters.
    // FIXME: One way to avoid this may be including an Emitter object as a part of
    // START message. Why not?
    GAPI_Assert(m_threads.empty()); // FIXME: NOW WE CAN RUN ONLY ONCE!!!
    for (auto it : ade::util::indexed(m_emitters))
    {
        const auto id = ade::util::index(it); // = index in GComputation's protocol
        const auto eh = ade::util::value(it);

        // Prepare emitter thread parameters
        auto emitter = m_gim.metadata(eh).get<Emitter>().object;

        // Collect all reader queues from the emitter's the only output object
        auto out_queues = reader_queues(*m_island_graph, eh->outNodes().front());

        m_threads.emplace_back(emitterActorThread,
                               emitter,
                               std::ref(m_emitter_queues[id]),
                               out_queues);
    }

    // Now do this for every island (in a topological order)
    for (auto &&op : m_ops)
    {
        // Prepare island thread parameters
        auto island = m_gim.metadata(op.nh).get<IslandExec>().object;

        // Collect actor's input queues
        auto in_queues = input_queues(*m_island_graph, op.nh);

        // Collect actor's output queues.
        // This may be tricky...
        std::vector< std::vector<stream::Q*> > out_queues;
        for (auto &&out_eh : op.nh->outNodes()) {
            out_queues.push_back(reader_queues(*m_island_graph, out_eh));
        }

        m_threads.emplace_back(islandActorThread,
                               op.in_objects,
                               op.out_objects,
                               op.out_metas,
                               island,
                               in_queues,
                               out_queues);
    }

    // Finally, start a collector thread.
    m_threads.emplace_back(collectorThread,
                           m_sink_queues,
                           std::ref(m_out_queue));
}

void cv::gimpl::GStreamingExecutor::start()
{
    // FIXME: start/stop/pause/etc logic
    GAPI_Assert(state == State::STOPPED);

    // Currently just trigger our emitters to work
    state = State::RUNNING;
    for (auto &q : m_emitter_queues)
    {
        q.push(stream::Cmd{stream::Start{}});
    }
}

bool cv::gimpl::GStreamingExecutor::pull(cv::GRunArgsP &&outs)
{
    if (state == State::STOPPED)
        return false;

    GAPI_Assert(m_sink_queues.size() == outs.size());

    Cmd cmd;
    m_out_queue.pop(cmd);
    if (cmd.index() == cmd.index_of<Stop>())
    {
        for (auto &t : m_threads) t.join();
        state = State::STOPPED;
        return false;
    }

    GAPI_Assert(cmd.index() == cmd.index_of<cv::GRunArgs>());
    cv::GRunArgs &this_result = cv::util::get<cv::GRunArgs>(cmd);
    sync_data(this_result, outs);
    return true;
}

bool cv::gimpl::GStreamingExecutor::try_pull(cv::GRunArgsP &&outs)
{
    if (state == State::STOPPED)
        return false;

    GAPI_Assert(m_sink_queues.size() == outs.size());

    Cmd cmd;
    if (!m_out_queue.try_pop(cmd)) {
        return false;
    }
    if (cmd.index() == cmd.index_of<Stop>())
    {
        // FIXME: Unify with pull()
        for (auto &t : m_threads) t.join();
        state = State::STOPPED;
        return false;
    }

    GAPI_Assert(cmd.index() == cmd.index_of<cv::GRunArgs>());
    cv::GRunArgs &this_result = cv::util::get<cv::GRunArgs>(cmd);
    sync_data(this_result, outs);
    return true;
}

void cv::gimpl::GStreamingExecutor::stop()
{
    if (state == State::STOPPED)
        return;

    // FIXME: ...and how to deal with still-unread data then?
    // Push a Stop message to the every emitter,
    // wait until it broadcasts within the pipeline,
    // FIXME: worker threads could stuck on push()!
    // need to read the output queues until Stop!
    for (auto &q : m_emitter_queues) {
        q.push(stream::Cmd{stream::Stop{}});
    }

    // Pull messages from the final queue to ensure completion
    Cmd cmd;
    while (cmd.index() != cmd.index_of<Stop>()) {
        m_out_queue.pop(cmd);
    }
    GAPI_Assert(cmd.index() == cmd.index_of<Stop>());

    for (auto &t : m_threads) {
        t.join();
    }
    m_threads.clear();
    // FIXME: Auto-stop on object destruction?
    // There still must be a graceful shutdown!!!
}

bool cv::gimpl::GStreamingExecutor::running() const
{
    return (state == State::RUNNING);
}
