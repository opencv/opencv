// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2020 Intel Corporation

#include "precomp.hpp"

#include <iostream>

#include <ade/util/zip_range.hpp>

#include <opencv2/gapi/opencv_includes.hpp>

#include "api/gproto_priv.hpp" // ptr(GRunArgP)
#include "compiler/passes/passes.hpp"
#include "backends/common/gbackend.hpp" // createMat
#include "compiler/gcompiler.hpp" // for compileIslands

#include "executor/gstreamingexecutor.hpp"

namespace
{
using namespace cv::gimpl::stream;

#if !defined(GAPI_STANDALONE)
class VideoEmitter final: public cv::gimpl::GIslandEmitter {
    cv::gapi::wip::IStreamSource::Ptr src;

    virtual bool pull(cv::GRunArg &arg) override {
        // FIXME: probably we can maintain a pool of (then) pre-allocated
        // buffers to avoid runtime allocations.
        // Pool size can be determined given the internal queue size.
        cv::gapi::wip::Data newData;
        if (!src->pull(newData)) {
            return false;
        }
        arg = std::move(static_cast<cv::GRunArg&>(newData));
        return true;
    }
public:
    explicit VideoEmitter(const cv::GRunArg &arg) {
        src = cv::util::get<cv::gapi::wip::IStreamSource::Ptr>(arg);
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
        result.push_back(qgr.metadata(in_eh).contains<DataQueue>()
                         ? &qgr.metadata(in_eh).get<DataQueue>().q
                         : nullptr);
    }
    return result;
}

void sync_data(cv::GRunArgs &results, cv::GRunArgsP &outputs)
{
    for (auto && it : ade::util::zip(ade::util::toRange(outputs),
                                     ade::util::toRange(results)))
    {
        auto &out_obj = std::get<0>(it);
        auto &res_obj = std::get<1>(it);

        // FIXME: this conversion should be unified
        using T = cv::GRunArgP;
        switch (out_obj.index())
        {
        case T::index_of<cv::Mat*>():
            *cv::util::get<cv::Mat*>(out_obj) = std::move(cv::util::get<cv::Mat>(res_obj));
            break;
        case T::index_of<cv::Scalar*>():
            *cv::util::get<cv::Scalar*>(out_obj) = std::move(cv::util::get<cv::Scalar>(res_obj));
            break;
        case T::index_of<cv::detail::VectorRef>():
            cv::util::get<cv::detail::VectorRef>(out_obj).mov(cv::util::get<cv::detail::VectorRef>(res_obj));
            break;
        case T::index_of<cv::detail::OpaqueRef>():
            cv::util::get<cv::detail::OpaqueRef>(out_obj).mov(cv::util::get<cv::detail::OpaqueRef>(res_obj));
            break;
        default:
            GAPI_Assert(false && "This value type is not supported!"); // ...maybe because of STANDALONE mode.
            break;
        }
    }
}

// Pops an item from every input queue and combine it to the final
// result.  Blocks the current thread.  Returns true if the vector has
// been obtained successfully and false if a Stop message has been
// received. Handles Stop x-queue synchronization gracefully.
//
// In fact, the logic behind this method is a little bit more complex.
// The complexity comes from handling the pipeline termination
// messages.  This version if GStreamerExecutable is running every
// graph island in its own thread, and threads communicate via bounded
// (limited in size) queues.  Threads poll their input queues in the
// infinite loops and pass the data to their Island executables when
// the full input vector (a "stack frame") arrives.
//
// If the input stream is over or stop() is called, "Stop" messages
// are broadcasted in the graph from island to island via queues,
// starting with the emitters (sources). Since queues are bounded,
// thread may block on push() if the queue is full already and is not
// popped for some reason in the reader thread. In order to avoid
// this, once an Island gets Stop on an input, it start reading all
// other input queues until it reaches Stop messages there as well.
// Only then the thread terminates so in theory queues are left
// free'd.
//
// "Stop" messages are sent to the pipeline in these three cases:
// 1. User has called stop(): a "Stop" message is sent to every input
//    queue.
// 2. Input video stream has reached its end -- its emitter sends Stop
//    to its readers AND asks constant emitters (emitters attached to
//    const data -- infinite data generators) to push Stop messages as
//    well - in order to maintain a regular Stop procedure as defined
//    above.
// 3. "Stop" message coming from a constant emitter after triggering an
//    EOS notification -- see (2).
//
// There is a problem with (3). Sometimes it terminates the pipeline
// too early while some frames could still be produced with no issue,
// and our test fails with error like "got 99 frames, expected 100".
// This is how it reproduces:
//
//                   q1
//   [const input]   -----------------------> [ ISL2 ] --> [output]
//                   q0             q2    .->
//   [stream input]  ---> [ ISL1 ] -------'
//
// Video emitter is pushing frames to q0, and ISL1 is taking every
// frame from this queue and processes it. Meanwhile, q1 is a
// const-input-queue staffed with const data already, ISL2 already
// popped one, and is waiting for data from q2 (of ISL1) to arrive.
//
// When the stream is over, stream emitter pushes the last frame to
// q0, followed by a Stop sign, and _immediately_ notifies const
// emitters to broadcast Stop messages as well.  In the above
// configuration, the replicated Stop message via q1 may reach ISL2
// faster than the real Stop message via q2 -- moreover, somewhere in
// q1 or q2 there may be real frames awaiting processing. ISL2 gets
// Stop via q1 and _discards_ any pending data coming from q2 -- so a
// last frame or two may be lost.
//
// A working but not very elegant solution to this problem is to tag
// Stop messages. Stop got via stop() is really a hard stop, while
// broadcasted Stop from a Const input shouldn't initiate the Island
// execution termination. Instead, its associated const data should
// remain somewhere in islands' thread local storage until a real
// "Stop" is received.
//
// Queue reader is the class which encapsulates all this logic and
// provides threads with a managed storage and an easy API to obtain
// data.
class QueueReader
{
    bool m_finishing = false; // Set to true once a "soft" stop is received
    std::vector<Cmd> m_cmd;

public:
    bool getInputVector(std::vector<Q*> &in_queues,
                        cv::GRunArgs    &in_constants,
                        cv::GRunArgs    &isl_inputs);
};

bool QueueReader::getInputVector(std::vector<Q*> &in_queues,
                                 cv::GRunArgs    &in_constants,
                                 cv::GRunArgs    &isl_inputs)
{
    // NOTE: in order to maintain the GRunArg's underlying object
    // lifetime, keep the whole cmd vector (of size == # of inputs)
    // in memory.
    m_cmd.resize(in_queues.size());
    isl_inputs.resize(in_queues.size());

    for (auto &&it : ade::util::indexed(in_queues))
    {
        auto id = ade::util::index(it);
        auto &q = ade::util::value(it);

        if (q == nullptr)
        {
            GAPI_Assert(!in_constants.empty());
            // NULL queue means a graph-constant value (like a
            // value-initialized scalar)
            // It can also hold a constant value received with
            // Stop::Kind::CNST message (see above).
            // FIXME: Variant move problem
            isl_inputs[id] = const_cast<const cv::GRunArg&>(in_constants[id]);
            continue;
        }

        q->pop(m_cmd[id]);
        if (!cv::util::holds_alternative<Stop>(m_cmd[id]))
        {
            // FIXME: Variant move problem
            isl_inputs[id] = const_cast<const cv::GRunArg &>(cv::util::get<cv::GRunArg>(m_cmd[id]));
        }
        else // A Stop sign
        {
            const auto &stop = cv::util::get<Stop>(m_cmd[id]);
            if (stop.kind == Stop::Kind::CNST)
            {
                // We've got a Stop signal from a const source,
                // propagated as a result of real stream reaching its
                // end.  Sometimes these signals come earlier than
                // real EOS Stops so are deprioritized -- just
                // remember the Const value here and continue
                // processing other queues. Set queue pointer to
                // nullptr and update the const_val vector
                // appropriately
                m_finishing = true;
                in_queues[id] = nullptr;
                in_constants.resize(in_queues.size());
                in_constants[id] = std::move(stop.cdata);

                // NEXT time (on a next call to getInputVector()), the
                // "q==nullptr" check above will be triggered, but now
                // we need to make it manually:
                isl_inputs[id] = const_cast<const cv::GRunArg&>(in_constants[id]);
            }
            else
            {
                GAPI_Assert(stop.kind == Stop::Kind::HARD);
                // Just got a stop sign. Reiterate through all
                // _remaining valid_ queues (some of them can be
                // set to nullptr already -- see above) and rewind
                // data to every Stop sign per queue
                for (auto &&qit : ade::util::indexed(in_queues))
                {
                    auto id2 = ade::util::index(qit);
                    auto &q2 = ade::util::value(qit);
                    if (id == id2) continue;

                    Cmd cmd2;
                    while (q2 && !cv::util::holds_alternative<Stop>(cmd2))
                        q2->pop(cmd2);
                }
                // After queues are read to the proper indicator,
                // indicate end-of-stream
                return false;
            } // if(Cnst)
        } // if(Stop)
    } // for(in_queues)

    if (m_finishing)
    {
        // If the process is about to end (a soft Stop was received
        // already) and an island has no other inputs than constant
        // inputs, its queues may all become nullptrs. Indicate it as
        // "no data".
        return !ade::util::all_of(in_queues, [](Q *ptr){return ptr == nullptr;});
    }
    return true; // A regular case - there is data to process.
}


// This thread is a plain dump source actor. What it do is just:
// - Check input queue (the only one) for a control command
// - Depending on the state, obtains next data object and pushes it to the
//   pipeline.
void emitterActorThread(std::shared_ptr<cv::gimpl::GIslandEmitter> emitter,
                        Q& in_queue,
                        std::vector<Q*> out_queues,
                        std::function<void()> cb_completion)
{
    // Wait for the explicit Start command.
    // ...or Stop command, this also happens.
    Cmd cmd;
    in_queue.pop(cmd);
    GAPI_Assert(   cv::util::holds_alternative<Start>(cmd)
                || cv::util::holds_alternative<Stop>(cmd));
    if (cv::util::holds_alternative<Stop>(cmd))
    {
        for (auto &&oq : out_queues) oq->push(cmd);
        return;
    }

    // Now start emitting the data from the source to the pipeline.
    while (true)
    {
        Cmd cancel;
        if (in_queue.try_pop(cancel))
        {
            // if we just popped a cancellation command...
            GAPI_Assert(cv::util::holds_alternative<Stop>(cancel));
            // Broadcast it to the readers and quit.
            for (auto &&oq : out_queues) oq->push(cancel);
            return;
        }

        // Try to obrain next data chunk from the source
        cv::GRunArg data;
        if (emitter->pull(data))
        {
            // // On success, broadcast it to our readers
            for (auto &&oq : out_queues)
            {
                // FIXME: FOR SOME REASON, oq->push(Cmd{data}) doesn't work!!
                // empty mats are arrived to the receivers!
                // There may be a fatal bug in our variant!
                const auto tmp = data;
                oq->push(Cmd{tmp});
            }
        }
        else
        {
            // Otherwise, broadcast STOP message to our readers and quit.
            // This usually means end-of-stream, so trigger a callback
            for (auto &&oq : out_queues) oq->push(Cmd{Stop{}});
            if (cb_completion) cb_completion();
            return;
        }
    }
}

class StreamingInput final: public cv::gimpl::GIslandExecutable::IInput
{
    QueueReader &qr;
    std::vector<Q*> &in_queues; // FIXME: This can be part of QueueReader
    cv::GRunArgs &in_constants; // FIXME: This can be part of QueueReader

    virtual cv::gimpl::StreamMsg get() override
    {
        cv::GRunArgs isl_input_args;
        if (!qr.getInputVector(in_queues, in_constants, isl_input_args))
        {
            // Stop case
            return cv::gimpl::StreamMsg{cv::gimpl::EndOfStream{}};
        }
        return cv::gimpl::StreamMsg{std::move(isl_input_args)};
    }
    virtual cv::gimpl::StreamMsg try_get() override
    {
        // FIXME: This is not very usable at the moment!
        return get();
    }
 public:
    explicit StreamingInput(QueueReader &rdr,
                            std::vector<Q*> &inq,
                            cv::GRunArgs &inc,
                            const std::vector<cv::gimpl::RcDesc> &in_descs)
        : qr(rdr), in_queues(inq), in_constants(inc)
    {
        set(in_descs);
    }
};

class StreamingOutput final: public cv::gimpl::GIslandExecutable::IOutput
{
    // These objects form an internal state of the StreamingOutput
    struct Posting
    {
        using V = cv::util::variant<cv::GRunArg, cv::gimpl::EndOfStream>;
        V data;
        bool ready = false;
    };
    using PostingList = std::list<Posting>;
    std::vector<PostingList> m_postings;
    std::unordered_map< const void*
                      , std::pair<int, PostingList::iterator>
                      > m_postIdx;
    std::size_t m_stops_sent = 0u;

    // These objects are owned externally
    const cv::GMetaArgs &m_metas;
    std::vector< std::vector<Q*> > &m_out_queues;

    // Allocate a new data object for output under idx
    // Prepare this object for posting
    virtual cv::GRunArgP get(int idx) override
    {
        using MatType = cv::Mat;
        using SclType = cv::Scalar;

        // Allocate a new posting first, then bind this GRunArgP to this item
        auto iter    = m_postings[idx].insert(m_postings[idx].end(), Posting{});
        const auto r = desc()[idx];
        cv::GRunArg& out_arg = cv::util::get<cv::GRunArg>(iter->data);
        cv::GRunArgP ret_val;
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
                cv::gimpl::createMat(cv::util::get<cv::GMatDesc>(m_metas[idx]), newMat);
                out_arg = cv::GRunArg(std::move(newMat));
                ret_val = cv::GRunArgP(&cv::util::get<MatType>(out_arg));
            }
            break;
        case cv::GShape::GSCALAR:
            {
                SclType newScl;
                out_arg = cv::GRunArg(std::move(newScl));
                ret_val = cv::GRunArgP(&cv::util::get<SclType>(out_arg));
            }
            break;
        case cv::GShape::GARRAY:
            {
                cv::detail::VectorRef newVec;
                cv::util::get<cv::detail::ConstructVec>(r.ctor)(newVec);
                out_arg = cv::GRunArg(std::move(newVec));
                // VectorRef is implicitly shared so no pointer is taken here
                // FIXME: that variant MOVE problem again
                const auto &rr = cv::util::get<cv::detail::VectorRef>(out_arg);
                ret_val = cv::GRunArgP(rr);
            }
            break;
        case cv::GShape::GOPAQUE:
            {
                cv::detail::OpaqueRef newOpaque;
                cv::util::get<cv::detail::ConstructOpaque>(r.ctor)(newOpaque);
                out_arg = cv::GRunArg(std::move(newOpaque));
                // OpaqueRef is implicitly shared so no pointer is taken here
                // FIXME: that variant MOVE problem again
                const auto &rr = cv::util::get<cv::detail::OpaqueRef>(out_arg);
                ret_val = cv::GRunArgP(rr);
            }
            break;
        default:
            cv::util::throw_error(std::logic_error("Unsupported GShape"));
        }
        m_postIdx[cv::gimpl::proto::ptr(ret_val)] = std::make_pair(idx, iter);
        return ret_val;
    }
    virtual void post(cv::GRunArgP&& argp) override
    {
        // Mark the output ready for posting. If it is the first in the line,
        // actually post it and all its successors which are ready for posting too.
        auto it = m_postIdx.find(cv::gimpl::proto::ptr(argp));
        GAPI_Assert(it != m_postIdx.end());
        const int out_idx = it->second.first;
        const auto out_iter = it->second.second;
        out_iter->ready = true;
        m_postIdx.erase(it); // Drop the link from the cache anyway
        if (out_iter != m_postings[out_idx].begin())
        {
            return; // There are some pending postings in the beginning, return
        }

        GAPI_Assert(out_iter == m_postings[out_idx].begin());
        auto post_iter = m_postings[out_idx].begin();
        while (post_iter != m_postings[out_idx].end() && post_iter->ready == true)
        {
            Cmd cmd;
            if (cv::util::holds_alternative<cv::GRunArg>(post_iter->data))
            {
                // FIXME: That ugly VARIANT problem
                cmd = Cmd{const_cast<const cv::GRunArg&>(cv::util::get<cv::GRunArg>(post_iter->data))};
            }
            else
            {
                GAPI_Assert(cv::util::holds_alternative<cv::gimpl::EndOfStream>(post_iter->data));
                cmd = Cmd{Stop{}};
                m_stops_sent++;
            }
            for (auto &&q : m_out_queues[out_idx])
            {
                // FIXME: This ugly VARIANT problem
                q->push(const_cast<const Cmd&>(cmd));
            }
            post_iter = m_postings[out_idx].erase(post_iter);
        }
    }
    virtual void post(cv::gimpl::EndOfStream&&) override
    {
        // If the posting list is empty, just broadcast the stop message.
        // If it is not, enqueue the Stop message in the postings list.
        for (auto &&it : ade::util::indexed(m_postings))
        {
            const auto  idx = ade::util::index(it);
                  auto &lst = ade::util::value(it);
            if (lst.empty())
            {
                for (auto &&q : m_out_queues[idx])
                {
                    q->push(Cmd(Stop{}));
                }
                m_stops_sent++;
            }
            else
            {
                Posting p;
                p.data = Posting::V{cv::gimpl::EndOfStream{}};
                p.ready = true;
                lst.push_back(std::move(p)); // FIXME: For some reason {}-ctor didn't work here
            }
        }
    }
public:
    explicit StreamingOutput(const cv::GMetaArgs &metas,
                             std::vector< std::vector<Q*> > &out_queues,
                             const std::vector<cv::gimpl::RcDesc> &out_descs)
        : m_metas(metas)
        , m_out_queues(out_queues)
    {
        set(out_descs);
        m_postings.resize(out_descs.size());
    }

    bool done() const
    {
        // The streaming actor work is considered DONE for this stream
        // when it posted/resent all STOP messages to all its outputs.
        return m_stops_sent == desc().size();
    }
};

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
                       cv::GRunArgs in_constants,
                       std::vector< std::vector<Q*> > out_queues)
{
    GAPI_Assert(in_queues.size() == in_rcs.size());
    GAPI_Assert(out_queues.size() == out_rcs.size());
    GAPI_Assert(out_queues.size() == out_metas.size());
    QueueReader qr;
    StreamingInput input(qr, in_queues, in_constants, in_rcs);
    StreamingOutput output(out_metas, out_queues, out_rcs);
    while (!output.done())
    {
        island->run(input, output);
    }
}

// The idea of collectorThread is easy.  If there're multiple outputs
// in the graph, we need to pull an object from every associated queue
// and then put the resulting vector into one single queue.  While it
// looks redundant, it simplifies dramatically the way how try_pull()
// is implemented - we need to check one queue instead of many.
void collectorThread(std::vector<Q*> in_queues,
                     Q&              out_queue)
{
    QueueReader qr;
    while (true)
    {
        cv::GRunArgs this_result(in_queues.size());
        cv::GRunArgs this_const(in_queues.size());
        if (!qr.getInputVector(in_queues, this_const, this_result))
        {
            out_queue.push(Cmd{Stop{}});
            return;
        }
        out_queue.push(Cmd{this_result});
    }
}
} // anonymous namespace

// GStreamingExecutor expects compile arguments as input to have possibility to do
// proper graph reshape and islands recompilation
cv::gimpl::GStreamingExecutor::GStreamingExecutor(std::unique_ptr<ade::Graph> &&g_model,
                                                  const GCompileArgs &comp_args)
    : m_orig_graph(std::move(g_model))
    , m_island_graph(GModel::Graph(*m_orig_graph).metadata()
                     .get<IslandModel>().model)
    , m_comp_args(comp_args)
    , m_gim(*m_island_graph)
{
    GModel::Graph gm(*m_orig_graph);
    // NB: Right now GIslandModel is acyclic, and all the below code assumes that.
    // NB: This naive execution code is taken from GExecutor nearly "as-is"

    const auto proto = gm.metadata().get<Protocol>();
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

    // If metadata was not passed to compileStreaming, Islands are not compiled at this point.
    // It is fine -- Islands are then compiled in setSource (at the first valid call).
    const bool islands_compiled = m_gim.metadata().contains<IslandsCompiled>();

    auto sorted = m_gim.metadata().get<ade::passes::TopologicalSortData>();
    for (auto nh : sorted.nodes())
    {
        switch (m_gim.metadata(nh).get<NodeKind>().k)
        {
        case NodeKind::ISLAND:
            {
                std::vector<RcDesc> input_rcs;
                std::vector<RcDesc> output_rcs;
                std::vector<GRunArg> in_constants;
                cv::GMetaArgs output_metas;
                input_rcs.reserve(nh->inNodes().size());
                in_constants.reserve(nh->inNodes().size()); // FIXME: Ugly
                output_rcs.reserve(nh->outNodes().size());
                output_metas.reserve(nh->outNodes().size());

                std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node> > const_ins;

                // FIXME: THIS ORDER IS IRRELEVANT TO PROTOCOL OR ANY OTHER ORDER!
                // FIXME: SAME APPLIES TO THE REGULAR GEEXECUTOR!!
                auto xtract_in = [&](ade::NodeHandle slot_nh, std::vector<RcDesc> &vec)
                {
                    const auto orig_data_nh
                        = m_gim.metadata(slot_nh).get<DataSlot>().original_data_node;
                    const auto &orig_data_info
                        = gm.metadata(orig_data_nh).get<Data>();
                    if (orig_data_info.storage == Data::Storage::CONST_VAL) {
                        const_ins.insert(slot_nh);
                        // FIXME: Variant move issue
                        in_constants.push_back(const_cast<const cv::GRunArg&>(gm.metadata(orig_data_nh).get<ConstValue>().arg));
                    } else in_constants.push_back(cv::GRunArg{}); // FIXME: Make it in some smarter way pls
                    if (orig_data_info.shape == GShape::GARRAY) {
                        // FIXME: GArray lost host constructor problem
                        GAPI_Assert(!cv::util::holds_alternative<cv::util::monostate>(orig_data_info.ctor));
                    }
                    vec.emplace_back(RcDesc{ orig_data_info.rc
                                           , orig_data_info.shape
                                           , orig_data_info.ctor});
                };
                auto xtract_out = [&](ade::NodeHandle slot_nh, std::vector<RcDesc> &vec, cv::GMetaArgs &metas)
                {
                    const auto orig_data_nh
                        = m_gim.metadata(slot_nh).get<DataSlot>().original_data_node;
                    const auto &orig_data_info
                        = gm.metadata(orig_data_nh).get<Data>();
                    if (orig_data_info.shape == GShape::GARRAY) {
                        // FIXME: GArray lost host constructor problem
                        GAPI_Assert(!cv::util::holds_alternative<cv::util::monostate>(orig_data_info.ctor));
                    }
                    vec.emplace_back(RcDesc{ orig_data_info.rc
                                           , orig_data_info.shape
                                           , orig_data_info.ctor});
                    metas.emplace_back(orig_data_info.meta);
                };
                // FIXME: JEZ IT WAS SO AWFUL!!!!
                for (auto in_slot_nh  : nh->inNodes())  xtract_in(in_slot_nh,  input_rcs);
                for (auto out_slot_nh : nh->outNodes()) xtract_out(out_slot_nh, output_rcs, output_metas);

                std::shared_ptr<GIslandExecutable> isl_exec = islands_compiled
                    ? m_gim.metadata(nh).get<IslandExec>().object
                    : nullptr;
                m_ops.emplace_back(OpDesc{ std::move(input_rcs)
                                         , std::move(output_rcs)
                                         , std::move(output_metas)
                                         , nh
                                         , in_constants
                                         , isl_exec
                                         });
                // Initialize queues for every operation's input
                ade::TypedGraph<DataQueue> qgr(*m_island_graph);
                for (auto eh : nh->inEdges())
                {
                    // ...only if the data is not compile-const
                    if (const_ins.count(eh->srcNode()) == 0) {
                        qgr.metadata(eh).set(DataQueue(queue_capacity));
                        m_internal_queues.insert(&qgr.metadata(eh).get<DataQueue>().q);
                    }
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

cv::gimpl::GStreamingExecutor::~GStreamingExecutor()
{
    if (state == State::READY || state == State::RUNNING)
        stop();
}

void cv::gimpl::GStreamingExecutor::setSource(GRunArgs &&ins)
{
    GAPI_Assert(state == State::READY || state == State::STOPPED);

    const auto is_video = [](const GRunArg &arg)
    {
        return util::holds_alternative<cv::gapi::wip::IStreamSource::Ptr>(arg);
    };
    const auto num_videos = std::count_if(ins.begin(), ins.end(), is_video);
    if (num_videos > 1)
    {
        // See below why (another reason - no documented behavior
        // on handling videos streams of different length)
        util::throw_error(std::logic_error("Only one video source is"
                                           " currently supported!"));
    }

    GModel::ConstGraph gm(*m_orig_graph);
    // Now the tricky-part: completing Islands compilation if compileStreaming
    // has been called without meta arguments.
    // The logic is basically the following:
    // - (0) Collect metadata from input vector;
    // - (1) If graph is compiled with meta
    //   - (2) Just check if the passed objects have correct meta.
    // - (3) Otherwise:
    //   - (4) Run metadata inference;
    //   - (5) If islands are not compiled at this point OR are not reshapeable:
    //     - (6) Compile them for a first time with this meta;
    //     - (7) Update internal structures with this island information
    //   - (8) Otherwise:
    //     - (9) Reshape islands to this new metadata.
    //     - (10) Update internal structures again
    const auto update_int_metas = [&]()
    {
        for (auto& op : m_ops)
        {
            op.out_metas.resize(0);
            for (auto out_slot_nh : op.nh->outNodes())
            {
                const auto &orig_nh = m_gim.metadata(out_slot_nh).get<DataSlot>().original_data_node;
                const auto &orig_info = gm.metadata(orig_nh).get<Data>();
                op.out_metas.emplace_back(orig_info.meta);
            }
        }
    };
    bool islandsRecompiled = false;
    const auto new_meta = cv::descr_of(ins); // 0
    if (gm.metadata().contains<OriginalInputMeta>()) // (1)
    {
        // NB: Metadata is tested in setSource() already - just put an assert here
        GAPI_Assert(new_meta == gm.metadata().get<OriginalInputMeta>().inputMeta); // (2)
    }
    else // (3)
    {
        GCompiler::runMetaPasses(*m_orig_graph.get(), new_meta); // (4)
        if (!m_gim.metadata().contains<IslandsCompiled>()
            || (m_reshapable.has_value() && m_reshapable.value() == false)) // (5)
        {
            bool is_reshapable = true;
            GCompiler::compileIslands(*m_orig_graph.get(), m_comp_args); // (6)
            for (auto& op : m_ops)
            {
                op.isl_exec = m_gim.metadata(op.nh).get<IslandExec>().object;
                is_reshapable = is_reshapable && op.isl_exec->canReshape();
            }
            update_int_metas(); // (7)
            m_reshapable = util::make_optional(is_reshapable);

            islandsRecompiled = true;
        }
        else // (8)
        {
            for (auto& op : m_ops)
            {
                op.isl_exec->reshape(*m_orig_graph, m_comp_args); // (9)
            }
            update_int_metas(); // (10)
        }
    }
    // Metadata handling is done!

    // Walk through the protocol, set-up emitters appropriately
    // There's a 1:1 mapping between emitters and corresponding data inputs.
    for (auto it : ade::util::zip(ade::util::toRange(m_emitters),
                                  ade::util::toRange(ins),
                                  ade::util::iota(m_emitters.size())))
    {
        auto  emit_nh  = std::get<0>(it);
        auto& emit_arg = std::get<1>(it);
        auto  emit_idx = std::get<2>(it);
        auto& emitter  = m_gim.metadata(emit_nh).get<Emitter>().object;

        using T = GRunArg;
        switch (emit_arg.index())
        {
        // Create a streaming emitter.
        // Produces the next video frame when pulled.
        case T::index_of<cv::gapi::wip::IStreamSource::Ptr>():
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
            m_const_vals.push_back(const_cast<cv::GRunArg &>(emit_arg)); // FIXME: move problem
            m_const_emitter_queues.push_back(&m_emitter_queues[emit_idx]);
            break;
        }
    }

    // FIXME: The below code assumes our graph may have only one
    // real video source (and so, only one stream which may really end)
    // all other inputs are "constant" generators.
    // Craft here a completion callback to notify Const emitters that
    // a video source is over
    GAPI_Assert(m_const_emitter_queues.size() == m_const_vals.size());
    auto real_video_completion_cb = [this]()
    {
        for (auto it : ade::util::zip(ade::util::toRange(m_const_emitter_queues),
                                      ade::util::toRange(m_const_vals)))
        {
            Stop stop;
            stop.kind = Stop::Kind::CNST;
            stop.cdata = std::get<1>(it);
            std::get<0>(it)->push(Cmd{std::move(stop)});
        }
    };

    // FIXME: ONLY now, after all executable objects are created,
    // we can set up our execution threads. Let's do it.
    // First create threads for all the emitters.
    // FIXME: One way to avoid this may be including an Emitter object as a part of
    // START message. Why not?
    if (state == State::READY)
    {
        stop();
    }

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
                               out_queues,
                               real_video_completion_cb);
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

        // If Island Executable is recompiled, all its stuff including internal kernel states
        // are recreated and re-initialized automatically.
        // But if not, we should notify Island Executable about new started stream to let it update
        // its internal variables.
        if (!islandsRecompiled)
        {
            op.isl_exec->handleNewStream();
        }

        m_threads.emplace_back(islandActorThread,
                               op.in_objects,
                               op.out_objects,
                               op.out_metas,
                               island,
                               in_queues,
                               op.in_constants,
                               out_queues);
    }

    // Finally, start a collector thread.
    m_threads.emplace_back(collectorThread,
                           m_sink_queues,
                           std::ref(m_out_queue));
    state = State::READY;
}

void cv::gimpl::GStreamingExecutor::start()
{
    if (state == State::STOPPED)
    {
        util::throw_error(std::logic_error("Please call setSource() before start() "
                                           "if the pipeline has been already stopped"));
    }
    GAPI_Assert(state == State::READY);

    // Currently just trigger our emitters to work
    state = State::RUNNING;
    for (auto &q : m_emitter_queues)
    {
        q.push(stream::Cmd{stream::Start{}});
    }
}

void cv::gimpl::GStreamingExecutor::wait_shutdown()
{
    // This utility is used by pull/try_pull/stop() to uniformly
    // shutdown the worker threads.
    // FIXME: Of course it can be designed much better
    for (auto &t : m_threads) t.join();
    m_threads.clear();

    // Clear all queues
    // If there are constant emitters, internal queues
    // may be polluted with constant values and have extra
    // data at the point of shutdown.
    // It usually happens when there's multiple inputs,
    // one constant and one is not, and the latter ends (e.g.
    // with end-of-stream).
    for (auto &q : m_emitter_queues) q.clear();
    for (auto &q : m_sink_queues) q->clear();
    for (auto &q : m_internal_queues) q->clear();
    m_out_queue.clear();

    state = State::STOPPED;
}

bool cv::gimpl::GStreamingExecutor::pull(cv::GRunArgsP &&outs)
{
    if (state == State::STOPPED)
        return false;
    GAPI_Assert(state == State::RUNNING);
    GAPI_Assert(m_sink_queues.size() == outs.size());

    Cmd cmd;
    m_out_queue.pop(cmd);
    if (cv::util::holds_alternative<Stop>(cmd))
    {
        wait_shutdown();
        return false;
    }

    GAPI_Assert(cv::util::holds_alternative<cv::GRunArgs>(cmd));
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
    if (cv::util::holds_alternative<Stop>(cmd))
    {
        wait_shutdown();
        return false;
    }

    GAPI_Assert(cv::util::holds_alternative<cv::GRunArgs>(cmd));
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
    while (!cv::util::holds_alternative<Stop>(cmd))
    {
        m_out_queue.pop(cmd);
    }
    GAPI_Assert(cv::util::holds_alternative<Stop>(cmd));
    wait_shutdown();
}

bool cv::gimpl::GStreamingExecutor::running() const
{
    return (state == State::RUNNING);
}
