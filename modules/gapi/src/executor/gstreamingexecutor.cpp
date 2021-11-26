// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2021 Intel Corporation

#include "precomp.hpp"

#include <memory> // make_shared

#include <ade/util/zip_range.hpp>

#include <opencv2/gapi/opencv_includes.hpp>

#if !defined(GAPI_STANDALONE)
#include <opencv2/gapi/core.hpp> // GCopy -- FIXME - to be removed!
#endif // GAPI_STANDALONE

#include "utils/itt.hpp"

#include "api/gproto_priv.hpp" // ptr(GRunArgP)
#include "compiler/passes/passes.hpp"
#include "backends/common/gbackend.hpp" // createMat
#include "backends/streaming/gstreamingbackend.hpp" // GCopy
#include "compiler/gcompiler.hpp" // for compileIslands

#include <logger.hpp>

#include "executor/gstreamingexecutor.hpp"

#include <opencv2/gapi/streaming/meta.hpp>
#include <opencv2/gapi/streaming/sync.hpp>

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
    enum tag { DESYNC }; // Enum of 1 element: purely a syntax sugar

    explicit DataQueue(std::size_t capacity) {
        // Note: `ptr` is shared<SyncQueue>, while the `q` is a shared<Q>
        auto ptr = std::make_shared<cv::gimpl::stream::SyncQueue>();
        if (capacity != 0) {
            ptr->set_capacity(capacity);
        }
        q = std::move(ptr);
    }
    explicit DataQueue(tag t)
        : q(new cv::gimpl::stream::DesyncQueue()) {
        GAPI_Assert(t == DESYNC);
    }

    // FIXME: ADE metadata requires types to be copiable
    std::shared_ptr<cv::gimpl::stream::Q> q;
};

struct DesyncSpecialCase {
    static const char *name() { return "DesyncSpecialCase"; }
};

std::vector<cv::gimpl::stream::Q*> reader_queues(      ade::Graph &g,
                                                 const ade::NodeHandle &obj)
{
    ade::TypedGraph<DataQueue> qgr(g);
    std::vector<cv::gimpl::stream::Q*> result;
    for (auto &&out_eh : obj->outEdges())
    {
        result.push_back(qgr.metadata(out_eh).get<DataQueue>().q.get());
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
                         ? qgr.metadata(in_eh).get<DataQueue>().q.get()
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
        {
            auto out_mat_p = cv::util::get<cv::Mat*>(out_obj);
            auto view = cv::util::get<cv::RMat>(res_obj).access(cv::RMat::Access::R);
            *out_mat_p = cv::gimpl::asMat(view).clone();
        } break;
        case T::index_of<cv::RMat*>():
            *cv::util::get<cv::RMat*>(out_obj) = std::move(cv::util::get<cv::RMat>(res_obj));
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
        case T::index_of<cv::MediaFrame*>():
            *cv::util::get<cv::MediaFrame*>(out_obj) = std::move(cv::util::get<cv::MediaFrame>(res_obj));
            break;
        default:
            GAPI_Assert(false && "This value type is not supported!"); // ...maybe because of STANDALONE mode.
            break;
        }
    }
}

// FIXME: Is there a way to derive function from its GRunArgsP version?
template<class C> using O = cv::util::optional<C>;
void sync_data(cv::gimpl::stream::Result &r, cv::GOptRunArgsP &outputs)
{
    namespace own = cv::gapi::own;

    for (auto && it : ade::util::zip(ade::util::toRange(outputs),
                                     ade::util::toRange(r.args),
                                     ade::util::toRange(r.flags)))
    {
        auto &out_obj  = std::get<0>(it);
        auto &res_obj  = std::get<1>(it);
        bool available = std::get<2>(it);

        using T = cv::GOptRunArgP;
#define HANDLE_CASE(Type)                                               \
        case T::index_of<O<Type>*>():                                   \
            if (available) {                                            \
                *cv::util::get<O<Type>*>(out_obj)                       \
                    = cv::util::make_optional(std::move(cv::util::get<Type>(res_obj))); \
            } else {                                                    \
                cv::util::get<O<Type>*>(out_obj)->reset();              \
            }

        // FIXME: this conversion should be unified
        switch (out_obj.index())
        {
            HANDLE_CASE(cv::Scalar);     break;
            HANDLE_CASE(cv::RMat);       break;
            HANDLE_CASE(cv::MediaFrame); break;

        case T::index_of<O<cv::Mat>*>(): {
            // Mat: special handling.
            auto &mat_opt = *cv::util::get<O<cv::Mat>*>(out_obj);
            if (available) {
                auto q_map = cv::util::get<cv::RMat>(res_obj).access(cv::RMat::Access::R);
                // FIXME: Copy! Maybe we could do some optimization for this case!
                // e.g. don't handle RMat for last ilsand in the graph.
                // It is not always possible though.
                mat_opt = cv::util::make_optional(cv::gimpl::asMat(q_map).clone());
            } else {
                mat_opt.reset();
            }
        } break;
        case T::index_of<cv::detail::OptionalVectorRef>(): {
            // std::vector<>: special handling
            auto &vec_opt = cv::util::get<cv::detail::OptionalVectorRef>(out_obj);
            if (available) {
                vec_opt.mov(cv::util::get<cv::detail::VectorRef>(res_obj));
            } else {
                vec_opt.reset();
            }
        } break;
        case T::index_of<cv::detail::OptionalOpaqueRef>(): {
            // std::vector<>: special handling
            auto &opq_opt = cv::util::get<cv::detail::OptionalOpaqueRef>(out_obj);
            if (available) {
                opq_opt.mov(cv::util::get<cv::detail::OpaqueRef>(res_obj));
            } else {
                opq_opt.reset();
            }
        } break;
        default:
            // ...maybe because of STANDALONE mode.
            GAPI_Assert(false && "This value type is not supported!");
            break;
        }
    }
#undef HANDLE_CASE
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

    void rewindToStop(std::vector<Q*>   &in_queues,
                      const std::size_t  this_id);

public:
    bool getInputVector  (std::vector<Q*>   &in_queues,
                          cv::GRunArgs      &in_constants,
                          cv::GRunArgs      &isl_inputs);

    bool getResultsVector(std::vector<Q*>         &in_queues,
                          const std::vector<int>  &in_mapping,
                          const std::size_t        out_size,
                          cv::GRunArgs            &out_results);
};

void rewindToStop(std::vector<Q*> &in_queues,
                  const std::size_t  this_id)
{
    for (auto &&qit : ade::util::indexed(in_queues))
    {
        auto id2 = ade::util::index(qit);
        auto &q2 = ade::util::value(qit);
        if (this_id == id2) continue;

        Cmd cmd;
        while (q2 && !cv::util::holds_alternative<Stop>(cmd))
            q2->pop(cmd);
    }
}

// This method handles a stop sign got from some input
// island. Reiterate through all _remaining valid_ queues (some of
// them can be set to nullptr already -- see handling in
// getInputVector) and rewind data to every Stop sign per queue.
void QueueReader::rewindToStop(std::vector<Q*>   &in_queues,
                               const std::size_t  this_id)
{
    ::rewindToStop(in_queues, this_id);
}

bool QueueReader::getInputVector(std::vector<Q*> &in_queues,
                                 cv::GRunArgs    &in_constants,
                                 cv::GRunArgs    &isl_inputs)
{
    // NB: Need to release resources from the previous step, to fetch new ones.
    // On some systems it might be impossible to allocate new memory
    // until the old one is released.
    m_cmd.clear();
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
            isl_inputs[id] = in_constants[id];
            continue;
        }

        q->pop(m_cmd[id]);
        if (!cv::util::holds_alternative<Stop>(m_cmd[id]))
        {
            isl_inputs[id] = cv::util::get<cv::GRunArg>(m_cmd[id]);
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
                isl_inputs[id] = in_constants[id];
            }
            else
            {
                GAPI_Assert(stop.kind == Stop::Kind::HARD);
                rewindToStop(in_queues, id);
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

// This is a special method to obtain a result vector
// for the entire pipeline's outputs.
//
// After introducing desync(), the pipeline output's vector
// can be produced just partially. Also, if a desynchronized
// path has multiple outputs for the pipeline, _these_ outputs
// should still come synchronized to the end user (via pull())
//
//
// This method handles all this.
// It takes a number of input queues, which may or may not be
// equal to the number of pipeline outputs (<=).
// It also takes indexes saying which queue produces which
// output in the resulting pipeline.
//
// `out_results` is always produced with the size of full output
// vector. In the desync case, the number of in_queues will
// be less than this size and some of the items won't be produced.
// In the sync case, there will be a 1-1 mapping.
//
// In the desync case, there _will be_ multiple collector threads
// calling this method, and pushing their whole-pipeline outputs
// (_may be_ partially filled) to the same final output queue.
// The receiver part at the GStreamingExecutor level won't change
// because of that.
bool QueueReader::getResultsVector(std::vector<Q*>   &in_queues,
                                   const std::vector<int>  &in_mapping,
                                   const std::size_t  out_size,
                                   cv::GRunArgs      &out_results)
{
    m_cmd.resize(out_size);
    for (auto &&it : ade::util::indexed(in_queues))
    {
        auto ii = ade::util::index(it);
        auto oi = in_mapping[ii];
        auto &q = ade::util::value(it);
        q->pop(m_cmd[oi]);
        if (!cv::util::holds_alternative<Stop>(m_cmd[oi]))
        {
            out_results[oi] = std::move(cv::util::get<cv::GRunArg>(m_cmd[oi]));
        }
        else // A Stop sign
        {
            // In theory, the CNST should never reach here.
            // Collector thread never handles the inputs directly
            // (collector's input queues are always produced by
            // islands in the graph).
            rewindToStop(in_queues, ii);
            return false;
        } // if(Stop)
    } // for(in_queues)
    return true;
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

    GAPI_ITT_STATIC_LOCAL_HANDLE(emitter_hndl, "emitter");
    GAPI_ITT_STATIC_LOCAL_HANDLE(emitter_pull_hndl, "emitter_pull");
    GAPI_ITT_STATIC_LOCAL_HANDLE(emitter_push_hndl, "emitter_push");

    // Now start emitting the data from the source to the pipeline.
    while (true)
    {
        GAPI_ITT_AUTO_TRACE_GUARD(emitter_hndl);

        Cmd cancel;
        if (in_queue.try_pop(cancel))
        {
            // if we just popped a cancellation command...
            GAPI_Assert(cv::util::holds_alternative<Stop>(cancel));
            // Broadcast it to the readers and quit.
            for (auto &&oq : out_queues) oq->push(cancel);
            return;
        }

        // Try to obtain next data chunk from the source
        cv::GRunArg data;

        const bool result = [&](){
            GAPI_ITT_AUTO_TRACE_GUARD(emitter_pull_hndl);
            return emitter->pull(data);
        }();

        if (result)
        {
            GAPI_ITT_AUTO_TRACE_GUARD(emitter_push_hndl);
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

// This thread pulls data from the assigned input queues and makes sure that
// all input args are in sync (timestamps are equal), dropping some inputs if required.
// After getting synchronized inputs from all input queues, the thread pushes them to out queues
void syncActorThread(std::vector<Q*> in_queues,
                     std::vector<std::vector<Q*>> out_queues) {
    using timestamp_t = int64_t;
    std::vector<bool> pop_nexts(in_queues.size());
    std::vector<Cmd> cmds(in_queues.size());

    GAPI_ITT_STATIC_LOCAL_HANDLE(sync_hndl, "sync_actor");
    GAPI_ITT_STATIC_LOCAL_HANDLE(sync_pull_1_queue_hndl, "sync_actor_pull_from_1_queue");
    GAPI_ITT_STATIC_LOCAL_HANDLE(sync_push_hndl, "sync_actor_push");
    while (true) {
        GAPI_ITT_AUTO_TRACE_GUARD(sync_hndl);
        // pop_nexts indicates which queue still contains earlier timestamps and
        // needs to be popped at least one more time.
        // For each iteration (frame) we need to pull from each input queue at least once,
        // so switch all to true when start processing new frame
        for (auto&& p : pop_nexts) {
            p = true;
        }
        timestamp_t max_ts = 0u;
        // Iterate through all input queues, pop GRunArg's and compare timestamps.
        // Continue pulling from queues whose timestamps are smaller.
        // Finish when all timestamps are equal.
        do {
            for (auto&& it : ade::util::indexed(
                                 ade::util::zip(pop_nexts, in_queues, cmds))) {
                auto& val = ade::util::value(it);
                auto& pop_next = std::get<0>(val);
                if (!pop_next) {
                    continue;
                }
                auto& q   = std::get<1>(val);
                auto& cmd = std::get<2>(val);

                {
                    GAPI_ITT_AUTO_TRACE_GUARD(sync_pull_1_queue_hndl);
                    q->pop(cmd);
                }
                if (cv::util::holds_alternative<Stop>(cmd)) {
                    // We got a stop command from one of the input queues.
                    // Rewind all input queues till Stop command,
                    // Push Stop command down the graph, finish the thread
                    rewindToStop(in_queues, ade::util::index(it));
                    for (auto &&oqs : out_queues) {
                        for (auto &&oq : oqs) {
                            oq->push(Cmd{Stop{}});
                        }
                    }
                    return;
                }

                // Extract the timestamp
                auto& arg = cv::util::get<cv::GRunArg>(cmd);
                auto ts = cv::util::any_cast<int64_t>(arg.meta[cv::gapi::streaming::meta_tag::timestamp]);
                GAPI_Assert(ts >= 0u);

                // TODO: this whole drop logic can be imported via compile args
                // to give a user a way to customize it
                if (ts < max_ts) {
                    // Continue popping from this queue
                    pop_next = true;
                } else if (ts == max_ts) {
                    // Stop popping from this queue
                    pop_next = false;
                } else if (ts > max_ts) {
                    // We got a timestamp which is greater than timestamps from other queues.
                    // It means that we need to reiterate through all the queues one more time
                    // (except the current one)
                    max_ts = ts;
                    for (auto&& p : pop_nexts) {
                        p = true;
                    }
                    pop_next = false;
                }
            }
        } while (ade::util::any_of(pop_nexts, [](bool v){ return v; }));

        // Finally we got all our inputs synchronized, push them further down the graph
        {
            GAPI_ITT_AUTO_TRACE_GUARD(sync_push_hndl);
            for (auto &&it : ade::util::zip(out_queues, cmds)) {
                for (auto &&q : std::get<0>(it)) {
                    q->push(std::get<1>(it));
                }
            }
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
        GAPI_ITT_STATIC_LOCAL_HANDLE(inputs_get_hndl, "StreamingInput::get");
        GAPI_ITT_AUTO_TRACE_GUARD(inputs_get_hndl);

        cv::GRunArgs isl_input_args;

        if (!qr.getInputVector(in_queues, in_constants, isl_input_args))
        {
            // Stop case
            return cv::gimpl::StreamMsg{cv::gimpl::EndOfStream{}};
        }
        // Wrap all input cv::Mats with RMats
        for (auto& arg : isl_input_args) {
            if (arg.index() == cv::GRunArg::index_of<cv::Mat>()) {
                arg = cv::GRunArg{ cv::make_rmat<cv::gimpl::RMatOnMat>(cv::util::get<cv::Mat>(arg))
                                 , arg.meta
                                 };
            }
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
    std::shared_ptr<cv::gimpl::GIslandExecutable> m_island;

    // NB: StreamingOutput have to be thread-safe.
    // Now synchronization approach is quite poor and inefficient.
    mutable std::mutex m_mutex;

    // Allocate a new data object for output under idx
    // Prepare this object for posting
    virtual cv::GRunArgP get(int idx) override
    {
        GAPI_ITT_STATIC_LOCAL_HANDLE(outputs_get_hndl, "StreamingOutput::get (alloc)");
        GAPI_ITT_AUTO_TRACE_GUARD(outputs_get_hndl);

        std::lock_guard<std::mutex> lock{m_mutex};

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
                auto desc = cv::util::get<cv::GMatDesc>(m_metas[idx]);
                if (m_island->allocatesOutputs())
                {
                    out_arg = cv::GRunArg(m_island->allocate(desc));
                }
                else
                {
                    MatType newMat;
                    cv::gimpl::createMat(desc, newMat);
                    auto rmat = cv::make_rmat<cv::gimpl::RMatOnMat>(newMat);
                    out_arg = cv::GRunArg(std::move(rmat));
                }
                ret_val = cv::GRunArgP(&cv::util::get<cv::RMat>(out_arg));
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
        case cv::GShape::GFRAME:
            {
                cv::MediaFrame frame;
                out_arg = cv::GRunArg(std::move(frame));
                ret_val = cv::GRunArgP(&cv::util::get<cv::MediaFrame>(out_arg));
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
        GAPI_ITT_STATIC_LOCAL_HANDLE(outputs_post_hndl, "StreamingOutput::post");
        GAPI_ITT_AUTO_TRACE_GUARD(outputs_post_hndl);

        std::lock_guard<std::mutex> lock{m_mutex};

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
                cmd = Cmd{cv::util::get<cv::GRunArg>(post_iter->data)};
            }
            else
            {
                GAPI_Assert(cv::util::holds_alternative<cv::gimpl::EndOfStream>(post_iter->data));
                cmd = Cmd{Stop{}};
                m_stops_sent++;
            }
            for (auto &&q : m_out_queues[out_idx])
            {
                q->push(cmd);
            }
            post_iter = m_postings[out_idx].erase(post_iter);
        }
    }

    virtual void post(cv::gimpl::EndOfStream&&) override
    {
        std::lock_guard<std::mutex> lock{m_mutex};
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
    void meta(const cv::GRunArgP &out, const cv::GRunArg::Meta &m) override
    {
        std::lock_guard<std::mutex> lock{m_mutex};
        const auto it = m_postIdx.find(cv::gimpl::proto::ptr(out));
        GAPI_Assert(it != m_postIdx.end());

        const auto out_iter = it->second.second;
        cv::util::get<cv::GRunArg>(out_iter->data).meta = m;
    }

public:
    explicit StreamingOutput(const cv::GMetaArgs &metas,
                             std::vector< std::vector<Q*> > &out_queues,
                             const std::vector<cv::gimpl::RcDesc> &out_descs,
                             std::shared_ptr<cv::gimpl::GIslandExecutable> island)
        : m_metas(metas)
        , m_out_queues(out_queues)
        , m_island(island)
    {
        set(out_descs);
        m_postings.resize(out_descs.size());
    }

    bool done() const
    {
        std::lock_guard<std::mutex> lock{m_mutex};
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
void islandActorThread(std::vector<cv::gimpl::RcDesc> in_rcs,                     // FIXME: this is...
                       std::vector<cv::gimpl::RcDesc> out_rcs,                    // FIXME: ...basically just...
                       cv::GMetaArgs out_metas,                                   // ...
                       std::shared_ptr<cv::gimpl::GIslandExecutable> island_exec, // FIXME: ...a copy of OpDesc{}.
                       std::vector<Q*> in_queues,
                       cv::GRunArgs in_constants,
                       std::vector< std::vector<Q*> > out_queues,
                       const std::string& island_meta_info)
{
    GAPI_Assert(in_queues.size() == in_rcs.size());
    GAPI_Assert(out_queues.size() == out_rcs.size());
    GAPI_Assert(out_queues.size() == out_metas.size());
    QueueReader qr;
    StreamingInput input(qr, in_queues, in_constants, in_rcs);
    StreamingOutput output(out_metas, out_queues, out_rcs, island_exec);

    GAPI_ITT_DYNAMIC_LOCAL_HANDLE(island_hndl, island_meta_info.c_str());
    while (!output.done())
    {
        GAPI_ITT_AUTO_TRACE_GUARD(island_hndl);
        island_exec->run(input, output);
    }
}

// The idea of collectorThread is easy.  If there're multiple outputs
// in the graph, we need to pull an object from every associated queue
// and then put the resulting vector into one single queue.  While it
// looks redundant, it simplifies dramatically the way how try_pull()
// is implemented - we need to check one queue instead of many.
//
// After desync() is added, there may be multiple collector threads
// running, every thread producing its own part of the partial
// pipeline output (optional<T>...). All partial outputs are pushed
// to the same output queue and then picked by GStreamingExecutor
// in the end.
void collectorThread(std::vector<Q*>   in_queues,
                     std::vector<int>  in_mapping,
                     const std::size_t out_size,
                     const bool        handle_stop,
                     Q&                out_queue)
{
    // These flags are static now: regardless if the sync or
    // desync branch is collected by this thread, all in_queue
    // data should come in sync.
    std::vector<bool> flags(out_size, false);
    for (auto idx : in_mapping) {
        flags[idx] = true;
    }

    GAPI_ITT_STATIC_LOCAL_HANDLE(collector_hndl, "collector");
    GAPI_ITT_STATIC_LOCAL_HANDLE(collector_get_results_hndl, "collector_get_results");
    GAPI_ITT_STATIC_LOCAL_HANDLE(collector_push_hndl, "collector_push");

    QueueReader qr;
    while (true)
    {
        GAPI_ITT_AUTO_TRACE_GUARD(collector_hndl);
        cv::GRunArgs this_result(out_size);

        const bool ok = [&](){
            GAPI_ITT_AUTO_TRACE_GUARD(collector_get_results_hndl);
            return qr.getResultsVector(in_queues, in_mapping, out_size, this_result);
        }();

        if (!ok)
        {
            if (handle_stop)
            {
                out_queue.push(Cmd{Stop{}});
            }
            // Terminate the thread anyway
            return;
        }

        {
            GAPI_ITT_AUTO_TRACE_GUARD(collector_push_hndl);
            out_queue.push(Cmd{Result{std::move(this_result), flags}});
        }
    }
}

void check_DesyncObjectConsumedByMultipleIslands(const cv::gimpl::GIslandModel::Graph &gim) {
    using namespace cv::gimpl;

    // Since the limitation exists only in this particular
    // implementation, the check is also done only here but not at the
    // graph compiler level.
    //
    // See comment in desync(GMat) src/api/kernels_streaming.cpp for details.
    for (auto &&nh : gim.nodes()) {
        if (gim.metadata(nh).get<NodeKind>().k == NodeKind::SLOT) {
            // SLOTs are read by ISLANDs, so look for the metadata
            // of the outbound edges
            std::unordered_map<int, GIsland*> out_desync_islands;
            for (auto &&out_eh : nh->outEdges()) {
                if (gim.metadata(out_eh).contains<DesyncIslEdge>()) {
                    // This is a desynchronized edge
                    // Look what Island it leads to
                    const auto out_desync_idx = gim.metadata(out_eh)
                        .get<DesyncIslEdge>().index;
                    const auto out_island = gim.metadata(out_eh->dstNode())
                        .get<FusedIsland>().object;

                    auto it = out_desync_islands.find(out_desync_idx);
                    if (it != out_desync_islands.end()) {
                        // If there's already an edge with this desync
                        // id, it must point to the same island object
                        GAPI_Assert(it->second == out_island.get()
                                    && "A single desync object may only be used by a single island!");
                    } else {
                        // Store the island pointer for the further check
                        out_desync_islands[out_desync_idx] = out_island.get();
                    }
                } // if(desync)
            } // for(out_eh)
            // There must be only one backend in the end of the day
            // (under this desync path)
        } // if(SLOT)
    } // for(nodes)
}

// NB: Construct GRunArgsP based on passed info and store the memory in passed cv::GRunArgs.
// Needed for python bridge, because in case python user doesn't pass output arguments to apply.
void constructOptGraphOutputs(const cv::GTypesInfo &out_info,
                                    cv::GOptRunArgs &args,
                                    cv::GOptRunArgsP &outs)
{
    for (auto&& info : out_info)
    {
        switch (info.shape)
        {
            case cv::GShape::GMAT:
            {
                args.emplace_back(cv::optional<cv::Mat>{});
                outs.emplace_back(&cv::util::get<cv::optional<cv::Mat>>(args.back()));
                break;
            }
            case cv::GShape::GSCALAR:
            {
                args.emplace_back(cv::optional<cv::Scalar>{});
                outs.emplace_back(&cv::util::get<cv::optional<cv::Scalar>>(args.back()));
                break;
            }
            case cv::GShape::GARRAY:
            {
                cv::detail::VectorRef ref;
                cv::util::get<cv::detail::ConstructVec>(info.ctor)(ref);
                args.emplace_back(cv::util::make_optional(std::move(ref)));
                outs.emplace_back(wrap_opt_arg(cv::util::get<cv::optional<cv::detail::VectorRef>>(args.back())));
                break;
            }
            case cv::GShape::GOPAQUE:
            {
                cv::detail::OpaqueRef ref;
                cv::util::get<cv::detail::ConstructOpaque>(info.ctor)(ref);
                args.emplace_back(cv::util::make_optional(std::move(ref)));
                outs.emplace_back(wrap_opt_arg(cv::util::get<cv::optional<cv::detail::OpaqueRef>>(args.back())));
                break;
            }
            default:
                cv::util::throw_error(std::logic_error("Unsupported optional output shape for Python"));
        }
    }
}
} // anonymous namespace

class cv::gimpl::GStreamingExecutor::Synchronizer final {
    gapi::streaming::sync_policy m_sync_policy = gapi::streaming::sync_policy::dont_sync;
    ade::Graph& m_island_graph;
    cv::gimpl::GIslandModel::Graph m_gim;
    std::size_t m_queue_capacity = 0u;
    std::thread m_thread;

    std::vector<ade::NodeHandle> m_synchronized_emitters;
    std::vector<stream::SyncQueue> m_sync_queues;

    std::vector<stream::Q*> newSyncQueue() {
        m_sync_queues.emplace_back(SyncQueue{});
        m_sync_queues.back().set_capacity(m_queue_capacity);
        return std::vector<Q*>{&m_sync_queues.back()};
    }
public:
    Synchronizer(gapi::streaming::sync_policy sync_policy,
                 ade::Graph& island_graph,
                 std::size_t queue_capacity)
        : m_sync_policy(sync_policy)
        , m_island_graph(island_graph)
        , m_gim(m_island_graph)
        , m_queue_capacity(queue_capacity) {
    }

    void registerVideoEmitters(std::vector<ade::NodeHandle>&& emitters) {
        // There is no point to make synchronization for the one video input
        // so do nothing in this case
        if (   m_sync_policy == cv::gapi::streaming::sync_policy::drop
            && emitters.size() > 1u) {
            m_synchronized_emitters = std::move(emitters);
            m_sync_queues.reserve(m_synchronized_emitters.size());
        }
    }

    std::vector<stream::Q*> outQueues(const ade::NodeHandle& emitter) {
        // If the emitter was registered previously (which means it needs to be synchronized),
        // create a new queue for this emitter to push the data to. Sync thread will
        // pop from this queue and push data to emitter's readers.
        // If the emitter was not registered, direct emitter output to its immediate readers right away
        return m_synchronized_emitters.end() != std::find(m_synchronized_emitters.begin(),
                                                          m_synchronized_emitters.end(),
                                                          emitter)
               ? newSyncQueue()
               : reader_queues(m_island_graph, emitter->outNodes().front());
    }

    // Start a thread which will handle the synchronization.
    // Do nothing if synchronization is not needed
    void start() {
        if (m_synchronized_emitters.size() != 0) {
            GAPI_Assert(m_synchronized_emitters.size() > 1u);
            std::vector<Q*> sync_in_queues(m_synchronized_emitters.size());
            std::vector<std::vector<Q*>> sync_out_queues(m_synchronized_emitters.size());
            for (auto it : ade::util::indexed(m_synchronized_emitters)) {
                const auto id = ade::util::index(it);
                const auto eh = ade::util::value(it);
                sync_in_queues[id] = &m_sync_queues[id];
                sync_out_queues[id] = reader_queues(m_island_graph, eh->outNodes().front());
            }
            m_thread = std::thread(syncActorThread,
                                   std::move(sync_in_queues),
                                   std::move(sync_out_queues));
        }
    }

    void join() {
        if (m_synchronized_emitters.size() != 0) {
            m_thread.join();
        }
    }

    void clear() {
        for (auto &q : m_sync_queues) q.clear();
        m_sync_queues.clear();
        m_synchronized_emitters.clear();
    }
};

// GStreamingExecutor expects compile arguments as input to have possibility to do
// proper graph reshape and islands recompilation
cv::gimpl::GStreamingExecutor::GStreamingExecutor(std::unique_ptr<ade::Graph> &&g_model,
                                                  const GCompileArgs &comp_args)
    : m_orig_graph(std::move(g_model))
    , m_island_graph(GModel::Graph(*m_orig_graph).metadata()
                     .get<IslandModel>().model)
    , m_comp_args(comp_args)
    , m_gim(*m_island_graph)
    , m_desync(GModel::Graph(*m_orig_graph).metadata()
               .contains<Desynchronized>())
{
    GModel::Graph gm(*m_orig_graph);
    // NB: Right now GIslandModel is acyclic, and all the below code assumes that.
    // NB: This naive execution code is taken from GExecutor nearly
    // "as-is"

    if (m_desync) {
        check_DesyncObjectConsumedByMultipleIslands(m_gim);
    }

    const auto proto = gm.metadata().get<Protocol>();
    m_emitters      .resize(proto.in_nhs.size());
    m_emitter_queues.resize(proto.in_nhs.size());
    m_sinks         .resize(proto.out_nhs.size());
    m_sink_queues   .resize(proto.out_nhs.size(), nullptr);
    m_sink_sync     .resize(proto.out_nhs.size(), -1);

    // Very rough estimation to limit internal queue sizes if not specified by the user.
    // Pipeline depth is equal to number of its (pipeline) steps.
    auto has_queue_capacity = cv::gapi::getCompileArg<cv::gapi::streaming::queue_capacity>(m_comp_args);
    const auto queue_capacity = has_queue_capacity ? has_queue_capacity->capacity :
            3*std::count_if
            (m_gim.nodes().begin(),
            m_gim.nodes().end(),
            [&](ade::NodeHandle nh) {
                return m_gim.metadata(nh).get<NodeKind>().k == NodeKind::ISLAND;
            });
    GAPI_Assert(queue_capacity != 0u);

    auto sync_policy = cv::gimpl::getCompileArg<cv::gapi::streaming::sync_policy>(m_comp_args)
                       .value_or(cv::gapi::streaming::sync_policy::dont_sync);
    m_sync.reset(new Synchronizer(sync_policy, *m_island_graph, queue_capacity));

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
                // FIXME: SAME APPLIES TO THE REGULAR GEXECUTOR!!
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
                ade::TypedGraph<DataQueue, DesyncSpecialCase> qgr(*m_island_graph);
                bool is_desync_start = false;
                for (auto eh : nh->inEdges())
                {
                    // ...only if the data is not compile-const
                    if (const_ins.count(eh->srcNode()) == 0) {
                        if (m_gim.metadata(eh).contains<DesyncIslEdge>()) {
                            qgr.metadata(eh).set(DataQueue(DataQueue::DESYNC));
                            is_desync_start = true;
                        } else if (qgr.metadata(eh).contains<DesyncSpecialCase>()) {
                            // See comment below
                            // Limit queue size to 1 in this case
                            qgr.metadata(eh).set(DataQueue(1u));
                        } else {
                            qgr.metadata(eh).set(DataQueue(queue_capacity));
                        }
                        m_internal_queues.insert(qgr.metadata(eh).get<DataQueue>().q.get());
                    }
                }
                // WORKAROUND:
                // Since now we always know desync() is followed by copy(),
                // copy is always the island with DesyncIslEdge.
                // Mark the node's outputs a special way so then its following
                // queue sizes will be limited to 1 (to avoid copy reading more
                // data in advance - as there's no other way for the underlying
                // "slow" part to control it)
                if (is_desync_start) {
                    auto isl = m_gim.metadata(nh).get<FusedIsland>().object;
                    // In the current implementation, such islands
                    // _must_ start with copy
                    GAPI_Assert(isl->in_ops().size() == 1u);
                    GAPI_Assert(GModel::Graph(*m_orig_graph)
                                .metadata(*isl->in_ops().begin())
                                .get<cv::gimpl::Op>()
                                .k.name == cv::gimpl::streaming::GCopy::id());
                    for (auto out_nh : nh->outNodes()) {
                        for (auto out_eh : out_nh->outEdges()) {
                            qgr.metadata(out_eh).set(DesyncSpecialCase{});
                        }
                    }
                }
                // It is ok to do it here since the graph is visited in
                // a topologic order and its consumers (those checking
                // their input edges & initializing queues) are yet to be
                // visited
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
                m_sink_queues[sink_idx] = qgr.metadata(nh->inEdges().front()).get<DataQueue>().q.get();

                // Assign a desync tag
                const auto sink_out_nh = gm.metadata().get<Protocol>().out_nhs[sink_idx];
                if (gm.metadata(sink_out_nh).contains<DesyncPath>()) {
                    // metadata().get_or<> could make this thing better
                    m_sink_sync[sink_idx] = gm.metadata(sink_out_nh).get<DesyncPath>().index;
                }
            }
            break;
        default:
            GAPI_Assert(false);
            break;
        } // switch(kind)
    } // for(gim nodes)

    // If there are desynchronized parts in the graph, there may be
    // multiple theads polling every separate (desynchronized)
    // branch in the graph individually. Prepare a mapping information
    // for any such thread
    for (auto &&idx : ade::util::iota(m_sink_queues.size())) {
        auto  path_id = m_sink_sync[idx];
        auto &info    = m_collector_map[path_id];
        info.queues.push_back(m_sink_queues[idx]);
        info.mapping.push_back(static_cast<int>(idx));
    }

    // Reserve space in the final queue based on the number
    // of desync parts (they can generate output individually
    // per the same input frame, so the output traffic multiplies)
    GAPI_Assert(m_collector_map.size() > 0u);
    m_out_queue.set_capacity(queue_capacity * m_collector_map.size());

    // FIXME: The code duplicates logic of collectGraphInfo()
    cv::gimpl::GModel::ConstGraph cgr(*m_orig_graph);
    auto meta = cgr.metadata().get<cv::gimpl::Protocol>().out_nhs;
    out_info.reserve(meta.size());

    ade::util::transform(meta, std::back_inserter(out_info), [&cgr](const ade::NodeHandle& nh) {
        const auto& data = cgr.metadata(nh).get<cv::gimpl::Data>();
        return cv::GTypeInfo{data.shape, data.kind, data.ctor};
    });
}

cv::gimpl::GStreamingExecutor::~GStreamingExecutor()
{
    // FIXME: this is a temporary try-catch exception hadling.
    // Need to eliminate throwings from stop()
    try {
        if (state == State::READY || state == State::RUNNING)
            stop();
    } catch (const std::exception& e) {
        std::stringstream message;
        message << "~GStreamingExecutor() threw exception with message '" << e.what() << "'\n";
        GAPI_LOG_WARNING(NULL, message.str());
    }
}

void cv::gimpl::GStreamingExecutor::setSource(GRunArgs &&ins)
{
    GAPI_Assert(state == State::READY || state == State::STOPPED);

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
    // Also collect video emitter nodes to use them later in synchronization
    std::vector<ade::NodeHandle> video_emitters;
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
            // Currently all video inputs are syncronized if sync policy is to drop,
            // there is no different fps branches etc, so all video emitters are registered
            video_emitters.emplace_back(emit_nh);
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

    m_sync->registerVideoEmitters(std::move(video_emitters));

    // Craft here a completion callback to notify Const emitters that
    // any of video sources is over
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
        auto out_queues = m_sync->outQueues(eh);

        m_threads.emplace_back(emitterActorThread,
                               emitter,
                               std::ref(m_emitter_queues[id]),
                               out_queues,
                               real_video_completion_cb);
    }

    m_sync->start();

    // Now do this for every island (in a topological order)
    for (auto &&op : m_ops)
    {
        // Prepare island thread parameters
        auto island_exec = m_gim.metadata(op.nh).get<IslandExec>().object;

        // Collect actor's input queues
        auto in_queues = input_queues(*m_island_graph, op.nh);

        // Collect actor's output queues.
        // This may be tricky...
        std::vector< std::vector<stream::Q*> > out_queues;
        for (auto &&out_eh : op.nh->outNodes()) {
            out_queues.push_back(reader_queues(*m_island_graph, out_eh));
        }

        // Create just empty island meta information
        std::string island_meta_info { };
#if defined(OPENCV_WITH_ITT)
        // In case if ITT tracing is enabled fill meta information with the built island name
        island_meta_info = GIslandModel::traceIslandName(op.nh, m_gim);
#endif // OPENCV_WITH_ITT

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
                               island_exec,
                               in_queues,
                               op.in_constants,
                               out_queues,
                               island_meta_info);
    }

    // Finally, start collector thread(s).
    // If there are desynchronized parts in the graph, there may be
    // multiple theads polling every separate (desynchronized)
    // branch in the graph individually.
    const bool has_main_path = m_sink_sync.end() !=
        std::find(m_sink_sync.begin(), m_sink_sync.end(), -1);
    for (auto &&info : m_collector_map) {
        m_threads.emplace_back(collectorThread,
                               info.second.queues,
                               info.second.mapping,
                               m_sink_queues.size(),
                               has_main_path ? info.first == -1 : true, // see below (*)
                               std::ref(m_out_queue));

        // (*) - there may be a problem with desynchronized paths when those work
        // faster than the main path. In this case, the desync paths get "Stop" message
        // earlier and thus broadcast it down to pipeline gets stopped when there is
        // some "main path" data to process. This new collectorThread's flag regulates it:
        // - desync paths should never post Stop message if there is a main path.
        // - if there is no main path, than any desync path can terminate the execution.
    }
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
    m_sync->join();

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
    m_const_emitter_queues.clear();
    m_const_vals.clear();
    m_out_queue.clear();
    m_sync->clear();

    for (auto &&op : m_ops) {
        op.isl_exec->handleStopStream();
    }

    state = State::STOPPED;
}

bool cv::gimpl::GStreamingExecutor::pull(cv::GRunArgsP &&outs)
{
    GAPI_ITT_STATIC_LOCAL_HANDLE(pull_hndl, "GStreamingExecutor::pull");
    GAPI_ITT_AUTO_TRACE_GUARD(pull_hndl);

    // This pull() can only be called when there's no desynchronized
    // parts in the graph.
    GAPI_Assert(!m_desync &&
                "This graph has desynchronized parts! Please use another pull()");

    if (state == State::STOPPED)
        return false;
    GAPI_Assert(state == State::RUNNING);
    GAPI_Assert(m_sink_queues.size() == outs.size() &&
                "Number of data objects in cv::gout() must match the number of graph outputs in cv::GOut()");

    Cmd cmd;
    m_out_queue.pop(cmd);
    if (cv::util::holds_alternative<Stop>(cmd))
    {
        wait_shutdown();
        return false;
    }

    GAPI_Assert(cv::util::holds_alternative<Result>(cmd));
    cv::GRunArgs &this_result = cv::util::get<Result>(cmd).args;
    sync_data(this_result, outs);
    return true;
}

bool cv::gimpl::GStreamingExecutor::pull(cv::GOptRunArgsP &&outs)
{
    // This pull() can only be called in both cases: if there are
    // desyncrhonized parts or not.

    // FIXME: so far it is a full duplicate of standard pull except
    // the sync_data version called.
    if (state == State::STOPPED)
        return false;
    GAPI_Assert(state == State::RUNNING);
    GAPI_Assert(m_sink_queues.size() == outs.size() &&
                "Number of data objects in cv::gout() must match the number of graph outputs in cv::GOut()");

    Cmd cmd;
    m_out_queue.pop(cmd);
    if (cv::util::holds_alternative<Stop>(cmd))
    {
        wait_shutdown();
        return false;
    }

    GAPI_Assert(cv::util::holds_alternative<Result>(cmd));
    sync_data(cv::util::get<Result>(cmd), outs);
    return true;
}

std::tuple<bool, cv::util::variant<cv::GRunArgs, cv::GOptRunArgs>> cv::gimpl::GStreamingExecutor::pull()
{
    using RunArgs = cv::util::variant<cv::GRunArgs, cv::GOptRunArgs>;
    bool is_over = false;

    if (m_desync) {
        GOptRunArgs opt_run_args;
        GOptRunArgsP opt_outs;
        opt_outs.reserve(out_info.size());
        opt_run_args.reserve(out_info.size());

        constructOptGraphOutputs(out_info, opt_run_args, opt_outs);
        is_over = pull(std::move(opt_outs));
        return std::make_tuple(is_over, RunArgs(opt_run_args));
    }

    GRunArgs run_args;
    GRunArgsP outs;
    run_args.reserve(out_info.size());
    outs.reserve(out_info.size());

    constructGraphOutputs(out_info, run_args, outs);
    is_over = pull(std::move(outs));
    return std::make_tuple(is_over, RunArgs(run_args));
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

    GAPI_Assert(cv::util::holds_alternative<Result>(cmd));
    cv::GRunArgs &this_result = cv::util::get<Result>(cmd).args;
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
