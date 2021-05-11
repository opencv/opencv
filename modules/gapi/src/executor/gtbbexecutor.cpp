// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2021 Intel Corporation

#include "gtbbexecutor.hpp"

#if defined(HAVE_TBB) && (TBB_INTERFACE_VERSION < 12000)
// TODO: TBB task API has been deprecated and removed in 12000

#include "utils/itt.hpp"

#include <opencv2/gapi/own/assert.hpp>
#include <opencv2/gapi/util/copy_through_move.hpp>
#include "logger.hpp" // GAPI_LOG

#include <tbb/task.h>
#include <memory> // unique_ptr

#include <atomic>
#include <condition_variable>

#include <chrono>

#define ASSERT(expr)          GAPI_DbgAssert(expr)

#define LOG_INFO(tag, ...)    GAPI_LOG_INFO(tag, __VA_ARGS__)
#define LOG_WARNING(tag, ...) GAPI_LOG_WARNING(tag, __VA_ARGS__)
#define LOG_DEBUG(tag, ...)   GAPI_LOG_DEBUG(tag, __VA_ARGS__)


namespace cv { namespace gimpl { namespace parallel {

namespace detail {
// some helper staff to deal with tbb::task related entities
namespace tasking {

enum class use_tbb_scheduler_bypass {
   NO,
   YES
};

inline void assert_graph_is_running(tbb::task* root) {
   // tbb::task::wait_for_all block calling thread until task ref_count is dropped to 1
   // So if the root task ref_count is greater than 1 graph still has a job to do and
   // according wait_for_all() has not yet returned
   ASSERT(root->ref_count() > 1);
}

// made template to break circular dependencies
template<typename body_t>
struct functor_task : tbb::task {
   body_t body;

   template<typename arg_t>
   functor_task(arg_t&& a) : body(std::forward<arg_t>(a)) {}

   tbb::task * execute() override {
      assert_graph_is_running(parent());

      auto reuse_current_task = body();
      // if needed, say TBB to execute current task once again
      return (use_tbb_scheduler_bypass::YES ==  reuse_current_task) ? (recycle_as_continuation(), this) : nullptr;
   }
   ~functor_task() {
      assert_graph_is_running(parent());
   }
};

template<typename body_t>
auto allocate_task(tbb::task* root, body_t const& body) -> functor_task<body_t>* {
    return new(tbb::task::allocate_additional_child_of(*root)) functor_task<body_t>{body};
}

template<typename body_t>
void spawn_no_assert(tbb::task* root, body_t const& body) {
   tbb::task::spawn(* allocate_task(root, body));
}

template<typename body_t>
void batch_spawn(size_t count, tbb::task* root, body_t const& body, bool do_assert_graph_is_running = true) {
   GAPI_ITT_STATIC_LOCAL_HANDLE(ittTbbSpawnReadyBlocks, "spawn ready blocks");
   GAPI_ITT_AUTO_TRACE_GUARD(ittTbbSpawnReadyBlocks);
   if (do_assert_graph_is_running) {
       assert_graph_is_running(root);
   }

   for (size_t i=0; i<count; i++) {
       spawn_no_assert(root, body);
   }
}


struct destroy_tbb_task {
    void operator()(tbb::task* t) const { if (t) tbb::task::destroy(*t);};
};

using root_t = std::unique_ptr<tbb::task, destroy_tbb_task>;

root_t inline create_root(tbb::task_group_context& ctx) {
    root_t  root{new (tbb::task::allocate_root(ctx)) tbb::empty_task};
    root->set_ref_count(1); // required by wait_for_all, as it waits until counter drops to 1
    return root;
}

std::size_t inline tg_context_traits() {
    // Specify tbb::task_group_context::concurrent_wait in the traits to ask TBB scheduler not to change
    // ref_count of the task we wait on (root) when wait is complete.
    return tbb::task_group_context::default_traits | tbb::task_group_context::concurrent_wait;
}

} // namespace tasking

namespace async {
struct async_tasks_t {
    std::atomic<size_t>         count {0};
    std::condition_variable     cv;
    std::mutex                  mtx;
};

enum class wake_tbb_master {
   NO,
   YES
};

void inline wake_master(async_tasks_t& async_tasks, wake_tbb_master wake_master) {
    // TODO: seems that this can be relaxed
    auto active_async_tasks = --async_tasks.count;

    if ((active_async_tasks == 0) || (wake_master == wake_tbb_master::YES)) {
        // Was the last async task or asked to wake TBB master up(e.g. there are new TBB tasks to execute)
        GAPI_ITT_STATIC_LOCAL_HANDLE(ittTbbUnlockMasterThread, "Unlocking master thread");
        GAPI_ITT_AUTO_TRACE_GUARD(ittTbbUnlockMasterThread);
        // While decrement of async_tasks_t::count is atomic, it might occur after the waiting
        // thread has read its value but _before_ it actually starts waiting on the condition variable.
        // So, lock acquire is needed to guarantee that current condition check (if any) in the waiting thread
        // (possibly ran in parallel to async_tasks_t::count decrement above) is completed _before_ signal is issued.
        // Therefore when notify_one is called, waiting thread is either sleeping on the condition variable or
        // running a new check which is guaranteed to pick the new value and return from wait().

        // There is no need to _hold_ the lock while signaling, only to acquire it.
        std::unique_lock<std::mutex> {async_tasks.mtx};   // Acquire and release the lock.
        async_tasks.cv.notify_one();
    }
}

struct master_thread_sleep_lock_t
{
    struct sleep_unlock {
       void operator()(async_tasks_t* t) const {
          ASSERT(t);
          wake_master(*t, wake_tbb_master::NO);
       }
    };

    std::unique_ptr<async_tasks_t, sleep_unlock>  guard;

    master_thread_sleep_lock_t() = default;
    master_thread_sleep_lock_t(async_tasks_t*  async_tasks_ptr) : guard(async_tasks_ptr) {
        // TODO: seems that this can be relaxed
        ++(guard->count);
    }

    void unlock(wake_tbb_master wake) {
        if (auto* p = guard.release()) {
            wake_master(*p, wake);
        }
    }
};

master_thread_sleep_lock_t inline lock_sleep_master(async_tasks_t& async_tasks) {
    return {&async_tasks};
}

enum class is_tbb_work_present {
   NO,
   YES
};

//RAII object to block TBB master thread (one that does wait_for_all())
//N.B. :wait_for_all() return control when root ref_count drops to 1,
struct root_wait_lock_t {
    struct root_decrement_ref_count{
        void operator()(tbb::task* t) const {
            ASSERT(t);
            auto result = t->decrement_ref_count();
            ASSERT(result >= 1);
        }
    };

    std::unique_ptr<tbb::task, root_decrement_ref_count> guard;

    root_wait_lock_t() = default;
    root_wait_lock_t(tasking::root_t& root, is_tbb_work_present& previous_state) : guard{root.get()} {
        // Block the master thread while the *this object is alive.
        auto new_root_ref_count = root->add_ref_count(1);
        previous_state = (new_root_ref_count == 2) ? is_tbb_work_present::NO : is_tbb_work_present::YES;
    }

};

root_wait_lock_t inline lock_wait_master(tasking::root_t& root, is_tbb_work_present& previous_state) {
    return root_wait_lock_t{root, previous_state};
}

} // namespace async

inline tile_node*  pop(prio_items_queue_t& q) {
    tile_node* node = nullptr;
    bool popped = q.try_pop(node);
    ASSERT(popped && "queue should be non empty as we push items to it before we spawn");
    return node;
}

namespace graph {
    // Returns : number of items actually pushed into the q
    std::size_t inline push_ready_dependants(prio_items_queue_t& q, tile_node* node) {
        GAPI_ITT_STATIC_LOCAL_HANDLE(ittTbbAddReadyBlocksToQueue, "add ready blocks to queue");
        GAPI_ITT_AUTO_TRACE_GUARD(ittTbbAddReadyBlocksToQueue);
        std::size_t ready_items = 0;
        // enable dependent tasks
        for (auto* dependant : node->dependants) {
            // fetch_and_sub returns previous value
            if (1 == dependant->dependency_count.fetch_sub(1)) {
                // tile node is ready for execution, add it to the queue
                q.push(dependant);
                ++ready_items;
            }
        }
        return ready_items;
    }

    struct exec_ctx {
        tbb::task_arena&                arena;
        prio_items_queue_t&             q;
        tbb::task_group_context         tg_ctx;
        tasking::root_t                 root;
        detail::async::async_tasks_t    async_tasks;
        std::atomic<size_t>             executed {0};

        exec_ctx(tbb::task_arena& arena_, prio_items_queue_t& q_)
            : arena(arena_), q(q_),
              // As the traits is last argument, explicitly specify (default) value for first argument
              tg_ctx{tbb::task_group_context::bound, tasking::tg_context_traits()},
              root(tasking::create_root(tg_ctx))
        {}
    };

    // At the moment there are no suitable tools to  manage TBB priorities on task by task basis.
    // Instead priority queue is used to respect tile_node priorities.
    // As well, TBB task is not bound to any particular tile_node until actually executed.

    // Strictly speaking there are two graphs here:
    // - G-API one, described by the connected tile_node instances.
    //   This graph is :
    //    - Known beforehand, and do not change during the execution (i.e. static)
    //    - Contains both TBB non-TBB parts
    //    - prioritized, (i.e. all nodes has assigned priority of execution)
    //
    // - TBB task tree, which is :
    //    - flat (Has only two levels : root and leaves)
    //    - dynamic, i.e. new leaves are added on demand when new tbb tasks are spawned
    //    - describes only TBB/CPU part of the whole graph
    //    - non-prioritized (i.e. all tasks are created equal)

    // Class below represents TBB task payload.
    //
    // Each instance basically does the three things :
    // 1. Gets the tile_node item from the top of the queue
    // 2. Executes its body
    // 3. Pushes dependent tile_nodes to the queue once they are ready
    //
    struct task_body {
        exec_ctx& ctx;

        std::size_t push_ready_dependants(tile_node* node) const {
            return graph::push_ready_dependants(ctx.q, node);
        }

        void spawn_clones(std::size_t items) const {
            tasking::batch_spawn(items, ctx.root.get(), *this);
        }

        task_body(exec_ctx& ctx_) : ctx(ctx_) {}
        tasking::use_tbb_scheduler_bypass operator()() const {
            ASSERT(!ctx.q.empty() && "Spawned task with no job to do ? ");

            tile_node* node = detail::pop(ctx.q);

            auto result = tasking::use_tbb_scheduler_bypass::NO;
            // execute the task

            if (auto p = util::get_if<tile_node::sync_task_body>(&(node->task_body))) {
                // synchronous task
                p->body();

                std::size_t ready_items = push_ready_dependants(node);

                if (ready_items > 0) {
                    // spawn one less tasks and say TBB to reuse(recycle) current task
                    spawn_clones(ready_items - 1);
                    result = tasking::use_tbb_scheduler_bypass::YES;
                }
            }
            else {
                LOG_DEBUG(NULL, "Async task");
                using namespace detail::async;
                using util::copy_through_move;

                auto block_master = copy_through_move(lock_sleep_master(ctx.async_tasks));

                auto self_copy = *this;
                auto callback = [node, block_master, self_copy] () mutable /*due to block_master.get().unlock()*/ {
                    LOG_DEBUG(NULL, "Async task callback is called");
                    // Implicitly unlock master right in the end of callback
                    auto master_sleep_lock = std::move(block_master);
                    std::size_t ready_items = self_copy.push_ready_dependants(node);
                    if (ready_items > 0) {
                        auto master_was_active = is_tbb_work_present::NO;
                        {
                            GAPI_ITT_STATIC_LOCAL_HANDLE(ittTbbEnqueueSpawnReadyBlocks, "enqueueing a spawn of ready blocks");
                            GAPI_ITT_AUTO_TRACE_GUARD(ittTbbEnqueueSpawnReadyBlocks);
                            // Force master thread (one that does wait_for_all()) to (actively) wait for enqueued tasks
                            // and unlock it right after all dependent tasks are spawned.

                            auto root_wait_lock = copy_through_move(lock_wait_master(self_copy.ctx.root, master_was_active));

                            // TODO: add test to cover proper holding of root_wait_lock
                            // As the calling thread most likely is not TBB one, instead of spawning TBB tasks directly we
                            // enqueue a task which will spawn them.
                            // For master thread to not leave wait_for_all() prematurely,
                            // hold the root_wait_lock until need tasks are actually spawned.
                            self_copy.ctx.arena.enqueue([ready_items, self_copy, root_wait_lock]() {
                                self_copy.spawn_clones(ready_items);
                                // TODO: why we need this? Either write a descriptive comment or remove it
                                volatile auto unused = root_wait_lock.get().guard.get();
                                util::suppress_unused_warning(unused);
                            });
                        }
                        // Wake master thread (if any) to pick up the enqueued tasks iff:
                        // 1. there is new TBB work to do, and
                        // 2. Master thread was sleeping on condition variable waiting for async tasks to complete
                        //   (There was no active work before (i.e. root->ref_count() was == 1))
                        auto wake_master = (master_was_active == is_tbb_work_present::NO) ?
                                wake_tbb_master::YES : wake_tbb_master::NO;
                        master_sleep_lock.get().unlock(wake_master);
                    }
                };

                auto& body = util::get<tile_node::async_task_body>(node->task_body).body;
                body(std::move(callback), node->total_order_index);
            }

            ctx.executed++;
            // reset dependecy_count to initial state to simplify re-execution of the same graph
            node->dependency_count = node->dependencies;

            return result;
        }
    };
}
} // namespace detail
}}}  // namespace cv::gimpl::parallel

void cv::gimpl::parallel::execute(prio_items_queue_t& q) {
    // get the reference to current task_arena (i.e. one we are running in)
#if TBB_INTERFACE_VERSION > 9002
    using attach_t = tbb::task_arena::attach;
#else
    using attach_t = tbb::internal::attach;
#endif

    tbb::task_arena arena{attach_t{}};
    execute(q, arena);
}

void cv::gimpl::parallel::execute(prio_items_queue_t& q, tbb::task_arena& arena) {
    using namespace detail;
    graph::exec_ctx ctx{arena, q};

    arena.execute(
        [&]() {
            // Passed in queue is assumed to contain starting tasks, i.e. ones with no (or resolved) dependencies
            auto num_start_tasks = q.size();

            // TODO: use recursive spawning and task soft affinity for faster task distribution
            // As graph is starting and no task has been spawned yet
            // assert_graph_is_running(root) will not hold, so spawn without assert
            tasking::batch_spawn(num_start_tasks, ctx.root.get(), graph::task_body{ctx}, /* assert_graph_is_running*/false);

            using namespace std::chrono;
            high_resolution_clock timer;

            auto tbb_work_done   = [&ctx]() { return 1 == ctx.root->ref_count(); };
            auto async_work_done = [&ctx]() { return 0 == ctx.async_tasks.count; };
            do {
               // First participate in execution of TBB graph till there are no more ready tasks.
               ctx.root->wait_for_all();

               if (!async_work_done()) { // Wait on the conditional variable iff there is active async work
                   auto start = timer.now();
                   std::unique_lock<std::mutex> lk(ctx.async_tasks.mtx);
                   // Wait (probably by sleeping) until all async tasks are completed or new TBB tasks are created.
                   // FIXME: Use TBB resumable tasks here to avoid blocking TBB thread
                   ctx.async_tasks.cv.wait(lk, [&]{return async_work_done() || !tbb_work_done() ;});

                   LOG_INFO(NULL, "Slept for " << duration_cast<milliseconds>(timer.now() - start).count() << " ms \n");
               }
            }
            while(!tbb_work_done() || !async_work_done());

            ASSERT(tbb_work_done() && async_work_done() && "Graph is still running?");
        }
    );

    LOG_INFO(NULL, "Done. Executed " << ctx.executed << " tasks");
}

std::ostream& cv::gimpl::parallel::operator<<(std::ostream& o, tile_node const& n) {
    o << "("
            << " at:"    << &n << ","
            << "indx: "  << n.total_order_index << ","
            << "deps #:" << n.dependency_count.value << ", "
            << "prods:"  << n.dependants.size();

    o << "[";
    for (auto* d: n.dependants) {
        o << d << ",";
    }
    o << "]";

    o << ")";
    return o;
}

#endif // HAVE_TBB && TBB_INTERFACE_VERSION
