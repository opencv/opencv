#if defined(USE_GAPI_TBB_EXECUTOR)
#include "gapi_tbb_executor.hpp"
#include "gapi_itt.hpp"

#include <tbb/task.h>
#include <tbb/task_arena.h>
#include <memory> //unique_ptr

#include <atomic>
#include <condition_variable>

#include <chrono>

//#include <cassert>

#include <cstdio>
#include <cstdlib> // abort()

inline void dev_assert(bool val, const char* str, int line, const char* file)
{
    if (!val)
    {
        fprintf(stderr, "%s:%d: Assertion \"%s\" failed\n", file, line, str);
        fflush(stderr);
        abort();
    }
}
#define ASSERT(arg) dev_assert(static_cast<bool>(arg), #arg, __LINE__, __FILE__)

#ifdef OPENCV_WITH_ITT
const __itt_domain* cv::gimpl::parallel::gapi_itt_domain = __itt_domain_create("GAPI Context");
#endif

namespace cv{ namespace gimpl { namespace parallel {

inline void assert_graph_is_running(tbb::task* root)
{
   //tbb::task::wait_for_all block calling thread until task ref_count is dropped to 1
   //So if the root task refc_count is greater than 1 graph still has a job to do and
   //according wait_for_all has not yet returned
   ASSERT(root->ref_count() > 1);
}

enum class use_tbb_scheduler_bypass {
   yes,
   no
};

enum class wake_tbb_master {
   yes,
   no
};

//made template to break circular dependencies
template<typename body_t>
struct functor_task : tbb::task {
   body_t body;

   template<typename arg_t>
   functor_task(arg_t&& a) : body(std::forward<arg_t>(a)){}

   tbb::task * execute() override
   {
      assert_graph_is_running(parent());

      auto reuse_current_task = body();
      //if needed, say TBB to execute current task once again
      return (use_tbb_scheduler_bypass::yes ==  reuse_current_task) ? (recycle_as_continuation(), this) : nullptr;
   }
   ~functor_task(){
      assert_graph_is_running(parent());
   }
};

template<typename body_t>
auto allocate_task(tbb::task* root, body_t const& body) -> functor_task<body_t>*
{
    return new(tbb::task::allocate_additional_child_of(*root) ) functor_task<body_t>{body};
}

template<typename body_t>
void spawn_no_assert(tbb::task* root, body_t const& body)
{
   tbb::task::spawn(* allocate_task(root, body));
}

#ifdef OPENCV_WITH_ITT
namespace {
    static __itt_string_handle* ittTbbAddReadyBlocksToQueue       = __itt_string_handle_create( "add ready blocks to queue" );
    static __itt_string_handle* ittTbbSpawnReadyBlocks            = __itt_string_handle_create( "spawn ready blocks" );
    static __itt_string_handle* ittTbbEnqueueSpawnReadyBlocks     = __itt_string_handle_create( "enqueing a spawn of ready blocks" );
    static __itt_string_handle* ittTbbUnlockMasterThread          = __itt_string_handle_create( "Unlocking master thread" );
}
#endif //OPENCV_WITH_ITT


template<typename body_t>
void batch_spawn(size_t count, tbb::task* root, body_t const& body, bool do_assert_graph_is_running = true)
{
   ITT_AUTO_TRACE_GUARD(ittTbbSpawnReadyBlocks);
   if (do_assert_graph_is_running){
       assert_graph_is_running(root);
   }

   for (size_t i=0; i<count; i++)
   {
       spawn_no_assert(root, body);
   }
}


}}}  // namespace cv::gimpl::parallel

void cv::gimpl::parallel::execute(tbb::concurrent_priority_queue<tile_node* , tile_node_indirect_priority_comparator> & q)
{
    struct destroy_tbb_task{
        void operator()(tbb::task* t) const {tbb::task::destroy(*t);};
    };
    std::unique_ptr<tbb::task, destroy_tbb_task>  root;

    //Specify tbb::task_group_context::concurrent_wait in the traits to ask TBB scheduler not to change ref_count of the task we wait on (root) when wait is complete.
    //As the traits is last argument explicitly specify (default) value for first argument
    tbb::task_group_context m_context = {tbb::task_group_context::bound, tbb::task_group_context::default_traits | tbb::task_group_context::concurrent_wait};


    root.reset(new (tbb::task::allocate_root(m_context)) tbb::empty_task);
    root->set_ref_count(1); //required by wait_for_all, as it waits until counter drops to 1

    unsigned int reserved_for_master_threads = 1;
             int max_concurrency             = tbb::task_arena::automatic; // make equal to 1 for single thread debugging;

    if (max_concurrency == 1)
    {
        //Leave no room for TBB worker threads, by reserving all to masters.
        //TBB runtime guarantees that no worker threads will join the arena if max_concurrency is equal to reserved_for_master_threads
        //except 1:1 + use of enqueued tasks for safety guarantee.
        //So deliberately make it 2:2 to force TBB not to create extra thread.
        //N.B. one slot will left empty as only one master thread(one that calls root->wait_for_all()) will join the arena
        //FIXME: strictly speaking master can take any free slot, not the first one.
        //However at the moment master seems to pick 0 slot all the time.
        max_concurrency = 2;
        reserved_for_master_threads = 2;
    }
    tbb::task_arena arena{max_concurrency, reserved_for_master_threads};


    std::atomic<size_t>         executed {0};

    struct async_tasks_t {
        std::atomic<size_t>         count {0};//count {0};
        std::condition_variable     cv;     //cv;
        std::mutex                  mtx; //mtx;
    };

    async_tasks_t async_tasks;

    struct master_thread_sleep_lock_t
    {
        async_tasks_t*         async_tasks_p  = nullptr;

       //C++11 does not allow explicit lambda capture initialization we need a copy constructor on this to capture it into lambda
       master_thread_sleep_lock_t(master_thread_sleep_lock_t& src) : master_thread_sleep_lock_t(std::move(src)) {};
       master_thread_sleep_lock_t(master_thread_sleep_lock_t&& source)             { std::swap(async_tasks_p, source.async_tasks_p);}
       master_thread_sleep_lock_t& operator=(master_thread_sleep_lock_t&& source)  { std::swap(async_tasks_p, source.async_tasks_p); return *this;}

       master_thread_sleep_lock_t() = default;
       master_thread_sleep_lock_t(async_tasks_t*  async_tasks_p_ ) : async_tasks_p(async_tasks_p_)
       {
          //TODO: seems that this can be relaxed
          ++(async_tasks_p->count);
       }

       void unlock(wake_tbb_master wake_master)
       {
          if (async_tasks_p)
          {
             //TODO: seems that this can be relaxed
             auto result = --(async_tasks_p->count);

             ASSERT(result >= 0);
             //TODO: checks performance gains of minimizing number of call to notify (i.e. only if this is the last one async node or there are new tbb task to execute)
             if ((result == 0) || (wake_master == wake_tbb_master::yes) )//was the last or there is the new TBB tasks to execute
             {
                 ITT_AUTO_TRACE_GUARD(ittTbbUnlockMasterThread);
                //Wile decrement of tbb_executor->async_tasks is atomic it might be done after waiting thread checked it value but _before_ it actually start waiting on the condition variable.
                //So lock acquire of is needed guarantee that current condition check (if any) in waiting thread (possibly ran in parallel to tbb_executor->async_tasks decrement above) is completed before
                //signal is issued. Therefore then notify_one is called waiting thread is either sleeping on the condition variable or running a new check which is guaranteed to pick new value and return
                //from wait().
                //There is no need to _hold_ the lock while signaling only to acquire it.
                std::unique_lock<std::mutex> {async_tasks_p->mtx};   //Acquire and release the lock.
                (async_tasks_p->cv).notify_one();
             }
             async_tasks_p = nullptr;
          }
       }

       ~master_thread_sleep_lock_t()
       {
          unlock(wake_tbb_master::no);
       }
    };

    struct unlock_root_wait_t {
        struct root_decrement_ref_count{
           void operator()(tbb::task* t) const{
              auto result = t->decrement_ref_count();
              ASSERT(result >= 1);
           }
        };

        std::unique_ptr<tbb::task, root_decrement_ref_count> guard;

        unlock_root_wait_t(tbb::task* root):guard{root} {}
        unlock_root_wait_t(unlock_root_wait_t&&) = default;
        //C++11 does not allow explicit lambda capture initialization we need a copy constructor on this to capture it into lambda
        unlock_root_wait_t(unlock_root_wait_t& src) : unlock_root_wait_t(std::move(src)) {}
    };

//    arena.execute(
//        [&](){
            //FIXME: avoid using std::function here due to extra indirection it impose
            //using std::function to allow self reference
            std::function<parallel::use_tbb_scheduler_bypass()> body = [&](){
                while(!q.empty()){

                    auto push_ready_dependees = [](tile_node* node, decltype(q) & q) -> std::size_t
                    {
                        ITT_AUTO_TRACE_GUARD(ittTbbAddReadyBlocksToQueue);
                        std::size_t ready_items = 0;
                        //then enable task dependees
                        for (auto* dependee : node->dependees)
                        {
                            //fetch_and_sub returns previous value
                            if (1 == dependee->dependency_count.fetch_sub(1))
                            {
                                //tile node is ready for execution, add it to the queue
                                q.push(dependee);
                                ++ready_items;
                            }
                        }
                        return ready_items;
                    };

                    tile_node* node = nullptr;
                    bool poped = q.try_pop(node);
                    ASSERT(poped && "queue should be non empty as we push items to it before we spawn");

                    auto result = parallel::use_tbb_scheduler_bypass::no;
                    //execute the task
                    if (!node->async)
                    {
                        node->task_body();

                        std::size_t ready_items = push_ready_dependees(node, q);

                        if (ready_items > 0){
                            //spawn one less tasks and say TBB to reuse(recycle) current task
                            parallel::batch_spawn(ready_items - 1,root.get(), body);
                            result = parallel::use_tbb_scheduler_bypass::yes;
                        }
                    }
                    else
                    {
//                        std::cout<<"async task "<<std::endl;
                        //move through copy this lock into the callback to block master until async tasks completes
                        master_thread_sleep_lock_t block_master{&async_tasks};

                        auto callback = [push_ready_dependees, node, &q, &arena, &root, block_master, body] () mutable /*due to block_master.unlock()*/
                        {
//                            std::cout<<"async task callback is called "<<std::endl;
                            std::size_t ready_items = push_ready_dependees(node, q);
                            if (ready_items > 0)
                            {
                                //force master thread (one that do wait_for_all()) to (actively) wait for enqueued tasks
                                //and unlock it right after all dependee tasks are spawn (and therefore ref_count of root has been increased accordingly)
                                auto new_root_ref_count = root->add_ref_count(1);
                                unlock_root_wait_t unlock_root_wait_guard{root.get()};

//                                std::chrono::high_resolution_clock timer;
                                {
                                    ITT_AUTO_TRACE_GUARD(ittTbbEnqueueSpawnReadyBlocks);
//                                    auto start = timer.now();
                                    arena.enqueue([ready_items, &root, body, unlock_root_wait_guard](){
                                        parallel::batch_spawn(ready_items, root.get(), body);
                                        //TODO: why we need this? Either write a descriptive comment or remove it
                                        volatile auto p = unlock_root_wait_guard.guard.get();
                                        util::suppress_unused_warning(p);
                                    });
//                                    auto end = timer.now();
//                                    std::cout<<"enqued in "<< std::chrono::duration_cast<std::chrono::microseconds>(end -start).count() <<" mks \n";
                                }
                                //unlock master thread waiting on conditional variable (if any) to pick up enqueued tasks
                                //wake master only if there is new work and there were no work before (i.e. root->ref_count() was 1 and master was
                                //waiting on cv for async tasks to complete)
                                block_master.unlock((new_root_ref_count == 2) ? wake_tbb_master::yes : wake_tbb_master::no);
                            }
                        };

                        node->async_task_body(callback, node->total_order_index);
                    }

                    executed++;
                    //reset dependecy_count to initial state
                    node->dependency_count = node->dependencies;

                    return result;
                }

                return parallel::use_tbb_scheduler_bypass::no;
            };

            auto num_start_tasks = q.size();

            //TODO: use recursive spawning and task soft affinity for faster task distribution
            //Spawn one less tasks and say TBB to reuse(recycle) current task.
            //As graph is starting and there has no task been spawned yet
            //assert_graph_is_running(root) will not hold, so spawn without assert
            parallel::batch_spawn(num_start_tasks - 1,root.get(), body, /* assert_graph_is_running*/false);
            root->spawn_and_wait_for_all(*parallel::allocate_task(root.get(), body));

            //so wait for all tasks to complete
//            root->wait_for_all();

//            std::chrono::high_resolution_clock timer;
            auto tbb_work_done   = [&root]       (){ return root->ref_count() == 1; };
            auto async_work_done = [&async_tasks](){ return 0 == async_tasks.count; };
            do {
//               //First participate in execution of TBB graph till there are no more ready tasks.
               root->wait_for_all();
               //TODO: add optimization to bypass waiting on cv if there are no async work in the graph
//
//               auto start = timer.now();
               std::unique_lock<std::mutex> lk(async_tasks.mtx);
               //then wait (probably by sleeping) until all async tasks completed or new TBB tasks created.
               async_tasks.cv.wait(lk, [&]{return async_work_done() || !tbb_work_done() ;});
//               auto end = timer.now();
//               std::cout<<"slept for "<< std::chrono::duration_cast<std::chrono::milliseconds>(end -start).count() <<" ms \n";
            }
            while(!tbb_work_done() || !async_work_done());
//
            ASSERT(tbb_work_done() && async_work_done() && "Graph is still running?");
//        }
//    );

//    std::cout<<"executed " <<executed<<" tasks"<<std::endl;
//            return executed;
}
#endif //USE_GAPI_TBB_EXECUTOR
