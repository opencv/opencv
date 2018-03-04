// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "parallel_impl.hpp"

#ifdef HAVE_PTHREADS_PF
#include <pthread.h>

#include <opencv2/core/utils/configuration.private.hpp>

#include <opencv2/core/utils/logger.defines.hpp>
//#undef CV_LOG_STRIP_LEVEL
//#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include <opencv2/core/utils/logger.hpp>

//#define CV_PROFILE_THREADS 64
//#define getTickCount getCPUTickCount  // use this if getTickCount() calls are expensive (and getCPUTickCount() is accurate)

//#define CV_USE_GLOBAL_WORKERS_COND_VAR  // not effective on many-core systems (10+)

#ifdef CV_CXX11
#include <atomic>
#else
#include <unistd.h>  // _POSIX_PRIORITY_SCHEDULING
#endif

// Spin lock's OS-level yield
#ifdef DECLARE_CV_YIELD
DECLARE_CV_YIELD
#endif
#ifndef CV_YIELD
# ifdef CV_CXX11
#   include <thread>
#   define CV_YIELD() std::this_thread::yield()
# elif defined(_POSIX_PRIORITY_SCHEDULING)
#   include <sched.h>
#   define CV_YIELD() sched_yield()
# else
#   warning "Can't detect sched_yield() on the target platform. Specify CV_YIELD() definition via compiler flags."
#   define CV_YIELD() /* no-op: works, but not effective */
# endif
#endif // CV_YIELD

// Spin lock's CPU-level yield (required for Hyper-Threading)
#ifdef DECLARE_CV_PAUSE
DECLARE_CV_PAUSE
#endif
#ifndef CV_PAUSE
#if defined __GNUC__ && (defined __i386__ || defined __x86_64__)
#   define CV_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { _mm_pause(); } } while (0)
# elif defined __GNUC__ && defined __aarch64__
#   define CV_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("yield" ::: "memory"); } } while (0)
# elif defined __GNUC__ && defined __arm__
#   define CV_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("" ::: "memory"); } } while (0)
# elif defined __GNUC__ && defined __PPC64__
#   define CV_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("or 27,27,27" ::: "memory"); } } while (0)
# else
#   warning "Can't detect 'pause' (CPU-yield) instruction on the target platform. Specify CV_PAUSE() definition via compiler flags."
#   define CV_PAUSE(...) do { /* no-op: works, but not effective */ } while (0)
# endif
#endif // CV_PAUSE


namespace cv
{

static int CV_ACTIVE_WAIT_PAUSE_LIMIT = (int)utils::getConfigurationParameterSizeT("OPENCV_THREAD_POOL_ACTIVE_WAIT_PAUSE_LIMIT", 16);  // iterations
static int CV_WORKER_ACTIVE_WAIT = (int)utils::getConfigurationParameterSizeT("OPENCV_THREAD_POOL_ACTIVE_WAIT_WORKER", 2000);  // iterations
static int CV_MAIN_THREAD_ACTIVE_WAIT = (int)utils::getConfigurationParameterSizeT("OPENCV_THREAD_POOL_ACTIVE_WAIT_MAIN", 10000); // iterations

static int CV_WORKER_ACTIVE_WAIT_THREADS_LIMIT = (int)utils::getConfigurationParameterSizeT("OPENCV_THREAD_POOL_ACTIVE_WAIT_THREADS_LIMIT", 0); // number of real cores

class WorkerThread;
class ParallelJob;

class ThreadPool
{
public:
    static ThreadPool& instance()
    {
        CV_SINGLETON_LAZY_INIT_REF(ThreadPool, new ThreadPool())
    }

    static void stop()
    {
        ThreadPool& manager = instance();
        manager.reconfigure(0);
    }

    void reconfigure(unsigned new_threads_count)
    {
        if (new_threads_count == threads.size())
            return;
        pthread_mutex_lock(&mutex);
        reconfigure_(new_threads_count);
        pthread_mutex_unlock(&mutex);
    }
    bool reconfigure_(unsigned new_threads_count); // internal implementation

    void run(const Range& range, const ParallelLoopBody& body, double nstripes);

    size_t getNumOfThreads();

    void setNumOfThreads(unsigned n);

    ThreadPool();

    ~ThreadPool();

    unsigned num_threads;

    pthread_mutex_t mutex;  // guards fields (job/threads) from non-worker threads (concurrent parallel_for calls)
#if defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
    pthread_cond_t cond_thread_wake;
#endif

    pthread_mutex_t mutex_notify;
    pthread_cond_t cond_thread_task_complete;

    std::vector< Ptr<WorkerThread> > threads;

    Ptr<ParallelJob> job;

#ifdef CV_PROFILE_THREADS
    double tickFreq;
    int64 jobSubmitTime;
    struct ThreadStatistics
    {
        ThreadStatistics() : threadWait(0)
        {
            reset();
        }
        void reset()
        {
            threadWake = 0;
            threadExecuteStart = 0;
            threadExecuteStop = 0;
            executedTasks = 0;
            keepActive = false;
            threadPing = getTickCount();
        }
        int64 threadWait; // don't reset by default
        int64 threadPing; // don't reset by default
        int64 threadWake;
        int64 threadExecuteStart;
        int64 threadExecuteStop;
        int64 threadFree;
        unsigned executedTasks;
        bool keepActive;

        int64 dummy_[8]; // separate cache lines

        void dump(int id, int64 baseTime, double tickFreq)
        {
            if (id < 0)
                std::cout << "Main: ";
            else
                printf("T%03d: ", id + 2);
            printf("wait=% 10.1f   ping=% 6.1f",
                    threadWait > 0 ? (threadWait - baseTime) / tickFreq * 1e6 : -0.0,
                    threadPing > 0 ? (threadPing - baseTime) / tickFreq * 1e6 : -0.0);
            if (threadWake > 0)
                printf("   wake=% 6.1f",
                    (threadWake > 0 ? (threadWake - baseTime) / tickFreq * 1e6 : -0.0));
            if (threadExecuteStart > 0)
            {
                printf("   exec=% 6.1f - % 6.1f   tasksDone=%5u   free=% 6.1f",
                    (threadExecuteStart > 0 ? (threadExecuteStart - baseTime) / tickFreq * 1e6 : -0.0),
                    (threadExecuteStop > 0 ? (threadExecuteStop - baseTime) / tickFreq * 1e6 : -0.0),
                    executedTasks,
                    (threadFree > 0 ? (threadFree - baseTime) / tickFreq * 1e6 : -0.0));
                if (id >= 0)
                    printf(" active=%s\n", keepActive ? "true" : "false");
                else
                    printf("\n");
            }
            else
                printf("   ------------------------------------------------------------------------------\n");
        }
    };
    ThreadStatistics threads_stat[CV_PROFILE_THREADS]; // 0 - main thread, 1..N - worker threads
#endif

};

class WorkerThread
{
public:
    ThreadPool& thread_pool;
    const unsigned id;
    pthread_t posix_thread;
    bool is_created;

    volatile bool stop_thread;

    volatile bool has_wake_signal;

    Ptr<ParallelJob> job;

    pthread_mutex_t mutex;
#if !defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
    volatile bool isActive;
    pthread_cond_t cond_thread_wake;
#endif

    WorkerThread(ThreadPool& thread_pool_, unsigned id_) :
        thread_pool(thread_pool_),
        id(id_),
        posix_thread(0),
        is_created(false),
        stop_thread(false),
        has_wake_signal(false)
#if !defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
        , isActive(true)
#endif
    {
        CV_LOG_VERBOSE(NULL, 1, "MainThread: initializing new worker: " << id);
        int res = pthread_mutex_init(&mutex, NULL);
        if (res != 0)
        {
            CV_LOG_ERROR(NULL, id << ": Can't create thread mutex: res = " << res);
            return;
        }
#if !defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
        res = pthread_cond_init(&cond_thread_wake, NULL);
        if (res != 0)
        {
            CV_LOG_ERROR(NULL, id << ": Can't create thread condition variable: res = " << res);
            return;
        }
#endif
        res = pthread_create(&posix_thread, NULL, thread_loop_wrapper, (void*)this);
        if (res != 0)
        {
            CV_LOG_ERROR(NULL, id << ": Can't spawn new thread: res = " << res);
        }
        else
        {
            is_created = true;
        }
    }

    ~WorkerThread()
    {
        CV_LOG_VERBOSE(NULL, 1, "MainThread: destroy worker thread: " << id);
        if (is_created)
        {
            if (!stop_thread)
            {
                pthread_mutex_lock(&mutex);  // to avoid signal miss due pre-check
                stop_thread = true;
                pthread_mutex_unlock(&mutex);
#if defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
                pthread_cond_broadcast(&thread_pool.cond_thread_wake);
#else
                pthread_cond_signal(&cond_thread_wake);
#endif
            }
            pthread_join(posix_thread, NULL);
        }
#if !defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
        pthread_cond_destroy(&cond_thread_wake);
#endif
        pthread_mutex_destroy(&mutex);
    }

    void thread_body();
    static void* thread_loop_wrapper(void* thread_object)
    {
        ((WorkerThread*)thread_object)->thread_body();
        return 0;
    }
};

class ParallelJob
{
public:
    ParallelJob(const ThreadPool& thread_pool_, const Range& range_, const ParallelLoopBody& body_, int nstripes_) :
        thread_pool(thread_pool_),
        body(body_),
        range(range_),
        nstripes((unsigned)nstripes_),
        is_completed(false)
    {
        CV_LOG_VERBOSE(NULL, 5, "ParallelJob::ParallelJob(" << (void*)this << ")");
#ifdef CV_CXX11
        current_task.store(0, std::memory_order_relaxed);
        active_thread_count.store(0, std::memory_order_relaxed);
        completed_thread_count.store(0, std::memory_order_relaxed);
#else
        current_task = 0;
        active_thread_count = 0;
        completed_thread_count = 0;
#endif
        dummy0_[0] = 0, dummy1_[0] = 0, dummy2_[0] = 0; // compiler warning
    }

    ~ParallelJob()
    {
        CV_LOG_VERBOSE(NULL, 5, "ParallelJob::~ParallelJob(" << (void*)this << ")");
    }

    unsigned execute(bool is_worker_thread)
    {
        unsigned executed_tasks = 0;
        const int task_count = range.size();
        const int remaining_multiplier = std::min(nstripes,
                std::max(
                        std::min(100u, thread_pool.num_threads * 4),
                        thread_pool.num_threads * 2
                ));  // experimental value
        for (;;)
        {
            int chunk_size = std::max(1, (task_count - current_task) / remaining_multiplier);
#ifdef CV_CXX11
            int id = current_task.fetch_add(chunk_size, std::memory_order_seq_cst);
#else
            int id = (int)CV_XADD(&current_task, chunk_size);
#endif
            if (id >= task_count)
                break; // no more free tasks

            executed_tasks += chunk_size;
            int start_id = id;
            int end_id = std::min(task_count, id + chunk_size);
            CV_LOG_VERBOSE(NULL, 9, "Thread: job " << start_id << "-" << end_id);

            //TODO: if (not pending exception)
            {
                body.operator()(Range(range.start + start_id, range.start + end_id));
            }
            if (is_worker_thread && is_completed)
            {
                CV_LOG_ERROR(NULL, "\t\t\t\tBUG! Job: " << (void*)this << " " << id << " " << active_thread_count << " " << completed_thread_count);
                CV_Assert(!is_completed); // TODO Dbg this
            }
        }
        return executed_tasks;
    }

    const ThreadPool& thread_pool;
    const ParallelLoopBody& body;
    const Range range;
    const unsigned nstripes;
#ifdef CV_CXX11
    std::atomic<int> current_task;  // next free part of job
    int64 dummy0_[8];  // avoid cache-line reusing for the same atomics

    std::atomic<int> active_thread_count;  // number of threads worked on this job
    int64 dummy1_[8];  // avoid cache-line reusing for the same atomics

    std::atomic<int> completed_thread_count;  // number of threads completed any activities on this job
    int64 dummy2_[8];  // avoid cache-line reusing for the same atomics
#else
    /*CV_DECL_ALIGNED(64)*/ volatile int current_task;  // next free part of job
    int64 dummy0_[8];  // avoid cache-line reusing for the same atomics

    /*CV_DECL_ALIGNED(64)*/ volatile int active_thread_count;  // number of threads worked on this job
    int64 dummy1_[8];  // avoid cache-line reusing for the same atomics

    /*CV_DECL_ALIGNED(64)*/ volatile int completed_thread_count;  // number of threads completed any activities on this job
    int64 dummy2_[8];  // avoid cache-line reusing for the same atomics
#endif

    volatile bool is_completed;  // std::atomic_flag ?

    // TODO exception handling
};


void WorkerThread::thread_body()
{
    (void)cv::utils::getThreadID(); // notify OpenCV about new thread
    CV_LOG_VERBOSE(NULL, 5, "Thread: new thread: " << id);

    bool allow_active_wait = true;

#ifdef CV_PROFILE_THREADS
    ThreadPool::ThreadStatistics& stat = thread_pool.threads_stat[id + 1];
#endif

    while (!stop_thread)
    {
        CV_LOG_VERBOSE(NULL, 5, "Thread: ... loop iteration: allow_active_wait=" << allow_active_wait << "   has_wake_signal=" << has_wake_signal);
        if (allow_active_wait && CV_WORKER_ACTIVE_WAIT > 0)
        {
            allow_active_wait = false;
            for (int i = 0; i < CV_WORKER_ACTIVE_WAIT; i++)
            {
                if (has_wake_signal)
                    break;
                if (CV_ACTIVE_WAIT_PAUSE_LIMIT > 0 && (i < CV_ACTIVE_WAIT_PAUSE_LIMIT || (i & 1)))
                    CV_PAUSE(16);
                else
                    CV_YIELD();
            }
        }
        pthread_mutex_lock(&mutex);
#ifdef CV_PROFILE_THREADS
        stat.threadWait = getTickCount();
#endif
        while (!has_wake_signal) // to handle spurious wakeups
        {
            //CV_LOG_VERBOSE(NULL, 5, "Thread: wait (sleep) ...");
#if defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
            pthread_cond_wait(&thread_pool.cond_thread_wake, &mutex);
#else
            isActive = false;
            pthread_cond_wait(&cond_thread_wake, &mutex);
            isActive = true;
#endif
            CV_LOG_VERBOSE(NULL, 5, "Thread: wake ... (has_wake_signal=" << has_wake_signal << " stop_thread=" << stop_thread << ")")
        }
#ifdef CV_PROFILE_THREADS
        stat.threadWake = getTickCount();
#endif

        CV_LOG_VERBOSE(NULL, 5, "Thread: checking for new job");
        if (CV_WORKER_ACTIVE_WAIT_THREADS_LIMIT == 0)
            allow_active_wait = true;
        Ptr<ParallelJob> j_ptr; swap(j_ptr, job);
        has_wake_signal = false;    // TODO .store(false, std::memory_order_release)
        pthread_mutex_unlock(&mutex);

        if (!stop_thread)
        {
            ParallelJob* j = j_ptr;
            if (j)
            {
                CV_LOG_VERBOSE(NULL, 5, "Thread: job size=" << j->range.size() << " done=" << j->current_task);
                if (j->current_task < j->range.size())
                {
#ifdef CV_CXX11
                    int other = j->active_thread_count.fetch_add(1, std::memory_order_seq_cst);
#else
                    int other = CV_XADD(&j->active_thread_count, 1);
#endif
                    CV_LOG_VERBOSE(NULL, 5, "Thread: processing new job (with " << other << " other threads)"); CV_UNUSED(other);
#ifdef CV_PROFILE_THREADS
                    stat.threadExecuteStart = getTickCount();
                    stat.executedTasks = j->execute(true);
                    stat.threadExecuteStop = getTickCount();
#else
                    j->execute(true);
#endif
#ifdef CV_CXX11
                    int completed = j->completed_thread_count.fetch_add(1, std::memory_order_seq_cst) + 1;
                    int active = j->active_thread_count.load(std::memory_order_acquire);
#else
                    int completed = (int)CV_XADD(&j->completed_thread_count, 1) + 1;
                    int active = j->active_thread_count;
#endif
                    if (CV_WORKER_ACTIVE_WAIT_THREADS_LIMIT > 0)
                    {
                        allow_active_wait = true;
                        if (active >= CV_WORKER_ACTIVE_WAIT_THREADS_LIMIT && (id & 1) == 0) // turn off a half of threads
                            allow_active_wait = false;
                    }
                    CV_LOG_VERBOSE(NULL, 5, "Thread: completed job processing: " << active << " " << completed);
                    if (active == completed)
                    {
                        bool need_signal = !j->is_completed;
                        j->is_completed = true;
                        j = NULL; j_ptr.release();
                        if (need_signal)
                        {
                            CV_LOG_VERBOSE(NULL, 5, "Thread: job finished => notifying the main thread");
                            pthread_mutex_lock(&thread_pool.mutex_notify);  // to avoid signal miss due pre-check condition
                            // empty
                            pthread_mutex_unlock(&thread_pool.mutex_notify);
                            pthread_cond_broadcast/*pthread_cond_signal*/(&thread_pool.cond_thread_task_complete);
                        }
                    }
                }
                else
                {
                    CV_LOG_VERBOSE(NULL, 5, "Thread: no free job tasks");
                }
            }
        }
#ifdef CV_PROFILE_THREADS
        stat.threadFree = getTickCount();
        stat.keepActive = allow_active_wait;
#endif
    }
}

ThreadPool::ThreadPool()
{
#ifdef CV_PROFILE_THREADS
    tickFreq = getTickFrequency();
#endif

    int res = 0;
    res |= pthread_mutex_init(&mutex, NULL);
    res |= pthread_mutex_init(&mutex_notify, NULL);
#if defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
    res |= pthread_cond_init(&cond_thread_wake, NULL);
#endif
    res |= pthread_cond_init(&cond_thread_task_complete, NULL);

    if (0 != res)
    {
        CV_LOG_FATAL(NULL, "Failed to initialize ThreadPool (pthreads)");
    }
    num_threads = defaultNumberOfThreads();
}

bool ThreadPool::reconfigure_(unsigned new_threads_count)
{
    if (new_threads_count == threads.size())
        return false;

    if (new_threads_count < threads.size())
    {
        CV_LOG_VERBOSE(NULL, 1, "MainThread: reduce worker pool: " << threads.size() << " => " << new_threads_count);
        std::vector< Ptr<WorkerThread> > release_threads(threads.size() - new_threads_count);
        for (size_t i = new_threads_count; i < threads.size(); ++i)
        {
            pthread_mutex_lock(&threads[i]->mutex);  // to avoid signal miss due pre-check
            threads[i]->stop_thread = true;
            threads[i]->has_wake_signal = true;
#if !defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
            pthread_mutex_unlock(&threads[i]->mutex);
            pthread_cond_broadcast/*pthread_cond_signal*/(&threads[i]->cond_thread_wake); // wake thread
#else
            pthread_mutex_unlock(&threads[i]->mutex);
#endif
            std::swap(threads[i], release_threads[i - new_threads_count]);
        }
#if defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
        CV_LOG_VERBOSE(NULL, 1, "MainThread: notify worker threads about termination...");
        pthread_cond_broadcast(&cond_thread_wake); // wake all threads
#endif
        threads.resize(new_threads_count);
        release_threads.clear();  // calls thread_join which want to lock mutex
        return false;
    }
    else
    {
        CV_LOG_VERBOSE(NULL, 1, "MainThread: upgrade worker pool: " << threads.size() << " => " << new_threads_count);
        for (size_t i = threads.size(); i < new_threads_count; ++i)
        {
            threads.push_back(Ptr<WorkerThread>(new WorkerThread(*this, (unsigned)i))); // spawn more threads
        }
    }
    return false;
}

ThreadPool::~ThreadPool()
{
    reconfigure(0);
    pthread_cond_destroy(&cond_thread_task_complete);
#if defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
    pthread_cond_destroy(&cond_thread_wake);
#endif
    pthread_mutex_destroy(&mutex);
    pthread_mutex_destroy(&mutex_notify);
}

void ThreadPool::run(const Range& range, const ParallelLoopBody& body, double nstripes)
{
    CV_LOG_VERBOSE(NULL, 1, "MainThread: new parallel job: num_threads=" << num_threads << "   range=" << range.size() << "   nstripes=" << nstripes << "   job=" << (void*)job);
#ifdef CV_PROFILE_THREADS
    jobSubmitTime = getTickCount();
    threads_stat[0].reset();
    threads_stat[0].threadWait = jobSubmitTime;
    threads_stat[0].threadWake = jobSubmitTime;
#endif
    if (getNumOfThreads() > 1 &&
        job == NULL &&
        (range.size() * nstripes >= 2 || (range.size() > 1 && nstripes <= 0))
    )
    {
        pthread_mutex_lock(&mutex);
        if (job != NULL)
        {
            pthread_mutex_unlock(&mutex);
            body(range);
            return;
        }
        reconfigure_(num_threads - 1);

        {
            CV_LOG_VERBOSE(NULL, 1, "MainThread: initialize parallel job: " << range.size());
            job = Ptr<ParallelJob>(new ParallelJob(*this, range, body, nstripes));
            pthread_mutex_unlock(&mutex);

            CV_LOG_VERBOSE(NULL, 5, "MainThread: wake worker threads...");
            for (size_t i = 0; i < threads.size(); ++i)
            {
                WorkerThread& thread = *(threads[i].get());
                if (
#if !defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
                        thread.isActive ||
#endif
                        thread.has_wake_signal
                        || !thread.job.empty()  // #10881
                )
                {
                    pthread_mutex_lock(&thread.mutex);
                    thread.job = job;
#if !defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
                    bool isActive = thread.isActive;
#endif
                    thread.has_wake_signal = true;
#ifdef CV_PROFILE_THREADS
                    threads_stat[i + 1].reset();
#endif
                    pthread_mutex_unlock(&thread.mutex);
#if !defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
                    if (!isActive)
                    {
                        pthread_cond_broadcast/*pthread_cond_signal*/(&thread.cond_thread_wake); // wake thread
                    }
#endif
                }
                else
                {
                    CV_Assert(thread.job.empty());
                    thread.job = job;
                    thread.has_wake_signal = true;
#ifdef CV_PROFILE_THREADS
                    threads_stat[i + 1].reset();
#endif
#if !defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
                    pthread_cond_broadcast/*pthread_cond_signal*/(&thread.cond_thread_wake); // wake thread
#endif
                }
            }
#ifdef CV_PROFILE_THREADS
            threads_stat[0].threadPing = getTickCount();
#endif
#if defined(CV_USE_GLOBAL_WORKERS_COND_VAR)
            pthread_cond_broadcast(&cond_thread_wake); // wake all threads
#endif
#ifdef CV_PROFILE_THREADS
            threads_stat[0].threadWake = getTickCount();
#endif
            CV_LOG_VERBOSE(NULL, 5, "MainThread: wake worker threads... (done)");

            {
                ParallelJob& j = *(this->job);
#ifdef CV_PROFILE_THREADS
                threads_stat[0].threadExecuteStart = getTickCount();
                threads_stat[0].executedTasks = j.execute(false);
                threads_stat[0].threadExecuteStop = getTickCount();
#else
                j.execute(false);
#endif
                CV_Assert(j.current_task >= j.range.size());
                CV_LOG_VERBOSE(NULL, 5, "MainThread: complete self-tasks: " << j.active_thread_count << " " << j.completed_thread_count);
                if (job->is_completed || j.active_thread_count == 0)
                {
                    job->is_completed = true;
                    CV_LOG_VERBOSE(NULL, 5, "MainThread: no WIP worker threads");
                }
                else
                {
                    if (CV_MAIN_THREAD_ACTIVE_WAIT > 0)
                    {
                        for (int i = 0; i < CV_MAIN_THREAD_ACTIVE_WAIT; i++)  // don't spin too much in any case (inaccurate getTickCount())
                        {
                            if (job->is_completed)
                            {
                                CV_LOG_VERBOSE(NULL, 5, "MainThread: job finalize (active wait) " << j.active_thread_count << " " << j.completed_thread_count);
                                break;
                            }
                            if (CV_ACTIVE_WAIT_PAUSE_LIMIT > 0 && (i < CV_ACTIVE_WAIT_PAUSE_LIMIT || (i & 1)))
                                CV_PAUSE(16);
                            else
                                CV_YIELD();
                        }
                    }
                    if (!job->is_completed)
                    {
                        CV_LOG_VERBOSE(NULL, 5, "MainThread: prepare wait " << j.active_thread_count << " " << j.completed_thread_count);
                        pthread_mutex_lock(&mutex_notify);
                        for (;;)
                        {
                            if (job->is_completed)
                            {
                                CV_LOG_VERBOSE(NULL, 5, "MainThread: job finalize (wait) " << j.active_thread_count << " " << j.completed_thread_count);
                                break;
                            }
                            CV_LOG_VERBOSE(NULL, 5, "MainThread: wait completion (sleep) ...");
                            pthread_cond_wait(&cond_thread_task_complete, &mutex_notify);
                            CV_LOG_VERBOSE(NULL, 5, "MainThread: wake");
                        }
                        pthread_mutex_unlock(&mutex_notify);
                    }
                }
            }
#ifdef CV_PROFILE_THREADS
            threads_stat[0].threadFree = getTickCount();
            std::cout << "Job: sz=" << range.size() << " nstripes=" << nstripes << "    Time: " << (threads_stat[0].threadFree - jobSubmitTime) / tickFreq * 1e6 << " usec" << std::endl;
            for (int i = 0; i < (int)threads.size() + 1; i++)
            {
                threads_stat[i].dump(i - 1, jobSubmitTime, tickFreq);
            }
#endif
            if (job)
            {
                pthread_mutex_lock(&mutex);
                CV_LOG_VERBOSE(NULL, 5, "MainThread: job release");
                CV_Assert(job->is_completed);
                job.release();
                pthread_mutex_unlock(&mutex);
            }
        }
    }
    else
    {
        body(range);
    }
}

size_t ThreadPool::getNumOfThreads()
{
    return num_threads;
}

void ThreadPool::setNumOfThreads(unsigned n)
{
    if (n != num_threads)
    {
        num_threads = n;
        if (n == 1)
           if (job == NULL) reconfigure(0);  // stop worker threads immediately
    }
}

size_t parallel_pthreads_get_threads_num()
{
    return ThreadPool::instance().getNumOfThreads();
}

void parallel_pthreads_set_threads_num(int num)
{
    if(num < 0)
    {
        ThreadPool::instance().setNumOfThreads(0);
    }
    else
    {
        ThreadPool::instance().setNumOfThreads(unsigned(num));
    }
}

void parallel_for_pthreads(const Range& range, const ParallelLoopBody& body, double nstripes)
{
    ThreadPool::instance().run(range, body, nstripes);
}

}

#endif
