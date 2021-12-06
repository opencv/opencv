// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_PARALLEL_FOR_TBB_HPP
#define OPENCV_CORE_PARALLEL_FOR_TBB_HPP

#include "opencv2/core/parallel/parallel_backend.hpp"
#include <opencv2/core/utils/logger.hpp>

#ifndef TBB_SUPPRESS_DEPRECATED_MESSAGES  // supress warning
#define TBB_SUPPRESS_DEPRECATED_MESSAGES 1
#endif
#include "tbb/tbb.h"
#if !defined(TBB_INTERFACE_VERSION)
#error "Unknows/unsupported TBB version"
#endif

#if TBB_INTERFACE_VERSION >= 8000
#include "tbb/task_arena.h"
#endif

namespace cv { namespace parallel { namespace tbb {

using namespace ::tbb;

#if TBB_INTERFACE_VERSION >= 8000
static tbb::task_arena& getArena()
{
    static tbb::task_arena tbbArena(tbb::task_arena::automatic);
    return tbbArena;
}
#else
static tbb::task_scheduler_init& getScheduler()
{
    static tbb::task_scheduler_init tbbScheduler(tbb::task_scheduler_init::deferred);
    return tbbScheduler;
}
#endif

/** TBB parallel_for API implementation
 *
 * @sa setParallelForBackend
 * @ingroup core_parallel_backend
 */
class ParallelForBackend : public ParallelForAPI
{
protected:
    int numThreads;
    int numThreadsMax;
public:
    ParallelForBackend()
    {
        CV_LOG_INFO(NULL, "Initializing TBB parallel backend: TBB_INTERFACE_VERSION=" << TBB_INTERFACE_VERSION);
        numThreads = 0;
#if TBB_INTERFACE_VERSION >= 8000
        (void)getArena();
#else
        (void)getScheduler();
#endif
    }

    virtual ~ParallelForBackend() {}

    class CallbackProxy
    {
        const FN_parallel_for_body_cb_t& callback;
        void* const callback_data;
        const int tasks;
    public:
        inline CallbackProxy(int tasks_, FN_parallel_for_body_cb_t& callback_, void* callback_data_)
            : callback(callback_), callback_data(callback_data_), tasks(tasks_)
        {
            // nothing
        }

        void operator()(const tbb::blocked_range<int>& range) const
        {
            this->callback(range.begin(), range.end(), callback_data);
        }

        void operator()() const
        {
            tbb::parallel_for(tbb::blocked_range<int>(0, tasks), *this);
        }
    };

    virtual void parallel_for(int tasks, FN_parallel_for_body_cb_t body_callback, void* callback_data) CV_OVERRIDE
    {
        CallbackProxy task(tasks, body_callback, callback_data);
#if TBB_INTERFACE_VERSION >= 8000
        getArena().execute(task);
#else
        task();
#endif
    }

    virtual int getThreadNum() const CV_OVERRIDE
    {
#if TBB_INTERFACE_VERSION >= 9100
        return tbb::this_task_arena::current_thread_index();
#elif TBB_INTERFACE_VERSION >= 8000
        return tbb::task_arena::current_thread_index();
#else
        return 0;
#endif
    }

    virtual int getNumThreads() const CV_OVERRIDE
    {
#if TBB_INTERFACE_VERSION >= 9100
    return getArena().max_concurrency();
#elif TBB_INTERFACE_VERSION >= 8000
    return numThreads > 0
        ? numThreads
        : tbb::task_scheduler_init::default_num_threads();
#else
    return getScheduler().is_active()
           ? numThreads
           : tbb::task_scheduler_init::default_num_threads();
#endif
    }

    virtual int setNumThreads(int nThreads) CV_OVERRIDE
    {
        int oldNumThreads = numThreads;
        numThreads = nThreads;

#if TBB_INTERFACE_VERSION >= 8000
        auto& tbbArena = getArena();
        if (tbbArena.is_active())
            tbbArena.terminate();
        if (numThreads > 0)
            tbbArena.initialize(numThreads);
#else
        auto& tbbScheduler = getScheduler();
        if (tbbScheduler.is_active())
            tbbScheduler.terminate();
        if (numThreads > 0)
            tbbScheduler.initialize(numThreads);
#endif
        return oldNumThreads;
    }

    const char* getName() const CV_OVERRIDE
    {
        return "tbb";
    }
};

}}}  // namespace

#endif  // OPENCV_CORE_PARALLEL_FOR_TBB_HPP
