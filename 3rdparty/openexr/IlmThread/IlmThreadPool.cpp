//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//  class Task, class ThreadPool, class TaskGroup
//
//-----------------------------------------------------------------------------

#include "IlmThreadPool.h"
#include "Iex.h"
#include "IlmThread.h"
#include "IlmThreadSemaphore.h"

#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if (defined(_WIN32) || defined(_WIN64))
#    include <windows.h>
#else
#    include <unistd.h>
#endif

ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_ENTER

#if ILMTHREAD_THREADING_ENABLED
#    define ENABLE_THREADING
#endif

namespace
{

static inline void
handleProcessTask (Task* task)
{
    if (task)
    {
        TaskGroup* taskGroup = task->group ();

        task->execute ();

        // kill the task prior to notifying the group
        // such that any internal reference-based
        // semantics will be handled prior to
        // the task group destructor letting it out
        // of the scope of those references
        delete task;

        if (taskGroup) taskGroup->finishOneTask ();
    }
}

struct DefaultThreadPoolData
{
    Semaphore          _taskSemaphore; // threads wait on this for ready tasks
    mutable std::mutex _taskMutex;     // mutual exclusion for the tasks list
    std::vector<Task*> _tasks;         // the list of tasks to execute

    mutable std::mutex       _threadMutex; // mutual exclusion for threads list
    std::vector<std::thread> _threads;     // the list of all threads

    std::atomic<int>  _threadCount;
    std::atomic<bool> _stopping;

    inline bool stopped () const
    {
        return _stopping.load (std::memory_order_relaxed);
    }

    inline void stop () { _stopping = true; }

    inline void resetAtomics ()
    {
        _threadCount = 0;
        _stopping    = false;
    }
};

} // namespace

#ifdef ENABLE_THREADING

struct TaskGroup::Data
{
    Data ();
    ~Data ();
    Data (const Data&)            = delete;
    Data& operator= (const Data&) = delete;
    Data (Data&&)                 = delete;
    Data& operator= (Data&&)      = delete;

    void addTask ();
    void removeTask ();

    void waitForEmpty ();

    std::atomic<int> numPending;
    std::atomic<int> inFlight;
    Semaphore        isEmpty; // used to signal that the taskgroup is empty
};

struct ThreadPool::Data
{
    using ProviderPtr = std::shared_ptr<ThreadPoolProvider>;

    Data ();
    ~Data ();
    Data (const Data&)            = delete;
    Data& operator= (const Data&) = delete;
    Data (Data&&)                 = delete;
    Data& operator= (Data&&)      = delete;

    ProviderPtr getProvider () const { return std::atomic_load (&_provider); }

    void setProvider (ProviderPtr provider)
    {
        ProviderPtr curp = std::atomic_exchange (&_provider, provider);
        if (curp && curp != provider) curp->finish ();
    }

    std::shared_ptr<ThreadPoolProvider> _provider;
};

namespace
{

//
// class DefaultThreadPoolProvider
//
class DefaultThreadPoolProvider : public ThreadPoolProvider
{
public:
    DefaultThreadPoolProvider (int count);
    DefaultThreadPoolProvider (const DefaultThreadPoolProvider&) = delete;
    DefaultThreadPoolProvider&
    operator= (const DefaultThreadPoolProvider&)                       = delete;
    DefaultThreadPoolProvider (DefaultThreadPoolProvider&&)            = delete;
    DefaultThreadPoolProvider& operator= (DefaultThreadPoolProvider&&) = delete;
    ~DefaultThreadPoolProvider () override;

    int  numThreads () const override;
    void setNumThreads (int count) override;
    void addTask (Task* task) override;

    void finish () override;

private:
    void lockedFinish ();
    void threadLoop (std::shared_ptr<DefaultThreadPoolData> d);

    std::shared_ptr<DefaultThreadPoolData> _data;
};

DefaultThreadPoolProvider::DefaultThreadPoolProvider (int count)
    : _data (std::make_shared<DefaultThreadPoolData> ())
{
    _data->resetAtomics ();
    setNumThreads (count);
}

DefaultThreadPoolProvider::~DefaultThreadPoolProvider ()
{}

int
DefaultThreadPoolProvider::numThreads () const
{
    return _data->_threadCount.load ();
}

void
DefaultThreadPoolProvider::setNumThreads (int count)
{
    // since we're a private class, the thread pool won't call us if
    // we aren't changing size so no need to check that...

    std::lock_guard<std::mutex> lock (_data->_threadMutex);

    size_t curThreads = _data->_threads.size ();
    size_t nToAdd     = static_cast<size_t> (count);

    if (nToAdd < curThreads)
    {
        // no easy way to only shutdown the n threads at the end of
        // the vector (well, really, guaranteeing they are the ones to
        // be woken up), so just kill all of the threads
        lockedFinish ();
        curThreads = 0;
    }

    _data->_threads.resize (nToAdd);
    for (size_t i = curThreads; i < nToAdd; ++i)
    {
        _data->_threads[i] =
            std::thread (&DefaultThreadPoolProvider::threadLoop, this, _data);
    }
    _data->_threadCount = static_cast<int> (_data->_threads.size ());
}

void
DefaultThreadPoolProvider::addTask (Task* task)
{
    // the thread pool will kill us and switch to a null provider
    // if the thread count is set to 0, so we can always
    // go ahead and lock and assume we have a thread to do the
    // processing
    {
        std::lock_guard<std::mutex> taskLock (_data->_taskMutex);

        //
        // Push the new task into the FIFO
        //
        _data->_tasks.push_back (task);
    }

    //
    // Signal that we have a new task to process
    //
    _data->_taskSemaphore.post ();
}

void
DefaultThreadPoolProvider::finish ()
{
    std::lock_guard<std::mutex> lock (_data->_threadMutex);

    lockedFinish ();
}

void
DefaultThreadPoolProvider::lockedFinish ()
{
    _data->stop ();

    //
    // Signal enough times to allow all threads to stop.
    //
    // NB: we must do this as many times as we have threads.
    //
    // If there is still work in the queue, or this call happens "too
    // quickly", threads will not be waiting on the semaphore, so we
    // need to ensure the semaphore is at a count equal to the amount
    // of work left plus the number of threads to ensure exit of a
    // thread. There can be threads in a few states:
    //   - still starting up (successive calls to setNumThreads)
    //   - in the middle of processing a task / looping
    //   - waiting in the semaphore
    size_t curT = _data->_threads.size ();
    for (size_t i = 0; i != curT; ++i)
        _data->_taskSemaphore.post ();

    //
    // We should not need to check joinability, they should all, by
    // definition, be joinable (assuming normal start)
    //
    for (size_t i = 0; i != curT; ++i)
    {
        // This isn't quite right in that the thread may have actually
        // be in an exited / signalled state (needing the
        // WaitForSingleObject call), and so already have an exit code
        // (I think, but the docs are vague), but if we don't do the
        // join, the stl thread seems to then throw an exception. The
        // join should just return invalid handle and continue, and is
        // more of a windows bug... except maybe someone needs to work
        // around it...
        //#    ifdef TEST_FOR_WIN_THREAD_STATUS
        //
        //        // per OIIO issue #2038, on exit / dll unload, windows may
        //        // kill the thread, double check that it is still active prior
        //        // to joining.
        //        DWORD tstatus;
        //        if (GetExitCodeThread (_threads[i].native_handle (), &tstatus))
        //        {
        //            if (tstatus != STILL_ACTIVE) { continue; }
        //        }
        //#    endif

        _data->_threads[i].join ();
    }

    _data->_threads.clear ();

    _data->resetAtomics ();
}

void
DefaultThreadPoolProvider::threadLoop (
    std::shared_ptr<DefaultThreadPoolData> data)
{
    while (true)
    {
        //
        // Wait for a task to become available
        //

        data->_taskSemaphore.wait ();

        {
            std::unique_lock<std::mutex> taskLock (data->_taskMutex);

            //
            // If there is a task pending, pop off the next task in the FIFO
            //

            if (!data->_tasks.empty ())
            {
                Task* task = data->_tasks.back ();
                data->_tasks.pop_back ();

                // release the mutex while we process
                taskLock.unlock ();

                handleProcessTask (task);

                // do not need to reacquire the lock at all since we
                // will just loop around, pull any other task
            }
            else if (data->stopped ()) { break; }
        }
    }
}

} //namespace

//
// struct TaskGroup::Data
//

TaskGroup::Data::Data () : numPending (0), inFlight (0), isEmpty (1)
{}

TaskGroup::Data::~Data ()
{}

void
TaskGroup::Data::waitForEmpty ()
{
    //
    // A TaskGroup acts like an "inverted" semaphore: if the count
    // is above 0 then waiting on the taskgroup will block.  The
    // destructor waits until the taskgroup is empty before returning.
    //

    isEmpty.wait ();

    // pseudo spin to wait for the notifying thread to finish the post
    // to avoid a premature deletion of the semaphore
    int count = 0;
    while (inFlight.load () > 0)
    {
        ++count;
        if (count > 100)
        {
            std::this_thread::yield ();
            count = 0;
        }
    }
}

void
TaskGroup::Data::addTask ()
{
    inFlight.fetch_add (1);

    // if we are the first task off the rank, clear the
    // isEmpty semaphore such that the group will actually pause
    // until the task finishes
    if (numPending.fetch_add (1) == 0) { isEmpty.wait (); }
}

void
TaskGroup::Data::removeTask ()
{
    // if we are the last task, notify the group we're done
    if (numPending.fetch_sub (1) == 1) { isEmpty.post (); }

    // in theory, a background thread could actually finish a task
    // prior to the next task being added. The fetch_add / fetch_sub
    // logic between addTask and removeTask are fine to keep the
    // inverted semaphore straight. All addTask must happen prior to
    // the TaskGroup destructor.
    //
    // But to let the taskgroup thread waiting know we're actually
    // finished with the last one and finished posting (the semaphore
    // might wake up the other thread while in the middle of post) so
    // we don't destroy the semaphore while posting to it, keep a
    // separate counter that is modified pre / post semaphore
    inFlight.fetch_sub (1);
}

//
// struct ThreadPool::Data
//

ThreadPool::Data::Data ()
{
    // empty
}

ThreadPool::Data::~Data ()
{
    setProvider (nullptr);
}

#endif // ENABLE_THREADING

//
// class Task
//

Task::Task (TaskGroup* g) : _group (g)
{
#ifdef ENABLE_THREADING
    if (g) g->_data->addTask ();
#endif
}

Task::~Task ()
{
    // empty
}

TaskGroup*
Task::group ()
{
    return _group;
}

TaskGroup::TaskGroup ()
    :
#ifdef ENABLE_THREADING
    _data (new Data)
#else
    _data (nullptr)
#endif
{
    // empty
}

TaskGroup::~TaskGroup ()
{
#ifdef ENABLE_THREADING
    _data->waitForEmpty ();
    delete _data;
#endif
}

void
TaskGroup::finishOneTask ()
{
#ifdef ENABLE_THREADING
    _data->removeTask ();
#endif
}

//
// class ThreadPoolProvider
//

ThreadPoolProvider::ThreadPoolProvider ()
{}

ThreadPoolProvider::~ThreadPoolProvider ()
{}

//
// class ThreadPool
//

ThreadPool::ThreadPool (unsigned nthreads)
    :
#ifdef ENABLE_THREADING
    _data (new Data)
#else
    _data (nullptr)
#endif
{
#ifdef ENABLE_THREADING
    setNumThreads (static_cast<int> (nthreads));
#endif
}

ThreadPool::~ThreadPool ()
{
#ifdef ENABLE_THREADING
    // ensures any jobs / threads are finished & shutdown
    _data->setProvider (nullptr);
    delete _data;
#endif
}

int
ThreadPool::numThreads () const
{
#ifdef ENABLE_THREADING
    Data::ProviderPtr sp = _data->getProvider ();
    return (sp) ? sp->numThreads () : 0;
#else
    return 0;
#endif
}

void
ThreadPool::setNumThreads (int count)
{
#ifdef ENABLE_THREADING
    if (count < 0)
        throw IEX_INTERNAL_NAMESPACE::ArgExc (
            "Attempt to set the number of threads "
            "in a thread pool to a negative value.");

    {
        Data::ProviderPtr sp = _data->getProvider ();
        if (sp)
        {
            int  curT    = sp->numThreads ();
            if (curT == count) return;

            if (count != 0)
            {
                sp->setNumThreads (count);
                return;
            }
        }
    }

    // either a null provider or a case where we should switch from
    // a default provider to a null one or vice-versa
    if (count == 0)
        _data->setProvider (nullptr);
    else
        _data->setProvider (
            std::make_shared<DefaultThreadPoolProvider> (count));

#else
    // just blindly ignore
    (void) count;
#endif
}

void
ThreadPool::setThreadProvider (ThreadPoolProvider* provider)
{
#ifdef ENABLE_THREADING
    // contract is we take ownership and will free the provider
    _data->setProvider (Data::ProviderPtr (provider));
#else
    throw IEX_INTERNAL_NAMESPACE::ArgExc (
        "Attempt to set a thread provider on a system with threads"
        " disabled / not available");
#endif
}

void
ThreadPool::addTask (Task* task)
{
    if (task)
    {
#ifdef ENABLE_THREADING
        Data::ProviderPtr p = _data->getProvider ();
        if (p)
        {
            p->addTask (task);
            return;
        }
#endif

        handleProcessTask (task);
    }
}

ThreadPool&
ThreadPool::globalThreadPool ()
{
    //
    // The global thread pool
    //

    static ThreadPool gThreadPool (0);

    return gThreadPool;
}

void
ThreadPool::addGlobalTask (Task* task)
{
    globalThreadPool ().addTask (task);
}

unsigned
ThreadPool::estimateThreadCountForFileIO ()
{
#ifdef ENABLE_THREADING
    unsigned rv = std::thread::hardware_concurrency ();
    // hardware concurrency is not required to work
    if (rv == 0 ||
        rv > static_cast<unsigned> (std::numeric_limits<int>::max ()))
    {
        rv = 1;
#    if (defined(_WIN32) || defined(_WIN64))
        SYSTEM_INFO si;
        GetNativeSystemInfo (&si);

        rv = si.dwNumberOfProcessors;
#    else
        // linux, bsd, and mac are fine with this
        // other *nix should be too, right?
        rv = sysconf (_SC_NPROCESSORS_ONLN);
#    endif
    }
    return rv;
#else
    return 0;
#endif
}

ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_EXIT
