//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//  class Task, class ThreadPool, class TaskGroup
//
//-----------------------------------------------------------------------------

#include "IlmThread.h"
#include "IlmThreadSemaphore.h"
#include "IlmThreadPool.h"
#include "Iex.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#include <thread>

using namespace std;

ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_ENTER

#if ILMTHREAD_THREADING_ENABLED
# define ENABLE_THREADING
#endif

#if defined(__GNU_LIBRARY__) && ( __GLIBC__ < 2 || ( __GLIBC__ == 2 && __GLIBC_MINOR__ < 21 ) )
# define ENABLE_SEM_DTOR_WORKAROUND
#endif

#ifdef ENABLE_THREADING

struct TaskGroup::Data
{
     Data ();
    ~Data ();
    
    void    addTask () ;
    void    removeTask ();
    std::atomic<int> numPending;
    Semaphore        isEmpty;        // used to signal that the taskgroup is empty
#if defined(ENABLE_SEM_DTOR_WORKAROUND)
    // this mutex is also used to lock numPending in the legacy c++ mode...
    std::mutex       dtorMutex;      // used to work around the glibc bug:
                                     // http://sources.redhat.com/bugzilla/show_bug.cgi?id=12674
#endif
};


struct ThreadPool::Data
{
    typedef ThreadPoolProvider *TPPointer;

     Data ();
    ~Data();
    Data (const Data&) = delete;
    Data &operator= (const Data&)  = delete;
    Data (Data&&) = delete;
    Data &operator= (Data&&)  = delete;

    struct SafeProvider
    {
        SafeProvider (Data *d, ThreadPoolProvider *p) : _data( d ), _ptr( p )
        {
        }

        ~SafeProvider()
        {
            if ( _data )
                _data->coalesceProviderUse();
        }
        SafeProvider (const SafeProvider &o)
            : _data( o._data ), _ptr( o._ptr )
        {
            if ( _data )
                _data->bumpProviderUse();
        }
        SafeProvider &operator= (const SafeProvider &o)
        {
            if ( this != &o )
            {
                if ( o._data )
                    o._data->bumpProviderUse();
                if ( _data )
                    _data->coalesceProviderUse();
                _data = o._data;
                _ptr = o._ptr;
            }
            return *this;
        }
        SafeProvider( SafeProvider &&o )
            : _data( o._data ), _ptr( o._ptr )
        {
            o._data = nullptr;
        }
        SafeProvider &operator=( SafeProvider &&o )
        {
            std::swap( _data, o._data );
            std::swap( _ptr, o._ptr );
            return *this;
        }

        inline ThreadPoolProvider *get () const
        {
            return _ptr;
        }
        ThreadPoolProvider *operator-> () const
        {
            return get();
        }

        Data *_data;
        ThreadPoolProvider *_ptr;
    };

    // NB: In C++20, there is full support for atomic shared_ptr, but that is not
    // yet in use or finalized. Once stabilized, add appropriate usage here
    inline SafeProvider getProvider ();
    inline void coalesceProviderUse ();
    inline void bumpProviderUse ();
    inline void setProvider (ThreadPoolProvider *p);

    std::atomic<int> provUsers;
    std::atomic<ThreadPoolProvider *> provider;
};



namespace {

class DefaultWorkerThread;

struct DefaultWorkData
{
    Semaphore taskSemaphore;        // threads wait on this for ready tasks
    mutable std::mutex taskMutex;        // mutual exclusion for the tasks list
    vector<Task*> tasks;            // the list of tasks to execute

    Semaphore threadSemaphore;      // signaled when a thread starts executing
    mutable std::mutex threadMutex;      // mutual exclusion for threads list
    vector<DefaultWorkerThread*> threads;  // the list of all threads
    
    std::atomic<bool> hasThreads;
    std::atomic<bool> stopping;

    inline bool stopped () const
    {
        return stopping.load( std::memory_order_relaxed );
    }

    inline void stop ()
    {
        stopping = true;
    }
};

//
// class WorkerThread
//
class DefaultWorkerThread: public Thread
{
  public:

    DefaultWorkerThread (DefaultWorkData* data);

    virtual void    run ();
    
  private:

    DefaultWorkData *  _data;
};


DefaultWorkerThread::DefaultWorkerThread (DefaultWorkData* data):
    _data (data)
{
    start();
}


void
DefaultWorkerThread::run ()
{
    //
    // Signal that the thread has started executing
    //

    _data->threadSemaphore.post();

    while (true)
    {
        //
        // Wait for a task to become available
        //

        _data->taskSemaphore.wait();

        {
            std::unique_lock<std::mutex> taskLock (_data->taskMutex);
    
            //
            // If there is a task pending, pop off the next task in the FIFO
            //

            if (!_data->tasks.empty())
            {
                Task* task = _data->tasks.back();
                _data->tasks.pop_back();
                // release the mutex while we process
                taskLock.unlock();

                TaskGroup* taskGroup = task->group();
                task->execute();

                delete task;

                taskGroup->_data->removeTask ();
            }
            else if (_data->stopped())
            {
                break;
            }
        }
    }
}


//
// class DefaultThreadPoolProvider
//
class DefaultThreadPoolProvider : public ThreadPoolProvider
{
  public:
    DefaultThreadPoolProvider(int count);
    virtual ~DefaultThreadPoolProvider();

    virtual int numThreads() const;
    virtual void setNumThreads(int count);
    virtual void addTask(Task *task);

    virtual void finish();

  private:
    DefaultWorkData _data;
};

DefaultThreadPoolProvider::DefaultThreadPoolProvider (int count)
{
    setNumThreads(count);
}

DefaultThreadPoolProvider::~DefaultThreadPoolProvider ()
{
    finish();
}

int
DefaultThreadPoolProvider::numThreads () const
{
    std::lock_guard<std::mutex> lock (_data.threadMutex);
    return static_cast<int> (_data.threads.size());
}

void
DefaultThreadPoolProvider::setNumThreads (int count)
{
    //
    // Lock access to thread list and size
    //

    std::lock_guard<std::mutex> lock (_data.threadMutex);

    size_t desired = static_cast<size_t>(count);
    if (desired > _data.threads.size())
    {
        //
        // Add more threads
        //

        while (_data.threads.size() < desired)
            _data.threads.push_back (new DefaultWorkerThread (&_data));
    }
    else if ((size_t)count < _data.threads.size())
    {
        //
        // Wait until all existing threads are finished processing,
        // then delete all threads.
        //
        finish ();

        //
        // Add in new threads
        //

        while (_data.threads.size() < desired)
            _data.threads.push_back (new DefaultWorkerThread (&_data));
    }

    _data.hasThreads = !(_data.threads.empty());
}

void
DefaultThreadPoolProvider::addTask (Task *task)
{
    //
    // Lock the threads, needed to access numThreads
    //
    bool doPush = _data.hasThreads.load( std::memory_order_relaxed );

    if ( doPush )
    {
        //
        // Get exclusive access to the tasks queue
        //

        {
            std::lock_guard<std::mutex> taskLock (_data.taskMutex);

            //
            // Push the new task into the FIFO
            //
            _data.tasks.push_back (task);
        }
        
        //
        // Signal that we have a new task to process
        //
        _data.taskSemaphore.post ();
    }
    else
    {
        // this path shouldn't normally happen since we have the
        // NullThreadPoolProvider, but just in case...
        task->execute ();
        task->group()->_data->removeTask ();
        delete task;
    }
}

void
DefaultThreadPoolProvider::finish ()
{
    _data.stop();

    //
    // Signal enough times to allow all threads to stop.
    //
    // Wait until all threads have started their run functions.
    // If we do not wait before we destroy the threads then it's
    // possible that the threads have not yet called their run
    // functions.
    // If this happens then the run function will be called off
    // of an invalid object and we will crash, most likely with
    // an error like: "pure virtual method called"
    //

    size_t curT = _data.threads.size();
    for (size_t i = 0; i != curT; ++i)
    {
        if (_data.threads[i]->joinable())
        {
            _data.taskSemaphore.post();
            _data.threadSemaphore.wait();
        }
    }

    //
    // Join all the threads
    //
    for (size_t i = 0; i != curT; ++i)
    {
        if (_data.threads[i]->joinable())
            _data.threads[i]->join();
        delete _data.threads[i];
    }

    std::lock_guard<std::mutex> lk( _data.taskMutex );

    _data.threads.clear();
    _data.tasks.clear();

    _data.stopping = false;
}


class NullThreadPoolProvider : public ThreadPoolProvider
{
    virtual ~NullThreadPoolProvider() {}
    virtual int numThreads () const { return 0; }
    virtual void setNumThreads (int count)
    {
    }
    virtual void addTask (Task *t)
    {
        t->execute ();
        t->group()->_data->removeTask ();
        delete t;
    }
    virtual void finish () {}
}; 

} //namespace

//
// struct TaskGroup::Data
//

TaskGroup::Data::Data () : numPending (0), isEmpty (1)
{
    // empty
}


TaskGroup::Data::~Data ()
{
    //
    // A TaskGroup acts like an "inverted" semaphore: if the count
    // is above 0 then waiting on the taskgroup will block.  This
    // destructor waits until the taskgroup is empty before returning.
    //

    isEmpty.wait ();

#ifdef ENABLE_SEM_DTOR_WORKAROUND
    // Update: this was fixed in v. 2.2.21, so this ifdef checks for that
    //
    // Alas, given the current bug in glibc we need a secondary
    // syncronisation primitive here to account for the fact that
    // destructing the isEmpty Semaphore in this thread can cause
    // an error for a separate thread that is issuing the post() call.
    // We are entitled to destruct the semaphore at this point, however,
    // that post() call attempts to access data out of the associated
    // memory *after* it has woken the waiting threads, including this one,
    // potentially leading to invalid memory reads.
    // http://sources.redhat.com/bugzilla/show_bug.cgi?id=12674

    std::lock_guard<std::mutex> lock (dtorMutex);
#endif
}


void
TaskGroup::Data::addTask () 
{
    //
    // in c++11, we use an atomic to protect numPending to avoid the
    // extra lock but for c++98, to add the ability for custom thread
    // pool we add the lock here
    //
    if (numPending++ == 0)
        isEmpty.wait ();
}


void
TaskGroup::Data::removeTask ()
{
    // Alas, given the current bug in glibc we need a secondary
    // syncronisation primitive here to account for the fact that
    // destructing the isEmpty Semaphore in a separate thread can
    // cause an error. Issuing the post call here the current libc
    // implementation attempts to access memory *after* it has woken
    // waiting threads.
    // Since other threads are entitled to delete the semaphore the
    // access to the memory location can be invalid.
    // http://sources.redhat.com/bugzilla/show_bug.cgi?id=12674
    // Update: this bug has been fixed, but how do we know which
    // glibc version we're in?

    // Further update:
    //
    // we could remove this if it is a new enough glibc, however 
    // we've changed the API to enable a custom override of a
    // thread pool. In order to provide safe access to the numPending,
    // we need the lock anyway, except for c++11 or newer
    if (--numPending == 0)
    {
#ifdef ENABLE_SEM_DTOR_WORKAROUND
        std::lock_guard<std::mutex> lk (dtorMutex);
#endif
        isEmpty.post ();
    }
}
    

//
// struct ThreadPool::Data
//

ThreadPool::Data::Data ():
    provUsers (0), provider (NULL)
{
    // empty
}


ThreadPool::Data::~Data()
{
    ThreadPoolProvider *p = provider.load( std::memory_order_relaxed );
    p->finish();
    delete p;
}

inline ThreadPool::Data::SafeProvider
ThreadPool::Data::getProvider ()
{
    provUsers.fetch_add( 1, std::memory_order_relaxed );
    return SafeProvider( this, provider.load( std::memory_order_relaxed ) );
}


inline void
ThreadPool::Data::coalesceProviderUse ()
{
    int ov = provUsers.fetch_sub( 1, std::memory_order_relaxed );
    // ov is the previous value, so one means that now it might be 0
    if ( ov == 1 )
    {
        // do we have anything to do here?
    }
}


inline void
ThreadPool::Data::bumpProviderUse ()
{
    provUsers.fetch_add( 1, std::memory_order_relaxed );
}


inline void
ThreadPool::Data::setProvider (ThreadPoolProvider *p)
{
    ThreadPoolProvider *old = provider.load( std::memory_order_relaxed );
    // work around older gcc bug just in case
    do
    {
        if ( ! provider.compare_exchange_weak( old, p, std::memory_order_release, std::memory_order_relaxed ) )
            continue;
    } while (false); // NOSONAR - suppress SonarCloud bug report.

    // wait for any other users to finish prior to deleting, given
    // that these are just mostly to query the thread count or push a
    // task to the queue (so fast), just spin...
    //
    // (well, and normally, people don't do this mid stream anyway, so
    // this will be 0 99.999% of the time, but just to be safe)
    // 
    while ( provUsers.load( std::memory_order_relaxed ) > 0 )
        std::this_thread::yield();

    if ( old )
    {
        old->finish();
        delete old;
    }

    // NB: the shared_ptr mechanism is safer and means we don't have
    // to have the provUsers counter since the shared_ptr keeps that
    // for us. However, gcc 4.8/9 compilers which many people are
    // still using even though it is 2018 forgot to add the shared_ptr
    // functions... once that compiler is fully deprecated, switch to
    // using the below, change provider to a std::shared_ptr and remove
    // provUsers...
    //
//    std::shared_ptr<ThreadPoolProvider> newp( p );
//    std::shared_ptr<ThreadPoolProvider> curp = std::atomic_load_explicit( &provider, std::memory_order_relaxed );
//    do
//    {
//        if ( ! std::atomic_compare_exchange_weak_explicit( &provider, &curp, newp, std::memory_order_release, std::memory_order_relaxed ) )
//            continue;
//    } while ( false );
//    if ( curp )
//        curp->finish();
}

#endif // ENABLE_THREADING

//
// class Task
//

Task::Task (TaskGroup* g): _group(g)
{
#ifdef ENABLE_THREADING
    if ( g )
        g->_data->addTask ();
#endif
}


Task::~Task()
{
    // empty
}


TaskGroup*
Task::group ()
{
    return _group;
}


TaskGroup::TaskGroup ():
#ifdef ENABLE_THREADING
    _data (new Data())
#else
    _data (nullptr)
#endif
{
    // empty
}


TaskGroup::~TaskGroup ()
{
#ifdef ENABLE_THREADING
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


ThreadPoolProvider::ThreadPoolProvider()
{
}


ThreadPoolProvider::~ThreadPoolProvider()
{
}


//
// class ThreadPool
//

ThreadPool::ThreadPool (unsigned nthreads):
#ifdef ENABLE_THREADING
    _data (new Data)
#else
    _data (nullptr)
#endif
{
#ifdef ENABLE_THREADING
    if ( nthreads == 0 )
        _data->setProvider( new NullThreadPoolProvider );
    else
        _data->setProvider( new DefaultThreadPoolProvider( int(nthreads) ) );
#endif
}


ThreadPool::~ThreadPool ()
{
#ifdef ENABLE_THREADING
    delete _data;
#endif
}


int
ThreadPool::numThreads () const
{
#ifdef ENABLE_THREADING
    return _data->getProvider ()->numThreads ();
#else
    return 0;
#endif
}


void
ThreadPool::setNumThreads (int count)
{
#ifdef ENABLE_THREADING
    if (count < 0)
        throw IEX_INTERNAL_NAMESPACE::ArgExc ("Attempt to set the number of threads "
               "in a thread pool to a negative value.");

    bool doReset = false;
    {
        Data::SafeProvider sp = _data->getProvider ();
        int curT = sp->numThreads ();
        if ( curT == count )
            return;

        if ( curT == 0 )
        {
            NullThreadPoolProvider *npp = dynamic_cast<NullThreadPoolProvider *>( sp.get() );
            if ( npp )
                doReset = true;
        }
        else if ( count == 0 )
        {
            DefaultThreadPoolProvider *dpp = dynamic_cast<DefaultThreadPoolProvider *>( sp.get() );
            if ( dpp )
                doReset = true;
        }
        if ( ! doReset )
            sp->setNumThreads( count );
    }

    if ( doReset )
    {
        if ( count == 0 )
            _data->setProvider( new NullThreadPoolProvider );
        else
            _data->setProvider( new DefaultThreadPoolProvider( count ) );
    }
#else
    // just blindly ignore
    (void)count;
#endif
}


void
ThreadPool::setThreadProvider (ThreadPoolProvider *provider)
{
#ifdef ENABLE_THREADING
    _data->setProvider (provider);
#else
    throw IEX_INTERNAL_NAMESPACE::ArgExc (
        "Attempt to set a thread provider on a system with threads"
        " disabled / not available");
#endif
}


void
ThreadPool::addTask (Task* task) 
{
#ifdef ENABLE_THREADING
    _data->getProvider ()->addTask (task);
#else
    task->execute ();
    delete task;
#endif
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
    globalThreadPool().addTask (task);
}

unsigned
ThreadPool::estimateThreadCountForFileIO ()
{
#ifdef ENABLE_THREADING
    return std::thread::hardware_concurrency ();
#else
    return 0;
#endif
}

ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_EXIT
