///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2005-2012, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
//
//  class Task, class ThreadPool, class TaskGroup
//
//-----------------------------------------------------------------------------

#include "IlmThread.h"
#include "IlmThreadMutex.h"
#include "IlmThreadSemaphore.h"
#include "IlmThreadPool.h"
#include "Iex.h"
#include <vector>
#ifndef ILMBASE_FORCE_CXX03
# include <memory>
# include <atomic>
# include <thread>
#endif

using namespace std;

ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_ENTER

#if defined(__GNU_LIBRARY__) && ( __GLIBC__ < 2 || ( __GLIBC__ == 2 && __GLIBC_MINOR__ < 21 ) )
# define ENABLE_SEM_DTOR_WORKAROUND
#endif

struct TaskGroup::Data
{
     Data ();
    ~Data ();
    
    void    addTask () ;
    void    removeTask ();
#ifndef ILMBASE_FORCE_CXX03
    std::atomic<int> numPending;
#else
    int              numPending;     // number of pending tasks to still execute
#endif
    Semaphore        isEmpty;        // used to signal that the taskgroup is empty
#if defined(ENABLE_SEM_DTOR_WORKAROUND) || defined(ILMBASE_FORCE_CXX03)
    // this mutex is also used to lock numPending in the legacy c++ mode...
    Mutex            dtorMutex;      // used to work around the glibc bug:
                                     // http://sources.redhat.com/bugzilla/show_bug.cgi?id=12674
#endif
};


struct ThreadPool::Data
{
    typedef ThreadPoolProvider *TPPointer;

     Data ();
    ~Data();

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
#ifndef ILMBASE_FORCE_CXX03
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
#endif
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

#ifdef ILMBASE_FORCE_CXX03
    Semaphore provSem;
    Mutex provMutex;
    int provUsers;
    ThreadPoolProvider *provider;
    ThreadPoolProvider *oldprovider;
#else
    std::atomic<ThreadPoolProvider *> provider;
    std::atomic<int> provUsers;
#endif
};



namespace {

class DefaultWorkerThread;

struct DefaultWorkData
{
    Semaphore taskSemaphore;        // threads wait on this for ready tasks
    mutable Mutex taskMutex;        // mutual exclusion for the tasks list
    vector<Task*> tasks;            // the list of tasks to execute

    Semaphore threadSemaphore;      // signaled when a thread starts executing
    mutable Mutex threadMutex;      // mutual exclusion for threads list
    vector<DefaultWorkerThread*> threads;  // the list of all threads
    
#ifdef ILMBASE_FORCE_CXX03
    bool stopping;                  // flag indicating whether to stop threads
    mutable Mutex stopMutex;        // mutual exclusion for stopping flag
#else
    std::atomic<bool> hasThreads;
    std::atomic<bool> stopping;
#endif

    inline bool stopped () const
    {
#ifdef ILMBASE_FORCE_CXX03
        Lock lock (stopMutex);
        return stopping;
#else
        return stopping.load( std::memory_order_relaxed );
#endif
    }

    inline void stop ()
    {
#ifdef ILMBASE_FORCE_CXX03
        Lock lock (stopMutex);
#endif
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
            Lock taskLock (_data->taskMutex);
    
            //
            // If there is a task pending, pop off the next task in the FIFO
            //

            if (!_data->tasks.empty())
            {
                Task* task = _data->tasks.back();
                _data->tasks.pop_back();
                taskLock.release();

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
    Lock lock (_data.threadMutex);
    return static_cast<int> (_data.threads.size());
}

void
DefaultThreadPoolProvider::setNumThreads (int count)
{
    //
    // Lock access to thread list and size
    //

    Lock lock (_data.threadMutex);

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
#ifndef ILMBASE_FORCE_CXX03
    _data.hasThreads = !(_data.threads.empty());
#endif
}

void
DefaultThreadPoolProvider::addTask (Task *task)
{
    //
    // Lock the threads, needed to access numThreads
    //
#ifdef ILMBASE_FORCE_CXX03
    bool doPush;
    {
        Lock lock (_data.threadMutex);
        doPush = !_data.threads.empty();
    }
#else
    bool doPush = _data.hasThreads.load( std::memory_order_relaxed );
#endif

    if ( doPush )
    {
        //
        // Get exclusive access to the tasks queue
        //

        {
            Lock taskLock (_data.taskMutex);

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
        _data.taskSemaphore.post();
        _data.threadSemaphore.wait();
    }

    //
    // Join all the threads
    //
    for (size_t i = 0; i != curT; ++i)
        delete _data.threads[i];

    Lock lock1 (_data.taskMutex);
#ifdef ILMBASE_FORCE_CXX03
    Lock lock2 (_data.stopMutex);
#endif
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

TaskGroup::Data::Data (): isEmpty (1), numPending (0)
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

    Lock lock (dtorMutex);
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
#if ILMBASE_FORCE_CXX03
    Lock lock (dtorMutex);
#endif
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
#ifdef ILMBASE_FORCE_CXX03
    Lock lock (dtorMutex);

    if (--numPending == 0)
        isEmpty.post ();
#else
    if (--numPending == 0)
    {
#ifdef ENABLE_SEM_DTOR_WORKAROUND
        Lock lock (dtorMutex);
#endif
        isEmpty.post ();
    }
#endif
}
    

//
// struct ThreadPool::Data
//

ThreadPool::Data::Data ():
    provUsers (0), provider (NULL)
#ifdef ILMBASE_FORCE_CXX03
    , oldprovider (NULL)
#else
#endif
{
    // empty
}


ThreadPool::Data::~Data()
{
#ifdef ILMBASE_FORCE_CXX03
    provider->finish();
#else
    ThreadPoolProvider *p = provider.load( std::memory_order_relaxed );
    p->finish();
#endif
}

inline ThreadPool::Data::SafeProvider
ThreadPool::Data::getProvider ()
{
#ifdef ILMBASE_FORCE_CXX03
    Lock provLock( provMutex );
    ++provUsers;
    return SafeProvider( this, provider );
#else
    provUsers.fetch_add( 1, std::memory_order_relaxed );
    return SafeProvider( this, provider.load( std::memory_order_relaxed ) );
#endif
}


inline void
ThreadPool::Data::coalesceProviderUse ()
{
#ifdef ILMBASE_FORCE_CXX03
    Lock provLock( provMutex );
    --provUsers;
    if ( provUsers == 0 )
    {
        if ( oldprovider )
            provSem.post();
    }
#else
    int ov = provUsers.fetch_sub( 1, std::memory_order_relaxed );
    // ov is the previous value, so one means that now it might be 0
    if ( ov == 1 )
    {
        
    }
#endif
}


inline void
ThreadPool::Data::bumpProviderUse ()
{
#ifdef ILMBASE_FORCE_CXX03
    Lock lock (provMutex);
    ++provUsers;
#else
    provUsers.fetch_add( 1, std::memory_order_relaxed );
#endif
}


inline void
ThreadPool::Data::setProvider (ThreadPoolProvider *p)
{
#ifdef ILMBASE_FORCE_CXX03
    Lock provLock( provMutex );

    if ( oldprovider )
        throw IEX_INTERNAL_NAMESPACE::ArgExc ("Attempt to set the thread pool provider while"
                                              " another thread is currently setting the provider.");

    oldprovider = provider;
    provider = p;

    while ( provUsers > 0 )
    {
        provLock.release();
        provSem.wait();
        provLock.acquire();
    }
    if ( oldprovider )
    {
        oldprovider->finish();
        delete oldprovider;
        oldprovider = NULL;
    }
#else
    ThreadPoolProvider *old = provider.load( std::memory_order_relaxed );
    do
    {
        if ( ! provider.compare_exchange_weak( old, p, std::memory_order_release, std::memory_order_relaxed ) )
            continue;
    } while ( false );

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
#endif
}

//
// class Task
//

Task::Task (TaskGroup* g): _group(g)
{
    if ( g )
        g->_data->addTask ();
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
    _data (new Data())
{
    // empty
}


TaskGroup::~TaskGroup ()
{
    delete _data;
}


void
TaskGroup::finishOneTask ()
{
    _data->removeTask ();
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
    _data (new Data)
{
    if ( nthreads == 0 )
        _data->setProvider( new NullThreadPoolProvider );
    else
        _data->setProvider( new DefaultThreadPoolProvider( int(nthreads) ) );
}


ThreadPool::~ThreadPool ()
{
    delete _data;
}


int
ThreadPool::numThreads () const
{
    return _data->getProvider ()->numThreads ();
}


void
ThreadPool::setNumThreads (int count)
{
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
}


void
ThreadPool::setThreadProvider (ThreadPoolProvider *provider)
{
    _data->setProvider (provider);
}


void
ThreadPool::addTask (Task* task) 
{
    _data->getProvider ()->addTask (task);
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


ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_EXIT
