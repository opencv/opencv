//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_ILM_THREAD_POOL_H
#define INCLUDED_ILM_THREAD_POOL_H


//-----------------------------------------------------------------------------
//
//	class Task, class ThreadPool, class TaskGroup
//
//	Class ThreadPool manages a set of worker threads and accepts
//	tasks for processing.  Tasks added to the thread pool are
//	executed concurrently by the worker threads.  
//	
//	Class Task provides an abstract interface for a task which
//	a ThreadPool works on.  Derived classes need to implement the
//	execute() function which performs the actual task.
//
//	Class TaskGroup allows synchronization on the completion of a set
//	of tasks.  Every task that is added to a ThreadPool belongs to a
//	single TaskGroup.  The destructor of the TaskGroup waits for all
//	tasks in the group to finish.
//
//	Note: if you plan to use the ThreadPool interface in your own
//	applications note that the implementation of the ThreadPool calls
//	operator delete on tasks as they complete.  If you define a custom
//	operator new for your tasks, for instance to use a custom heap,
//	then you must also write an appropriate operator delete.
//
//-----------------------------------------------------------------------------

#include "IlmThreadNamespace.h"
#include "IlmThreadExport.h"
#include "IlmThreadConfig.h"

ILMTHREAD_INTERNAL_NAMESPACE_HEADER_ENTER

class TaskGroup;
class Task;

//-------------------------------------------------------
// ThreadPoolProvider -- this is a pure virtual interface
// enabling custom overloading of the threads used and how
// the implementation of the processing of tasks
// is implemented
//-------------------------------------------------------
class ILMTHREAD_EXPORT_TYPE ThreadPoolProvider
{
  public:
    ILMTHREAD_EXPORT ThreadPoolProvider();
    ILMTHREAD_EXPORT virtual ~ThreadPoolProvider();

    // as in ThreadPool below
    virtual int numThreads () const = 0;
    // as in ThreadPool below
    virtual void setNumThreads (int count) = 0;
    // as in ThreadPool below
    virtual void addTask (Task* task) = 0;

    // Ensure that all tasks in this set are finished
    // and threads shutdown
    virtual void finish () = 0;

    // Make the provider non-copyable
    ThreadPoolProvider (const ThreadPoolProvider &) = delete;
    ThreadPoolProvider &operator= (const ThreadPoolProvider &) = delete;
    ThreadPoolProvider (ThreadPoolProvider &&) = delete;
    ThreadPoolProvider &operator= (ThreadPoolProvider &&) = delete;
};  

class ILMTHREAD_EXPORT_TYPE ThreadPool  
{
  public:
    //-------------------------------------------------------
    // static routine to query how many processors should be
    // used for processing exr files. The user of ThreadPool
    // is free to use std::thread::hardware_concurrency or
    // whatever number of threads is appropriate based on the
    // application. However, this routine exists such that
    // in the future, if core counts expand faster than
    // memory bandwidth, or higher order NUMA machines are built
    // that we can query, this routine gives a place where we
    // can centralize that logic
    //-------------------------------------------------------
    ILMTHREAD_EXPORT
    static unsigned estimateThreadCountForFileIO ();

    //-------------------------------------------------------
    // Constructor -- creates numThreads worker threads which
    // wait until a task is available,
    // using a default ThreadPoolProvider
    //-------------------------------------------------------

    ILMTHREAD_EXPORT ThreadPool (unsigned numThreads = 0);


    //-----------------------------------------------------------
    // Destructor -- waits for all tasks to complete, joins all
    // the threads to the calling thread, and then destroys them.
    //-----------------------------------------------------------

    ILMTHREAD_EXPORT virtual ~ThreadPool ();
    ThreadPool (const ThreadPool&) = delete;
    ThreadPool& operator= (const ThreadPool&) = delete;
    ThreadPool (ThreadPool&&) = delete;
    ThreadPool& operator= (ThreadPool&&) = delete;

    //--------------------------------------------------------
    // Query and set the number of worker threads in the pool.
    //
    // Warning: never call setNumThreads from within a worker
    // thread as this will almost certainly cause a deadlock
    // or crash.
    //--------------------------------------------------------
    
    ILMTHREAD_EXPORT int  numThreads () const;
    ILMTHREAD_EXPORT void setNumThreads (int count);

    //--------------------------------------------------------
    // Set the thread provider for the pool.
    //
    // The ThreadPool takes ownership of the ThreadPoolProvider
    // and will call delete on it when it is finished or when
    // it is changed
    //
    // Warning: never call setThreadProvider from within a worker
    // thread as this will almost certainly cause a deadlock
    // or crash.
    //--------------------------------------------------------
    ILMTHREAD_EXPORT void setThreadProvider (ThreadPoolProvider *provider);

    //------------------------------------------------------------
    // Add a task for processing.  The ThreadPool can handle any
    // number of tasks regardless of the number of worker threads.
    // The tasks are first added onto a queue, and are executed
    // by threads as they become available, in FIFO order.
    //------------------------------------------------------------

    ILMTHREAD_EXPORT void addTask (Task* task);
    

    //-------------------------------------------
    // Access functions for the global threadpool
    //-------------------------------------------
    
    ILMTHREAD_EXPORT static ThreadPool&	globalThreadPool ();
    ILMTHREAD_EXPORT static void		addGlobalTask (Task* task);

    struct ILMTHREAD_HIDDEN Data;

  protected:

    Data *		_data;
};


class ILMTHREAD_EXPORT_TYPE Task
{
  public:

    ILMTHREAD_EXPORT Task (TaskGroup* g);
    ILMTHREAD_EXPORT virtual ~Task ();
    Task (const Task&) = delete;
    Task &operator= (const Task&) = delete;
    Task (Task&&) = delete;
    Task& operator= (Task&&) = delete;

    virtual void	execute () = 0;
    ILMTHREAD_EXPORT
    TaskGroup *		group();

  protected:

    TaskGroup *		_group;
};


class ILMTHREAD_EXPORT_TYPE TaskGroup
{
  public:

    ILMTHREAD_EXPORT TaskGroup();
    ILMTHREAD_EXPORT ~TaskGroup();

    TaskGroup (const TaskGroup& other) = delete;
    TaskGroup& operator = (const TaskGroup& other) = delete;
    TaskGroup (TaskGroup&& other) = delete;
    TaskGroup& operator = (TaskGroup&& other) = delete;
    
    // marks one task as finished
    // should be used by the thread pool provider to notify
    // as it finishes tasks
    ILMTHREAD_EXPORT void finishOneTask ();

    struct ILMTHREAD_HIDDEN Data;
    Data* const		_data;
};


ILMTHREAD_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_ILM_THREAD_POOL_H
