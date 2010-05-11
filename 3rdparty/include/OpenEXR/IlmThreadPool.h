///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2005, Industrial Light & Magic, a division of Lucas
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
//	Class Thread provides an abstract interface for a task which
//	a ThreadPool works on.  Derived classes need to implement the
//	execute() function which performs the actual task.
//
//	Class TaskTroup allows synchronization on the completion of a set
//	of tasks.  Every task that is added to a ThreadPool belongs to a
//	single TaskGroup.  The destructor of the TaskGroup waits for all
//	tasks in the group to finish.
//
//	Note: if you plan to use the ThreadPool interface in your own
//	applications note that the implementation of the ThreadPool calls
//	opertor delete on tasks as they complete.  If you define a custom
//	operator new for your tasks, for instance to use a custom heap,
//	then you must also write an appropriate operator delete.
//
//-----------------------------------------------------------------------------

namespace IlmThread {

class TaskGroup;
class Task;


class ThreadPool  
{
  public:

    //-------------------------------------------------------
    // Constructor -- creates numThreads worker threads which
    // wait until a task is available. 
    //-------------------------------------------------------

    ThreadPool (unsigned numThreads = 0);
    
    
    //-----------------------------------------------------------
    // Destructor -- waits for all tasks to complete, joins all
    // the threads to the calling thread, and then destroys them.
    //-----------------------------------------------------------

    virtual ~ThreadPool ();
    

    //--------------------------------------------------------
    // Query and set the number of worker threads in the pool.
    //
    // Warning: never call setNumThreads from within a worker
    // thread as this will almost certainly cause a deadlock
    // or crash.
    //--------------------------------------------------------
    
    int		numThreads () const;
    void	setNumThreads (int count);
    
    
    //------------------------------------------------------------
    // Add a task for processing.  The ThreadPool can handle any
    // number of tasks regardless of the number of worker threads.
    // The tasks are first added onto a queue, and are executed
    // by threads as they become available, in FIFO order.
    //------------------------------------------------------------

    void addTask (Task* task);
    

    //-------------------------------------------
    // Access functions for the global threadpool
    //-------------------------------------------
    
    static ThreadPool&	globalThreadPool ();
    static void		addGlobalTask (Task* task);

    struct Data;

  protected:

    Data *		_data;
};


class Task
{
  public:

    Task (TaskGroup* g);
    virtual ~Task ();

    virtual void	execute () = 0;
    TaskGroup *		group();

  protected:

    TaskGroup *		_group;
};


class TaskGroup
{
  public:

     TaskGroup();
    ~TaskGroup();

    struct Data;
    Data* const		_data;
};


} // namespace IlmThread

#endif
