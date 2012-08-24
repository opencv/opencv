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

//-----------------------------------------------------------------------------
//
//	class Task, class ThreadPool, class TaskGroup
//
//-----------------------------------------------------------------------------

#include "IlmThread.h"
#include "IlmThreadMutex.h"
#include "IlmThreadSemaphore.h"
#include "IlmThreadPool.h"
#include "Iex.h"
#include <list>

using namespace std;

namespace IlmThread {
namespace {

class WorkerThread: public Thread
{
  public:

    WorkerThread (ThreadPool::Data* data);

    virtual void	run ();
    
  private:

    ThreadPool::Data *	_data;
};

} //namespace


struct TaskGroup::Data
{
     Data ();
    ~Data ();
    
    void	addTask () ;
    void	removeTask ();
    
    Semaphore	isEmpty;	// used to signal that the taskgroup is empty
    int		numPending;	// number of pending tasks to still execute
};


struct ThreadPool::Data
{
     Data ();
    ~Data();
    
    void	finish ();
    bool	stopped () const;
    void	stop ();

    Semaphore taskSemaphore;        // threads wait on this for ready tasks
    Mutex taskMutex;                // mutual exclusion for the tasks list
    list<Task*> tasks;              // the list of tasks to execute
    size_t numTasks;                // fast access to list size
                                    //   (list::size() can be O(n))

    Semaphore threadSemaphore;      // signaled when a thread starts executing
    Mutex threadMutex;              // mutual exclusion for threads list
    list<WorkerThread*> threads;    // the list of all threads
    size_t numThreads;              // fast access to list size
    
    bool stopping;                  // flag indicating whether to stop threads
    Mutex stopMutex;                // mutual exclusion for stopping flag
};



//
// class WorkerThread
//

WorkerThread::WorkerThread (ThreadPool::Data* data):
    _data (data)
{
    start();
}


void
WorkerThread::run ()
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

            if (_data->numTasks > 0)
            {
                Task* task = _data->tasks.front();
		TaskGroup* taskGroup = task->group();
                _data->tasks.pop_front();
                _data->numTasks--;

                taskLock.release();
                task->execute();
                taskLock.acquire();

                delete task;
                taskGroup->_data->removeTask();
            }
            else if (_data->stopped())
	    {
                break;
	    }
        }
    }
}


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
}


void
TaskGroup::Data::addTask () 
{
    //
    // Any access to the taskgroup is protected by a mutex that is
    // held by the threadpool.  Therefore it is safe to access
    // numPending before we wait on the semaphore.
    //

    if (numPending++ == 0)
	isEmpty.wait ();
}


void
TaskGroup::Data::removeTask ()
{
    if (--numPending == 0)
	isEmpty.post ();
}
    

//
// struct ThreadPool::Data
//

ThreadPool::Data::Data (): numTasks (0), numThreads (0), stopping (false)
{
    // empty
}


ThreadPool::Data::~Data()
{
    Lock lock (threadMutex);
    finish ();
}


void
ThreadPool::Data::finish ()
{
    stop();

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

    for (size_t i = 0; i < numThreads; i++)
    {
	taskSemaphore.post();
	threadSemaphore.wait();
    }

    //
    // Join all the threads
    //

    for (list<WorkerThread*>::iterator i = threads.begin();
	 i != threads.end();
	 ++i)
    {
	delete (*i);
    }

    Lock lock1 (taskMutex);
    Lock lock2 (stopMutex);
    threads.clear();
    tasks.clear();
    numThreads = 0;
    numTasks = 0;
    stopping = false;
}


bool
ThreadPool::Data::stopped () const
{
    Lock lock (stopMutex);
    return stopping;
}


void
ThreadPool::Data::stop ()
{
    Lock lock (stopMutex);
    stopping = true;
}


//
// class Task
//

Task::Task (TaskGroup* g): _group(g)
{
    // empty
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


//
// class ThreadPool
//

ThreadPool::ThreadPool (unsigned nthreads):
    _data (new Data())
{
    setNumThreads (nthreads);
}


ThreadPool::~ThreadPool ()
{
    delete _data;
}


int
ThreadPool::numThreads () const
{
    Lock lock (_data->threadMutex);
    return _data->numThreads;
}


void
ThreadPool::setNumThreads (int count)
{
    if (count < 0)
        throw Iex::ArgExc ("Attempt to set the number of threads "
			   "in a thread pool to a negative value.");

    //
    // Lock access to thread list and size
    //

    Lock lock (_data->threadMutex);

    if ((size_t)count > _data->numThreads)
    {
	//
        // Add more threads
	//

        while (_data->numThreads < (size_t)count)
        {
            _data->threads.push_back (new WorkerThread (_data));
            _data->numThreads++;
        }
    }
    else if ((size_t)count < _data->numThreads)
    {
	//
	// Wait until all existing threads are finished processing,
	// then delete all threads.
	//

        _data->finish ();

	//
        // Add in new threads
	//

        while (_data->numThreads < (size_t)count)
        {
            _data->threads.push_back (new WorkerThread (_data));
            _data->numThreads++;
        }
    }
}


void
ThreadPool::addTask (Task* task) 
{
    //
    // Lock the threads, needed to access numThreads
    //

    Lock lock (_data->threadMutex);

    if (_data->numThreads == 0)
    {
        task->execute ();
        delete task;
    }
    else
    {
	//
        // Get exclusive access to the tasks queue
	//

        {
            Lock taskLock (_data->taskMutex);

	    //
            // Push the new task into the FIFO
	    //

            _data->tasks.push_back (task);
            _data->numTasks++;
            task->group()->_data->addTask();
        }
        
	//
        // Signal that we have a new task to process
	//

        _data->taskSemaphore.post ();
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
    globalThreadPool().addTask (task);
}


} // namespace IlmThread
