//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_ILM_THREAD_H
#define INCLUDED_ILM_THREAD_H

//-----------------------------------------------------------------------------
//
//	class Thread
//
//	Class Thread is a portable interface to a system-dependent thread
//	primitive.  In order to make a thread actually do something useful,
//	you must derive a subclass from class Thread and implement the
//	run() function.  If the operating system supports threading then
//	the run() function will be executed int a new thread.
//
//	The actual creation of the thread is done by the start() routine
//	which then calls the run() function.  In general the start()
//	routine should be called from the constructor of the derived class.
//
//	The base-class thread destructor will join/destroy the thread.
//
//	IMPORTANT: Due to the mechanisms that encapsulate the low-level
//	threading primitives in a C++ class there is a race condition
//	with code resembling the following:
//
//	    {
//		WorkerThread myThread;
//	    } // myThread goes out of scope, is destroyed
//	      // and the thread is joined
//
//	The race is between the parent thread joining the child thread
//	in the destructor of myThread, and the run() function in the
//	child thread.  If the destructor gets executed first then run()
//	will be called with an invalid "this" pointer.
//
//	This issue can be fixed by using a Semaphore to keep track of
//	whether the run() function has already been called.  You can
//	include a Semaphore member variable within your derived class
//	which you post() on in the run() function, and wait() on in the
//	destructor before the thread is joined.  Alternatively you could
//	do something like this:
//
//	    Semaphore runStarted;
//
//	    void WorkerThread::run ()
//	    {
//		runStarted.post()
//		// do some work
//		...
//	    }
//
//	    {
//		WorkerThread myThread;
//		runStarted.wait ();    // ensure that we have started
//				       // the run function
//	    } // myThread goes out of scope, is destroyed
//	      // and the thread is joined
//
//-----------------------------------------------------------------------------

#include "IlmThreadConfig.h"
#include "IlmThreadExport.h"
#include "IlmThreadNamespace.h"

#if ILMTHREAD_THREADING_ENABLED
#include <thread>
#endif

ILMTHREAD_INTERNAL_NAMESPACE_HEADER_ENTER

//
// Query function to determine if the current platform supports
// threads AND this library was compiled with threading enabled.
//

ILMTHREAD_EXPORT bool supportsThreads ();


class ILMTHREAD_EXPORT_TYPE Thread
{
  public:

    ILMTHREAD_EXPORT Thread ();
    ILMTHREAD_EXPORT virtual ~Thread ();

    ILMTHREAD_EXPORT void         start ();

    virtual void run () = 0;

    //
    // wait for thread to exit - must be called before deleting thread
    //
    ILMTHREAD_EXPORT void join();
    ILMTHREAD_EXPORT bool joinable() const;

  private:

#if ILMTHREAD_THREADING_ENABLED
    std::thread _thread;
#endif

    Thread &operator= (const Thread& t) = delete;
    Thread &operator= (Thread&& t) = delete;
    Thread (const Thread& t) = delete;
    Thread (Thread&& t) = delete;
};


ILMTHREAD_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_ILM_THREAD_H
