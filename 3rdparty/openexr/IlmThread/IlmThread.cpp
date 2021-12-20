//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class Thread -- this file contains two implementations of thread:
//	- dummy implementation for platforms that disable / do not support threading
//	- c++11 and newer version
//
//-----------------------------------------------------------------------------

#include "IlmThreadConfig.h"
#include "IlmThread.h"
#include "Iex.h"

ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_ENTER

#if ILMTHREAD_THREADING_ENABLED

bool
supportsThreads ()
{
    return true;
}

Thread::Thread ()
{
    // empty
}


Thread::~Thread ()
{
    // hopefully the thread has basically exited and we are just
    // cleaning up, because run is a virtual function, so the v-table
    // has already been partly destroyed...
    if ( _thread.joinable () )
        _thread.join ();
}

void
Thread::join()
{
    if ( _thread.joinable () )
        _thread.join ();
}

bool
Thread::joinable() const
{
    return _thread.joinable();
}

void
Thread::start ()
{
    _thread = std::thread (&Thread::run, this);
}

#else

bool
supportsThreads ()
{
    return false;
}


Thread::Thread ()
{
    throw IEX_NAMESPACE::NoImplExc ("Threads not supported / enabled on this platform.");
}


Thread::~Thread ()
{
}


void
Thread::start ()
{
    throw IEX_NAMESPACE::NoImplExc ("Threads not supported / enabled on this platform.");
}

void
Thread::join ()
{
    throw IEX_NAMESPACE::NoImplExc ("Threads not supported / enabled on this platform.");
}

bool
Thread::joinable () const
{
    throw IEX_NAMESPACE::NoImplExc ("Threads not supported / enabled on this platform.");
}

#endif


ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_EXIT

