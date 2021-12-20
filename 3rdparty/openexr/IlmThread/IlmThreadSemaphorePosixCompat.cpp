//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class Semaphore -- implementation for for platforms that do
//	support Posix threads but do not support Posix semaphores
//
//-----------------------------------------------------------------------------

#include "IlmThreadConfig.h"

#if ILMTHREAD_THREADING_ENABLED
#if ( !(ILMTHREAD_HAVE_POSIX_SEMAPHORES) && !defined (__APPLE__) && !defined (_WIN32) && !defined (_WIN64) )

#include "IlmThreadSemaphore.h"

ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_ENTER

Semaphore::Semaphore (unsigned int value)
{
    _semaphore.count = value;
    _semaphore.numWaiting = 0;
}


Semaphore::~Semaphore ()
{
}


void
Semaphore::wait ()
{
    std::unique_lock<std::mutex> lk(_semaphore.mutex);

    _semaphore.numWaiting++;

    while (_semaphore.count == 0)
        _semaphore.nonZero.wait (lk);

    _semaphore.numWaiting--;
    _semaphore.count--;
}


bool
Semaphore::tryWait ()
{
    std::lock_guard<std::mutex> lk(_semaphore.mutex);
    
    if (_semaphore.count == 0)
        return false;

    _semaphore.count--;
    return true;
}


void
Semaphore::post ()
{
    std::lock_guard<std::mutex> lk(_semaphore.mutex);

    _semaphore.count++;
    if (_semaphore.numWaiting > 0)
    {
        if (_semaphore.count > 1)
            _semaphore.nonZero.notify_all();
        else
            _semaphore.nonZero.notify_one();
    }
}


int
Semaphore::value () const
{
    std::lock_guard<std::mutex> lk(_semaphore.mutex);
    return _semaphore.count;
}

ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_EXIT

#endif // posix semaphore compat
#endif // enable threading
