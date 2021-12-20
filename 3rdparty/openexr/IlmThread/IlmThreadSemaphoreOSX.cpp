//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class Semaphore -- implementation for OSX platform(it don't support unnamed Posix semaphores)
//	std::condition_variable + std::mutex emulation show poor performance
//
//-----------------------------------------------------------------------------

#if defined(__APPLE__) && !ILMTHREAD_HAVE_POSIX_SEMAPHORES

#include "IlmThreadSemaphore.h"
#include "Iex.h"

ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_ENTER


Semaphore::Semaphore (unsigned int value)
{
    // Calls to dispatch_semaphore_signal must be balanced with calls to wait().
    // Attempting to dispose of a semaphore with a count lower than value causes an EXC_BAD_INSTRUCTION exception.
    _semaphore = dispatch_semaphore_create (0);
    while (value--)
        post ();
}


Semaphore::~Semaphore ()
{
    dispatch_release (_semaphore);
}


void
Semaphore::wait ()
{
    dispatch_semaphore_wait (_semaphore, DISPATCH_TIME_FOREVER);
}


bool
Semaphore::tryWait ()
{
    return dispatch_semaphore_wait (_semaphore, DISPATCH_TIME_NOW) == 0;
}


void
Semaphore::post ()
{
    dispatch_semaphore_signal (_semaphore);
}


int
Semaphore::value () const
{
    throw IEX_NAMESPACE::NoImplExc ("Not implemented on this platform");

    return 0;
}


ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_EXIT

#endif
