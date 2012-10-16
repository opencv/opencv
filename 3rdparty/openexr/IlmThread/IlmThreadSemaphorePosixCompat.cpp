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
//	class Semaphore -- implementation for for platforms that do
//	support Posix threads but do not support Posix semaphores,
//	for example, OS X
//
//-----------------------------------------------------------------------------

#include "IlmBaseConfig.h"

#if HAVE_PTHREAD && !HAVE_POSIX_SEMAPHORES

#include "IlmThreadSemaphore.h"
#include "Iex.h"
#include <assert.h>

namespace IlmThread {


Semaphore::Semaphore (unsigned int value)
{
    if (int error = ::pthread_mutex_init (&_semaphore.mutex, 0))
        Iex::throwErrnoExc ("Cannot initialize mutex (%T).", error);

    if (int error = ::pthread_cond_init (&_semaphore.nonZero, 0))
        Iex::throwErrnoExc ("Cannot initialize condition variable (%T).",
                            error);

    _semaphore.count = value;
    _semaphore.numWaiting = 0;
}


Semaphore::~Semaphore ()
{
    int error = ::pthread_cond_destroy (&_semaphore.nonZero);
    assert (error == 0);
    error = ::pthread_mutex_destroy (&_semaphore.mutex);
    assert (error == 0);
}


void
Semaphore::wait ()
{
    ::pthread_mutex_lock (&_semaphore.mutex);

    _semaphore.numWaiting++;

    while (_semaphore.count == 0)
    {
        if (int error = ::pthread_cond_wait (&_semaphore.nonZero,
                                             &_semaphore.mutex))
    {
            ::pthread_mutex_unlock (&_semaphore.mutex);

            Iex::throwErrnoExc ("Cannot wait on condition variable (%T).",
                                error);
    }
    }

    _semaphore.numWaiting--;
    _semaphore.count--;

    ::pthread_mutex_unlock (&_semaphore.mutex);
}


bool
Semaphore::tryWait ()
{
    ::pthread_mutex_lock (&_semaphore.mutex);

    if (_semaphore.count == 0)
    {
        ::pthread_mutex_unlock (&_semaphore.mutex);
        return false;
    }
    else
    {
        _semaphore.count--;
        ::pthread_mutex_unlock (&_semaphore.mutex);
        return true;
    }
}


void
Semaphore::post ()
{
    ::pthread_mutex_lock (&_semaphore.mutex);

    if (_semaphore.numWaiting > 0)
    {
        if (int error = ::pthread_cond_signal (&_semaphore.nonZero))
    {
            ::pthread_mutex_unlock (&_semaphore.mutex);

            Iex::throwErrnoExc ("Cannot signal condition variable (%T).",
                                error);
    }
    }

    _semaphore.count++;
    ::pthread_mutex_unlock (&_semaphore.mutex);
}


int
Semaphore::value () const
{
    ::pthread_mutex_lock (&_semaphore.mutex);
    int value = _semaphore.count;
    ::pthread_mutex_unlock (&_semaphore.mutex);
    return value;
}


} // namespace IlmThread

#endif
