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

#ifndef INCLUDED_ILM_THREAD_SEMAPHORE_H
#define INCLUDED_ILM_THREAD_SEMAPHORE_H

//-----------------------------------------------------------------------------
//
//	class Semaphore -- a wrapper class for
//	system-dependent counting semaphores
//
//-----------------------------------------------------------------------------

#include "IlmBaseConfig.h"

#if defined _WIN32 || defined _WIN64
    #ifdef NOMINMAX
        #undef NOMINMAX
    #endif
    #define NOMINMAX
    #include <windows.h>
#elif HAVE_PTHREAD && !HAVE_POSIX_SEMAPHORES
    #include <pthread.h>
#elif HAVE_PTHREAD && HAVE_POSIX_SEMAPHORES
    #include <semaphore.h>
#endif

namespace IlmThread {


class Semaphore
{
  public:

    Semaphore (unsigned int value = 0);
    virtual ~Semaphore();

    void	wait();
    bool	tryWait();
    void	post();
    int		value() const;

  private:

    #if defined _WIN32 || defined _WIN64

    mutable HANDLE _semaphore;

    #elif HAVE_PTHREAD && !HAVE_POSIX_SEMAPHORES

    //
    // If the platform has Posix threads but no semapohores,
    // then we implement them ourselves using condition variables
    //

    struct sema_t
    {
        unsigned int count;
        unsigned long numWaiting;
        pthread_mutex_t mutex;
        pthread_cond_t nonZero;
    };

    mutable sema_t _semaphore;

    #elif HAVE_PTHREAD && HAVE_POSIX_SEMAPHORES

    mutable sem_t _semaphore;

    #endif

    void operator = (const Semaphore& s);	// not implemented
    Semaphore (const Semaphore& s);		// not implemented
};


} // namespace IlmThread

#endif
