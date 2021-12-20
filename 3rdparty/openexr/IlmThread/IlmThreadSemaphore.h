//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_ILM_THREAD_SEMAPHORE_H
#define INCLUDED_ILM_THREAD_SEMAPHORE_H

//-----------------------------------------------------------------------------
//
//	class Semaphore -- a wrapper class for
//	system-dependent counting semaphores
//
//-----------------------------------------------------------------------------

#include "IlmThreadExport.h"

#include "IlmThreadConfig.h"
#include "IlmThreadNamespace.h"

#if ILMTHREAD_THREADING_ENABLED
#   if ILMTHREAD_HAVE_POSIX_SEMAPHORES
#      include <semaphore.h>
#   elif defined(__APPLE__)
#      include <dispatch/dispatch.h>
#   elif (defined (_WIN32) || defined (_WIN64))
#      ifdef NOMINMAX
#         undef NOMINMAX
#      endif
#      define NOMINMAX
#      include <windows.h>
#   else
#      include <mutex>
#      include <condition_variable>
#   endif
#endif

ILMTHREAD_INTERNAL_NAMESPACE_HEADER_ENTER


class ILMTHREAD_EXPORT_TYPE Semaphore
{
  public:

    ILMTHREAD_EXPORT Semaphore (unsigned int value = 0);
    ILMTHREAD_EXPORT virtual ~Semaphore();

    ILMTHREAD_EXPORT void wait();
    ILMTHREAD_EXPORT bool tryWait();
    ILMTHREAD_EXPORT void post();
    ILMTHREAD_EXPORT int  value() const;

  private:

#if ILMTHREAD_HAVE_POSIX_SEMAPHORES

	mutable sem_t _semaphore;

#elif defined(__APPLE__)
	mutable dispatch_semaphore_t _semaphore;

#elif (defined (_WIN32) || defined (_WIN64))

	mutable HANDLE _semaphore;

#elif ILMTHREAD_THREADING_ENABLED
	//
	// If the platform has threads but no semapohores,
	// then we implement them ourselves using condition variables
	//

	struct sema_t
	{
	    unsigned int count;
	    unsigned long numWaiting;
        std::mutex mutex;
        std::condition_variable nonZero;
	};

	mutable sema_t _semaphore;
  
#endif

    void operator = (const Semaphore& s) = delete;
    Semaphore (const Semaphore& s) = delete;
    void operator = (Semaphore&& s) = delete;
    Semaphore (Semaphore&& s) = delete;
};


ILMTHREAD_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_ILM_THREAD_SEMAPHORE_H
