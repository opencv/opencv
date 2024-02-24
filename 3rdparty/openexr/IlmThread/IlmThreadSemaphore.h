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

//
// Decipher the platform-specific threading support.
// Set the ILMTHREAD_SEMAPHORE_* defines to indicate the corresponding
// implementation of the Semaphore class. Only one of these should be
// defined.
//

#if ILMTHREAD_THREADING_ENABLED
#    if ILMTHREAD_HAVE_POSIX_SEMAPHORES
#        include <semaphore.h>
#        define ILMTHREAD_SEMAPHORE_POSIX 1
#    elif defined(__APPLE__)
#        include <AvailabilityMacros.h>
#        if MAC_OS_X_VERSION_MIN_REQUIRED > 1050 && !defined(__ppc__)
#            include <dispatch/dispatch.h>
#            define ILMTHREAD_SEMAPHORE_OSX 1
#        else
#            include <condition_variable>
#            include <mutex>
#            define ILMTHREAD_SEMAPHORE_OTHER 1
#        endif
#    elif (defined(_WIN32) || defined(_WIN64))
#        ifdef NOMINMAX
#            undef NOMINMAX
#        endif
#        define NOMINMAX
#        include <windows.h>
#        define ILMTHREAD_SEMAPHORE_WINDOWS 1
#    else
#        include <condition_variable>
#        include <mutex>
#        define ILMTHREAD_SEMAPHORE_OTHER 1
#    endif
#else
#    define ILMTHREAD_SEMAPHORE_DISABLED 1
#endif

ILMTHREAD_INTERNAL_NAMESPACE_HEADER_ENTER

class ILMTHREAD_EXPORT_TYPE Semaphore
{
public:
    ILMTHREAD_EXPORT Semaphore (unsigned int value = 0);
    ILMTHREAD_EXPORT virtual ~Semaphore ();

    ILMTHREAD_EXPORT void wait ();
    ILMTHREAD_EXPORT bool tryWait ();
    ILMTHREAD_EXPORT void post ();
    ILMTHREAD_EXPORT int  value () const;

private:

#if ILMTHREAD_SEMAPHORE_POSIX

    mutable sem_t _semaphore;

#elif ILMTHREAD_SEMAPHORE_OSX
    
    mutable dispatch_semaphore_t _semaphore;

#elif ILMTHREAD_SEMAPHORE_WINDOWS

    mutable HANDLE _semaphore;

#elif ILMTHREAD_SEMAPHORE_OTHER
    
    //
    // If the platform has threads but no semaphores,
    // then we implement them ourselves using condition variables
    //

    struct sema_t
    {
        unsigned int            count;
        unsigned long           numWaiting;
        std::mutex              mutex;
        std::condition_variable nonZero;
    };

    mutable sema_t _semaphore;

#endif

    void operator= (const Semaphore& s) = delete;
    Semaphore (const Semaphore& s)      = delete;
    void operator= (Semaphore&& s) = delete;
    Semaphore (Semaphore&& s)      = delete;
};

ILMTHREAD_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_ILM_THREAD_SEMAPHORE_H
