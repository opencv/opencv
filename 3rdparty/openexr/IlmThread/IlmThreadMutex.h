//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_ILM_THREAD_MUTEX_H
#define INCLUDED_ILM_THREAD_MUTEX_H

//-----------------------------------------------------------------------------
//
// NB: Maintained for backward compatibility with header files only. This
// has been entirely replaced by c++11 and the std::mutex layer
//
//-----------------------------------------------------------------------------

#include "IlmThreadExport.h"
#include "IlmThreadConfig.h"
#include "IlmThreadNamespace.h"

#if ILMTHREAD_THREADING_ENABLED
#include <mutex>
#endif

ILMTHREAD_INTERNAL_NAMESPACE_HEADER_ENTER

#if ILMTHREAD_THREADING_ENABLED
using Mutex ILMTHREAD_DEPRECATED ("replace with std::mutex") = std::mutex;

// unfortunately we can't use std::unique_lock as a replacement for Lock since
// they have different API. Let us deprecate for now and give people a chance
// to clean up their code.
class Lock
{
  public:

    ILMTHREAD_DEPRECATED ("replace with std::lock_guard or std::unique_lock")
    Lock (const Mutex& m, bool autoLock = true):
        _mutex (const_cast<Mutex &>(m)), _locked (false)
    {
        if (autoLock)
        {
            _mutex.lock();
            _locked = true;
        }
    }
    
    ~Lock ()
    {
        if (_locked)
            _mutex.unlock();
    }
    Lock (const Lock&) = delete;
    Lock &operator= (const Lock&) = delete;
    Lock (Lock&&) = delete;
    Lock& operator= (Lock&&) = delete;

    void acquire ()
    {
        _mutex.lock();
        _locked = true;
    }
    
    void release ()
    {
        _locked = false;
        _mutex.unlock();
    }
    
    bool locked ()
    {
        return _locked;
    }

  private:

    Mutex & _mutex;
    bool    _locked;
};
#endif

ILMTHREAD_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_ILM_THREAD_MUTEX_H
