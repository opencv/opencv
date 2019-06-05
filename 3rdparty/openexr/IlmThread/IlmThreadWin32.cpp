///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2005-2012, Industrial Light & Magic, a division of Lucas
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
//	class Thread -- implementation for Windows
//
//-----------------------------------------------------------------------------


#include "IlmBaseConfig.h"

#ifdef ILMBASE_FORCE_CXX03

#include "IlmThread.h"
#include "Iex.h"
#include <iostream>
#include <assert.h>

ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_ENTER


bool
supportsThreads ()
{
    return true;
}

namespace {

unsigned __stdcall
threadLoop (void * t)
{
    reinterpret_cast<Thread*>(t)->run();
    _endthreadex (0);
    return 0;
}

} // namespace


Thread::Thread ()
{
    // empty
}


Thread::~Thread ()
{
    DWORD status = ::WaitForSingleObject (_thread, INFINITE);
    assert (status ==  WAIT_OBJECT_0);
    bool ok = ::CloseHandle (_thread) != FALSE;
    assert (ok);
}


void
Thread::start ()
{
    unsigned id;
    _thread = (HANDLE)::_beginthreadex (0, 0, &threadLoop, this, 0, &id);

    if (_thread == 0)
        IEX_NAMESPACE::throwErrnoExc ("Cannot create new thread (%T).");
}


ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_EXIT

#endif
