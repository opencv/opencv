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
//	class Semaphore -- implementation for Windows
//
//-----------------------------------------------------------------------------

#include "IlmThreadSemaphore.h"
#include "Iex.h"
#include <string>
#include <assert.h>
#include <iostream>

namespace IlmThread {

using namespace Iex;

namespace {

std::string
errorString ()
{
    LPSTR messageBuffer;
    DWORD bufferLength;
    std::string message;

    //
    // Call FormatMessage() to allow for message 
    // text to be acquired from the system.
    //

    if (bufferLength = FormatMessageA (FORMAT_MESSAGE_ALLOCATE_BUFFER |
				       FORMAT_MESSAGE_IGNORE_INSERTS |
				       FORMAT_MESSAGE_FROM_SYSTEM,
				       0,
				       GetLastError (),
				       MAKELANGID (LANG_NEUTRAL,
						   SUBLANG_DEFAULT),
				       (LPSTR) &messageBuffer,
				       0,
				       NULL))
    {
	message = messageBuffer;
        LocalFree (messageBuffer);
    }

    return message;
}

} // namespace


Semaphore::Semaphore (unsigned int value)
{
    if ((_semaphore = ::CreateSemaphore (0, value, 0x7fffffff, 0)) == 0)
    {
	THROW (LogicExc, "Could not create semaphore "
			 "(" << errorString() << ").");
    }
}


Semaphore::~Semaphore()
{
    bool ok = ::CloseHandle (_semaphore) != FALSE;
    assert (ok);
}


void
Semaphore::wait()
{
    if (::WaitForSingleObject (_semaphore, INFINITE) != WAIT_OBJECT_0)
    {
	THROW (LogicExc, "Could not wait on semaphore "
			 "(" << errorString() << ").");
    }
}


bool
Semaphore::tryWait()
{
    return ::WaitForSingleObject (_semaphore, 0) == WAIT_OBJECT_0;
}


void
Semaphore::post()
{
    if (!::ReleaseSemaphore (_semaphore, 1, 0))
    {
	THROW (LogicExc, "Could not post on semaphore "
			 "(" << errorString() << ").");
    }
}


int
Semaphore::value() const
{
    LONG v = -1;

    if (!::ReleaseSemaphore (_semaphore, 0, &v) || v < 0)
    {
	THROW (LogicExc, "Could not get value of semaphore "
			 "(" << errorString () << ").");
    }

    return v;
}

} // namespace IlmThread
