//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class Semaphore -- implementation for Windows
//
//-----------------------------------------------------------------------------

#include "IlmThreadConfig.h"

#if (defined(_WIN32) || defined(_WIN64)) && !ILMTHREAD_HAVE_POSIX_SEMAPHORES

#include "IlmThreadSemaphore.h"
#include "Iex.h"
#include <string>
#include <assert.h>
#include <iostream>

ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace IEX_NAMESPACE;

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


ILMTHREAD_INTERNAL_NAMESPACE_SOURCE_EXIT

#endif // _WIN32
