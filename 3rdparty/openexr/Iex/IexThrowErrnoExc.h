///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IEXTHROWERRNOEXC_H
#define INCLUDED_IEXTHROWERRNOEXC_H

//----------------------------------------------------------
//
//	A function which throws ExcErrno exceptions
//
//----------------------------------------------------------

#include "IexBaseExc.h"

namespace Iex {


//--------------------------------------------------------------------------
//
// Function throwErrnoExc() throws an exception which corresponds to
// error code errnum.  The exception text is initialized with a copy
// of the string passed to throwErrnoExc(), where all occurrences of
// "%T" have been replaced with the output of strerror(oserror()).
//
// Example:
//   
// If opening file /tmp/output failed with an ENOENT error code,
// calling
//
//	throwErrnoExc ();
//
// or
//
//	throwErrnoExc ("%T.");
//
// will throw an EnoentExc whose text reads
//
//	No such file or directory.
//
// More detailed messages can be assembled using stringstreams:
//
//	std::stringstream s;
//	s << "Cannot open file " << name << " (%T).";
//	throwErrnoExc (s);
//
// The resulting exception contains the following text:
//
//	Cannot open file /tmp/output (No such file or directory).
//
// Alternatively, you may want to use the THROW_ERRNO macro defined
// in IexMacros.h:
//
//	THROW_ERRNO ("Cannot open file " << name << " (%T).")
//
//--------------------------------------------------------------------------

void throwErrnoExc (const std::string &txt, int errnum);
void throwErrnoExc (const std::string &txt = "%T." /*, int errnum = oserror() */);


} // namespace Iex

#endif
