///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002-2012, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IEXMACROS_H
#define INCLUDED_IEXMACROS_H

//--------------------------------------------------------------------
//
//	Macros which make throwing exceptions more convenient
//
//--------------------------------------------------------------------

#include <sstream>


//----------------------------------------------------------------------------
// A macro to throw exceptions whose text is assembled using stringstreams.
//
// Example:
//
//	THROW (InputExc, "Syntax error in line " << line ", " << file << ".");
//	
//----------------------------------------------------------------------------

#include "IexExport.h"
#include "IexForward.h"

IEX_EXPORT void iex_debugTrap();

#define THROW(type, text)       \
    do                          \
    {                           \
        iex_debugTrap();        \
        std::stringstream s;	\
        s << text;              \
        throw type (s);         \
    }                           \
    while (0)


//----------------------------------------------------------------------------
// Macros to add to or to replace the text of an exception.
// The new text is assembled using stringstreams.
//
// Examples:
//
// Append to end of an exception's text:
//
//	catch (BaseExc &e)
//	{
//	    APPEND_EXC (e, " Directory " << name << " does not exist.");
//	    throw;
//	}
//
// Replace an exception's text:
//
//	catch (BaseExc &e)
//	{
//	    REPLACE_EXC (e, "Directory " << name << " does not exist. " << e);
//	    throw;
//	}
//----------------------------------------------------------------------------

#define APPEND_EXC(exc, text)   \
    do                          \
    {                           \
        std::stringstream s;    \
        s << text;              \
        exc.append (s);         \
    }                           \
    while (0)

#define REPLACE_EXC(exc, text)  \
    do                          \
    {                           \
        std::stringstream s;    \
        s << text;              \
        exc.assign (s);         \
    }                           \
    while (0)


//-------------------------------------------------------------
// A macro to throw ErrnoExc exceptions whose text is assembled
// using stringstreams:
//
// Example:
//
//	THROW_ERRNO ("Cannot open file " << name << " (%T).");
//
//-------------------------------------------------------------

#define THROW_ERRNO(text)                         \
    do                                            \
    {                                             \
        std::stringstream s;                      \
        s << text;                                \
        ::IEX_NAMESPACE::throwErrnoExc (s.str()); \
    }                                             \
    while (0)


//-------------------------------------------------------------
// A macro to throw exceptions if an assertion is false.
//
// Example:
//
//	ASSERT (ptr != 0, NullExc, "Null pointer" );
//
//-------------------------------------------------------------

#define ASSERT(assertion, type, text)   \
    do                                  \
    {                                   \
        if( (assertion) == false )      \
        {                               \
            THROW( type, text );        \
        }                               \
    }                                   \
    while (0)

//-------------------------------------------------------------
// A macro to throw an IEX_NAMESPACE::LogicExc if an assertion is false,
// with the text composed from the source code file, line number,
// and assertion argument text.
//
// Example:
//
//      LOGIC_ASSERT (i < n);
//
//-------------------------------------------------------------
#define LOGIC_ASSERT(assertion)           \
    ASSERT(assertion,                     \
           IEX_NAMESPACE::LogicExc,       \
           __FILE__ << "(" << __LINE__ << "): logical assertion failed: " << #assertion )

#endif
