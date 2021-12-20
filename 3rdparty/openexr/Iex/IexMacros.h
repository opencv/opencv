//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


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

#define THROW(type, text)                  \
    do                                     \
    {                                      \
        iex_debugTrap();                   \
        std::stringstream _iex_throw_s;	   \
        _iex_throw_s << text;              \
        throw type (_iex_throw_s);         \
    }                                      \
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

#define APPEND_EXC(exc, text)               \
    do                                      \
    {                                       \
        std::stringstream _iex_append_s;    \
        _iex_append_s << text;              \
        exc.append (_iex_append_s);         \
    }                                       \
    while (0)

#define REPLACE_EXC(exc, text)               \
    do                                       \
    {                                        \
        std::stringstream _iex_replace_s;    \
        _iex_replace_s << text;              \
        exc.assign (_iex_replace_s);         \
    }                                        \
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

#define THROW_ERRNO(text)                                          \
    do                                                             \
    {                                                              \
        std::stringstream _iex_throw_errno_s;                      \
        _iex_throw_errno_s << text;                                \
        ::IEX_NAMESPACE::throwErrnoExc (_iex_throw_errno_s.str()); \
    }                                                              \
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
        if( bool(assertion) == false )      \
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
