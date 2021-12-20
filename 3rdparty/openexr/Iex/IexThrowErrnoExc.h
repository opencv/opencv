//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IEXTHROWERRNOEXC_H
#define INCLUDED_IEXTHROWERRNOEXC_H

//----------------------------------------------------------
//
//	A function which throws ExcErrno exceptions
//
//----------------------------------------------------------

#include "IexBaseExc.h"
#include "IexExport.h"

IEX_INTERNAL_NAMESPACE_HEADER_ENTER


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

IEX_EXPORT void throwErrnoExc(const std::string &txt, int errnum);
IEX_EXPORT void throwErrnoExc(const std::string &txt);
IEX_EXPORT void throwErrnoExc();

IEX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IEXTHROWERRNOEXC_H
