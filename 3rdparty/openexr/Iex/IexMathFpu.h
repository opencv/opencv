//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IEXMATHFPU_H
#define INCLUDED_IEXMATHFPU_H

//------------------------------------------------------------------------
//
//	Functions to control floating point exceptions.
//
//------------------------------------------------------------------------

#include "IexExport.h"
#include "IexNamespace.h"

#include "IexMathIeeeExc.h"

IEX_INTERNAL_NAMESPACE_HEADER_ENTER


//-----------------------------------------
// setFpExceptions() defines which floating
// point exceptions cause SIGFPE signals.
//-----------------------------------------

void setFpExceptions (int when = (IEEE_OVERFLOW | IEEE_DIVZERO | IEEE_INVALID));


//----------------------------------------
// fpExceptions() tells you which floating
// point exceptions cause SIGFPE signals.
//----------------------------------------

int fpExceptions ();


//------------------------------------------
// setFpExceptionHandler() defines a handler
// that will be called when SIGFPE occurs.
//------------------------------------------

extern "C" typedef void (* FpExceptionHandler) (int type, const char explanation[]);

void setFpExceptionHandler (FpExceptionHandler handler);

// -----------------------------------------
// handleExceptionsSetInRegisters() examines
// the exception registers and calls the
// floating point exception handler if the
// bits are set.  This function exists to 
// allow trapping of exception register states
// that can get set though no SIGFPE occurs.
// -----------------------------------------

void handleExceptionsSetInRegisters();


IEX_INTERNAL_NAMESPACE_HEADER_EXIT


#endif
