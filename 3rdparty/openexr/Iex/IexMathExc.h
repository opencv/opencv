//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IEXMATHEXC_H
#define INCLUDED_IEXMATHEXC_H

#include "IexBaseExc.h"

IEX_INTERNAL_NAMESPACE_HEADER_ENTER

//---------------------------------------------------------
// Exception classess which correspond to specific floating
// point exceptions.
//---------------------------------------------------------

DEFINE_EXC_EXP (IEX_EXPORT, OverflowExc,    MathExc)	// Overflow
DEFINE_EXC_EXP (IEX_EXPORT, UnderflowExc,   MathExc)	// Underflow
DEFINE_EXC_EXP (IEX_EXPORT, DivzeroExc,     MathExc)	// Division by zero
DEFINE_EXC_EXP (IEX_EXPORT, InexactExc,     MathExc)	// Inexact result
DEFINE_EXC_EXP (IEX_EXPORT, InvalidFpOpExc, MathExc)	// Invalid operation

IEX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IEXMATHEXC_H
