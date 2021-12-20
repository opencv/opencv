//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IEXMATHIEEE_EXC_H
#define INCLUDED_IEXMATHIEEE_EXC_H


//---------------------------------------------------------------------------
//
//	Names for the loating point exceptions defined by IEEE standard 754
//
//---------------------------------------------------------------------------

#include "IexExport.h"
#include "IexNamespace.h"

IEX_INTERNAL_NAMESPACE_HEADER_ENTER


enum IEX_EXPORT_ENUM IeeeExcType
{
    IEEE_OVERFLOW  = 1,
    IEEE_UNDERFLOW = 2,
    IEEE_DIVZERO   = 4,
    IEEE_INEXACT   = 8,
    IEEE_INVALID   = 16
};


IEX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
