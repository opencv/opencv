//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	Low-level file input and output for OpenEXR.
//
//-----------------------------------------------------------------------------

#include <ImfIO.h>
#include "Iex.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


IStream::IStream (const char fileName[]): _fileName (fileName)
{
    // empty
}


IStream::~IStream ()
{
    // empty
}


bool
IStream::isMemoryMapped () const
{
    return false;
}


char *
IStream::readMemoryMapped (int n)
{
    throw IEX_NAMESPACE::InputExc ("Attempt to perform a memory-mapped read "
			 "on a file that is not memory mapped.");
}


void
IStream::clear ()
{
    // empty
}


const char *
IStream::fileName () const
{
    return _fileName.c_str();
}


OStream::OStream (const char fileName[]): _fileName (fileName)
{
    // empty
}


OStream::~OStream ()
{
    // empty
}


const char *
OStream::fileName () const
{
    return _fileName.c_str();
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
