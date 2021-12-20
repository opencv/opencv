//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


//-----------------------------------------------------------------------------
//
//	class ZipCompressor
//
//-----------------------------------------------------------------------------

#include "ImfZipCompressor.h"
#include "ImfCheckedArithmetic.h"
#include "Iex.h"
#include <zlib.h>
#include "ImfNamespace.h"
#include "ImfHeader.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


ZipCompressor::ZipCompressor
    (const Header &hdr,
     size_t maxScanLineSize,
     size_t numScanLines)
:
    Compressor (hdr),
    _maxScanLineSize (maxScanLineSize),
    _numScanLines (numScanLines),
    _outBuffer (0),
    _zip(maxScanLineSize, numScanLines, hdr.zipCompressionLevel())
{
    // TODO: Remove this when we can change the ABI
    (void) _maxScanLineSize;
    _outBuffer = new char[_zip.maxCompressedSize ()];
}


ZipCompressor::~ZipCompressor ()
{
    delete [] _outBuffer;
}


int
ZipCompressor::numScanLines () const
{
    return _numScanLines;
}


int
ZipCompressor::compress (const char *inPtr,
			 int inSize,
			 int minY,
			 const char *&outPtr)
{
    //
    // Special case �- empty input buffer
    //

    if (inSize == 0)
    {
	outPtr = _outBuffer;
	return 0;
    }

    int outSize = _zip.compress(inPtr, inSize, _outBuffer);

    outPtr = _outBuffer;
    return outSize;
}


int
ZipCompressor::uncompress (const char *inPtr,
			   int inSize,
			   int minY,
			   const char *&outPtr)
{
    //
    // Special case �- empty input buffer
    //

    if (inSize == 0)
    {
	outPtr = _outBuffer;
	return 0;
    }

    int outSize = _zip.uncompress(inPtr, inSize, _outBuffer);

    outPtr = _outBuffer;
    return outSize;
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
