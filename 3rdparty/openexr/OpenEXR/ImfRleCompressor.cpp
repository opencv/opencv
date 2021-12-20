//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


//-----------------------------------------------------------------------------
//
//	class RleCompressor
//
//-----------------------------------------------------------------------------

#include "ImfRleCompressor.h"
#include "ImfCheckedArithmetic.h"
#include "ImfRle.h"
#include "Iex.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

RleCompressor::RleCompressor (const Header &hdr, size_t maxScanLineSize):
    Compressor (hdr),
    _maxScanLineSize (maxScanLineSize),
    _tmpBuffer (0),
    _outBuffer (0)
{
    _tmpBuffer = new char [maxScanLineSize];
    _outBuffer = new char [uiMult (maxScanLineSize, size_t (3)) / 2];
}


RleCompressor::~RleCompressor ()
{
    delete [] _tmpBuffer;
    delete [] _outBuffer;
}


int
RleCompressor::numScanLines () const
{
    //
    // This compressor compresses individual scan lines.
    //

    return 1;
}


int
RleCompressor::compress (const char *inPtr,
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

    //
    // Reorder the pixel data.
    //

    {
	char *t1 = _tmpBuffer;
	char *t2 = _tmpBuffer + (inSize + 1) / 2;
	const char *stop = inPtr + inSize;

	while (true)
	{
	    if (inPtr < stop)
		*(t1++) = *(inPtr++);
	    else
		break;

	    if (inPtr < stop)
		*(t2++) = *(inPtr++);
	    else
		break;
	}
    }

    //
    // Predictor.
    //

    {
	unsigned char *t = (unsigned char *) _tmpBuffer + 1;
	unsigned char *stop = (unsigned char *) _tmpBuffer + inSize;
	int p = t[-1];

	while (t < stop)
	{
	    int d = int (t[0]) - p + (128 + 256);
	    p = t[0];
	    t[0] = d;
	    ++t;
	}
    }

    //
    // Run-length encode the data.
    //

    outPtr = _outBuffer;
    return rleCompress (inSize, _tmpBuffer, (signed char *) _outBuffer);
}


int
RleCompressor::uncompress (const char *inPtr,
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

    //
    // Decode the run-length encoded data
    //

    int outSize;

    if (0 == (outSize = rleUncompress (inSize, _maxScanLineSize,
				       (const signed char *) inPtr,
				       _tmpBuffer)))
    {
	throw IEX_NAMESPACE::InputExc ("Data decoding (rle) failed.");
    }

    //
    // Predictor.
    //

    {
	unsigned char *t = (unsigned char *) _tmpBuffer + 1;
	unsigned char *stop = (unsigned char *) _tmpBuffer + outSize;

	while (t < stop)
	{
	    int d = int (t[-1]) + int (t[0]) - 128;
	    t[0] = d;
	    ++t;
	}
    }

    //
    // Reorder the pixel data.
    //

    {
	const char *t1 = _tmpBuffer;
	const char *t2 = _tmpBuffer + (outSize + 1) / 2;
	char *s = _outBuffer;
	char *stop = s + outSize;

	while (true)
	{
	    if (s < stop)
		*(s++) = *(t1++);
	    else
		break;

	    if (s < stop)
		*(s++) = *(t2++);
	    else
		break;
	}
    }

    outPtr = _outBuffer;
    return outSize;
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
