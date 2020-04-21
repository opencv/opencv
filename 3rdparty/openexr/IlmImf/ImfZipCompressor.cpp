///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004, Industrial Light & Magic, a division of Lucas
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
    _zip(maxScanLineSize, numScanLines)
{
    _outBuffer = new char[_zip.maxCompressedSize()];
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

