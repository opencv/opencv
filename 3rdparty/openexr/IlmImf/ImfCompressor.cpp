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
//	class Compressor
//
//-----------------------------------------------------------------------------

#include "ImfCompressor.h"
#include "ImfRleCompressor.h"
#include "ImfZipCompressor.h"
#include "ImfPizCompressor.h"
#include "ImfPxr24Compressor.h"
#include "ImfB44Compressor.h"
#include "ImfDwaCompressor.h"
#include "ImfCheckedArithmetic.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using IMATH_NAMESPACE::Box2i;


Compressor::Compressor (const Header &hdr): _header (hdr) {}


Compressor::~Compressor () {}


Compressor::Format
Compressor::format () const
{
    return XDR;
}


int
Compressor::compressTile (const char *inPtr,
			  int inSize,
			  Box2i range,
			  const char *&outPtr)
{
    return compress (inPtr, inSize, range.min.y, outPtr);
}

             
int
Compressor::uncompressTile (const char *inPtr,
			    int inSize,
			    Box2i range,
			    const char *&outPtr)
{
    return uncompress (inPtr, inSize, range.min.y, outPtr);
}


bool	
isValidCompression (Compression c)
{
    switch (c)
    {
      case NO_COMPRESSION:
      case RLE_COMPRESSION:
      case ZIPS_COMPRESSION:
      case ZIP_COMPRESSION:
      case PIZ_COMPRESSION:
      case PXR24_COMPRESSION:
      case B44_COMPRESSION:
      case B44A_COMPRESSION:
      case DWAA_COMPRESSION:
      case DWAB_COMPRESSION:

	return true;

      default:

	return false;
    }
}

bool isValidDeepCompression(Compression c)
{
  switch(c)
  {
      case NO_COMPRESSION:
      case RLE_COMPRESSION:
      case ZIPS_COMPRESSION:
          return true;
      default :
          return false;
  }
}


Compressor *
newCompressor (Compression c, size_t maxScanLineSize, const Header &hdr)
{
    switch (c)
    {
      case RLE_COMPRESSION:

	return new RleCompressor (hdr, maxScanLineSize);

      case ZIPS_COMPRESSION:

	return new ZipCompressor (hdr, maxScanLineSize, 1);

      case ZIP_COMPRESSION:

	return new ZipCompressor (hdr, maxScanLineSize, 16);

      case PIZ_COMPRESSION:

	return new PizCompressor (hdr, maxScanLineSize, 32);

      case PXR24_COMPRESSION:

	return new Pxr24Compressor (hdr, maxScanLineSize, 16);

      case B44_COMPRESSION:

	return new B44Compressor (hdr, maxScanLineSize, 32, false);

      case B44A_COMPRESSION:

	return new B44Compressor (hdr, maxScanLineSize, 32, true);

      case DWAA_COMPRESSION:

	return new DwaCompressor (hdr, maxScanLineSize, 32, 
                               DwaCompressor::STATIC_HUFFMAN);

      case DWAB_COMPRESSION:

	return new DwaCompressor (hdr, maxScanLineSize, 256, 
                               DwaCompressor::STATIC_HUFFMAN);

      default:

	return 0;
    }
}


Compressor *
newTileCompressor (Compression c,
		   size_t tileLineSize,
		   size_t numTileLines,
		   const Header &hdr)
{
    switch (c)
    {
      case RLE_COMPRESSION:

	return new RleCompressor (hdr, uiMult (tileLineSize, numTileLines));

      case ZIPS_COMPRESSION:
      case ZIP_COMPRESSION:

	return new ZipCompressor (hdr, tileLineSize, numTileLines);

      case PIZ_COMPRESSION:

	return new PizCompressor (hdr, tileLineSize, numTileLines);

      case PXR24_COMPRESSION:

	return new Pxr24Compressor (hdr, tileLineSize, numTileLines);

      case B44_COMPRESSION:

	return new B44Compressor (hdr, tileLineSize, numTileLines, false);

      case B44A_COMPRESSION:

	return new B44Compressor (hdr, tileLineSize, numTileLines, true);

      case DWAA_COMPRESSION:
      case DWAB_COMPRESSION:

	return new DwaCompressor (hdr, tileLineSize, numTileLines, 
                               DwaCompressor::DEFLATE);

      default:

	return 0;
    }
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT

