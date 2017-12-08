///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IMF_COMPRESSOR_H
#define INCLUDED_IMF_COMPRESSOR_H

//-----------------------------------------------------------------------------
//
//	class Compressor
//
//-----------------------------------------------------------------------------

#include "ImfCompression.h"
#include "ImathBox.h"
#include "ImfNamespace.h"
#include "ImfExport.h"
#include "ImfForward.h"

#include <stdlib.h>


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


class IMF_EXPORT Compressor
{
  public:

    //---------------------------------------------
    // Constructor -- hdr is the header of the file
    // that will be compressed or uncompressed
    //---------------------------------------------

    Compressor (const Header &hdr);


    //-----------
    // Destructor
    //-----------

    virtual ~Compressor ();


    //----------------------------------------------
    // Maximum number of scan lines processed by
    // a single call to compress() and uncompress().
    //----------------------------------------------

    virtual int		numScanLines () const = 0;


    //--------------------------------------------
    // Format of the pixel data read and written
    // by the compress() and uncompress() methods.
    // The default implementation of format()
    // returns XDR.
    //--------------------------------------------

    enum Format
    {
	NATIVE,		// the machine's native format
	XDR		// Xdr format
    };

    virtual Format	format () const;


    //----------------------------
    // Access to the file's header
    //----------------------------

    const Header &	header () const		{return _header;}


    //-------------------------------------------------------------------------
    // Compress an array of bytes that represents the contents of up to
    // numScanLines() scan lines:
    //
    //	    inPtr		Input buffer (uncompressed data).
    //
    //	    inSize		Number of bytes in the input buffer
    //
    //	    minY		Minimum y coordinate of the scan lines to
    //				be compressed
    //
    //	    outPtr		Pointer to output buffer
    //
    //	    return value	Size of compressed data in output buffer
    //
    // Arrangement of uncompressed pixel data in the input buffer:
    //
    //	Before calling
    //
    //	        compress (buf, size, minY, ...);
    //
    //	the InputFile::writePixels() method gathers pixel data from the
    // 	frame buffer, fb, and places them in buffer buf, like this:
    //
    //  char *endOfBuf = buf;
    //
    //	for (int y = minY;
    //	     y <= min (minY + numScanLines() - 1, header().dataWindow().max.y);
    //	     ++y)
    //	{
    //	    for (ChannelList::ConstIterator c = header().channels().begin();
    //		 c != header().channels().end();
    //		 ++c)
    //	    {
    //		if (modp (y, c.channel().ySampling) != 0)
    //		    continue;
    //
    //		for (int x = header().dataWindow().min.x;
    //		     x <= header().dataWindow().max.x;
    //		     ++x)
    //		{
    //		    if (modp (x, c.channel().xSampling) != 0)
    //			continue;
    //
    //		    Xdr::write<CharPtrIO> (endOfBuf, fb.pixel (c, x, y));
    //		}
    //	    }
    //	}
    //
    //	int size = endOfBuf - buf;
    //
    //-------------------------------------------------------------------------

    virtual int		compress (const char *inPtr,
				  int inSize,
				  int minY,
				  const char *&outPtr) = 0;

    virtual int		compressTile (const char *inPtr,
				      int inSize,
				      IMATH_NAMESPACE::Box2i range,
				      const char *&outPtr);

    //-------------------------------------------------------------------------
    // Uncompress an array of bytes that has been compressed by compress():
    //
    //	    inPtr		Input buffer (compressed data).
    //
    //	    inSize		Number of bytes in the input buffer
    //
    //	    minY		Minimum y coordinate of the scan lines to
    //				be uncompressed
    //
    //	    outPtr		Pointer to output buffer
    //
    //	    return value	Size of uncompressed data in output buffer
    //
    //-------------------------------------------------------------------------

    virtual int		uncompress (const char *inPtr,
				    int inSize,
				    int minY,
				    const char *&outPtr) = 0;

    virtual int		uncompressTile (const char *inPtr,
					int inSize,
					IMATH_NAMESPACE::Box2i range,
					const char *&outPtr);

  private:

    const Header &	_header;
};


//--------------------------------------
// Test if c is a valid compression type
//--------------------------------------

IMF_EXPORT 
bool isValidCompression (Compression c);

//--------------------------------------
// Test if c is valid for deep data
//--------------------------------------

IMF_EXPORT
bool            isValidDeepCompression (Compression c);


//-----------------------------------------------------------------
// Construct a Compressor for compression type c:
//
//  maxScanLineSize	Maximum number of bytes per uncompressed
//			scan line.
//
//  header		Header of the input or output file whose
//			pixels will be compressed or uncompressed.
//			
//  return value	A pointer to a new Compressor object (it
//			is the caller's responsibility to delete
//			the object), or 0 (if c is NO_COMPRESSION).
//
//-----------------------------------------------------------------

IMF_EXPORT 
Compressor *	newCompressor (Compression c,
			       size_t maxScanLineSize,
			       const Header &hdr);


//-----------------------------------------------------------------
// Construct a Compressor for compression type c for a tiled image:
//
//  tileLineSize	Maximum number of bytes per uncompressed
//			line in a tile.
//
//  numTileLines	Maximum number of lines in a tile.
//
//  header		Header of the input or output file whose
//			pixels will be compressed or uncompressed.
//
//  return value	A pointer to a new Compressor object (it
//			is the caller's responsibility to delete
//			the object), or 0 (if c is NO_COMPRESSION).
//
//-----------------------------------------------------------------

IMF_EXPORT 
Compressor *    newTileCompressor (Compression c,
				   size_t tileLineSize,
				   size_t numTileLines,
				   const Header &hdr);


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
