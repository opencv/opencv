///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2006, Industrial Light & Magic, a division of Lucas
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
//	class B44Compressor
//
//	This compressor is lossy for HALF channels; the compression rate
//	is fixed at 32/14 (approximately 2.28).  FLOAT and UINT channels
//	are not compressed; their data are preserved exactly.
//
//	Each HALF channel is split into blocks of 4 by 4 pixels.  An
//	uncompressed block occupies 32 bytes, which are re-interpreted
//	as sixteen 16-bit unsigned integers, t[0] ... t[15].  Compression
//	shrinks the block to 14 bytes.  The compressed 14-byte block
//	contains
//
//	 - t[0]
//
//	 - a 6-bit shift value
//
//	 - 15 densely packed 6-bit values, r[0] ... r[14], which are
//         computed by subtracting adjacent pixel values and right-
//	   shifting the differences according to the stored shift value.
//
//	   Differences between adjacent pixels are computed according
//	   to the following diagram:
//
//		 0 -------->  1 -------->  2 -------->  3
//               |     3            7           11
//               |
//               | 0
//               |
//               v 
//		 4 -------->  5 -------->  6 -------->  7
//               |     4            8           12
//               |
//               | 1
//               |
//               v
//		 8 -------->  9 --------> 10 --------> 11
//               |     5            9           13
//               |
//               | 2
//               |
//               v
//		12 --------> 13 --------> 14 --------> 15
//                     6           10           14
//
//	    Here
//
//               5 ---------> 6
//                     8
//
//	    means that r[8] is the difference between t[5] and t[6].
//
//	 - optionally, a 4-by-4 pixel block where all pixels have the
//	   same value can be treated as a special case, where the
//	   compressed block contains only 3 instead of 14 bytes:
//	   t[0], followed by an "impossible" 6-bit shift value and
//	   two padding bits.
//
//	This compressor can handle positive and negative pixel values.
//	NaNs and infinities are replaced with zeroes before compression.
//
//-----------------------------------------------------------------------------

#include "ImfB44Compressor.h"
#include "ImfHeader.h"
#include "ImfChannelList.h"
#include "ImfMisc.h"
#include "ImfCheckedArithmetic.h"
#include <ImathFun.h>
#include <ImathBox.h>
#include <Iex.h>
#include <ImfIO.h>
#include <ImfXdr.h>
#include <string.h>
#include <assert.h>
#include <algorithm>
#include "ImfNamespace.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


using IMATH_NAMESPACE::divp;
using IMATH_NAMESPACE::modp;
using IMATH_NAMESPACE::Box2i;
using IMATH_NAMESPACE::V2i;
using std::min;

namespace {

//
// Lookup tables for
//	y = exp (x / 8)
// and 
//	x = 8 * log (y)
//

#include "b44ExpLogTable.h"


inline void
convertFromLinear (unsigned short s[16])
{
    for (int i = 0; i < 16; ++i)
	s[i] = expTable[s[i]];
}


inline void
convertToLinear (unsigned short s[16])
{
    for (int i = 0; i < 16; ++i)
	s[i] = logTable[s[i]];
}


inline int
shiftAndRound (int x, int shift)
{
    //
    // Compute
    //
    //     y = x * pow (2, -shift),
    //
    // then round y to the nearest integer.
    // In case of a tie, where y is exactly
    // halfway between two integers, round
    // to the even one.
    //

    x <<= 1;
    int a = (1 << shift) - 1;
    shift += 1;
    int b = (x >> shift) & 1;
    return (x + a + b) >> shift;
}


int
pack (const unsigned short s[16],
      unsigned char b[14],
      bool optFlatFields,
      bool exactMax)
{
    //
    // Pack a block of 4 by 4 16-bit pixels (32 bytes) into
    // either 14 or 3 bytes.
    //

    //
    // Integers s[0] ... s[15] represent floating-point numbers
    // in what is essentially a sign-magnitude format.  Convert
    // s[0] .. s[15] into a new set of integers, t[0] ... t[15],
    // such that if t[i] is greater than t[j], the floating-point
    // number that corresponds to s[i] is always greater than
    // the floating-point number that corresponds to s[j].
    //
    // Also, replace any bit patterns that represent NaNs or
    // infinities with bit patterns that represent floating-point
    // zeroes.
    //
    //	bit pattern	floating-point		bit pattern
    //	in s[i]		value			in t[i]
    //
    //  0x7fff		NAN			0x8000
    //  0x7ffe		NAN			0x8000
    //	  ...					  ...
    //  0x7c01		NAN			0x8000
    //  0x7c00		+infinity		0x8000
    //  0x7bff		+HALF_MAX		0xfbff
    //  0x7bfe					0xfbfe
    //  0x7bfd					0xfbfd
    //	  ...					  ...
    //  0x0002		+2 * HALF_MIN		0x8002
    //  0x0001		+HALF_MIN		0x8001
    //  0x0000		+0.0			0x8000
    //  0x8000		-0.0			0x7fff
    //  0x8001		-HALF_MIN		0x7ffe
    //  0x8002		-2 * HALF_MIN		0x7ffd
    //	  ...					  ...
    //  0xfbfd					0x0f02
    //  0xfbfe					0x0401
    //  0xfbff		-HALF_MAX		0x0400
    //  0xfc00		-infinity		0x8000
    //  0xfc01		NAN			0x8000
    //	  ...					  ...
    //  0xfffe		NAN			0x8000
    //  0xffff		NAN			0x8000
    //

    unsigned short t[16];

    for (int i = 0; i < 16; ++i)
    {
	if ((s[i] & 0x7c00) == 0x7c00)
	    t[i] = 0x8000;
	else if (s[i] & 0x8000)
	    t[i] = ~s[i];
	else
	    t[i] = s[i] | 0x8000;
    }
    
    //
    // Find the maximum, tMax, of t[0] ... t[15].
    //

    unsigned short tMax = 0;

    for (int i = 0; i < 16; ++i)
	if (tMax < t[i])
	    tMax = t[i];

    //
    // Compute a set of running differences, r[0] ... r[14]:
    // Find a shift value such that after rounding off the
    // rightmost bits and shifting all differenes are between
    // -32 and +31.  Then bias the differences so that they
    // end up between 0 and 63.
    //

    int shift = -1;
    int d[16];
    int r[15];
    int rMin;
    int rMax;

    const int bias = 0x20;

    do
    {
        shift += 1;

        //
        // Compute absolute differences, d[0] ... d[15],
        // between tMax and t[0] ... t[15].
        //
        // Shift and round the absolute differences.
        //

        for (int i = 0; i < 16; ++i)
            d[i] = shiftAndRound (tMax - t[i], shift);

        //
        // Convert d[0] .. d[15] into running differences
        //

        r[ 0] = d[ 0] - d[ 4] + bias;
        r[ 1] = d[ 4] - d[ 8] + bias;
        r[ 2] = d[ 8] - d[12] + bias;

        r[ 3] = d[ 0] - d[ 1] + bias;
        r[ 4] = d[ 4] - d[ 5] + bias;
        r[ 5] = d[ 8] - d[ 9] + bias;
        r[ 6] = d[12] - d[13] + bias;

        r[ 7] = d[ 1] - d[ 2] + bias;
        r[ 8] = d[ 5] - d[ 6] + bias;
        r[ 9] = d[ 9] - d[10] + bias;
        r[10] = d[13] - d[14] + bias;

        r[11] = d[ 2] - d[ 3] + bias;
        r[12] = d[ 6] - d[ 7] + bias;
        r[13] = d[10] - d[11] + bias;
        r[14] = d[14] - d[15] + bias;

        rMin = r[0];
        rMax = r[0];

        for (int i = 1; i < 15; ++i)
        {
            if (rMin > r[i])
                rMin = r[i];

            if (rMax < r[i])
                rMax = r[i];
        }
    }
    while (rMin < 0 || rMax > 0x3f);

    if (rMin == bias && rMax == bias && optFlatFields)
    {
        //
        // Special case - all pixels have the same value.
        // We encode this in 3 instead of 14 bytes by
        // storing the value 0xfc in the third output byte,
        // which cannot occur in the 14-byte encoding.
        //

        b[0] = (t[0] >> 8);
        b[1] = (unsigned char) t[0];
        b[2] = 0xfc;

        return 3;
    }

    if (exactMax)
    {
        //
        // Adjust t[0] so that the pixel whose value is equal
        // to tMax gets represented as accurately as possible.
        //

        t[0] = tMax - (d[0] << shift);
    }

    //
    // Pack t[0], shift and r[0] ... r[14] into 14 bytes:
    //

    b[ 0] = (t[0] >> 8);
    b[ 1] = (unsigned char) t[0];

    b[ 2] = (unsigned char) ((shift << 2) | (r[ 0] >> 4));
    b[ 3] = (unsigned char) ((r[ 0] << 4) | (r[ 1] >> 2));
    b[ 4] = (unsigned char) ((r[ 1] << 6) |  r[ 2]      );

    b[ 5] = (unsigned char) ((r[ 3] << 2) | (r[ 4] >> 4));
    b[ 6] = (unsigned char) ((r[ 4] << 4) | (r[ 5] >> 2));
    b[ 7] = (unsigned char) ((r[ 5] << 6) |  r[ 6]      );

    b[ 8] = (unsigned char) ((r[ 7] << 2) | (r[ 8] >> 4));
    b[ 9] = (unsigned char) ((r[ 8] << 4) | (r[ 9] >> 2));
    b[10] = (unsigned char) ((r[ 9] << 6) |  r[10]      );

    b[11] = (unsigned char) ((r[11] << 2) | (r[12] >> 4));
    b[12] = (unsigned char) ((r[12] << 4) | (r[13] >> 2));
    b[13] = (unsigned char) ((r[13] << 6) |  r[14]      );

    return 14;
}


inline
void
unpack14 (const unsigned char b[14], unsigned short s[16])
{
    //
    // Unpack a 14-byte block into 4 by 4 16-bit pixels.
    //

    #if defined (DEBUG)
	assert (b[2] != 0xfc);
    #endif

    s[ 0] = (b[0] << 8) | b[1];

    unsigned short shift = (b[ 2] >> 2);
    unsigned short bias = (0x20 << shift);

    s[ 4] = s[ 0] + ((((b[ 2] << 4) | (b[ 3] >> 4)) & 0x3f) << shift) - bias;
    s[ 8] = s[ 4] + ((((b[ 3] << 2) | (b[ 4] >> 6)) & 0x3f) << shift) - bias;
    s[12] = s[ 8] +   ((b[ 4]                       & 0x3f) << shift) - bias;
    
    s[ 1] = s[ 0] +   ((b[ 5] >> 2)                         << shift) - bias;
    s[ 5] = s[ 4] + ((((b[ 5] << 4) | (b[ 6] >> 4)) & 0x3f) << shift) - bias;
    s[ 9] = s[ 8] + ((((b[ 6] << 2) | (b[ 7] >> 6)) & 0x3f) << shift) - bias;
    s[13] = s[12] +   ((b[ 7]                       & 0x3f) << shift) - bias;
    
    s[ 2] = s[ 1] +   ((b[ 8] >> 2)                         << shift) - bias;
    s[ 6] = s[ 5] + ((((b[ 8] << 4) | (b[ 9] >> 4)) & 0x3f) << shift) - bias;
    s[10] = s[ 9] + ((((b[ 9] << 2) | (b[10] >> 6)) & 0x3f) << shift) - bias;
    s[14] = s[13] +   ((b[10]                       & 0x3f) << shift) - bias;
    
    s[ 3] = s[ 2] +   ((b[11] >> 2)                         << shift) - bias;
    s[ 7] = s[ 6] + ((((b[11] << 4) | (b[12] >> 4)) & 0x3f) << shift) - bias;
    s[11] = s[10] + ((((b[12] << 2) | (b[13] >> 6)) & 0x3f) << shift) - bias;
    s[15] = s[14] +   ((b[13]                       & 0x3f) << shift) - bias;

    for (int i = 0; i < 16; ++i)
    {
	if (s[i] & 0x8000)
	    s[i] &= 0x7fff;
	else
	    s[i] = ~s[i];
    }
}


inline
void
unpack3 (const unsigned char b[3], unsigned short s[16])
{
    //
    // Unpack a 3-byte block into 4 by 4 identical 16-bit pixels.
    //

    #if defined (DEBUG)
	assert (b[2] == 0xfc);
    #endif

    s[0] = (b[0] << 8) | b[1];

    if (s[0] & 0x8000)
	s[0] &= 0x7fff;
    else
	s[0] = ~s[0];

    for (int i = 1; i < 16; ++i)
	s[i] = s[0];
}


void
notEnoughData ()
{
    throw IEX_NAMESPACE::InputExc ("Error decompressing data "
			 "(input data are shorter than expected).");
}


void
tooMuchData ()
{
    throw IEX_NAMESPACE::InputExc ("Error decompressing data "
			 "(input data are longer than expected).");
}

} // namespace


struct B44Compressor::ChannelData
{
    unsigned short *	start;
    unsigned short *	end;
    int			nx;
    int			ny;
    int			ys;
    PixelType		type;
    bool		pLinear;
    int			size;
};


B44Compressor::B44Compressor
    (const Header &hdr,
     size_t maxScanLineSize,
     size_t numScanLines,
     bool optFlatFields)
:
    Compressor (hdr),
    _maxScanLineSize (maxScanLineSize),
    _optFlatFields (optFlatFields),
    _format (XDR),
    _numScanLines (numScanLines),
    _tmpBuffer (0),
    _outBuffer (0),
    _numChans (0),
    _channels (hdr.channels()),
    _channelData (0)
{
    //
    // Allocate buffers for compressed an uncompressed pixel data,
    // allocate a set of ChannelData structs to help speed up the
    // compress() and uncompress() functions, below, and determine
    // if uncompressed pixel data should be in native or Xdr format.
    //

    _tmpBuffer = new unsigned short
        [checkArraySize (uiMult (maxScanLineSize, numScanLines),
                         sizeof (unsigned short))];

    const ChannelList &channels = header().channels();
    int numHalfChans = 0;

    for (ChannelList::ConstIterator c = channels.begin();
	 c != channels.end();
	 ++c)
    {
	assert (pixelTypeSize (c.channel().type) % pixelTypeSize (HALF) == 0);
	++_numChans;

	if (c.channel().type == HALF)
	    ++numHalfChans;
    }

    //
    // Compressed data may be larger than the input data
    //

    size_t padding = 12 * numHalfChans * (numScanLines + 3) / 4;

    _outBuffer = new char
        [uiAdd (uiMult (maxScanLineSize, numScanLines), padding)];

    _channelData = new ChannelData[_numChans];

    int i = 0;

    for (ChannelList::ConstIterator c = channels.begin();
	 c != channels.end();
	 ++c, ++i)
    {
	_channelData[i].ys = c.channel().ySampling;
	_channelData[i].type = c.channel().type;
	_channelData[i].pLinear = c.channel().pLinear;
	_channelData[i].size =
	    pixelTypeSize (c.channel().type) / pixelTypeSize (HALF);
    }

    const Box2i &dataWindow = hdr.dataWindow();

    _minX = dataWindow.min.x;
    _maxX = dataWindow.max.x;
    _maxY = dataWindow.max.y;

    //
    // We can support uncompressed data in the machine's native
    // format only if all image channels are of type HALF.
    //

    assert (sizeof (unsigned short) == pixelTypeSize (HALF));

    if (_numChans == numHalfChans)
	_format = NATIVE;
}


B44Compressor::~B44Compressor ()
{
    delete [] _tmpBuffer;
    delete [] _outBuffer;
    delete [] _channelData;
}


int
B44Compressor::numScanLines () const
{
    return _numScanLines;
}


Compressor::Format
B44Compressor::format () const
{
    return _format;
}


int
B44Compressor::compress (const char *inPtr,
			 int inSize,
			 int minY,
			 const char *&outPtr)
{
    return compress (inPtr,
		     inSize,
		     Box2i (V2i (_minX, minY),
			    V2i (_maxX, minY + numScanLines() - 1)),
		     outPtr);
}


int
B44Compressor::compressTile (const char *inPtr,
			     int inSize,
			     IMATH_NAMESPACE::Box2i range,
			     const char *&outPtr)
{
    return compress (inPtr, inSize, range, outPtr);
}


int
B44Compressor::uncompress (const char *inPtr,
			   int inSize,
			   int minY,
			   const char *&outPtr)
{
    return uncompress (inPtr,
		       inSize,
		       Box2i (V2i (_minX, minY),
			      V2i (_maxX, minY + numScanLines() - 1)),
		       outPtr);
}


int
B44Compressor::uncompressTile (const char *inPtr,
			       int inSize,
			       IMATH_NAMESPACE::Box2i range,
			       const char *&outPtr)
{
    return uncompress (inPtr, inSize, range, outPtr);
}


int
B44Compressor::compress (const char *inPtr,
			 int inSize,
			 IMATH_NAMESPACE::Box2i range,
			 const char *&outPtr)
{
    //
    // Compress a block of pixel data:  First copy the input pixels
    // from the input buffer into _tmpBuffer, rearranging them such
    // that blocks of 4x4 pixels of a single channel can be accessed
    // conveniently.  Then compress each 4x4 block of HALF pixel data
    // and append the result to the output buffer.  Copy UINT and
    // FLOAT data to the output buffer without compressing them.
    //

    outPtr = _outBuffer;

    if (inSize == 0)
    {
	//
	// Special case - empty input buffer.
	//

	return 0;
    }

    //
    // For each channel, detemine how many pixels are stored
    // in the input buffer, and where those pixels will be
    // placed in _tmpBuffer.
    //

    int minX = range.min.x;
    int maxX = min (range.max.x, _maxX);
    int minY = range.min.y;
    int maxY = min (range.max.y, _maxY);
    
    unsigned short *tmpBufferEnd = _tmpBuffer;
    int i = 0;

    for (ChannelList::ConstIterator c = _channels.begin();
	 c != _channels.end();
	 ++c, ++i)
    {
	ChannelData &cd = _channelData[i];

	cd.start = tmpBufferEnd;
	cd.end = cd.start;

	cd.nx = numSamples (c.channel().xSampling, minX, maxX);
	cd.ny = numSamples (c.channel().ySampling, minY, maxY);

	tmpBufferEnd += cd.nx * cd.ny * cd.size;
    }

    if (_format == XDR)
    {
	//
	// The data in the input buffer are in the machine-independent
	// Xdr format.  Copy the HALF channels into _tmpBuffer and
	// convert them back into native format for compression.
	// Copy UINT and FLOAT channels verbatim into _tmpBuffer.
	//

	for (int y = minY; y <= maxY; ++y)
	{
	    for (int i = 0; i < _numChans; ++i)
	    {
		ChannelData &cd = _channelData[i];

		if (modp (y, cd.ys) != 0)
		    continue;

		if (cd.type == HALF)
		{
		    for (int x = cd.nx; x > 0; --x)
		    {
			Xdr::read <CharPtrIO> (inPtr, *cd.end);
			++cd.end;
		    }
		}
		else
		{
		    int n = cd.nx * cd.size;
		    memcpy (cd.end, inPtr, n * sizeof (unsigned short));
		    inPtr += n * sizeof (unsigned short);
		    cd.end += n;
		}
	    }
	}
    }
    else
    {
	//
	// The input buffer contains only HALF channels, and they
	// are in native, machine-dependent format.  Copy the pixels
	// into _tmpBuffer.
	//

	for (int y = minY; y <= maxY; ++y)
	{
	    for (int i = 0; i < _numChans; ++i)
	    {
		ChannelData &cd = _channelData[i];

		#if defined (DEBUG)
		    assert (cd.type == HALF);
		#endif

		if (modp (y, cd.ys) != 0)
		    continue;

		int n = cd.nx * cd.size;
		memcpy (cd.end, inPtr, n * sizeof (unsigned short));
		inPtr  += n * sizeof (unsigned short);
		cd.end += n;
	    }
	}
    }

    //
    // The pixels for each channel have been packed into a contiguous
    // block in _tmpBuffer.  HALF channels are in native format; UINT
    // and FLOAT channels are in Xdr format.
    //

    #if defined (DEBUG)

	for (int i = 1; i < _numChans; ++i)
	    assert (_channelData[i-1].end == _channelData[i].start);

	assert (_channelData[_numChans-1].end == tmpBufferEnd);

    #endif

    //
    // For each HALF channel, split the data in _tmpBuffer into 4x4
    // pixel blocks.  Compress each block and append the compressed
    // data to the output buffer.
    //
    // UINT and FLOAT channels are copied from _tmpBuffer into the
    // output buffer without further processing.
    //

    char *outEnd = _outBuffer;

    for (int i = 0; i < _numChans; ++i)
    {
	ChannelData &cd = _channelData[i];
	
	if (cd.type != HALF)
	{
	    //
	    // UINT or FLOAT channel.
	    //

	    int n = cd.nx * cd.ny * cd.size * sizeof (unsigned short);
	    memcpy (outEnd, cd.start, n);
	    outEnd += n;

	    continue;
	}
	
	//
	// HALF channel
	//

	for (int y = 0; y < cd.ny; y += 4)
	{
	    //
	    // Copy the next 4x4 pixel block into array s.
	    // If the width, cd.nx, or the height, cd.ny, of
	    // the pixel data in _tmpBuffer is not divisible
	    // by 4, then pad the data by repeating the
	    // rightmost column and the bottom row.
	    // 

	    unsigned short *row0 = cd.start + y * cd.nx;
	    unsigned short *row1 = row0 + cd.nx;
	    unsigned short *row2 = row1 + cd.nx;
	    unsigned short *row3 = row2 + cd.nx;

	    if (y + 3 >= cd.ny)
	    {
		if (y + 1 >= cd.ny)
		    row1 = row0;

		if (y + 2 >= cd.ny)
		    row2 = row1;

		row3 = row2;
	    }

	    for (int x = 0; x < cd.nx; x += 4)
	    {
		unsigned short s[16];

		if (x + 3 >= cd.nx)
		{
		    int n = cd.nx - x;

		    for (int i = 0; i < 4; ++i)
		    {
			int j = min (i, n - 1);

			s[i +  0] = row0[j];
			s[i +  4] = row1[j];
			s[i +  8] = row2[j];
			s[i + 12] = row3[j];
		    }
		}
		else
		{
		    memcpy (&s[ 0], row0, 4 * sizeof (unsigned short));
		    memcpy (&s[ 4], row1, 4 * sizeof (unsigned short));
		    memcpy (&s[ 8], row2, 4 * sizeof (unsigned short));
		    memcpy (&s[12], row3, 4 * sizeof (unsigned short));
		}

		row0 += 4;
		row1 += 4;
		row2 += 4;
		row3 += 4;

		//
		// Compress the contents of array s and append the
		// results to the output buffer.
		//

		if (cd.pLinear)
		    convertFromLinear (s);

		outEnd += pack (s, (unsigned char *) outEnd,
				_optFlatFields, !cd.pLinear);
	    }
	}
    }

    return outEnd - _outBuffer;
}


int
B44Compressor::uncompress (const char *inPtr,
			   int inSize,
			   IMATH_NAMESPACE::Box2i range,
			   const char *&outPtr)
{
    //
    // This function is the reverse of the compress() function,
    // above.  First all pixels are moved from the input buffer
    // into _tmpBuffer.  UINT and FLOAT channels are copied
    // verbatim; HALF channels are uncompressed in blocks of
    // 4x4 pixels.  Then the pixels in _tmpBuffer are copied
    // into the output buffer and rearranged such that the data
    // for for each scan line form a contiguous block.
    //

    outPtr = _outBuffer;

    if (inSize == 0)
    {
	return 0;
    }

    int minX = range.min.x;
    int maxX = min (range.max.x, _maxX);
    int minY = range.min.y;
    int maxY = min (range.max.y, _maxY);
    
    unsigned short *tmpBufferEnd = _tmpBuffer;
    int i = 0;

    for (ChannelList::ConstIterator c = _channels.begin();
	 c != _channels.end();
	 ++c, ++i)
    {
	ChannelData &cd = _channelData[i];

	cd.start = tmpBufferEnd;
	cd.end = cd.start;

	cd.nx = numSamples (c.channel().xSampling, minX, maxX);
	cd.ny = numSamples (c.channel().ySampling, minY, maxY);

	tmpBufferEnd += cd.nx * cd.ny * cd.size;
    }

    for (int i = 0; i < _numChans; ++i)
    {
	ChannelData &cd = _channelData[i];

	if (cd.type != HALF)
	{
	    //
	    // UINT or FLOAT channel.
	    //

	    int n = cd.nx * cd.ny * cd.size * sizeof (unsigned short);

	    if (inSize < n)
		notEnoughData();

	    memcpy (cd.start, inPtr, n);
	    inPtr += n;
	    inSize -= n;

	    continue;
	}

	//
	// HALF channel
	//

	for (int y = 0; y < cd.ny; y += 4)
	{
	    unsigned short *row0 = cd.start + y * cd.nx;
	    unsigned short *row1 = row0 + cd.nx;
	    unsigned short *row2 = row1 + cd.nx;
	    unsigned short *row3 = row2 + cd.nx;

	    for (int x = 0; x < cd.nx; x += 4)
	    {
		unsigned short s[16]; 

		if (inSize < 3)
		    notEnoughData();

		if (((const unsigned char *)inPtr)[2] == 0xfc)
		{
		    unpack3 ((const unsigned char *)inPtr, s);
		    inPtr += 3;
		    inSize -= 3;
		}
		else
		{
		    if (inSize < 14)
			notEnoughData();

		    unpack14 ((const unsigned char *)inPtr, s);
		    inPtr += 14;
		    inSize -= 14;
		}

		if (cd.pLinear)
		    convertToLinear (s);

		int n = (x + 3 < cd.nx)?
			    4 * sizeof (unsigned short) :
			    (cd.nx - x) * sizeof (unsigned short);

		if (y + 3 < cd.ny)
		{
		    memcpy (row0, &s[ 0], n);
		    memcpy (row1, &s[ 4], n);
		    memcpy (row2, &s[ 8], n);
		    memcpy (row3, &s[12], n);
		}
		else
		{
		    memcpy (row0, &s[ 0], n);

		    if (y + 1 < cd.ny)
			memcpy (row1, &s[ 4], n);

		    if (y + 2 < cd.ny)
			memcpy (row2, &s[ 8], n);
		}

		row0 += 4;
		row1 += 4;
		row2 += 4;
		row3 += 4;
	    }
	}
    }

    char *outEnd = _outBuffer;

    if (_format == XDR)
    {
	for (int y = minY; y <= maxY; ++y)
	{
	    for (int i = 0; i < _numChans; ++i)
	    {
		ChannelData &cd = _channelData[i];

		if (modp (y, cd.ys) != 0)
		    continue;

		if (cd.type == HALF)
		{
		    for (int x = cd.nx; x > 0; --x)
		    {
			Xdr::write <CharPtrIO> (outEnd, *cd.end);
			++cd.end;
		    }
		}
		else
		{
		    int n = cd.nx * cd.size;
		    memcpy (outEnd, cd.end, n * sizeof (unsigned short));
		    outEnd += n * sizeof (unsigned short);
		    cd.end += n;
		}
	    }
	}
    }
    else
    {
	for (int y = minY; y <= maxY; ++y)
	{
	    for (int i = 0; i < _numChans; ++i)
	    {
		ChannelData &cd = _channelData[i];

		#if defined (DEBUG)
		    assert (cd.type == HALF);
		#endif

		if (modp (y, cd.ys) != 0)
		    continue;

		int n = cd.nx * cd.size;
		memcpy (outEnd, cd.end, n * sizeof (unsigned short));
		outEnd += n * sizeof (unsigned short);
		cd.end += n;
	    }
	}
    }

    #if defined (DEBUG)

	for (int i = 1; i < _numChans; ++i)
	    assert (_channelData[i-1].end == _channelData[i].start);

	assert (_channelData[_numChans-1].end == tmpBufferEnd);

    #endif

    if (inSize > 0)
	tooMuchData();

    outPtr = _outBuffer;
    return outEnd - _outBuffer;
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
