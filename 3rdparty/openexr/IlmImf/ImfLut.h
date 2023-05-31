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



#ifndef INCLUDED_IMF_LUT_H
#define INCLUDED_IMF_LUT_H

//-----------------------------------------------------------------------------
//
//	Lookup tables for efficient application
//	of half --> half functions to pixel data,
//	and some commonly applied functions.
//
//-----------------------------------------------------------------------------

#include "ImfRgbaFile.h"
#include "ImfFrameBuffer.h"
#include "ImathBox.h"
#include "halfFunction.h"
#include "ImfNamespace.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//
// Lookup table for individual half channels.
//

class HalfLut
{
  public:

    //------------
    // Constructor
    //------------

    template <class Function>
    HalfLut (Function f);


    //----------------------------------------------------------------------
    // Apply the table to data[0], data[stride] ... data[(nData-1) * stride]
    //----------------------------------------------------------------------

    IMF_EXPORT
    void apply (half *data,
		int nData,
		int stride = 1) const;


    //---------------------------------------------------------------
    // Apply the table to a frame buffer slice (see ImfFrameBuffer.h)
    //---------------------------------------------------------------

    IMF_EXPORT
    void apply (const Slice &data,
		const IMATH_NAMESPACE::Box2i &dataWindow) const;

  private:

    halfFunction <half>	_lut;
};


//
// Lookup table for combined RGBA data.
//

class RgbaLut
{
  public:

    //------------
    // Constructor
    //------------

    template <class Function>
    RgbaLut (Function f, RgbaChannels chn = WRITE_RGB);


    //----------------------------------------------------------------------
    // Apply the table to data[0], data[stride] ... data[(nData-1) * stride]
    //----------------------------------------------------------------------

    IMF_EXPORT
    void apply (Rgba *data,
		int nData,
		int stride = 1) const;


    //-----------------------------------------------------------------------
    // Apply the table to a frame buffer (see RgbaOutpuFile.setFrameBuffer())
    //-----------------------------------------------------------------------

    IMF_EXPORT
    void apply (Rgba *base,
		int xStride,
		int yStride,
		const IMATH_NAMESPACE::Box2i &dataWindow) const;

  private:

    halfFunction <half>	_lut;
    RgbaChannels	_chn;
};


//
// 12bit log rounding reduces data to 20 stops with 200 steps per stop.
// That makes 4000 numbers.  An extra 96 just come along for the ride.
// Zero explicitly remains zero.  The first non-zero half will map to 1
// in the 0-4095 12log space.  A nice power of two number is placed at
// the center [2000] and that number is near 0.18.
//

IMF_EXPORT 
half round12log (half x);


//
// Round to n-bit precision (n should be between 0 and 10).
// After rounding, the significand's 10-n least significant
// bits will be zero.
//

struct roundNBit
{
    roundNBit (int n): n(n) {}
    half operator () (half x) {return x.round(n);}
    int n;
};


//
// Template definitions
//


template <class Function>
HalfLut::HalfLut (Function f):
    _lut(f, -HALF_MAX, HALF_MAX, half (0),
	 half::posInf(), half::negInf(), half::qNan())
{
    // empty
}


template <class Function>
RgbaLut::RgbaLut (Function f, RgbaChannels chn):
    _lut(f, -HALF_MAX, HALF_MAX, half (0),
	 half::posInf(), half::negInf(), half::qNan()),
    _chn(chn)
{
    // empty
}


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
