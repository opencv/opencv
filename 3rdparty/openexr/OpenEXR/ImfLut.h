//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_LUT_H
#define INCLUDED_IMF_LUT_H

//-----------------------------------------------------------------------------
//
//	Lookup tables for efficient application
//	of half --> half functions to pixel data,
//	and some commonly applied functions.
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfRgbaFile.h"

#include <ImathBox.h>
#include <halfFunction.h>

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
