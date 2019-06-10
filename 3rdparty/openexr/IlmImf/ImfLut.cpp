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



//-----------------------------------------------------------------------------
//
//	Lookup tables for efficient application
//	of half --> half functions to pixel data,
//	and some commonly applied functions.
//
//-----------------------------------------------------------------------------

#include <ImfLut.h>
#include <math.h>
#include <assert.h>
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


void
HalfLut::apply (half *data, int nData, int stride) const
{
    while (nData)
    {
	*data = _lut (*data);
	data += stride;
	nData -= 1;
    }
}


void
HalfLut::apply (const Slice &data, const IMATH_NAMESPACE::Box2i &dataWindow) const
{
    assert (data.type == HALF);
    assert (dataWindow.min.x % data.xSampling == 0);
    assert (dataWindow.min.y % data.ySampling == 0);
    assert ((dataWindow.max.x - dataWindow.min.x + 1) % data.xSampling == 0);
    assert ((dataWindow.max.y - dataWindow.min.y + 1) % data.ySampling == 0);

    char *base = data.base + data.yStride *
		 (dataWindow.min.y / data.ySampling);

    for (int y = dataWindow.min.y;
	 y <= dataWindow.max.y;
	 y += data.ySampling)
    {
	char *pixel = base + data.xStride *
		      (dataWindow.min.x / data.xSampling);

	for (int x = dataWindow.min.x;
	     x <= dataWindow.max.x;
	     x += data.xSampling)
	{
	    *(half *)pixel = _lut (*(half *)pixel);
	    pixel += data.xStride;
	}

	base += data.yStride;
    }
}


void
RgbaLut::apply (Rgba *data, int nData, int stride) const
{
    while (nData)
    {
	if (_chn & WRITE_R)
	    data->r = _lut (data->r);

	if (_chn & WRITE_G)
	    data->g = _lut (data->g);

	if (_chn & WRITE_B)
	    data->b = _lut (data->b);

	if (_chn & WRITE_A)
	    data->a = _lut (data->a);

	data += stride;
	nData -= 1;
    }
}


void
RgbaLut::apply (Rgba *base,
		int xStride, int yStride,
		const IMATH_NAMESPACE::Box2i &dataWindow) const
{
    base += dataWindow.min.y * yStride;

    for (int y = dataWindow.min.y; y <= dataWindow.max.y; ++y)
    {
	Rgba *pixel = base + dataWindow.min.x * xStride;

	for (int x = dataWindow.min.x; x <= dataWindow.max.x; ++x)
	{
	    if (_chn & WRITE_R)
		pixel->r = _lut (pixel->r);

	    if (_chn & WRITE_G)
		pixel->g = _lut (pixel->g);

	    if (_chn & WRITE_B)
		pixel->b = _lut (pixel->b);

	    if (_chn & WRITE_A)
		pixel->a = _lut (pixel->a);

	    pixel += xStride;
	}

	base += yStride;
    }
}


half
round12log (half x)
{
    const float middleval = pow (2.0, -2.5);
    int int12log;

    if (x <= 0)
    {
	return 0;
    }
    else
    {
	int12log = int (2000.5 + 200.0 * log (x / middleval) / log (2.0));

	if (int12log > 4095)
	    int12log = 4095;

	if (int12log < 1)
	    int12log = 1;
    }

    return middleval * pow (2.0, (int12log - 2000.0) / 200.0);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT

