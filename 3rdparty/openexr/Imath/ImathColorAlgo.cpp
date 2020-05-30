///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002-2012, Industrial Light & Magic, a division of Lucas
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


//----------------------------------------------------------------------------
//
//	Implementation of non-template items declared in ImathColorAlgo.h
//
//----------------------------------------------------------------------------

#include "ImathColorAlgo.h"

IMATH_INTERNAL_NAMESPACE_SOURCE_ENTER


Vec3<double>
hsv2rgb_d(const Vec3<double> &hsv)
{
    double hue = hsv.x;
    double sat = hsv.y;
    double val = hsv.z;

    double x = 0.0, y = 0.0, z = 0.0;
    
    if (hue == 1) hue = 0;
    else hue *= 6;

    int i = int(Math<double>::floor(hue));
    double f = hue-i;
    double p = val*(1-sat);
    double q = val*(1-(sat*f));
    double t = val*(1-(sat*(1-f)));

    switch (i) 
    {
      case 0: x = val; y = t; z = p; break;
      case 1: x = q; y = val; z = p; break;
      case 2: x = p; y = val; z = t; break;
      case 3: x = p; y = q; z = val; break;
      case 4: x = t; y = p; z = val; break;
      case 5: x = val; y = p; z = q; break;
    }

    return Vec3<double>(x,y,z);
}


Color4<double>	
hsv2rgb_d(const Color4<double> &hsv)
{
    double hue = hsv.r;
    double sat = hsv.g;
    double val = hsv.b;

    double   r = 0.0, g = 0.0, b = 0.0;
    
    if (hue == 1) hue = 0;
    else hue *= 6;

    int i = int(Math<double>::floor(hue));
    double f = hue-i;
    double p = val*(1-sat);
    double q = val*(1-(sat*f));
    double t = val*(1-(sat*(1-f)));

    switch (i) 
    {
      case 0: r = val; g = t; b = p; break;
      case 1: r = q; g = val; b = p; break;
      case 2: r = p; g = val; b = t; break;
      case 3: r = p; g = q; b = val; break;
      case 4: r = t; g = p; b = val; break;
      case 5: r = val; g = p; b = q; break;
    }

    return Color4<double>(r,g,b,hsv.a);
}



Vec3<double>
rgb2hsv_d(const Vec3<double> &c)
{
    const double &x = c.x;
    const double &y = c.y;
    const double &z = c.z;

    double max	 = (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
    double min	 = (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
    double range = max - min;
    double val	 = max;
    double sat   = 0;
    double hue   = 0;
    
    if (max != 0)   sat = range/max;
    
    if (sat != 0) 
    {
	double h;
	
	if      (x == max)	h =     (y - z) / range;
	else if (y == max)	h = 2 + (z - x) / range;
	else		h = 4 + (x - y) / range;

	hue = h/6.;
	    
	if (hue < 0.)
	    hue += 1.0;
    }
    return Vec3<double>(hue,sat,val);
}


Color4<double>
rgb2hsv_d(const Color4<double> &c)
{
    const double &r = c.r;
    const double &g = c.g;
    const double &b = c.b;

    double max	 = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
    double min	 = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);
    double range = max - min;
    double val	 = max;
    double sat   = 0;
    double hue   = 0;
    
    if (max != 0)   sat = range/max;
    
    if (sat != 0) 
    {
	double h;
	
	if      (r == max)	h =     (g - b) / range;
	else if (g == max)	h = 2 + (b - r) / range;
	else		h = 4 + (r - g) / range;

	hue = h/6.;
	    
	if (hue < 0.)
	    hue += 1.0;
    }
    return Color4<double>(hue,sat,val,c.a);
}


IMATH_INTERNAL_NAMESPACE_SOURCE_EXIT
