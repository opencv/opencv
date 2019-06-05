///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2003, Industrial Light & Magic, a division of Lucas
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


#ifndef INCLUDED_IMF_CHROMATICITIES_H
#define INCLUDED_IMF_CHROMATICITIES_H

//-----------------------------------------------------------------------------
//
//	CIE (x,y) chromaticities, and conversions between
//	RGB tiples and CIE XYZ tristimulus values.
//
//-----------------------------------------------------------------------------

#include "ImathVec.h"
#include "ImathMatrix.h"
#include "ImfNamespace.h"
#include "ImfExport.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

   
struct Chromaticities
{
    //-----------------------------------------------
    // The CIE x and y coordinates of the RGB triples
    // (1,0,0), (0,1,0), (0,0,1) and (1,1,1).
    //-----------------------------------------------

    IMATH_NAMESPACE::V2f	red;
    IMATH_NAMESPACE::V2f	green;
    IMATH_NAMESPACE::V2f	blue;
    IMATH_NAMESPACE::V2f	white;

    //--------------------------------------------
    // Default constructor produces chromaticities
    // according to Rec. ITU-R BT.709-3
    //--------------------------------------------

    IMF_EXPORT
    Chromaticities (const IMATH_NAMESPACE::V2f &red   = IMATH_NAMESPACE::V2f (0.6400f, 0.3300f),
		    const IMATH_NAMESPACE::V2f &green = IMATH_NAMESPACE::V2f (0.3000f, 0.6000f),
		    const IMATH_NAMESPACE::V2f &blue  = IMATH_NAMESPACE::V2f (0.1500f, 0.0600f),
		    const IMATH_NAMESPACE::V2f &white = IMATH_NAMESPACE::V2f (0.3127f, 0.3290f));
    
    
    //---------
    // Equality
    //---------
    
    IMF_EXPORT
    bool		operator == (const Chromaticities &v) const;    
    IMF_EXPORT
    bool		operator != (const Chromaticities &v) const;
};


//
// Conversions between RGB and CIE XYZ
//
// RGB to XYZ:
//
// 	Given a set of chromaticities, c, and the luminance, Y, of the RGB
// 	triple (1,1,1), or "white", RGBtoXYZ(c,Y) computes a matrix, M, so
// 	that multiplying an RGB value, v, with M produces an equivalent
// 	XYZ value, w.  (w == v * M)
// 
// 	If we define that
// 
// 	   (Xr, Yr, Zr) == (1, 0, 0) * M
// 	   (Xg, Yg, Zg) == (0, 1, 0) * M
// 	   (Xb, Yb, Zb) == (0, 0, 1) * M
// 	   (Xw, Yw, Zw) == (1, 1, 1) * M,
// 
// 	then the following statements are true:
// 
// 	   Xr / (Xr + Yr + Zr) == c.red.x
// 	   Yr / (Xr + Yr + Zr) == c.red.y
// 
// 	   Xg / (Xg + Yg + Zg) == c.red.x
// 	   Yg / (Xg + Yg + Zg) == c.red.y
// 
// 	   Xb / (Xb + Yb + Zb) == c.red.x
// 	   Yb / (Xb + Yb + Zb) == c.red.y
// 
// 	   Xw / (Xw + Yw + Zw) == c.red.x
// 	   Yw / (Xw + Yw + Zw) == c.red.y
// 
// 	   Yw == Y.
// 
// XYZ to RGB:
// 
// 	YYZtoRGB(c,Y) returns RGBtoXYZ(c,Y).inverse().
// 

IMF_EXPORT IMATH_NAMESPACE::M44f    RGBtoXYZ (const Chromaticities chroma, float Y);
IMF_EXPORT IMATH_NAMESPACE::M44f    XYZtoRGB (const Chromaticities chroma, float Y);


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
