//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_CHROMATICITIES_H
#define INCLUDED_IMF_CHROMATICITIES_H

//-----------------------------------------------------------------------------
//
//	CIE (x,y) chromaticities, and conversions between
//	RGB tiples and CIE XYZ tristimulus values.
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImathMatrix.h"
#include "ImathVec.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

   
struct IMF_EXPORT_TYPE Chromaticities
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
// 	   Xg / (Xg + Yg + Zg) == c.green.x
// 	   Yg / (Xg + Yg + Zg) == c.green.y
// 
// 	   Xb / (Xb + Yb + Zb) == c.blue.x
// 	   Yb / (Xb + Yb + Zb) == c.blue.y
// 
// 	   Xw / (Xw + Yw + Zw) == c.white.x
// 	   Yw / (Xw + Yw + Zw) == c.white.y
// 
// 	   Yw == Y.
// 
// XYZ to RGB:
// 
// 	XYZtoRGB(c,Y) returns RGBtoXYZ(c,Y).inverse().
// 

IMF_EXPORT IMATH_NAMESPACE::M44f    RGBtoXYZ (const Chromaticities &chroma, float Y);
IMF_EXPORT IMATH_NAMESPACE::M44f    XYZtoRGB (const Chromaticities &chroma, float Y);


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
