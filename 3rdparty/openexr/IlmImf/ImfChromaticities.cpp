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


//-----------------------------------------------------------------------------
//
//	CIE (x,y) chromaticities, and conversions between
//	RGB tiples and CIE XYZ tristimulus values.
//
//-----------------------------------------------------------------------------

#include <ImfChromaticities.h>
#include "ImfNamespace.h"
#include <string.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

   
Chromaticities::Chromaticities (const IMATH_NAMESPACE::V2f &red,
				const IMATH_NAMESPACE::V2f &green,
				const IMATH_NAMESPACE::V2f &blue,
				const IMATH_NAMESPACE::V2f &white)
:
    red (red),
    green (green),
    blue (blue),
    white (white)
{
    // empty
}

    
bool
Chromaticities::operator == (const Chromaticities & c) const
{
    return red == c.red && green == c.green && blue == c.blue;
}

    
bool
Chromaticities::operator != (const Chromaticities & c) const
{
    return red != c.red || green != c.green || blue != c.blue;
}
    
    
IMATH_NAMESPACE::M44f
RGBtoXYZ (const Chromaticities chroma, float Y)
{
    //
    // For an explanation of how the color conversion matrix is derived,
    // see Roy Hall, "Illumination and Color in Computer Generated Imagery",
    // Springer-Verlag, 1989, chapter 3, "Perceptual Response"; and 
    // Charles A. Poynton, "A Technical Introduction to Digital Video",
    // John Wiley & Sons, 1996, chapter 7, "Color science for video".
    //

    //
    // X and Z values of RGB value (1, 1, 1), or "white"
    //

    float X = chroma.white.x * Y / chroma.white.y;
    float Z = (1 - chroma.white.x - chroma.white.y) * Y / chroma.white.y;

    //
    // Scale factors for matrix rows
    //

    float d = chroma.red.x   * (chroma.blue.y  - chroma.green.y) +
	      chroma.blue.x  * (chroma.green.y - chroma.red.y) +
	      chroma.green.x * (chroma.red.y   - chroma.blue.y);

    float Sr = (X * (chroma.blue.y - chroma.green.y) -
	        chroma.green.x * (Y * (chroma.blue.y - 1) +
		chroma.blue.y  * (X + Z)) +
		chroma.blue.x  * (Y * (chroma.green.y - 1) +
		chroma.green.y * (X + Z))) / d;

    float Sg = (X * (chroma.red.y - chroma.blue.y) +
		chroma.red.x   * (Y * (chroma.blue.y - 1) +
		chroma.blue.y  * (X + Z)) -
		chroma.blue.x  * (Y * (chroma.red.y - 1) +
		chroma.red.y   * (X + Z))) / d;

    float Sb = (X * (chroma.green.y - chroma.red.y) -
		chroma.red.x   * (Y * (chroma.green.y - 1) +
		chroma.green.y * (X + Z)) +
		chroma.green.x * (Y * (chroma.red.y - 1) +
		chroma.red.y   * (X + Z))) / d;

    //
    // Assemble the matrix
    //

    IMATH_NAMESPACE::M44f M;

    M[0][0] = Sr * chroma.red.x;
    M[0][1] = Sr * chroma.red.y;
    M[0][2] = Sr * (1 - chroma.red.x - chroma.red.y);

    M[1][0] = Sg * chroma.green.x;
    M[1][1] = Sg * chroma.green.y;
    M[1][2] = Sg * (1 - chroma.green.x - chroma.green.y);

    M[2][0] = Sb * chroma.blue.x;
    M[2][1] = Sb * chroma.blue.y;
    M[2][2] = Sb * (1 - chroma.blue.x - chroma.blue.y);

    return M;
}


IMATH_NAMESPACE::M44f
XYZtoRGB (const Chromaticities chroma, float Y)
{
    return RGBtoXYZ (chroma, Y).inverse();
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
