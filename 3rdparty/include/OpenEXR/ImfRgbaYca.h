#ifndef INCLUDED_IMF_RGBA_YCA_H
#define INCLUDED_IMF_RGBA_YCA_H

//////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004, Industrial Light & Magic, a division of Lucasfilm
// Entertainment Company Ltd.  Portions contributed and copyright held by
// others as indicated.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above
//       copyright notice, this list of conditions and the following
//       disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided with
//       the distribution.
//
//     * Neither the name of Industrial Light & Magic nor the names of
//       any other contributors to this software may be used to endorse or
//       promote products derived from this software without specific prior
//       written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
//
//	Conversion between RGBA (red, green, blue alpha)
//	and YCA (luminance, subsampled chroma, alpha) data:
//
//	Luminance, Y, is computed as a weighted sum of R, G, and B:
//
//		Y = yw.x * R + yw.y * G + yw.z * B
//
//	Function computeYw() computes a set of RGB-to-Y weights, yw,
//	from a set of primary and white point chromaticities.
//
//	Chroma, C, consists of two components, RY and BY:
//
//		RY = (R - Y) / Y
//		BY = (B - Y) / Y
//
//	For efficiency, the x and y subsampling rates for chroma are
//	hardwired to 2, and the chroma subsampling and reconstruction
//	filters are fixed 27-pixel wide windowed sinc functions.
//
//	Starting with an image that has RGBA data for all pixels,
//
//		RGBA RGBA RGBA RGBA ... RGBA RGBA
//		RGBA RGBA RGBA RGBA ... RGBA RGBA
//		RGBA RGBA RGBA RGBA ... RGBA RGBA
//		RGBA RGBA RGBA RGBA ... RGBA RGBA
//		...
//		RGBA RGBA RGBA RGBA ... RGBA RGBA
//		RGBA RGBA RGBA RGBA ... RGBA RGBA
//
//	function RGBAtoYCA() converts the pixels to YCA format:
//
//		YCA  YCA  YCA  YCA  ... YCA  YCA
//		YCA  YCA  YCA  YCA  ... YCA  YCA
//		YCA  YCA  YCA  YCA  ... YCA  YCA
//		YCA  YCA  YCA  YCA  ... YCA  YCA
//		...
//		YCA  YCA  YCA  YCA  ... YCA  YCA
//		YCA  YCA  YCA  YCA  ... YCA  YCA
//
//	Next, decimateChomaHoriz() eliminates the chroma values from
//	the odd-numbered pixels in every scan line:
//
//		YCA  YA   YCA  YA   ... YCA  YA  
//		YCA  YA   YCA  YA   ... YCA  YA  
//		YCA  YA   YCA  YA   ... YCA  YA  
//		YCA  YA   YCA  YA   ... YCA  YA  
//		...
//		YCA  YA   YCA  YA   ... YCA  YA  
//		YCA  YA   YCA  YA   ... YCA  YA  
//
//	decimateChromaVert() eliminates all chroma values from the
//	odd-numbered scan lines:
//
//		YCA  YA   YCA  YA   ... YCA  YA  
//		YA   YA   YA   YA   ... YA   YA  
//		YCA  YA   YCA  YA   ... YCA  YA  
//		YA   YA   YA   YA   ... YA   YA  
//		...
//		YCA  YA   YCA  YA   ... YCA  YA  
//		YA   YA   YA   YA   ... YA   YA  
//
//	Finally, roundYCA() reduces the precision of the luminance
//	and chroma values so that the pixel data shrink more when
//	they are saved in a compressed file.
//
//	The output of roundYCA() can be converted back to a set
//	of RGBA pixel data that is visually very similar to the
//	original RGBA image, by calling reconstructChromaHoriz(),
//	reconstructChromaVert(), YCAtoRGBA(), and finally
//	fixSaturation().
//
//-----------------------------------------------------------------------------

#include <ImfRgba.h>
#include <ImfChromaticities.h>

namespace Imf {
namespace RgbaYca {


//
// Width of the chroma subsampling and reconstruction filters
//

static const int N = 27;
static const int N2 = N / 2;


//
// Convert a set of primary chromaticities into a set of weighting
// factors for computing a pixels's luminance, Y, from R, G and B
// 

Imath::V3f computeYw (const Chromaticities &cr);


//
// Convert an array of n RGBA pixels, rgbaIn, to YCA (luminance/chroma/alpha):
//
//	ycaOut[i].g = Y (rgbaIn[i]);
//	ycaOut[i].r = RY (rgbaIn[i]);
//	ycaOut[i].b = BY (rgbaIn[i]);
//	ycaOut[i].a = aIsValid? rgbaIn[i].a: 1
//
// yw is a set of RGB-to-Y weighting factors, as computed by computeYw().
//

void RGBAtoYCA (const Imath::V3f &yw,
		int n,
	        bool aIsValid,
		const Rgba rgbaIn[/*n*/],
		Rgba ycaOut[/*n*/]);

//
// Perform horizontal low-pass filtering and subsampling of
// the chroma channels of an array of n pixels.  In order
// to avoid indexing off the ends of the input array during
// low-pass filtering, ycaIn must have N2 extra pixels at
// both ends.  Before calling decimateChromaHoriz(), the extra
// pixels should be filled with copies of the first and last
// "real" input pixel.
//

void decimateChromaHoriz (int n,
			  const Rgba ycaIn[/*n+N-1*/],
			  Rgba ycaOut[/*n*/]);

//
// Perform vertical chroma channel low-pass filtering and subsampling.
// N scan lines of input pixels are combined into a single scan line
// of output pixels.
//

void decimateChromaVert (int n,
			 const Rgba * const ycaIn[N],
			 Rgba ycaOut[/*n*/]);

//
// Round the luminance and chroma channels of an array of YCA
// pixels that has already been filtered and subsampled.
// The signifcands of the pixels' luminance and chroma values
// are rounded to roundY and roundC bits respectively.
//

void roundYCA (int n,
	       unsigned int roundY,
	       unsigned int roundC,
	       const Rgba ycaIn[/*n*/],
	       Rgba ycaOut[/*n*/]);

//
// For a scan line that has valid chroma data only for every other pixel,
// reconstruct the missing chroma values.
//

void reconstructChromaHoriz (int n,
			     const Rgba ycaIn[/*n+N-1*/],
			     Rgba ycaOut[/*n*/]);

//
// For a scan line that has only luminance and no valid chroma data,
// reconstruct chroma from the surronding N scan lines.
//

void reconstructChromaVert (int n,
			    const Rgba * const ycaIn[N],
			    Rgba ycaOut[/*n*/]);
			 
//
// Convert an array of n YCA (luminance/chroma/alpha) pixels to RGBA.
// This function is the inverse of RGBAtoYCA().
// yw is a set of RGB-to-Y weighting factors, as computed by computeYw().
//

void YCAtoRGBA (const Imath::V3f &yw,
		int n,
		const Rgba ycaIn[/*n*/],
		Rgba rgbaOut[/*n*/]);
			 
//
// Eliminate super-saturated pixels:
//
// Converting an image from RGBA to YCA, low-pass filtering chroma,
// and converting the result back to RGBA can produce pixels with
// super-saturated colors, where one or two of the RGB components
// become zero or negative.  (The low-pass and reconstruction filters
// introduce some amount of ringing into the chroma components.
// This can lead to negative RGB values near high-contrast edges.)
//
// The fixSaturation() function finds super-saturated pixels and
// corrects them by desaturating their colors while maintaining
// their luminance.  fixSaturation() takes three adjacent input
// scan lines, rgbaIn[0], rgbaIn[1], rgbaIn[2], adjusts the
// saturation of rgbaIn[1], and stores the result in rgbaOut.
//

void fixSaturation (const Imath::V3f &yw,
		    int n,
		    const Rgba * const rgbaIn[3],
		    Rgba rgbaOut[/*n*/]);

} // namespace RgbaYca
} // namespace Imf

#endif
