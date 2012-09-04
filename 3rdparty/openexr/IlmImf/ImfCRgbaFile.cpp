///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004, Industrial Light & Magic, a division of Lucas
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
//	C interface to C++ classes Imf::RgbaOutputFile and Imf::RgbaInputFile
//
//-----------------------------------------------------------------------------


#include <ImfCRgbaFile.h>
#include <ImfRgbaFile.h>
#include <ImfTiledRgbaFile.h>
#include <ImfIntAttribute.h>
#include <ImfFloatAttribute.h>
#include <ImfDoubleAttribute.h>
#include <ImfStringAttribute.h>
#include <ImfBoxAttribute.h>
#include <ImfVecAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfChannelList.h>
#include <ImfLut.h>
#include "half.h"
#include <string.h>

using Imath::Box2i;
using Imath::Box2f;
using Imath::V2i;
using Imath::V2f;
using Imath::V3i;
using Imath::V3f;
using Imath::M33f;
using Imath::M44f;


namespace {


const int MAX_ERR_LENGTH = 1024;
char errorMessage[MAX_ERR_LENGTH];


void
setErrorMessage (const std::exception &e)
{
    strncpy (errorMessage, e.what(), MAX_ERR_LENGTH - 1);
    errorMessage[MAX_ERR_LENGTH - 1] = 0;
}


inline Imf::Header *
header (ImfHeader *hdr)
{
    return (Imf::Header *)(hdr);
}


inline const Imf::Header *
header (const ImfHeader *hdr)
{
    return (const Imf::Header *)(hdr);
}


inline Imf::RgbaOutputFile *
outfile (ImfOutputFile *out)
{
    return (Imf::RgbaOutputFile *) out;
}


inline const Imf::RgbaOutputFile *
outfile (const ImfOutputFile *out)
{
    return (const Imf::RgbaOutputFile *) out;
}


inline Imf::TiledRgbaOutputFile *
outfile (ImfTiledOutputFile *out)
{
    return (Imf::TiledRgbaOutputFile *) out;
}


inline const Imf::TiledRgbaOutputFile *
outfile (const ImfTiledOutputFile *out)
{
    return (const Imf::TiledRgbaOutputFile *) out;
}


inline Imf::RgbaInputFile *
infile (ImfInputFile *in)
{
    return (Imf::RgbaInputFile *) in;
}


inline const Imf::RgbaInputFile *
infile (const ImfInputFile *in)
{
    return (const Imf::RgbaInputFile *) in;
}


inline Imf::TiledRgbaInputFile *
infile (ImfTiledInputFile *in)
{
    return (Imf::TiledRgbaInputFile *) in;
}


inline const Imf::TiledRgbaInputFile *
infile (const ImfTiledInputFile *in)
{
    return (const Imf::TiledRgbaInputFile *) in;
}


} // namespace


void	
ImfFloatToHalf (float f, ImfHalf *h)
{
    *h = half(f).bits();
}


void	
ImfFloatToHalfArray (int n, const float f[/*n*/], ImfHalf h[/*n*/])
{
    for (int i = 0; i < n; ++i)
	h[i] = half(f[i]).bits();
}


float	
ImfHalfToFloat (ImfHalf h)
{
    return float (*((half *)&h));
}


void
ImfHalfToFloatArray (int n, const ImfHalf h[/*n*/], float f[/*n*/])
{
    for (int i = 0; i < n; ++i)
	f[i] = float (*((half *)(h + i)));
}


ImfHeader *
ImfNewHeader (void)
{
    try
    {
	return (ImfHeader *) new Imf::Header;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


void	
ImfDeleteHeader (ImfHeader *hdr)
{
    delete header (hdr);
}


ImfHeader *
ImfCopyHeader (const ImfHeader *hdr)
{
    try
    {
	return (ImfHeader *) new Imf::Header (*header (hdr));
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


void	
ImfHeaderSetDisplayWindow (ImfHeader *hdr,
			   int xMin, int yMin,
			   int xMax, int yMax)
{
    header(hdr)->displayWindow() = Box2i (V2i (xMin, yMin), V2i (xMax, yMax));
}


void	
ImfHeaderDisplayWindow (const ImfHeader *hdr,
			int *xMin, int *yMin,
			int *xMax, int *yMax)
{
    const Box2i dw = header(hdr)->displayWindow();
    *xMin = dw.min.x;
    *yMin = dw.min.y;
    *xMax = dw.max.x;
    *yMax = dw.max.y;
}


void
ImfHeaderSetDataWindow (ImfHeader *hdr,
			int xMin, int yMin,
			int xMax, int yMax)
{
    header(hdr)->dataWindow() = Box2i (V2i (xMin, yMin), V2i (xMax, yMax));
}


void	
ImfHeaderDataWindow (const ImfHeader *hdr,
		     int *xMin, int *yMin,
		     int *xMax, int *yMax)
{
    const Box2i dw = header(hdr)->dataWindow();
    *xMin = dw.min.x;
    *yMin = dw.min.y;
    *xMax = dw.max.x;
    *yMax = dw.max.y;
}


void	
ImfHeaderSetPixelAspectRatio (ImfHeader *hdr, float pixelAspectRatio)
{
    header(hdr)->pixelAspectRatio() = pixelAspectRatio;
}


float	
ImfHeaderPixelAspectRatio (const ImfHeader *hdr)
{
    return header(hdr)->pixelAspectRatio();
}


void	
ImfHeaderSetScreenWindowCenter (ImfHeader *hdr, float x, float y)
{
    header(hdr)->screenWindowCenter() = V2f (x, y);
}


void	
ImfHeaderScreenWindowCenter (const ImfHeader *hdr, float *x, float *y)
{
    const V2i &swc = header(hdr)->screenWindowCenter();
    *x = swc.x;
    *y = swc.y;
}


void	
ImfHeaderSetScreenWindowWidth (ImfHeader *hdr, float width)
{
    header(hdr)->screenWindowWidth() = width;
}


float	
ImfHeaderScreenWindowWidth (const ImfHeader *hdr)
{
    return header(hdr)->screenWindowWidth();
}


void	
ImfHeaderSetLineOrder (ImfHeader *hdr, int lineOrder)
{
    header(hdr)->lineOrder() = Imf::LineOrder (lineOrder);
}


int	
ImfHeaderLineOrder (const ImfHeader *hdr)
{
    return header(hdr)->lineOrder();
}

			    
void	
ImfHeaderSetCompression (ImfHeader *hdr, int compression)
{
    header(hdr)->compression() = Imf::Compression (compression);
}


int	
ImfHeaderCompression (const ImfHeader *hdr)
{
    return header(hdr)->compression();
}


int	
ImfHeaderSetIntAttribute (ImfHeader *hdr, const char name[], int value)
{
    try
    {
	if (header(hdr)->find(name) == header(hdr)->end())
	{
	    header(hdr)->insert (name, Imf::IntAttribute (value));
	}
	else
	{
	    header(hdr)->typedAttribute<Imf::IntAttribute>(name).value() =
		value;
	}

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int	
ImfHeaderIntAttribute (const ImfHeader *hdr, const char name[], int *value)
{
    try
    {
	*value = header(hdr)->typedAttribute<Imf::IntAttribute>(name).value();
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int	
ImfHeaderSetFloatAttribute (ImfHeader *hdr, const char name[], float value)
{
    try
    {
	if (header(hdr)->find(name) == header(hdr)->end())
	{
	    header(hdr)->insert (name, Imf::FloatAttribute (value));
	}
	else
	{
	    header(hdr)->typedAttribute<Imf::FloatAttribute>(name).value() =
		value;
	}

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int	
ImfHeaderSetDoubleAttribute (ImfHeader *hdr, const char name[], double value)
{
    try
    {
	if (header(hdr)->find(name) == header(hdr)->end())
	{
	    header(hdr)->insert (name, Imf::DoubleAttribute (value));
	}
	else
	{
	    header(hdr)->typedAttribute<Imf::DoubleAttribute>(name).value() =
		value;
	}

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderFloatAttribute (const ImfHeader *hdr, const char name[], float *value)
{
    try
    {
	*value = header(hdr)->typedAttribute<Imf::FloatAttribute>(name).value();
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderDoubleAttribute (const ImfHeader *hdr,
			  const char name[],
			  double *value)
{
    try
    {
	*value = header(hdr)->
	    typedAttribute<Imf::DoubleAttribute>(name).value();

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderSetStringAttribute (ImfHeader *hdr,
			     const char name[],
			     const char value[])
{
    try
    {
	if (header(hdr)->find(name) == header(hdr)->end())
	{
	    header(hdr)->insert (name, Imf::StringAttribute (value));
	}
	else
	{
	    header(hdr)->typedAttribute<Imf::StringAttribute>(name).value() =
		value;
	}

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderStringAttribute (const ImfHeader *hdr,
			  const char name[],
			  const char **value)
{
    try
    {
	*value = header(hdr)->
	    typedAttribute<Imf::StringAttribute>(name).value().c_str();

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderSetBox2iAttribute (ImfHeader *hdr,
			    const char name[],
			    int xMin, int yMin,
			    int xMax, int yMax)
{
    try
    {
	Box2i box (V2i (xMin, yMin), V2i (xMax, yMax));

	if (header(hdr)->find(name) == header(hdr)->end())
	{
	    header(hdr)->insert (name, Imf::Box2iAttribute (box));
	}
	else
	{
	    header(hdr)->typedAttribute<Imf::Box2iAttribute>(name).value() =
		box;
	}

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderBox2iAttribute (const ImfHeader *hdr,
			 const char name[],
			 int *xMin, int *yMin,
			 int *xMax, int *yMax)
{
    try
    {
	const Box2i &box =
	    header(hdr)->typedAttribute<Imf::Box2iAttribute>(name).value();

	*xMin = box.min.x;
	*yMin = box.min.y;
	*xMax = box.max.x;
	*yMax = box.max.y;

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderSetBox2fAttribute (ImfHeader *hdr,
			    const char name[],
			    float xMin, float yMin,
			    float xMax, float yMax)
{
    try
    {
	Box2f box (V2f (xMin, yMin), V2f (xMax, yMax));

	if (header(hdr)->find(name) == header(hdr)->end())
	{
	    header(hdr)->insert (name, Imf::Box2fAttribute (box));
	}
	else
	{
	    header(hdr)->typedAttribute<Imf::Box2fAttribute>(name).value() =
		box;
	}

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderBox2fAttribute (const ImfHeader *hdr,
			 const char name[],
			 float *xMin, float *yMin,
			 float *xMax, float *yMax)
{
    try
    {
	const Box2f &box =
	    header(hdr)->typedAttribute<Imf::Box2fAttribute>(name).value();

	*xMin = box.min.x;
	*yMin = box.min.y;
	*xMax = box.max.x;
	*yMax = box.max.y;

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderSetV2iAttribute (ImfHeader *hdr,
			  const char name[],
			  int x, int y)
{
    try
    {
	V2i v (x, y);

	if (header(hdr)->find(name) == header(hdr)->end())
	    header(hdr)->insert (name, Imf::V2iAttribute (v));
	else
	    header(hdr)->typedAttribute<Imf::V2iAttribute>(name).value() = v;

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderV2iAttribute (const ImfHeader *hdr,
		       const char name[],
		       int *x, int *y)
{
    try
    {
	const V2i &v =
	    header(hdr)->typedAttribute<Imf::V2iAttribute>(name).value();

	*x = v.x;
	*y = v.y;

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int	
ImfHeaderSetV2fAttribute (ImfHeader *hdr,
			  const char name[],
			  float x, float y)
{
    try
    {
	V2f v (x, y);

	if (header(hdr)->find(name) == header(hdr)->end())
	    header(hdr)->insert (name, Imf::V2fAttribute (v));
	else
	    header(hdr)->typedAttribute<Imf::V2fAttribute>(name).value() = v;

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderV2fAttribute (const ImfHeader *hdr,
		       const char name[],
		       float *x, float *y)
{
    try
    {
	const V2f &v =
	    header(hdr)->typedAttribute<Imf::V2fAttribute>(name).value();

	*x = v.x;
	*y = v.y;

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderSetV3iAttribute (ImfHeader *hdr,
			  const char name[],
			  int x, int y, int z)
{
    try
    {
	V3i v (x, y, z);

	if (header(hdr)->find(name) == header(hdr)->end())
	    header(hdr)->insert (name, Imf::V3iAttribute (v));
	else
	    header(hdr)->typedAttribute<Imf::V3iAttribute>(name).value() = v;

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderV3iAttribute (const ImfHeader *hdr,
		       const char name[],
		       int *x, int *y, int *z)
{
    try
    {
	const V3i &v =
	    header(hdr)->typedAttribute<Imf::V3iAttribute>(name).value();

	*x = v.x;
	*y = v.y;
	*z = v.z;

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderSetV3fAttribute (ImfHeader *hdr,
			  const char name[],
			  float x, float y, float z)
{
    try
    {
	V3f v (x, y, z);

	if (header(hdr)->find(name) == header(hdr)->end())
	    header(hdr)->insert (name, Imf::V3fAttribute (v));
	else
	    header(hdr)->typedAttribute<Imf::V3fAttribute>(name).value() = v;

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderV3fAttribute (const ImfHeader *hdr,
		       const char name[],
		       float *x, float *y, float *z)
{
    try
    {
	const V3f &v =
	    header(hdr)->typedAttribute<Imf::V3fAttribute>(name).value();

	*x = v.x;
	*y = v.y;
	*z = v.z;

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderSetM33fAttribute (ImfHeader *hdr,
			   const char name[],
			   const float m[3][3])
{
    try
    {
	M33f m3 (m);

	if (header(hdr)->find(name) == header(hdr)->end())
	    header(hdr)->insert (name, Imf::M33fAttribute (m3));
	else
	    header(hdr)->typedAttribute<Imf::M33fAttribute>(name).value() = m3;

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderM33fAttribute (const ImfHeader *hdr,
			const char name[],
			float m[3][3])
{
    try
    {
	const M33f &m3 =
	    header(hdr)->typedAttribute<Imf::M33fAttribute>(name).value();

	m[0][0] = m3[0][0];
	m[0][1] = m3[0][1];
	m[0][2] = m3[0][2];

	m[1][0] = m3[1][0];
	m[1][1] = m3[1][1];
	m[1][2] = m3[1][2];

	m[2][0] = m3[2][0];
	m[2][1] = m3[2][1];
	m[2][2] = m3[2][2];

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderSetM44fAttribute (ImfHeader *hdr,
			   const char name[],
			   const float m[4][4])
{
    try
    {
	M44f m4 (m);

	if (header(hdr)->find(name) == header(hdr)->end())
	    header(hdr)->insert (name, Imf::M44fAttribute (m4));
	else
	    header(hdr)->typedAttribute<Imf::M44fAttribute>(name).value() = m4;

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfHeaderM44fAttribute (const ImfHeader *hdr,
			const char name[],
			float m[4][4])
{
    try
    {
	const M44f &m4 =
	    header(hdr)->typedAttribute<Imf::M44fAttribute>(name).value();

	m[0][0] = m4[0][0];
	m[0][1] = m4[0][1];
	m[0][2] = m4[0][2];
	m[0][3] = m4[0][3];

	m[1][0] = m4[1][0];
	m[1][1] = m4[1][1];
	m[1][2] = m4[1][2];
	m[1][3] = m4[1][3];

	m[2][0] = m4[2][0];
	m[2][1] = m4[2][1];
	m[2][2] = m4[2][2];
	m[2][3] = m4[2][3];

	m[3][0] = m4[3][0];
	m[3][1] = m4[3][1];
	m[3][2] = m4[3][2];
	m[3][3] = m4[3][3];

	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


ImfOutputFile *	
ImfOpenOutputFile (const char name[], const ImfHeader *hdr, int channels)
{
    try
    {
	return (ImfOutputFile *) new Imf::RgbaOutputFile
	    (name, *header(hdr), Imf::RgbaChannels (channels));
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int		
ImfCloseOutputFile (ImfOutputFile *out)
{
    try
    {
	delete outfile (out);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int		
ImfOutputSetFrameBuffer (ImfOutputFile *out,
			 const ImfRgba *base,
			 size_t xStride,
			 size_t yStride)
{
    try
    {
	outfile(out)->setFrameBuffer ((Imf::Rgba *)base, xStride, yStride);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int		
ImfOutputWritePixels (ImfOutputFile *out, int numScanLines)
{
    try
    {
	outfile(out)->writePixels (numScanLines);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfOutputCurrentScanLine (const ImfOutputFile *out)
{
    return outfile(out)->currentScanLine();
}


const ImfHeader *
ImfOutputHeader (const ImfOutputFile *out)
{
    return (const ImfHeader *) &outfile(out)->header();
}


int
ImfOutputChannels (const ImfOutputFile *out)
{
    return outfile(out)->channels();
}


ImfTiledOutputFile *	
ImfOpenTiledOutputFile (const char name[],
			const ImfHeader *hdr,
			int channels,
			int xSize, int ySize,
			int mode, int rmode)
{
    try
    {
	return (ImfTiledOutputFile *) new Imf::TiledRgbaOutputFile
		    (name, *header(hdr),
		     Imf::RgbaChannels (channels),
		     xSize, ySize,
		     Imf::LevelMode (mode),
		     Imf::LevelRoundingMode (rmode));
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int		
ImfCloseTiledOutputFile (ImfTiledOutputFile *out)
{
    try
    {
	delete outfile (out);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int		
ImfTiledOutputSetFrameBuffer (ImfTiledOutputFile *out,
			      const ImfRgba *base,
			      size_t xStride,
			      size_t yStride)
{
    try
    {
	outfile(out)->setFrameBuffer ((Imf::Rgba *)base, xStride, yStride);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int		
ImfTiledOutputWriteTile (ImfTiledOutputFile *out,
			 int dx, int dy,
			 int lx, int ly)
{
    try
    {
	outfile(out)->writeTile (dx, dy, lx, ly);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int		
ImfTiledOutputWriteTiles (ImfTiledOutputFile *out,
			  int dxMin, int dxMax,
                          int dyMin, int dyMax,
			  int lx, int ly)
{
    try
    {
	outfile(out)->writeTiles (dxMin, dxMax, dyMin, dyMax, lx, ly);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


const ImfHeader *
ImfTiledOutputHeader (const ImfTiledOutputFile *out)
{
    return (const ImfHeader *) &outfile(out)->header();
}


int
ImfTiledOutputChannels (const ImfTiledOutputFile *out)
{
    return outfile(out)->channels();
}


int
ImfTiledOutputTileXSize (const ImfTiledOutputFile *out)
{
    return outfile(out)->tileXSize();
}


int
ImfTiledOutputTileYSize (const ImfTiledOutputFile *out)
{
    return outfile(out)->tileYSize();
}


int
ImfTiledOutputLevelMode (const ImfTiledOutputFile *out)
{
    return outfile(out)->levelMode();
}


int
ImfTiledOutputLevelRoundingMode (const ImfTiledOutputFile *out)
{
    return outfile(out)->levelRoundingMode();
}


ImfInputFile *	
ImfOpenInputFile (const char name[])
{
    try
    {
	return (ImfInputFile *) new Imf::RgbaInputFile (name);
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfCloseInputFile (ImfInputFile *in)
{
    try
    {
	delete infile (in);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int		
ImfInputSetFrameBuffer (ImfInputFile *in,
			ImfRgba *base,
			size_t xStride,
			size_t yStride)
{
    try
    {
	infile(in)->setFrameBuffer ((Imf::Rgba *) base, xStride, yStride);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfInputReadPixels (ImfInputFile *in, int scanLine1, int scanLine2)
{
    try
    {
	infile(in)->readPixels (scanLine1, scanLine2);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


const ImfHeader *
ImfInputHeader (const ImfInputFile *in)
{
    return (const ImfHeader *) &infile(in)->header();
}


int
ImfInputChannels (const ImfInputFile *in)
{
    return infile(in)->channels();
}


const char *
ImfInputFileName (const ImfInputFile *in)
{
    return infile(in)->fileName();
}


ImfTiledInputFile *	
ImfOpenTiledInputFile (const char name[])
{
    try
    {
	return (ImfTiledInputFile *) new Imf::TiledRgbaInputFile (name);
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfCloseTiledInputFile (ImfTiledInputFile *in)
{
    try
    {
	delete infile (in);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int		
ImfTiledInputSetFrameBuffer (ImfTiledInputFile *in,
			     ImfRgba *base,
			     size_t xStride,
			     size_t yStride)
{
    try
    {
	infile(in)->setFrameBuffer ((Imf::Rgba *) base, xStride, yStride);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfTiledInputReadTile (ImfTiledInputFile *in,
		       int dx, int dy,
		       int lx, int ly)
{
    try
    {
	infile(in)->readTile (dx, dy, lx, ly);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


int
ImfTiledInputReadTiles (ImfTiledInputFile *in,
		        int dxMin, int dxMax,
                        int dyMin, int dyMax,
		        int lx, int ly)
{
    try
    {
	infile(in)->readTiles (dxMin, dxMax, dyMin, dyMax, lx, ly);
	return 1;
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


const ImfHeader *
ImfTiledInputHeader (const ImfTiledInputFile *in)
{
    return (const ImfHeader *) &infile(in)->header();
}


int
ImfTiledInputChannels (const ImfTiledInputFile *in)
{
    return infile(in)->channels();
}


const char *
ImfTiledInputFileName (const ImfTiledInputFile *in)
{
    return infile(in)->fileName();
}


int
ImfTiledInputTileXSize (const ImfTiledInputFile *in)
{
    return infile(in)->tileXSize();
}


int
ImfTiledInputTileYSize (const ImfTiledInputFile *in)
{
    return infile(in)->tileYSize();
}


int
ImfTiledInputLevelMode (const ImfTiledInputFile *in)
{
    return infile(in)->levelMode();
}


int
ImfTiledInputLevelRoundingMode (const ImfTiledInputFile *in)
{
    return infile(in)->levelRoundingMode();
}


ImfLut *
ImfNewRound12logLut (int channels)
{
    try
    {
	return (ImfLut *) new Imf::RgbaLut
	    (Imf::round12log, Imf::RgbaChannels (channels));
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


ImfLut *
ImfNewRoundNBitLut (unsigned int n, int channels)
{
    try
    {
	return (ImfLut *) new Imf::RgbaLut
	    (Imf::roundNBit (n), Imf::RgbaChannels (channels));
    }
    catch (const std::exception &e)
    {
	setErrorMessage (e);
	return 0;
    }
}


void
ImfDeleteLut (ImfLut *lut)
{
    delete (Imf::RgbaLut *) lut;
}


void
ImfApplyLut (ImfLut *lut, ImfRgba *data, int nData, int stride)
{
    ((Imf::RgbaLut *)lut)->apply ((Imf::Rgba *)data, nData, stride);
}


const char *	
ImfErrorMessage ()
{
    return errorMessage;
}
