///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2007, Industrial Light & Magic, a division of Lucas
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
//	ACES image file I/O.
//	
//-----------------------------------------------------------------------------

#include <ImfAcesFile.h>
#include <ImfRgbaFile.h>
#include <ImfStandardAttributes.h>
#include <Iex.h>
#include <algorithm>

using namespace std;
using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


const Chromaticities &
acesChromaticities ()
{
    static const Chromaticities acesChr 
	    (V2f (0.73470,  0.26530),	// red
	     V2f (0.00000,  1.00000),	// green
	     V2f (0.00010, -0.07700),	// blue
	     V2f (0.32168,  0.33767));	// white

    return acesChr;
}


class AcesOutputFile::Data
{
  public:

     Data();
    ~Data();

    RgbaOutputFile *	rgbaFile;
};


AcesOutputFile::Data::Data ():
    rgbaFile (0)
{
    // empty
}


AcesOutputFile::Data::~Data ()
{
    delete rgbaFile;
}


namespace {

void
checkCompression (Compression compression)
{
    //
    // Not all compression methods are allowed in ACES files.
    //

    switch (compression)
    {
      case NO_COMPRESSION:
      case PIZ_COMPRESSION:
      case B44A_COMPRESSION:
	break;

      default:
	throw ArgExc ("Invalid compression type for ACES file.");
    }
}

} // namespace


AcesOutputFile::AcesOutputFile
    (const std::string &name,
     const Header &header,
     RgbaChannels rgbaChannels,
     int numThreads)
:
    _data (new Data)
{
    checkCompression (header.compression());

    Header newHeader = header;
    addChromaticities (newHeader, acesChromaticities());
    addAdoptedNeutral (newHeader, acesChromaticities().white);

    _data->rgbaFile = new RgbaOutputFile (name.c_str(),
					  newHeader,
					  rgbaChannels,
					  numThreads);

    _data->rgbaFile->setYCRounding (7, 6);
}


AcesOutputFile::AcesOutputFile
    (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os,
     const Header &header,
     RgbaChannels rgbaChannels,
     int numThreads)
:
    _data (new Data)
{
    checkCompression (header.compression());

    Header newHeader = header;
    addChromaticities (newHeader, acesChromaticities());
    addAdoptedNeutral (newHeader, acesChromaticities().white);

    _data->rgbaFile = new RgbaOutputFile (os,
					  header,
					  rgbaChannels,
					  numThreads);

    _data->rgbaFile->setYCRounding (7, 6);
}


AcesOutputFile::AcesOutputFile
    (const std::string &name,
     const IMATH_NAMESPACE::Box2i &displayWindow,
     const IMATH_NAMESPACE::Box2i &dataWindow,
     RgbaChannels rgbaChannels,
     float pixelAspectRatio,
     const IMATH_NAMESPACE::V2f screenWindowCenter,
     float screenWindowWidth,
     LineOrder lineOrder,
     Compression compression,
     int numThreads)
:
    _data (new Data)
{
    checkCompression (compression);

    Header newHeader (displayWindow,
		      dataWindow.isEmpty()? displayWindow: dataWindow,
		      pixelAspectRatio,
		      screenWindowCenter,
		      screenWindowWidth,
		      lineOrder,
		      compression);

    addChromaticities (newHeader, acesChromaticities());
    addAdoptedNeutral (newHeader, acesChromaticities().white);

    _data->rgbaFile = new RgbaOutputFile (name.c_str(),
					  newHeader,
					  rgbaChannels,
					  numThreads);

    _data->rgbaFile->setYCRounding (7, 6);
}


AcesOutputFile::AcesOutputFile
    (const std::string &name,
     int width,
     int height,
     RgbaChannels rgbaChannels,
     float pixelAspectRatio,
     const IMATH_NAMESPACE::V2f screenWindowCenter,
     float screenWindowWidth,
     LineOrder lineOrder,
     Compression compression,
     int numThreads)
:
    _data (new Data)
{
    checkCompression (compression);

    Header newHeader (width,
		      height,
		      pixelAspectRatio,
		      screenWindowCenter,
		      screenWindowWidth,
		      lineOrder,
		      compression);

    addChromaticities (newHeader, acesChromaticities());
    addAdoptedNeutral (newHeader, acesChromaticities().white);

    _data->rgbaFile = new RgbaOutputFile (name.c_str(),
					  newHeader,
					  rgbaChannels,
					  numThreads);

    _data->rgbaFile->setYCRounding (7, 6);
}


AcesOutputFile::~AcesOutputFile ()
{
    delete _data;
}


void		
AcesOutputFile::setFrameBuffer
    (const Rgba *base,
     size_t xStride,
     size_t yStride)
{
    _data->rgbaFile->setFrameBuffer (base, xStride, yStride);
}


void		
AcesOutputFile::writePixels (int numScanLines)
{
    _data->rgbaFile->writePixels (numScanLines);
}


int			
AcesOutputFile::currentScanLine () const
{
    return _data->rgbaFile->currentScanLine();
}


const Header &
AcesOutputFile::header () const
{
    return _data->rgbaFile->header();
}


const IMATH_NAMESPACE::Box2i &
AcesOutputFile::displayWindow () const
{
    return _data->rgbaFile->displayWindow();
}


const IMATH_NAMESPACE::Box2i &
AcesOutputFile::dataWindow () const
{
    return _data->rgbaFile->dataWindow();
}


float		
AcesOutputFile::pixelAspectRatio () const
{
    return _data->rgbaFile->pixelAspectRatio();
}


const IMATH_NAMESPACE::V2f
AcesOutputFile::screenWindowCenter () const
{
    return _data->rgbaFile->screenWindowCenter();
}


float		
AcesOutputFile::screenWindowWidth () const
{
    return _data->rgbaFile->screenWindowWidth();
}


LineOrder		
AcesOutputFile::lineOrder () const
{
    return _data->rgbaFile->lineOrder();
}


Compression		
AcesOutputFile::compression () const
{
    return _data->rgbaFile->compression();
}


RgbaChannels
AcesOutputFile::channels () const
{
    return _data->rgbaFile->channels();
}


void		
AcesOutputFile::updatePreviewImage (const PreviewRgba pixels[])
{
    _data->rgbaFile->updatePreviewImage (pixels);
}


class AcesInputFile::Data
{
  public:

     Data();
    ~Data();

    void		initColorConversion ();

    RgbaInputFile *	rgbaFile;

    Rgba *		fbBase;
    size_t		fbXStride;
    size_t		fbYStride;
    int			minX;
    int			maxX;

    bool		mustConvertColor;
    M44f		fileToAces;
};


AcesInputFile::Data::Data ():
    rgbaFile (0),
    fbBase (0),
    fbXStride (0),
    fbYStride (0),
    minX (0),
    maxX (0),
    mustConvertColor (false)
{
    // empty
}


AcesInputFile::Data::~Data ()
{
    delete rgbaFile;
}


void
AcesInputFile::Data::initColorConversion ()
{
    const Header &header = rgbaFile->header();

    Chromaticities fileChr;

    if (hasChromaticities (header))
	fileChr = chromaticities (header);

    V2f fileNeutral = fileChr.white;

    if (hasAdoptedNeutral (header))
	fileNeutral = adoptedNeutral (header);

    const Chromaticities acesChr = acesChromaticities();

    V2f acesNeutral = acesChr.white;

    if (fileChr.red == acesChr.red &&
	fileChr.green == acesChr.green &&
	fileChr.blue == acesChr.blue &&
	fileChr.white == acesChr.white &&
	fileNeutral == acesNeutral)
    {
	//
	// The file already contains ACES data,
	// color conversion is not necessary.

	return;
    }

    mustConvertColor = true;
    minX = header.dataWindow().min.x;
    maxX = header.dataWindow().max.x;

    //
    // Create a matrix that transforms colors from the
    // RGB space of the input file into the ACES space
    // using a color adaptation transform to move the
    // white point.
    //

    //
    // We'll need the Bradford cone primary matrix and its inverse
    //

    static const M44f bradfordCPM
	    (0.895100, -0.750200,  0.038900,  0.000000,
	     0.266400,  1.713500, -0.068500,  0.000000,
	    -0.161400,  0.036700,  1.029600,  0.000000,
	     0.000000,  0.000000,  0.000000,  1.000000);

    const static M44f inverseBradfordCPM
	    (0.986993,  0.432305, -0.008529,  0.000000,
	    -0.147054,  0.518360,  0.040043,  0.000000,
	     0.159963,  0.049291,  0.968487,  0.000000,
	     0.000000,  0.000000,  0.000000,  1.000000);

    //
    // Convert the white points of the two RGB spaces to XYZ
    //

    float fx = fileNeutral.x;
    float fy = fileNeutral.y;
    V3f fileNeutralXYZ (fx / fy, 1, (1 - fx - fy) / fy);

    float ax = acesNeutral.x;
    float ay = acesNeutral.y;
    V3f acesNeutralXYZ (ax / ay, 1, (1 - ax - ay) / ay);

    //
    // Compute the Bradford transformation matrix
    //

    V3f ratio ((acesNeutralXYZ * bradfordCPM) /
	       (fileNeutralXYZ * bradfordCPM));

    M44f ratioMat (ratio[0], 0,        0,        0,
		   0,        ratio[1], 0,        0,
		   0,        0,        ratio[2], 0,
		   0,        0,        0,        1);

    M44f bradfordTrans = bradfordCPM *
                         ratioMat *
			 inverseBradfordCPM;

    //
    // Build a combined file-RGB-to-ACES-RGB conversion matrix
    //

    fileToAces = RGBtoXYZ (fileChr, 1) * bradfordTrans * XYZtoRGB (acesChr, 1);
}


AcesInputFile::AcesInputFile (const std::string &name, int numThreads):
    _data (new Data)
{
    _data->rgbaFile = new RgbaInputFile (name.c_str(), numThreads);
    _data->initColorConversion();
}


AcesInputFile::AcesInputFile (IStream &is, int numThreads):
    _data (new Data)
{
    _data->rgbaFile = new RgbaInputFile (is, numThreads);
    _data->initColorConversion();
}


AcesInputFile::~AcesInputFile ()
{
    delete _data;
}


void		
AcesInputFile::setFrameBuffer (Rgba *base, size_t xStride, size_t yStride)
{
    _data->rgbaFile->setFrameBuffer (base, xStride, yStride);
    _data->fbBase = base;
    _data->fbXStride = xStride;
    _data->fbYStride = yStride;
}


void		
AcesInputFile::readPixels (int scanLine1, int scanLine2)
{
    //
    // Copy the pixels from the RgbaInputFile into the frame buffer.
    //

    _data->rgbaFile->readPixels (scanLine1, scanLine2);

    //
    // If the RGB space of the input file is not the same as the ACES
    // RGB space, then the pixels in the frame buffer must be transformed
    // into the ACES RGB space.
    //

    if (!_data->mustConvertColor)
	return;

    int minY = min (scanLine1, scanLine2);
    int maxY = max (scanLine1, scanLine2);

    for (int y = minY; y <= maxY; ++y)
    {
	Rgba *base = _data->fbBase +
		     _data->fbXStride * _data->minX +
		     _data->fbYStride * y;

	for (int x = _data->minX; x <= _data->maxX; ++x)
	{
	    V3f aces = V3f (base->r, base->g, base->b) * _data->fileToAces;

	    base->r = aces[0];
	    base->g = aces[1];
	    base->b = aces[2];

	    base += _data->fbXStride;
	}
    }
}


void		
AcesInputFile::readPixels (int scanLine)
{
    readPixels (scanLine, scanLine);
}


const Header &
AcesInputFile::header () const
{
    return _data->rgbaFile->header();
}


const IMATH_NAMESPACE::Box2i &
AcesInputFile::displayWindow () const
{
    return _data->rgbaFile->displayWindow();
}


const IMATH_NAMESPACE::Box2i &
AcesInputFile::dataWindow () const
{
    return _data->rgbaFile->dataWindow();
}


float
AcesInputFile::pixelAspectRatio () const
{
    return _data->rgbaFile->pixelAspectRatio();
}


const IMATH_NAMESPACE::V2f
AcesInputFile::screenWindowCenter () const
{
    return _data->rgbaFile->screenWindowCenter();
}


float
AcesInputFile::screenWindowWidth () const
{
    return _data->rgbaFile->screenWindowWidth();
}


LineOrder
AcesInputFile::lineOrder () const
{
    return _data->rgbaFile->lineOrder();
}


Compression
AcesInputFile::compression () const
{
    return _data->rgbaFile->compression();
}


RgbaChannels
AcesInputFile::channels () const
{
    return _data->rgbaFile->channels();
}


const char *  
AcesInputFile::fileName () const
{
    return _data->rgbaFile->fileName();
}


bool
AcesInputFile::isComplete () const
{
    return _data->rgbaFile->isComplete();
}


int
AcesInputFile::version () const
{
    return _data->rgbaFile->version();
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
