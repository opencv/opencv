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
//	class TiledRgbaOutputFile
//	class TiledRgbaInputFile
//
//-----------------------------------------------------------------------------

#include <ImfTiledRgbaFile.h>
#include <ImfRgbaFile.h>
#include <ImfTiledOutputFile.h>
#include <ImfTiledInputFile.h>
#include <ImfChannelList.h>
#include <ImfTileDescriptionAttribute.h>
#include <ImfStandardAttributes.h>
#include <ImfRgbaYca.h>
#include <ImfArray.h>
#include "IlmThreadMutex.h"
#include "Iex.h"

#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace std;
using namespace IMATH_NAMESPACE;
using namespace RgbaYca;
using namespace ILMTHREAD_NAMESPACE;

namespace {

void
insertChannels (Header &header,
		RgbaChannels rgbaChannels,
		const char fileName[])
{
    ChannelList ch;

    if (rgbaChannels & (WRITE_Y | WRITE_C))
    {
	if (rgbaChannels & WRITE_Y)
	{
	    ch.insert ("Y", Channel (HALF, 1, 1));
	}

	if (rgbaChannels & WRITE_C)
	{
	    THROW (IEX_NAMESPACE::ArgExc, "Cannot open file \"" << fileName << "\" "
				"for writing.  Tiled image files do not "
				"support subsampled chroma channels.");
	}
    }
    else
    {
	if (rgbaChannels & WRITE_R)
	    ch.insert ("R", Channel (HALF, 1, 1));

	if (rgbaChannels & WRITE_G)
	    ch.insert ("G", Channel (HALF, 1, 1));

	if (rgbaChannels & WRITE_B)
	    ch.insert ("B", Channel (HALF, 1, 1));
    }

    if (rgbaChannels & WRITE_A)
	ch.insert ("A", Channel (HALF, 1, 1));

    header.channels() = ch;
}


RgbaChannels
rgbaChannels (const ChannelList &ch, const string &channelNamePrefix = "")
{
    int i = 0;

    if (ch.findChannel (channelNamePrefix + "R"))
	i |= WRITE_R;

    if (ch.findChannel (channelNamePrefix + "G"))
	i |= WRITE_G;
    
    if (ch.findChannel (channelNamePrefix + "B"))
	i |= WRITE_B;

    if (ch.findChannel (channelNamePrefix + "A"))
	i |= WRITE_A;

    if (ch.findChannel (channelNamePrefix + "Y"))
	i |= WRITE_Y;

    return RgbaChannels (i);
}


string
prefixFromLayerName (const string &layerName, const Header &header)
{
    if (layerName.empty())
	return "";

    if (hasMultiView (header) && multiView(header)[0] == layerName)
	return "";

    return layerName + ".";
}


V3f
ywFromHeader (const Header &header)
{
    Chromaticities cr;

    if (hasChromaticities (header))
	cr = chromaticities (header);

    return computeYw (cr);
}

} // namespace


class TiledRgbaOutputFile::ToYa: public Mutex
{
  public:

     ToYa (TiledOutputFile &outputFile, RgbaChannels rgbaChannels);

     void	setFrameBuffer (const Rgba *base,
				size_t xStride,
				size_t yStride);

     void	writeTile (int dx, int dy, int lx, int ly);

  private:

     TiledOutputFile &	_outputFile;
     bool		_writeA;
     unsigned int	_tileXSize;
     unsigned int	_tileYSize;
     V3f		_yw;
     Array2D <Rgba>	_buf;
     const Rgba *	_fbBase;
     size_t		_fbXStride;
     size_t		_fbYStride;
};


TiledRgbaOutputFile::ToYa::ToYa (TiledOutputFile &outputFile,
				 RgbaChannels rgbaChannels)
:
    _outputFile (outputFile)
{
    _writeA = (rgbaChannels & WRITE_A)? true: false;
    
    const TileDescription &td = outputFile.header().tileDescription();

    _tileXSize = td.xSize;
    _tileYSize = td.ySize;
    _yw = ywFromHeader (_outputFile.header());
    _buf.resizeErase (_tileYSize, _tileXSize);
    _fbBase = 0;
    _fbXStride = 0;
    _fbYStride = 0;
}


void
TiledRgbaOutputFile::ToYa::setFrameBuffer (const Rgba *base,
					   size_t xStride,
					   size_t yStride)
{
    _fbBase = base;
    _fbXStride = xStride;
    _fbYStride = yStride;
}


void
TiledRgbaOutputFile::ToYa::writeTile (int dx, int dy, int lx, int ly)
{
    if (_fbBase == 0)
    {
	THROW (IEX_NAMESPACE::ArgExc, "No frame buffer was specified as the "
			    "pixel data source for image file "
			    "\"" << _outputFile.fileName() << "\".");
    }

    //
    // Copy the tile's RGBA pixels into _buf and convert
    // them to luminance/alpha format
    //

    Box2i dw = _outputFile.dataWindowForTile (dx, dy, lx, ly);
    int width = dw.max.x - dw.min.x + 1;

    for (int y = dw.min.y, y1 = 0; y <= dw.max.y; ++y, ++y1)
    {
	for (int x = dw.min.x, x1 = 0; x <= dw.max.x; ++x, ++x1)
	    _buf[y1][x1] = _fbBase[x * _fbXStride + y * _fbYStride];

	RGBAtoYCA (_yw, width, _writeA, _buf[y1], _buf[y1]);
    }

    //
    // Store the contents of _buf in the output file
    //

    FrameBuffer fb;

    fb.insert ("Y", Slice (HALF,				   // type
			   (char *) &_buf[-dw.min.y][-dw.min.x].g, // base
			   sizeof (Rgba),			   // xStride
			   sizeof (Rgba) * _tileXSize));	   // yStride

    fb.insert ("A", Slice (HALF,				   // type
			   (char *) &_buf[-dw.min.y][-dw.min.x].a, // base
			   sizeof (Rgba),			   // xStride
			   sizeof (Rgba) * _tileXSize));	   // yStride

    _outputFile.setFrameBuffer (fb);
    _outputFile.writeTile (dx, dy, lx, ly);
}


TiledRgbaOutputFile::TiledRgbaOutputFile
    (const char name[],
     const Header &header,
     RgbaChannels rgbaChannels,
     int tileXSize,
     int tileYSize,
     LevelMode mode,
     LevelRoundingMode rmode,
     int numThreads)
:
    _outputFile (0),
    _toYa (0)
{
    Header hd (header);
    insertChannels (hd, rgbaChannels, name);
    hd.setTileDescription (TileDescription (tileXSize, tileYSize, mode, rmode));
    _outputFile = new TiledOutputFile (name, hd, numThreads);

    if (rgbaChannels & WRITE_Y)
	_toYa = new ToYa (*_outputFile, rgbaChannels);
}



TiledRgbaOutputFile::TiledRgbaOutputFile
    (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os,
     const Header &header,
     RgbaChannels rgbaChannels,
     int tileXSize,
     int tileYSize,
     LevelMode mode,
     LevelRoundingMode rmode,
     int numThreads)
:
    _outputFile (0),
    _toYa (0)
{
    Header hd (header);
    insertChannels (hd, rgbaChannels, os.fileName());
    hd.setTileDescription (TileDescription (tileXSize, tileYSize, mode, rmode));
    _outputFile = new TiledOutputFile (os, hd, numThreads);

    if (rgbaChannels & WRITE_Y)
	_toYa = new ToYa (*_outputFile, rgbaChannels);
}



TiledRgbaOutputFile::TiledRgbaOutputFile
    (const char name[],
     int tileXSize,
     int tileYSize,
     LevelMode mode,
     LevelRoundingMode rmode,
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
    _outputFile (0),
    _toYa (0)
{
    Header hd (displayWindow,
	       dataWindow.isEmpty()? displayWindow: dataWindow,
	       pixelAspectRatio,
	       screenWindowCenter,
	       screenWindowWidth,
	       lineOrder,
	       compression);

    insertChannels (hd, rgbaChannels, name);
    hd.setTileDescription (TileDescription (tileXSize, tileYSize, mode, rmode));
    _outputFile = new TiledOutputFile (name, hd, numThreads);

    if (rgbaChannels & WRITE_Y)
	_toYa = new ToYa (*_outputFile, rgbaChannels);
}


TiledRgbaOutputFile::TiledRgbaOutputFile
    (const char name[],
     int width,
     int height,
     int tileXSize,
     int tileYSize,
     LevelMode mode,
     LevelRoundingMode rmode,
     RgbaChannels rgbaChannels,
     float pixelAspectRatio,
     const IMATH_NAMESPACE::V2f screenWindowCenter,
     float screenWindowWidth,
     LineOrder lineOrder,
     Compression compression,
     int numThreads)
:
    _outputFile (0),
    _toYa (0)
{
    Header hd (width,
	       height,
	       pixelAspectRatio,
	       screenWindowCenter,
	       screenWindowWidth,
	       lineOrder,
	       compression);

    insertChannels (hd, rgbaChannels, name);
    hd.setTileDescription (TileDescription (tileXSize, tileYSize, mode, rmode));
    _outputFile = new TiledOutputFile (name, hd, numThreads);

    if (rgbaChannels & WRITE_Y)
	_toYa = new ToYa (*_outputFile, rgbaChannels);
}


TiledRgbaOutputFile::~TiledRgbaOutputFile ()
{
    delete _outputFile;
    delete _toYa;
}


void
TiledRgbaOutputFile::setFrameBuffer (const Rgba *base,
				     size_t xStride,
				     size_t yStride)
{
    if (_toYa)
    {
	Lock lock (*_toYa);
	_toYa->setFrameBuffer (base, xStride, yStride);
    }
    else
    {
	size_t xs = xStride * sizeof (Rgba);
	size_t ys = yStride * sizeof (Rgba);

	FrameBuffer fb;

	fb.insert ("R", Slice (HALF, (char *) &base[0].r, xs, ys));
	fb.insert ("G", Slice (HALF, (char *) &base[0].g, xs, ys));
	fb.insert ("B", Slice (HALF, (char *) &base[0].b, xs, ys));
	fb.insert ("A", Slice (HALF, (char *) &base[0].a, xs, ys));

	_outputFile->setFrameBuffer (fb);
    }
}


const Header &
TiledRgbaOutputFile::header () const
{
    return _outputFile->header();
}


const FrameBuffer &
TiledRgbaOutputFile::frameBuffer () const
{
    return _outputFile->frameBuffer();
}


const IMATH_NAMESPACE::Box2i &
TiledRgbaOutputFile::displayWindow () const
{
    return _outputFile->header().displayWindow();
}


const IMATH_NAMESPACE::Box2i &
TiledRgbaOutputFile::dataWindow () const
{
    return _outputFile->header().dataWindow();
}


float	
TiledRgbaOutputFile::pixelAspectRatio () const
{
    return _outputFile->header().pixelAspectRatio();
}


const IMATH_NAMESPACE::V2f
TiledRgbaOutputFile::screenWindowCenter () const
{
    return _outputFile->header().screenWindowCenter();
}


float	
TiledRgbaOutputFile::screenWindowWidth () const
{
    return _outputFile->header().screenWindowWidth();
}


LineOrder
TiledRgbaOutputFile::lineOrder () const
{
    return _outputFile->header().lineOrder();
}


Compression
TiledRgbaOutputFile::compression () const
{
    return _outputFile->header().compression();
}


RgbaChannels
TiledRgbaOutputFile::channels () const
{
    return rgbaChannels (_outputFile->header().channels());
}


unsigned int
TiledRgbaOutputFile::tileXSize () const
{
     return _outputFile->tileXSize();
}


unsigned int
TiledRgbaOutputFile::tileYSize () const
{
     return _outputFile->tileYSize();
}


LevelMode
TiledRgbaOutputFile::levelMode () const
{
     return _outputFile->levelMode();
}


LevelRoundingMode
TiledRgbaOutputFile::levelRoundingMode () const
{
     return _outputFile->levelRoundingMode();
}


int
TiledRgbaOutputFile::numLevels () const
{
     return _outputFile->numLevels();
}


int
TiledRgbaOutputFile::numXLevels () const
{
     return _outputFile->numXLevels();
}


int
TiledRgbaOutputFile::numYLevels () const
{
     return _outputFile->numYLevels();
}


bool
TiledRgbaOutputFile::isValidLevel (int lx, int ly) const
{
    return _outputFile->isValidLevel (lx, ly);
}


int
TiledRgbaOutputFile::levelWidth (int lx) const
{
     return _outputFile->levelWidth (lx);
}


int
TiledRgbaOutputFile::levelHeight (int ly) const
{
     return _outputFile->levelHeight (ly);
}


int
TiledRgbaOutputFile::numXTiles (int lx) const
{
     return _outputFile->numXTiles (lx);
}


int
TiledRgbaOutputFile::numYTiles (int ly) const
{
     return _outputFile->numYTiles (ly);
}


IMATH_NAMESPACE::Box2i
TiledRgbaOutputFile::dataWindowForLevel (int l) const
{
     return _outputFile->dataWindowForLevel (l);
}


IMATH_NAMESPACE::Box2i
TiledRgbaOutputFile::dataWindowForLevel (int lx, int ly) const
{
     return _outputFile->dataWindowForLevel (lx, ly);
}


IMATH_NAMESPACE::Box2i
TiledRgbaOutputFile::dataWindowForTile (int dx, int dy, int l) const
{
     return _outputFile->dataWindowForTile (dx, dy, l);
}


IMATH_NAMESPACE::Box2i
TiledRgbaOutputFile::dataWindowForTile (int dx, int dy, int lx, int ly) const
{
     return _outputFile->dataWindowForTile (dx, dy, lx, ly);
}


void
TiledRgbaOutputFile::writeTile (int dx, int dy, int l)
{
    if (_toYa)
    {
	Lock lock (*_toYa);
	_toYa->writeTile (dx, dy, l, l);
    }
    else
    {
	 _outputFile->writeTile (dx, dy, l);
    }
}


void
TiledRgbaOutputFile::writeTile (int dx, int dy, int lx, int ly)
{
    if (_toYa)
    {
	Lock lock (*_toYa);
	_toYa->writeTile (dx, dy, lx, ly);
    }
    else
    {
	 _outputFile->writeTile (dx, dy, lx, ly);
    }
}


void	
TiledRgbaOutputFile::writeTiles
    (int dxMin, int dxMax, int dyMin, int dyMax, int lx, int ly)
{
    if (_toYa)
    {
	Lock lock (*_toYa);

        for (int dy = dyMin; dy <= dyMax; dy++)
            for (int dx = dxMin; dx <= dxMax; dx++)
	        _toYa->writeTile (dx, dy, lx, ly);
    }
    else
    {
        _outputFile->writeTiles (dxMin, dxMax, dyMin, dyMax, lx, ly);
    }
}

void	
TiledRgbaOutputFile::writeTiles
    (int dxMin, int dxMax, int dyMin, int dyMax, int l)
{
    writeTiles (dxMin, dxMax, dyMin, dyMax, l, l);
}


class TiledRgbaInputFile::FromYa: public Mutex
{
  public:

     FromYa (TiledInputFile &inputFile);

     void	setFrameBuffer (Rgba *base,
				size_t xStride,
				size_t yStride,
				const string &channelNamePrefix);

     void	readTile (int dx, int dy, int lx, int ly);

  private:

     TiledInputFile &	_inputFile;
     unsigned int	_tileXSize;
     unsigned int	_tileYSize;
     V3f		_yw;
     Array2D <Rgba>	_buf;
     Rgba *		_fbBase;
     size_t		_fbXStride;
     size_t		_fbYStride;
};


TiledRgbaInputFile::FromYa::FromYa (TiledInputFile &inputFile)
:
    _inputFile (inputFile)
{
    const TileDescription &td = inputFile.header().tileDescription();

    _tileXSize = td.xSize;
    _tileYSize = td.ySize;
    _yw = ywFromHeader (_inputFile.header());
    _buf.resizeErase (_tileYSize, _tileXSize);
    _fbBase = 0;
    _fbXStride = 0;
    _fbYStride = 0;
}


void
TiledRgbaInputFile::FromYa::setFrameBuffer (Rgba *base,
					    size_t xStride,
					    size_t yStride,
					    const string &channelNamePrefix)
{
    if (_fbBase == 0)
{
	FrameBuffer fb;

	fb.insert (channelNamePrefix + "Y",
		   Slice (HALF,				// type
			  (char *) &_buf[0][0].g,	// base
			  sizeof (Rgba),		// xStride
			  sizeof (Rgba) * _tileXSize,	// yStride
			  1, 1,				// sampling
			  0.0,				// fillValue
			  true, true));			// tileCoordinates

	fb.insert (channelNamePrefix + "A",
		   Slice (HALF,				// type
			  (char *) &_buf[0][0].a,	// base
			  sizeof (Rgba),		// xStride
			  sizeof (Rgba) * _tileXSize,	// yStride
			  1, 1,				// sampling
			  1.0,				// fillValue
			  true, true));			// tileCoordinates

	_inputFile.setFrameBuffer (fb);
    }

    _fbBase = base;
    _fbXStride = xStride;
    _fbYStride = yStride;
}


void
TiledRgbaInputFile::FromYa::readTile (int dx, int dy, int lx, int ly)
{
    if (_fbBase == 0)
    {
	THROW (IEX_NAMESPACE::ArgExc, "No frame buffer was specified as the "
			    "pixel data destination for image file "
			    "\"" << _inputFile.fileName() << "\".");
    }

    //
    // Read the tile requested by the caller into _buf.
    //
    
    _inputFile.readTile (dx, dy, lx, ly);

    //
    // Convert the luminance/alpha pixels to RGBA
    // and copy them into the caller's frame buffer.
    //

    Box2i dw = _inputFile.dataWindowForTile (dx, dy, lx, ly);
    int width = dw.max.x - dw.min.x + 1;

    for (int y = dw.min.y, y1 = 0; y <= dw.max.y; ++y, ++y1)
    {
	for (int x1 = 0; x1 < width; ++x1)
	{
	    _buf[y1][x1].r = 0;
	    _buf[y1][x1].b = 0;
	}

	YCAtoRGBA (_yw, width, _buf[y1], _buf[y1]);

	for (int x = dw.min.x, x1 = 0; x <= dw.max.x; ++x, ++x1)
	{
	    _fbBase[x * _fbXStride + y * _fbYStride] = _buf[y1][x1];
	}
    }
}


TiledRgbaInputFile::TiledRgbaInputFile (const char name[], int numThreads):
    _inputFile (new TiledInputFile (name, numThreads)),
    _fromYa (0),
    _channelNamePrefix ("")
{
    if (channels() & WRITE_Y)
	_fromYa = new FromYa (*_inputFile);
}


TiledRgbaInputFile::TiledRgbaInputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int numThreads):
    _inputFile (new TiledInputFile (is, numThreads)),
    _fromYa (0),
    _channelNamePrefix ("")
{
    if (channels() & WRITE_Y)
	_fromYa = new FromYa (*_inputFile);
}


TiledRgbaInputFile::TiledRgbaInputFile (const char name[],
					const string &layerName,
					int numThreads)
:
    _inputFile (new TiledInputFile (name, numThreads)),
    _fromYa (0),
    _channelNamePrefix (prefixFromLayerName (layerName, _inputFile->header()))
{
    if (channels() & WRITE_Y)
	_fromYa = new FromYa (*_inputFile);
}


TiledRgbaInputFile::TiledRgbaInputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,
					const string &layerName,
					int numThreads)
:
    _inputFile (new TiledInputFile (is, numThreads)),
    _fromYa (0),
    _channelNamePrefix (prefixFromLayerName (layerName, _inputFile->header()))
{
    if (channels() & WRITE_Y)
	_fromYa = new FromYa (*_inputFile);
}


TiledRgbaInputFile::~TiledRgbaInputFile ()
{
    delete _inputFile;
    delete _fromYa;
}


void	
TiledRgbaInputFile::setFrameBuffer (Rgba *base, size_t xStride, size_t yStride)
{
    if (_fromYa)
    {
	Lock lock (*_fromYa);
	_fromYa->setFrameBuffer (base, xStride, yStride, _channelNamePrefix);
    }
    else
    {
	size_t xs = xStride * sizeof (Rgba);
	size_t ys = yStride * sizeof (Rgba);

	FrameBuffer fb;

	fb.insert (_channelNamePrefix + "R",
		   Slice (HALF,
			       (char *) &base[0].r,
			       xs, ys,
			       1, 1,	// xSampling, ySampling
			       0.0));	// fillValue

	fb.insert (_channelNamePrefix + "G",
		   Slice (HALF,
			       (char *) &base[0].g,
			       xs, ys,
			       1, 1,	// xSampling, ySampling
			       0.0));	// fillValue

	fb.insert (_channelNamePrefix + "B",
		   Slice (HALF,
			       (char *) &base[0].b,
			       xs, ys,
			       1, 1,	// xSampling, ySampling
			       0.0));	// fillValue

	fb.insert (_channelNamePrefix + "A",
		   Slice (HALF,
			       (char *) &base[0].a,
			       xs, ys,
			       1, 1,	// xSampling, ySampling
			       1.0));	// fillValue

	_inputFile->setFrameBuffer (fb);
    }
}


void		
TiledRgbaInputFile::setLayerName (const std::string &layerName)
{
    delete _fromYa;
    _fromYa = 0;
    
    _channelNamePrefix = prefixFromLayerName (layerName, _inputFile->header());

    if (channels() & WRITE_Y)
	_fromYa = new FromYa (*_inputFile);

    FrameBuffer fb;
    _inputFile->setFrameBuffer (fb);
}


const Header &
TiledRgbaInputFile::header () const
{
    return _inputFile->header();
}


const char *
TiledRgbaInputFile::fileName () const
{
    return _inputFile->fileName();
}


const FrameBuffer &	
TiledRgbaInputFile::frameBuffer () const
{
    return _inputFile->frameBuffer();
}


const IMATH_NAMESPACE::Box2i &
TiledRgbaInputFile::displayWindow () const
{
    return _inputFile->header().displayWindow();
}


const IMATH_NAMESPACE::Box2i &
TiledRgbaInputFile::dataWindow () const
{
    return _inputFile->header().dataWindow();
}


float	
TiledRgbaInputFile::pixelAspectRatio () const
{
    return _inputFile->header().pixelAspectRatio();
}


const IMATH_NAMESPACE::V2f	
TiledRgbaInputFile::screenWindowCenter () const
{
    return _inputFile->header().screenWindowCenter();
}


float	
TiledRgbaInputFile::screenWindowWidth () const
{
    return _inputFile->header().screenWindowWidth();
}


LineOrder
TiledRgbaInputFile::lineOrder () const
{
    return _inputFile->header().lineOrder();
}


Compression
TiledRgbaInputFile::compression () const
{
    return _inputFile->header().compression();
}


RgbaChannels	
TiledRgbaInputFile::channels () const
{
    return rgbaChannels (_inputFile->header().channels(), _channelNamePrefix);
}


int
TiledRgbaInputFile::version () const
{
    return _inputFile->version();
}


bool
TiledRgbaInputFile::isComplete () const
{
    return _inputFile->isComplete();
}


unsigned int
TiledRgbaInputFile::tileXSize () const
{
     return _inputFile->tileXSize();
}


unsigned int
TiledRgbaInputFile::tileYSize () const
{
     return _inputFile->tileYSize();
}


LevelMode
TiledRgbaInputFile::levelMode () const
{
     return _inputFile->levelMode();
}


LevelRoundingMode
TiledRgbaInputFile::levelRoundingMode () const
{
     return _inputFile->levelRoundingMode();
}


int
TiledRgbaInputFile::numLevels () const
{
     return _inputFile->numLevels();
}


int
TiledRgbaInputFile::numXLevels () const
{
     return _inputFile->numXLevels();
}


int
TiledRgbaInputFile::numYLevels () const
{
     return _inputFile->numYLevels();
}


bool
TiledRgbaInputFile::isValidLevel (int lx, int ly) const
{
    return _inputFile->isValidLevel (lx, ly);
}


int
TiledRgbaInputFile::levelWidth (int lx) const
{
     return _inputFile->levelWidth (lx);
}


int
TiledRgbaInputFile::levelHeight (int ly) const
{
     return _inputFile->levelHeight (ly);
}


int
TiledRgbaInputFile::numXTiles (int lx) const
{
     return _inputFile->numXTiles(lx);
}


int
TiledRgbaInputFile::numYTiles (int ly) const
{
     return _inputFile->numYTiles(ly);
}


IMATH_NAMESPACE::Box2i
TiledRgbaInputFile::dataWindowForLevel (int l) const
{
     return _inputFile->dataWindowForLevel (l);
}


IMATH_NAMESPACE::Box2i
TiledRgbaInputFile::dataWindowForLevel (int lx, int ly) const
{
     return _inputFile->dataWindowForLevel (lx, ly);
}


IMATH_NAMESPACE::Box2i
TiledRgbaInputFile::dataWindowForTile (int dx, int dy, int l) const
{
     return _inputFile->dataWindowForTile (dx, dy, l);
}


IMATH_NAMESPACE::Box2i
TiledRgbaInputFile::dataWindowForTile (int dx, int dy, int lx, int ly) const
{
     return _inputFile->dataWindowForTile (dx, dy, lx, ly);
}


void
TiledRgbaInputFile::readTile (int dx, int dy, int l)
{
    if (_fromYa)
    {
	Lock lock (*_fromYa);
	_fromYa->readTile (dx, dy, l, l);
    }
    else
    {
	 _inputFile->readTile (dx, dy, l);
    }
}


void
TiledRgbaInputFile::readTile (int dx, int dy, int lx, int ly)
{
    if (_fromYa)
    {
	Lock lock (*_fromYa);
	_fromYa->readTile (dx, dy, lx, ly);
    }
    else
    {
	 _inputFile->readTile (dx, dy, lx, ly);
    }
}


void	
TiledRgbaInputFile::readTiles (int dxMin, int dxMax, int dyMin, int dyMax,
                               int lx, int ly)
{
    if (_fromYa)
    {
	Lock lock (*_fromYa);

        for (int dy = dyMin; dy <= dyMax; dy++)
            for (int dx = dxMin; dx <= dxMax; dx++)
	        _fromYa->readTile (dx, dy, lx, ly);
    }
    else
    {
        _inputFile->readTiles (dxMin, dxMax, dyMin, dyMax, lx, ly);
    }
}

void	
TiledRgbaInputFile::readTiles (int dxMin, int dxMax, int dyMin, int dyMax,
                               int l)
{
    readTiles (dxMin, dxMax, dyMin, dyMax, l, l);
}


void		
TiledRgbaOutputFile::updatePreviewImage (const PreviewRgba newPixels[])
{
    _outputFile->updatePreviewImage (newPixels);
}


void	
TiledRgbaOutputFile::breakTile  (int dx, int dy, int lx, int ly,
				 int offset, int length, char c)
{
    _outputFile->breakTile (dx, dy, lx, ly, offset, length, c);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
