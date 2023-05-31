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
//	class RgbaOutputFile
//	class RgbaInputFile
//
//-----------------------------------------------------------------------------

#include <ImfRgbaFile.h>
#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImfRgbaYca.h>
#include <ImfStandardAttributes.h>
#include <ImathFun.h>
#include <IlmThreadMutex.h>
#include <Iex.h>
#include <string.h>
#include <algorithm>

#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace std;
using namespace IMATH_NAMESPACE;
using namespace RgbaYca;
using namespace ILMTHREAD_NAMESPACE;

namespace {

void
insertChannels (Header &header, RgbaChannels rgbaChannels)
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
	    ch.insert ("RY", Channel (HALF, 2, 2, true));
	    ch.insert ("BY", Channel (HALF, 2, 2, true));
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

    if (ch.findChannel (channelNamePrefix + "RY") ||
	ch.findChannel (channelNamePrefix + "BY"))
	i |= WRITE_C;

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


ptrdiff_t
cachePadding (ptrdiff_t size)
{
    //
    // Some of the buffers that are allocated by classes ToYca and
    // FromYca, below, may need to be padded to avoid cache thrashing.
    // If the difference between the buffer size and the nearest power
    // of two is less than CACHE_LINE_SIZE, then we add an appropriate
    // amount of padding.
    //
    // CACHE_LINE_SIZE must be a power of two, and it must be at
    // least as big as the true size of a cache line on the machine
    // we are running on.  (It is ok if CACHE_LINE_SIZE is larger
    // than a real cache line.)
    //

    static int LOG2_CACHE_LINE_SIZE = 8;
    static const ptrdiff_t CACHE_LINE_SIZE = (1 << LOG2_CACHE_LINE_SIZE);

    int i = LOG2_CACHE_LINE_SIZE + 2;

    while ((size >> i) > 1)
	++i;

    if (size > (1 << (i + 1)) - 64)
	return 64 + ((1 << (i + 1)) - size);

    if (size < (1 << i) + 64)
	return 64 + ((1 << i) - size);

    return 0;
}

} // namespace


class RgbaOutputFile::ToYca: public Mutex
{
  public:

     ToYca (OutputFile &outputFile, RgbaChannels rgbaChannels);
    ~ToYca ();

    void		setYCRounding (unsigned int roundY,
	    			       unsigned int roundC);

    void		setFrameBuffer (const Rgba *base,
					size_t xStride,
					size_t yStride);

    void		writePixels (int numScanLines);
    int			currentScanLine () const;

  private:

    void		padTmpBuf ();
    void		rotateBuffers ();
    void		duplicateLastBuffer ();
    void		duplicateSecondToLastBuffer ();
    void		decimateChromaVertAndWriteScanLine ();

    OutputFile &	_outputFile;
    bool		_writeY;
    bool		_writeC;
    bool		_writeA;
    int			_xMin;
    int			_width;
    int			_height;
    int			_linesConverted;
    LineOrder		_lineOrder;
    int			_currentScanLine;
    V3f			_yw;
    Rgba *		_bufBase;
    Rgba *		_buf[N];
    Rgba *		_tmpBuf;
    const Rgba *	_fbBase;
    size_t		_fbXStride;
    size_t		_fbYStride;
    int			_roundY;
    int			_roundC;
};


RgbaOutputFile::ToYca::ToYca (OutputFile &outputFile,
			      RgbaChannels rgbaChannels)
:
    _outputFile (outputFile)
{
    _writeY = (rgbaChannels & WRITE_Y)? true: false;
    _writeC = (rgbaChannels & WRITE_C)? true: false;
    _writeA = (rgbaChannels & WRITE_A)? true: false;

    const Box2i dw = _outputFile.header().dataWindow();

    _xMin = dw.min.x;
    _width  = dw.max.x - dw.min.x + 1;
    _height = dw.max.y - dw.min.y + 1;

    _linesConverted = 0;
    _lineOrder = _outputFile.header().lineOrder();
    
    if (_lineOrder == INCREASING_Y)
	_currentScanLine = dw.min.y;
    else
	_currentScanLine = dw.max.y;

    _yw = ywFromHeader (_outputFile.header());

    ptrdiff_t pad = cachePadding (_width * sizeof (Rgba)) / sizeof (Rgba);

    _bufBase = new Rgba[(_width + pad) * N];

    for (int i = 0; i < N; ++i)
	_buf[i] = _bufBase + (i * (_width + pad));

    _tmpBuf = new Rgba[_width + N - 1];

    _fbBase = 0;
    _fbXStride = 0;
    _fbYStride = 0;

    _roundY = 7;
    _roundC = 5;
}


RgbaOutputFile::ToYca::~ToYca ()
{
    delete [] _bufBase;
    delete [] _tmpBuf;
}


void
RgbaOutputFile::ToYca::setYCRounding (unsigned int roundY,
				      unsigned int roundC)
{
    _roundY = roundY;
    _roundC = roundC;
}


void
RgbaOutputFile::ToYca::setFrameBuffer (const Rgba *base,
				       size_t xStride,
				       size_t yStride)
{
    if (_fbBase == 0)
    {
	FrameBuffer fb;

	if (_writeY)
	{
	    fb.insert ("Y",
		       Slice (HALF,				// type
			      (char *) &_tmpBuf[-_xMin].g,	// base
			      sizeof (Rgba),			// xStride
			      0,				// yStride
			      1,				// xSampling
			      1));				// ySampling
	}

	if (_writeC)
	{
	    fb.insert ("RY",
		       Slice (HALF,				// type
			      (char *) &_tmpBuf[-_xMin].r,	// base
			      sizeof (Rgba) * 2,		// xStride
			      0,				// yStride
			      2,				// xSampling
			      2));				// ySampling

	    fb.insert ("BY",
		       Slice (HALF,				// type
			      (char *) &_tmpBuf[-_xMin].b,	// base
			      sizeof (Rgba) * 2,		// xStride
			      0,				// yStride
			      2,				// xSampling
			      2));				// ySampling
	}

	if (_writeA)
	{
	    fb.insert ("A",
		       Slice (HALF,				// type
			      (char *) &_tmpBuf[-_xMin].a,	// base
			      sizeof (Rgba),			// xStride
			      0,				// yStride
			      1,				// xSampling
			      1));				// ySampling
	}

	_outputFile.setFrameBuffer (fb);
    }

    _fbBase = base;
    _fbXStride = xStride;
    _fbYStride = yStride;
}


void
RgbaOutputFile::ToYca::writePixels (int numScanLines)
{
    if (_fbBase == 0)
    {
	THROW (IEX_NAMESPACE::ArgExc, "No frame buffer was specified as the "
			    "pixel data source for image file "
			    "\"" << _outputFile.fileName() << "\".");
    }

    if (_writeY && !_writeC)
    {
	//
	// We are writing only luminance; filtering
	// and subsampling are not necessary.
	//

	for (int i = 0; i < numScanLines; ++i)
	{
	    //
	    // Copy the next scan line from the caller's
	    // frame buffer into _tmpBuf.
	    //

	    for (int j = 0; j < _width; ++j)
	    {
		_tmpBuf[j] = _fbBase[_fbYStride * _currentScanLine +
				     _fbXStride * (j + _xMin)];
	    }

	    //
	    // Convert the scan line from RGB to luminance/chroma,
	    // and store the result in the output file.
	    //

	    RGBAtoYCA (_yw, _width, _writeA, _tmpBuf, _tmpBuf);
	    _outputFile.writePixels (1);

	    ++_linesConverted;

	    if (_lineOrder == INCREASING_Y)
		++_currentScanLine;
	    else
		--_currentScanLine;
	}
    }
    else
    {
	//
	// We are writing chroma; the pixels must be filtered and subsampled.
	//

	for (int i = 0; i < numScanLines; ++i)
	{
	    //
	    // Copy the next scan line from the caller's
	    // frame buffer into _tmpBuf.
	    //

	    for (int j = 0; j < _width; ++j)
	    {
		_tmpBuf[j + N2] = _fbBase[_fbYStride * _currentScanLine +
					  _fbXStride * (j + _xMin)];
	    }

	    //
	    // Convert the scan line from RGB to luminance/chroma.
	    //

	    RGBAtoYCA (_yw, _width, _writeA, _tmpBuf + N2, _tmpBuf + N2);

	    //
	    // Append N2 copies of the first and last pixel to the
	    // beginning and end of the scan line.
	    //

	    padTmpBuf ();

	    //
	    // Filter and subsample the scan line's chroma channels
	    // horizontally; store the result in _buf.
	    //

	    rotateBuffers();
	    decimateChromaHoriz (_width, _tmpBuf, _buf[N - 1]);

	    //
	    // If this is the first scan line in the image,
	    // store N2 more copies of the scan line in _buf.
	    //

	    if (_linesConverted == 0)
	    {
		for (int j = 0; j < N2; ++j)
		    duplicateLastBuffer();
	    }

	    ++_linesConverted;

	    //
	    // If we have have converted at least N2 scan lines from
	    // RGBA to luminance/chroma, then we can start to filter
	    // and subsample vertically, and store pixels in the
	    // output file.
	    //

	    if (_linesConverted > N2)
		decimateChromaVertAndWriteScanLine();

	    //
	    // If we have already converted the last scan line in
	    // the image to luminance/chroma, filter, subsample and
	    // store the remaining scan lines in _buf.
	    //

	    if (_linesConverted >= _height)
	    {
		for (int j = 0; j < N2 - _height; ++j)
		    duplicateLastBuffer();

		duplicateSecondToLastBuffer();
		++_linesConverted;
		decimateChromaVertAndWriteScanLine();

		for (int j = 1; j < min (_height, N2); ++j)
		{
		    duplicateLastBuffer();
		    ++_linesConverted;
		    decimateChromaVertAndWriteScanLine();
		}
	    }

	    if (_lineOrder == INCREASING_Y)
		++_currentScanLine;
	    else
		--_currentScanLine;
	}
    }
}


int
RgbaOutputFile::ToYca::currentScanLine () const
{
    return _currentScanLine;
}


void
RgbaOutputFile::ToYca::padTmpBuf ()
{
    for (int i = 0; i < N2; ++i)
    {
	_tmpBuf[i] = _tmpBuf[N2];
	_tmpBuf[_width + N2 + i] = _tmpBuf[_width + N2 - 2];
    }
}


void
RgbaOutputFile::ToYca::rotateBuffers ()
{
    Rgba *tmp = _buf[0];

    for (int i = 0; i < N - 1; ++i)
	_buf[i] = _buf[i + 1];

    _buf[N - 1] = tmp;
}


void
RgbaOutputFile::ToYca::duplicateLastBuffer ()
{
    rotateBuffers();
    memcpy (_buf[N - 1], _buf[N - 2], _width * sizeof (Rgba));
}


void
RgbaOutputFile::ToYca::duplicateSecondToLastBuffer ()
{
    rotateBuffers();
    memcpy (_buf[N - 1], _buf[N - 3], _width * sizeof (Rgba));
}


void
RgbaOutputFile::ToYca::decimateChromaVertAndWriteScanLine ()
{
    if (_linesConverted & 1)
	memcpy (_tmpBuf, _buf[N2], _width * sizeof (Rgba));
    else
	decimateChromaVert (_width, _buf, _tmpBuf);

    if (_writeY && _writeC)
	roundYCA (_width, _roundY, _roundC, _tmpBuf, _tmpBuf);

    _outputFile.writePixels (1);
}


RgbaOutputFile::RgbaOutputFile (const char name[],
				const Header &header,
				RgbaChannels rgbaChannels,
                                int numThreads):
    _outputFile (0),
    _toYca (0)
{
    Header hd (header);
    insertChannels (hd, rgbaChannels);
    _outputFile = new OutputFile (name, hd, numThreads);

    if (rgbaChannels & (WRITE_Y | WRITE_C))
	_toYca = new ToYca (*_outputFile, rgbaChannels);
}


RgbaOutputFile::RgbaOutputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os,
				const Header &header,
				RgbaChannels rgbaChannels,
                                int numThreads):
    _outputFile (0),
    _toYca (0)
{
    Header hd (header);
    insertChannels (hd, rgbaChannels);
    _outputFile = new OutputFile (os, hd, numThreads);

    if (rgbaChannels & (WRITE_Y | WRITE_C))
	_toYca = new ToYca (*_outputFile, rgbaChannels);
}


RgbaOutputFile::RgbaOutputFile (const char name[],
				const IMATH_NAMESPACE::Box2i &displayWindow,
				const IMATH_NAMESPACE::Box2i &dataWindow,
				RgbaChannels rgbaChannels,
				float pixelAspectRatio,
				const IMATH_NAMESPACE::V2f screenWindowCenter,
				float screenWindowWidth,
				LineOrder lineOrder,
				Compression compression,
                                int numThreads):
    _outputFile (0),
    _toYca (0)
{
    Header hd (displayWindow,
	       dataWindow.isEmpty()? displayWindow: dataWindow,
	       pixelAspectRatio,
	       screenWindowCenter,
	       screenWindowWidth,
	       lineOrder,
	       compression);

    insertChannels (hd, rgbaChannels);
    _outputFile = new OutputFile (name, hd, numThreads);

    if (rgbaChannels & (WRITE_Y | WRITE_C))
	_toYca = new ToYca (*_outputFile, rgbaChannels);
}


RgbaOutputFile::RgbaOutputFile (const char name[],
				int width,
				int height,
				RgbaChannels rgbaChannels,
				float pixelAspectRatio,
				const IMATH_NAMESPACE::V2f screenWindowCenter,
				float screenWindowWidth,
				LineOrder lineOrder,
				Compression compression,
                                int numThreads):
    _outputFile (0),
    _toYca (0)
{
    Header hd (width,
	       height,
	       pixelAspectRatio,
	       screenWindowCenter,
	       screenWindowWidth,
	       lineOrder,
	       compression);

    insertChannels (hd, rgbaChannels);
    _outputFile = new OutputFile (name, hd, numThreads);

    if (rgbaChannels & (WRITE_Y | WRITE_C))
	_toYca = new ToYca (*_outputFile, rgbaChannels);
}


RgbaOutputFile::~RgbaOutputFile ()
{
    delete _toYca;
    delete _outputFile;
}


void
RgbaOutputFile::setFrameBuffer (const Rgba *base,
				size_t xStride,
				size_t yStride)
{
    if (_toYca)
    {
	Lock lock (*_toYca);
	_toYca->setFrameBuffer (base, xStride, yStride);
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


void	
RgbaOutputFile::writePixels (int numScanLines)
{
    if (_toYca)
    {
	Lock lock (*_toYca);
	_toYca->writePixels (numScanLines);
    }
    else
    {
	_outputFile->writePixels (numScanLines);
    }
}


int	
RgbaOutputFile::currentScanLine () const
{
    if (_toYca)
    {
	Lock lock (*_toYca);
	return _toYca->currentScanLine();
    }
    else
    {
	return _outputFile->currentScanLine();
    }
}


const Header &
RgbaOutputFile::header () const
{
    return _outputFile->header();
}


const FrameBuffer &
RgbaOutputFile::frameBuffer () const
{
    return _outputFile->frameBuffer();
}


const IMATH_NAMESPACE::Box2i &
RgbaOutputFile::displayWindow () const
{
    return _outputFile->header().displayWindow();
}


const IMATH_NAMESPACE::Box2i &
RgbaOutputFile::dataWindow () const
{
    return _outputFile->header().dataWindow();
}


float	
RgbaOutputFile::pixelAspectRatio () const
{
    return _outputFile->header().pixelAspectRatio();
}


const IMATH_NAMESPACE::V2f
RgbaOutputFile::screenWindowCenter () const
{
    return _outputFile->header().screenWindowCenter();
}


float	
RgbaOutputFile::screenWindowWidth () const
{
    return _outputFile->header().screenWindowWidth();
}


LineOrder
RgbaOutputFile::lineOrder () const
{
    return _outputFile->header().lineOrder();
}


Compression
RgbaOutputFile::compression () const
{
    return _outputFile->header().compression();
}


RgbaChannels
RgbaOutputFile::channels () const
{
    return rgbaChannels (_outputFile->header().channels());
}


void		
RgbaOutputFile::updatePreviewImage (const PreviewRgba newPixels[])
{
    _outputFile->updatePreviewImage (newPixels);
}


void		
RgbaOutputFile::setYCRounding (unsigned int roundY, unsigned int roundC)
{
    if (_toYca)
    {
	Lock lock (*_toYca);
	_toYca->setYCRounding (roundY, roundC);
    }
}


void	
RgbaOutputFile::breakScanLine  (int y, int offset, int length, char c)
{
    _outputFile->breakScanLine (y, offset, length, c);
}


class RgbaInputFile::FromYca: public Mutex
{
  public:

     FromYca (InputFile &inputFile, RgbaChannels rgbaChannels);
    ~FromYca ();

    void		setFrameBuffer (Rgba *base,
					size_t xStride,
					size_t yStride,
					const string &channelNamePrefix);

    void		readPixels (int scanLine1, int scanLine2);

  private:

    void		readPixels (int scanLine);
    void		rotateBuf1 (int d);
    void		rotateBuf2 (int d);
    void		readYCAScanLine (int y, Rgba buf[]);
    void		padTmpBuf ();

    InputFile &		_inputFile;
    bool		_readC;
    int			_xMin;
    int			_yMin;
    int 		_yMax;
    int			_width;
    int			_height;
    int			_currentScanLine;
    LineOrder		_lineOrder;
    V3f			_yw;
    Rgba *		_bufBase;
    Rgba *		_buf1[N + 2];
    Rgba *		_buf2[3];
    Rgba *		_tmpBuf;
    Rgba *		_fbBase;
    size_t		_fbXStride;
    size_t		_fbYStride;
};


RgbaInputFile::FromYca::FromYca (InputFile &inputFile,
				 RgbaChannels rgbaChannels)
:
    _inputFile (inputFile)
{
    _readC = (rgbaChannels & WRITE_C)? true: false;

    const Box2i dw = _inputFile.header().dataWindow();

    _xMin = dw.min.x;
    _yMin = dw.min.y;
    _yMax = dw.max.y;
    _width  = dw.max.x - dw.min.x + 1;
    _height = dw.max.y - dw.min.y + 1;
    _currentScanLine = dw.min.y - N - 2;
    _lineOrder = _inputFile.header().lineOrder();
    _yw = ywFromHeader (_inputFile.header());

    ptrdiff_t pad = cachePadding (_width * sizeof (Rgba)) / sizeof (Rgba);

    _bufBase = new Rgba[(_width + pad) * (N + 2 + 3)];

    for (int i = 0; i < N + 2; ++i)
	_buf1[i] = _bufBase + (i * (_width + pad));
    
    for (int i = 0; i < 3; ++i)
	_buf2[i] = _bufBase + ((i + N + 2) * (_width + pad));

    _tmpBuf = new Rgba[_width + N - 1];

    _fbBase = 0;
    _fbXStride = 0;
    _fbYStride = 0;
}


RgbaInputFile::FromYca::~FromYca ()
{
    delete [] _bufBase;
    delete [] _tmpBuf;
}


void
RgbaInputFile::FromYca::setFrameBuffer (Rgba *base,
					size_t xStride,
					size_t yStride,
					const string &channelNamePrefix)
{
    if (_fbBase == 0)
    {
	FrameBuffer fb;

	fb.insert (channelNamePrefix + "Y",
		   Slice (HALF,					// type
			  (char *) &_tmpBuf[N2 - _xMin].g,	// base
			  sizeof (Rgba),			// xStride
			  0,					// yStride
			  1,					// xSampling
			  1,					// ySampling
			  0.5));				// fillValue

	if (_readC)
	{
	    fb.insert (channelNamePrefix + "RY",
		       Slice (HALF,				// type
			      (char *) &_tmpBuf[N2 - _xMin].r,	// base
			      sizeof (Rgba) * 2,		// xStride
			      0,				// yStride
			      2,				// xSampling
			      2,				// ySampling
			      0.0));				// fillValue

	    fb.insert (channelNamePrefix + "BY",
		       Slice (HALF,				// type
			      (char *) &_tmpBuf[N2 - _xMin].b,	// base
			      sizeof (Rgba) * 2,		// xStride
			      0,				// yStride
			      2,				// xSampling
			      2,				// ySampling
			      0.0));				// fillValue
	}

	fb.insert (channelNamePrefix + "A",
		   Slice (HALF,					// type
			  (char *) &_tmpBuf[N2 - _xMin].a,	// base
			  sizeof (Rgba),			// xStride
			  0,					// yStride
			  1,					// xSampling
			  1,					// ySampling
			  1.0));				// fillValue

	_inputFile.setFrameBuffer (fb);
    }

    _fbBase = base;
    _fbXStride = xStride;
    _fbYStride = yStride;
}


void	
RgbaInputFile::FromYca::readPixels (int scanLine1, int scanLine2)
{
    int minY = min (scanLine1, scanLine2);
    int maxY = max (scanLine1, scanLine2);

    if (_lineOrder == INCREASING_Y)
    {
	for (int y = minY; y <= maxY; ++y)
	    readPixels (y);
    }
    else
    {
	for (int y = maxY; y >= minY; --y)
	    readPixels (y);
    }
}


void	
RgbaInputFile::FromYca::readPixels (int scanLine)
{
    if (_fbBase == 0)
    {
	THROW (IEX_NAMESPACE::ArgExc, "No frame buffer was specified as the "
			    "pixel data destination for image file "
			    "\"" << _inputFile.fileName() << "\".");
    }

    //
    // In order to convert one scan line to RGB format, we need that
    // scan line plus N2+1 extra scan lines above and N2+1 scan lines
    // below in luminance/chroma format.
    //
    // We allow random access to scan lines, but we buffer partially
    // processed luminance/chroma data in order to make reading pixels
    // in increasing y or decreasing y order reasonably efficient:
    //
    //	_currentScanLine	holds the y coordinate of the scan line
    //				that was most recently read.
    //
    //	_buf1			contains scan lines _currentScanLine-N2-1
    //				through _currentScanLine+N2+1 in
    //				luminance/chroma format.  Odd-numbered
    //				lines contain no chroma data.  Even-numbered
    //				lines have valid chroma data for all pixels.
    //
    //  _buf2			contains scan lines _currentScanLine-1
    //  			through _currentScanLine+1, in RGB format.
    //				Super-saturated pixels (see ImfRgbaYca.h)
    //				have not yet been eliminated.
    //
    // If the scan line we are trying to read now is close enough to
    // _currentScanLine, we don't have to recompute the contents of _buf1
    // and _buf2 from scratch.  We can rotate _buf1 and _buf2, and fill
    // in the missing data.
    //

    int dy = scanLine - _currentScanLine;

    if (abs (dy) < N + 2)
	rotateBuf1 (dy);

    if (abs (dy) < 3)
	rotateBuf2 (dy);

    if (dy < 0)
    {
	{
	    int n = min (-dy, N + 2);
	    int yMin = scanLine - N2 - 1;

	    for (int i = n - 1; i >= 0; --i)
		readYCAScanLine (yMin + i, _buf1[i]);
	}

	{
	    int n = min (-dy, 3);

	    for (int i = 0; i < n; ++i)
	    {
		if ((scanLine + i) & 1)
		{
		    YCAtoRGBA (_yw, _width, _buf1[N2 + i], _buf2[i]);
		}
		else
		{
		    reconstructChromaVert (_width, _buf1 + i, _buf2[i]);
		    YCAtoRGBA (_yw, _width, _buf2[i], _buf2[i]);
		}
	    }
	}
    }
    else
    {
	{
	    int n = min (dy, N + 2);
	    int yMax = scanLine + N2 + 1;

	    for (int i = n - 1; i >= 0; --i)
		readYCAScanLine (yMax - i, _buf1[N + 1 - i]);
	}

	{
	    int n = min (dy, 3);

	    for (int i = 2; i > 2 - n; --i)
	    {
		if ((scanLine + i) & 1)
		{
		    YCAtoRGBA (_yw, _width, _buf1[N2 + i], _buf2[i]);
		}
		else
		{
		    reconstructChromaVert (_width, _buf1 + i, _buf2[i]);
		    YCAtoRGBA (_yw, _width, _buf2[i], _buf2[i]);
		}
	    }
	}
    }

    fixSaturation (_yw, _width, _buf2, _tmpBuf);

    for (int i = 0; i < _width; ++i)
	_fbBase[_fbYStride * scanLine + _fbXStride * (i + _xMin)] = _tmpBuf[i];

    _currentScanLine = scanLine;
}


void
RgbaInputFile::FromYca::rotateBuf1 (int d)
{
    d = modp (d, N + 2);

    Rgba *tmp[N + 2];

    for (int i = 0; i < N + 2; ++i)
	tmp[i] = _buf1[i];

    for (int i = 0; i < N + 2; ++i)
	_buf1[i] = tmp[(i + d) % (N + 2)];
}


void
RgbaInputFile::FromYca::rotateBuf2 (int d)
{
    d = modp (d, 3);

    Rgba *tmp[3];

    for (int i = 0; i < 3; ++i)
	tmp[i] = _buf2[i];

    for (int i = 0; i < 3; ++i)
	_buf2[i] = tmp[(i + d) % 3];
}


void
RgbaInputFile::FromYca::readYCAScanLine (int y, Rgba *buf)
{
    //
    // Clamp y.
    //

    if (y < _yMin)
	y = _yMin;
    else if (y > _yMax)
	y = _yMax - 1;

    //
    // Read scan line y into _tmpBuf.
    //

    _inputFile.readPixels (y);

    //
    // Reconstruct missing chroma samples and copy
    // the scan line into buf.
    //

    if (!_readC)
    {
	for (int i = 0; i < _width; ++i)
	{
	    _tmpBuf[i + N2].r = 0;
	    _tmpBuf[i + N2].b = 0;
	}
    }

    if (y & 1)
    {
	memcpy (buf, _tmpBuf + N2, _width * sizeof (Rgba));
    }
    else
    {
	padTmpBuf();
	reconstructChromaHoriz (_width, _tmpBuf, buf);
    }
}


void
RgbaInputFile::FromYca::padTmpBuf ()
{
    for (int i = 0; i < N2; ++i)
    {
	_tmpBuf[i] = _tmpBuf[N2];
	_tmpBuf[_width + N2 + i] = _tmpBuf[_width + N2 - 2];
    }
}


RgbaInputFile::RgbaInputFile (const char name[], int numThreads):
    _inputFile (new InputFile (name, numThreads)),
    _fromYca (0),
    _channelNamePrefix ("")
{
    RgbaChannels rgbaChannels = channels();

    if (rgbaChannels & (WRITE_Y | WRITE_C))
	_fromYca = new FromYca (*_inputFile, rgbaChannels);
}


RgbaInputFile::RgbaInputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int numThreads):
    _inputFile (new InputFile (is, numThreads)),
    _fromYca (0),
    _channelNamePrefix ("")
{
    RgbaChannels rgbaChannels = channels();

    if (rgbaChannels & (WRITE_Y | WRITE_C))
	_fromYca = new FromYca (*_inputFile, rgbaChannels);
}


RgbaInputFile::RgbaInputFile (const char name[],
			      const string &layerName,
			      int numThreads)
:
    _inputFile (new InputFile (name, numThreads)),
    _fromYca (0),
    _channelNamePrefix (prefixFromLayerName (layerName, _inputFile->header()))
{
    RgbaChannels rgbaChannels = channels();

    if (rgbaChannels & (WRITE_Y | WRITE_C))
	_fromYca = new FromYca (*_inputFile, rgbaChannels);
}


RgbaInputFile::RgbaInputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,
			      const string &layerName,
			      int numThreads)
:
    _inputFile (new InputFile (is, numThreads)),
    _fromYca (0),
    _channelNamePrefix (prefixFromLayerName (layerName, _inputFile->header()))
{
    RgbaChannels rgbaChannels = channels();

    if (rgbaChannels & (WRITE_Y | WRITE_C))
	_fromYca = new FromYca (*_inputFile, rgbaChannels);
}


RgbaInputFile::~RgbaInputFile ()
{
    delete _inputFile;
    delete _fromYca;
}


void	
RgbaInputFile::setFrameBuffer (Rgba *base, size_t xStride, size_t yStride)
{
    if (_fromYca)
    {
	Lock lock (*_fromYca);
	_fromYca->setFrameBuffer (base, xStride, yStride, _channelNamePrefix);
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
			  1, 1,		// xSampling, ySampling
			  0.0));	// fillValue

	fb.insert (_channelNamePrefix + "G",
		   Slice (HALF,
			  (char *) &base[0].g,
			  xs, ys,
			  1, 1,		// xSampling, ySampling
			  0.0));	// fillValue

	fb.insert (_channelNamePrefix + "B",
		   Slice (HALF,
			  (char *) &base[0].b,
			  xs, ys,
			  1, 1,		// xSampling, ySampling
			  0.0));	// fillValue

	fb.insert (_channelNamePrefix + "A",
		   Slice (HALF,
			  (char *) &base[0].a,
			  xs, ys,
			  1, 1,		// xSampling, ySampling
			  1.0));	// fillValue

	_inputFile->setFrameBuffer (fb);
    }
}


void
RgbaInputFile::setLayerName (const string &layerName)
{
    delete _fromYca;
    _fromYca = 0;

    _channelNamePrefix = prefixFromLayerName (layerName, _inputFile->header());

    RgbaChannels rgbaChannels = channels();

    if (rgbaChannels & (WRITE_Y | WRITE_C))
	_fromYca = new FromYca (*_inputFile, rgbaChannels);

    FrameBuffer fb;
    _inputFile->setFrameBuffer (fb);
}


void	
RgbaInputFile::readPixels (int scanLine1, int scanLine2)
{
    if (_fromYca)
    {
	Lock lock (*_fromYca);
	_fromYca->readPixels (scanLine1, scanLine2);
    }
    else
    {
	_inputFile->readPixels (scanLine1, scanLine2);
    }
}


void	
RgbaInputFile::readPixels (int scanLine)
{
    readPixels (scanLine, scanLine);
}


bool
RgbaInputFile::isComplete () const
{
    return _inputFile->isComplete();
}


const Header &
RgbaInputFile::header () const
{
    return _inputFile->header();
}


const char *
RgbaInputFile::fileName () const
{
    return _inputFile->fileName();
}


const FrameBuffer &	
RgbaInputFile::frameBuffer () const
{
    return _inputFile->frameBuffer();
}


const IMATH_NAMESPACE::Box2i &
RgbaInputFile::displayWindow () const
{
    return _inputFile->header().displayWindow();
}


const IMATH_NAMESPACE::Box2i &
RgbaInputFile::dataWindow () const
{
    return _inputFile->header().dataWindow();
}


float	
RgbaInputFile::pixelAspectRatio () const
{
    return _inputFile->header().pixelAspectRatio();
}


const IMATH_NAMESPACE::V2f	
RgbaInputFile::screenWindowCenter () const
{
    return _inputFile->header().screenWindowCenter();
}


float	
RgbaInputFile::screenWindowWidth () const
{
    return _inputFile->header().screenWindowWidth();
}


LineOrder
RgbaInputFile::lineOrder () const
{
    return _inputFile->header().lineOrder();
}


Compression
RgbaInputFile::compression () const
{
    return _inputFile->header().compression();
}


RgbaChannels	
RgbaInputFile::channels () const
{
    return rgbaChannels (_inputFile->header().channels(), _channelNamePrefix);
}


int
RgbaInputFile::version () const
{
    return _inputFile->version();
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
