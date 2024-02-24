//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class RgbaOutputFile
//	class RgbaInputFile
//
//-----------------------------------------------------------------------------

#include <Iex.h>
#include <ImathFun.h>
#include <ImfChannelList.h>
#include <ImfInputPart.h>
#include <ImfMultiPartInputFile.h>
#include <ImfOutputFile.h>
#include <ImfRgbaFile.h>
#include <ImfRgbaYca.h>
#include <ImfStandardAttributes.h>
#include <algorithm>
#include <mutex>
#include <string.h>

#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using namespace std;
using namespace IMATH_NAMESPACE;
using namespace RgbaYca;

namespace
{

void
insertChannels (Header& header, RgbaChannels rgbaChannels)
{
    ChannelList ch;

    if (rgbaChannels & (WRITE_Y | WRITE_C))
    {
        if (rgbaChannels & WRITE_Y) { ch.insert ("Y", Channel (HALF, 1, 1)); }

        if (rgbaChannels & WRITE_C)
        {
            ch.insert ("RY", Channel (HALF, 2, 2, true));
            ch.insert ("BY", Channel (HALF, 2, 2, true));
        }
    }
    else
    {
        if (rgbaChannels & WRITE_R) ch.insert ("R", Channel (HALF, 1, 1));

        if (rgbaChannels & WRITE_G) ch.insert ("G", Channel (HALF, 1, 1));

        if (rgbaChannels & WRITE_B) ch.insert ("B", Channel (HALF, 1, 1));
    }

    if (rgbaChannels & WRITE_A) ch.insert ("A", Channel (HALF, 1, 1));

    header.channels () = ch;
}

RgbaChannels
rgbaChannels (const ChannelList& ch, const string& channelNamePrefix = "")
{
    int i = 0;

    if (ch.findChannel (channelNamePrefix + "R")) i |= WRITE_R;

    if (ch.findChannel (channelNamePrefix + "G")) i |= WRITE_G;

    if (ch.findChannel (channelNamePrefix + "B")) i |= WRITE_B;

    if (ch.findChannel (channelNamePrefix + "A")) i |= WRITE_A;

    if (ch.findChannel (channelNamePrefix + "Y")) i |= WRITE_Y;

    if (ch.findChannel (channelNamePrefix + "RY") ||
        ch.findChannel (channelNamePrefix + "BY"))
        i |= WRITE_C;

    return RgbaChannels (i);
}

string
prefixFromLayerName (const string& layerName, const Header& header)
{
    if (layerName.empty ()) return "";

    if (hasMultiView (header) && multiView (header)[0] == layerName) return "";

    return layerName + ".";
}

V3f
ywFromHeader (const Header& header)
{
    Chromaticities cr;

    if (hasChromaticities (header)) cr = chromaticities (header);

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
    // CACHE_LINE_SIZE = (1 << LOG2_CACHE_LINE_SIZE)
    //

    static int LOG2_CACHE_LINE_SIZE = 8;

    size_t i = LOG2_CACHE_LINE_SIZE + 2;

    while ((size >> i) > 1)
        ++i;

    if (size > (1ll << (i + 1)) - 64ll)
        return 64ll + ((1ll << (i + 1ll)) - size);

    if (size < (1ll << i) + 64ll) return 64ll + ((1ll << i) - size);

    return 0;
}

} // namespace

class RgbaOutputFile::ToYca : public std::mutex
{
public:
    ToYca (OutputFile& outputFile, RgbaChannels rgbaChannels);
    ~ToYca ();

    ToYca (const ToYca& other) = delete;
    ToYca& operator= (const ToYca& other) = delete;
    ToYca (ToYca&& other)                 = delete;
    ToYca& operator= (ToYca&& other) = delete;

    void setYCRounding (unsigned int roundY, unsigned int roundC);

    void setFrameBuffer (const Rgba* base, size_t xStride, size_t yStride);

    void writePixels (int numScanLines);
    int  currentScanLine () const;

private:
    void padTmpBuf ();
    void rotateBuffers ();
    void duplicateLastBuffer ();
    void duplicateSecondToLastBuffer ();
    void decimateChromaVertAndWriteScanLine ();

    OutputFile& _outputFile;
    bool        _writeY;
    bool        _writeC;
    bool        _writeA;
    int         _xMin;
    int         _width;
    int         _height;
    int         _linesConverted;
    LineOrder   _lineOrder;
    int         _currentScanLine;
    V3f         _yw;
    Rgba*       _bufBase;
    Rgba*       _buf[N];
    Rgba*       _tmpBuf;
    const Rgba* _fbBase;
    size_t      _fbXStride;
    size_t      _fbYStride;
    int         _roundY;
    int         _roundC;
};

RgbaOutputFile::ToYca::ToYca (OutputFile& outputFile, RgbaChannels rgbaChannels)
    : _outputFile (outputFile)
{
    _writeY = (rgbaChannels & WRITE_Y) ? true : false;
    _writeC = (rgbaChannels & WRITE_C) ? true : false;
    _writeA = (rgbaChannels & WRITE_A) ? true : false;

    const Box2i dw = _outputFile.header ().dataWindow ();

    _xMin   = dw.min.x;
    _width  = dw.max.x - dw.min.x + 1;
    _height = dw.max.y - dw.min.y + 1;

    _linesConverted = 0;
    _lineOrder      = _outputFile.header ().lineOrder ();

    if (_lineOrder == INCREASING_Y)
        _currentScanLine = dw.min.y;
    else
        _currentScanLine = dw.max.y;

    _yw = ywFromHeader (_outputFile.header ());

    ptrdiff_t pad = cachePadding (_width * sizeof (Rgba)) / sizeof (Rgba);

    _bufBase = new Rgba[(_width + pad) * N];

    for (int i = 0; i < N; ++i)
        _buf[i] = _bufBase + (i * (_width + pad));

    _tmpBuf = new Rgba[_width + N - 1];

    _fbBase    = 0;
    _fbXStride = 0;
    _fbYStride = 0;

    _roundY = 7;
    _roundC = 5;
}

RgbaOutputFile::ToYca::~ToYca ()
{
    delete[] _bufBase;
    delete[] _tmpBuf;
}

void
RgbaOutputFile::ToYca::setYCRounding (unsigned int roundY, unsigned int roundC)
{
    _roundY = roundY;
    _roundC = roundC;
}

void
RgbaOutputFile::ToYca::setFrameBuffer (
    const Rgba* base, size_t xStride, size_t yStride)
{
    if (_fbBase == 0)
    {
        FrameBuffer fb;

        if (_writeY)
        {
            fb.insert (
                "Y",
                Slice (
                    HALF,                       // type
                    (char*) &_tmpBuf[-_xMin].g, // base
                    sizeof (Rgba),              // xStride
                    0,                          // yStride
                    1,                          // xSampling
                    1));                        // ySampling
        }

        if (_writeC)
        {
            fb.insert (
                "RY",
                Slice (
                    HALF,                       // type
                    (char*) &_tmpBuf[-_xMin].r, // base
                    sizeof (Rgba) * 2,          // xStride
                    0,                          // yStride
                    2,                          // xSampling
                    2));                        // ySampling

            fb.insert (
                "BY",
                Slice (
                    HALF,                       // type
                    (char*) &_tmpBuf[-_xMin].b, // base
                    sizeof (Rgba) * 2,          // xStride
                    0,                          // yStride
                    2,                          // xSampling
                    2));                        // ySampling
        }

        if (_writeA)
        {
            fb.insert (
                "A",
                Slice (
                    HALF,                       // type
                    (char*) &_tmpBuf[-_xMin].a, // base
                    sizeof (Rgba),              // xStride
                    0,                          // yStride
                    1,                          // xSampling
                    1));                        // ySampling
        }

        _outputFile.setFrameBuffer (fb);
    }

    _fbBase    = base;
    _fbXStride = xStride;
    _fbYStride = yStride;
}

void
RgbaOutputFile::ToYca::writePixels (int numScanLines)
{
    if (_fbBase == 0)
    {
        THROW (
            IEX_NAMESPACE::ArgExc,
            "No frame buffer was specified as the "
            "pixel data source for image file "
            "\"" << _outputFile.fileName ()
                 << "\".");
    }

    intptr_t base = reinterpret_cast<intptr_t> (_fbBase);
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
                _tmpBuf[j] = *reinterpret_cast<Rgba*> (
                    base + sizeof (Rgba) * (_fbYStride * _currentScanLine +
                                            _fbXStride * (j + _xMin)));
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

            intptr_t base = reinterpret_cast<intptr_t> (_fbBase);

            for (int j = 0; j < _width; ++j)
            {
                const Rgba* ptr = reinterpret_cast<const Rgba*> (
                    base + sizeof (Rgba) * (_fbYStride * _currentScanLine +
                                            _fbXStride * (j + _xMin)));
                _tmpBuf[j + N2] = *ptr;
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

            rotateBuffers ();
            decimateChromaHoriz (_width, _tmpBuf, _buf[N - 1]);

            //
            // If this is the first scan line in the image,
            // store N2 more copies of the scan line in _buf.
            //

            if (_linesConverted == 0)
            {
                for (int j = 0; j < N2; ++j)
                    duplicateLastBuffer ();
            }

            ++_linesConverted;

            //
            // If we have have converted at least N2 scan lines from
            // RGBA to luminance/chroma, then we can start to filter
            // and subsample vertically, and store pixels in the
            // output file.
            //

            if (_linesConverted > N2) decimateChromaVertAndWriteScanLine ();

            //
            // If we have already converted the last scan line in
            // the image to luminance/chroma, filter, subsample and
            // store the remaining scan lines in _buf.
            //

            if (_linesConverted >= _height)
            {
                for (int j = 0; j < N2 - _height; ++j)
                    duplicateLastBuffer ();

                duplicateSecondToLastBuffer ();
                ++_linesConverted;
                decimateChromaVertAndWriteScanLine ();

                for (int j = 1; j < min (_height, N2); ++j)
                {
                    duplicateLastBuffer ();
                    ++_linesConverted;
                    decimateChromaVertAndWriteScanLine ();
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
        _tmpBuf[i]               = _tmpBuf[N2];
        _tmpBuf[_width + N2 + i] = _tmpBuf[_width + N2 - 2];
    }
}

void
RgbaOutputFile::ToYca::rotateBuffers ()
{
    Rgba* tmp = _buf[0];

    for (int i = 0; i < N - 1; ++i)
        _buf[i] = _buf[i + 1];

    _buf[N - 1] = tmp;
}

void
RgbaOutputFile::ToYca::duplicateLastBuffer ()
{
    rotateBuffers ();
    memcpy (_buf[N - 1], _buf[N - 2], _width * sizeof (Rgba));
}

void
RgbaOutputFile::ToYca::duplicateSecondToLastBuffer ()
{
    rotateBuffers ();
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

RgbaOutputFile::RgbaOutputFile (
    const char    name[],
    const Header& header,
    RgbaChannels  rgbaChannels,
    int           numThreads)
    : _outputFile (0), _toYca (0)
{
    Header hd (header);
    insertChannels (hd, rgbaChannels);
    _outputFile = new OutputFile (name, hd, numThreads);

    if (rgbaChannels & (WRITE_Y | WRITE_C))
        _toYca = new ToYca (*_outputFile, rgbaChannels);
}

RgbaOutputFile::RgbaOutputFile (
    OPENEXR_IMF_INTERNAL_NAMESPACE::OStream& os,
    const Header&                            header,
    RgbaChannels                             rgbaChannels,
    int                                      numThreads)
    : _outputFile (0), _toYca (0)
{
    Header hd (header);
    insertChannels (hd, rgbaChannels);
    _outputFile = new OutputFile (os, hd, numThreads);

    if (rgbaChannels & (WRITE_Y | WRITE_C))
        _toYca = new ToYca (*_outputFile, rgbaChannels);
}

RgbaOutputFile::RgbaOutputFile (
    const char                    name[],
    const IMATH_NAMESPACE::Box2i& displayWindow,
    const IMATH_NAMESPACE::Box2i& dataWindow,
    RgbaChannels                  rgbaChannels,
    float                         pixelAspectRatio,
    const IMATH_NAMESPACE::V2f    screenWindowCenter,
    float                         screenWindowWidth,
    LineOrder                     lineOrder,
    Compression                   compression,
    int                           numThreads)
    : _outputFile (0), _toYca (0)
{
    Header hd (
        displayWindow,
        dataWindow.isEmpty () ? displayWindow : dataWindow,
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

RgbaOutputFile::RgbaOutputFile (
    const char                 name[],
    int                        width,
    int                        height,
    RgbaChannels               rgbaChannels,
    float                      pixelAspectRatio,
    const IMATH_NAMESPACE::V2f screenWindowCenter,
    float                      screenWindowWidth,
    LineOrder                  lineOrder,
    Compression                compression,
    int                        numThreads)
    : _outputFile (0), _toYca (0)
{
    Header hd (
        width,
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
RgbaOutputFile::setFrameBuffer (
    const Rgba* base, size_t xStride, size_t yStride)
{
    if (_toYca)
    {
        std::lock_guard<std::mutex> lock (*_toYca);
        _toYca->setFrameBuffer (base, xStride, yStride);
    }
    else
    {
        size_t xs = xStride * sizeof (Rgba);
        size_t ys = yStride * sizeof (Rgba);

        FrameBuffer fb;

        fb.insert ("R", Slice (HALF, (char*) &base[0].r, xs, ys));
        fb.insert ("G", Slice (HALF, (char*) &base[0].g, xs, ys));
        fb.insert ("B", Slice (HALF, (char*) &base[0].b, xs, ys));
        fb.insert ("A", Slice (HALF, (char*) &base[0].a, xs, ys));

        _outputFile->setFrameBuffer (fb);
    }
}

void
RgbaOutputFile::writePixels (int numScanLines)
{
    if (_toYca)
    {
        std::lock_guard<std::mutex> lock (*_toYca);
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
        std::lock_guard<std::mutex> lock (*_toYca);
        return _toYca->currentScanLine ();
    }
    else
    {
        return _outputFile->currentScanLine ();
    }
}

const Header&
RgbaOutputFile::header () const
{
    return _outputFile->header ();
}

const FrameBuffer&
RgbaOutputFile::frameBuffer () const
{
    return _outputFile->frameBuffer ();
}

const IMATH_NAMESPACE::Box2i&
RgbaOutputFile::displayWindow () const
{
    return _outputFile->header ().displayWindow ();
}

const IMATH_NAMESPACE::Box2i&
RgbaOutputFile::dataWindow () const
{
    return _outputFile->header ().dataWindow ();
}

float
RgbaOutputFile::pixelAspectRatio () const
{
    return _outputFile->header ().pixelAspectRatio ();
}

const IMATH_NAMESPACE::V2f
RgbaOutputFile::screenWindowCenter () const
{
    return _outputFile->header ().screenWindowCenter ();
}

float
RgbaOutputFile::screenWindowWidth () const
{
    return _outputFile->header ().screenWindowWidth ();
}

LineOrder
RgbaOutputFile::lineOrder () const
{
    return _outputFile->header ().lineOrder ();
}

Compression
RgbaOutputFile::compression () const
{
    return _outputFile->header ().compression ();
}

RgbaChannels
RgbaOutputFile::channels () const
{
    return rgbaChannels (_outputFile->header ().channels ());
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
        std::lock_guard<std::mutex> lock (*_toYca);
        _toYca->setYCRounding (roundY, roundC);
    }
}

void
RgbaOutputFile::breakScanLine (int y, int offset, int length, char c)
{
    _outputFile->breakScanLine (y, offset, length, c);
}

class RgbaInputFile::FromYca : public std::mutex
{
public:
    FromYca (InputPart& inputFile, RgbaChannels rgbaChannels);
    ~FromYca ();

    FromYca (const FromYca& other) = delete;
    FromYca& operator= (const FromYca& other) = delete;
    FromYca (FromYca&& other)                 = delete;
    FromYca& operator= (FromYca&& other) = delete;

    void setFrameBuffer (
        Rgba*         base,
        size_t        xStride,
        size_t        yStride,
        const string& channelNamePrefix);

    void readPixels (int scanLine1, int scanLine2);

private:
    void readPixels (int scanLine);
    void rotateBuf1 (int d);
    void rotateBuf2 (int d);
    void readYCAScanLine (int y, Rgba buf[]);
    void padTmpBuf ();

    InputPart& _inputPart;
    bool       _readC;
    int        _xMin;
    int        _yMin;
    int        _yMax;
    int        _width;
    int        _height;
    int        _currentScanLine;
    LineOrder  _lineOrder;
    V3f        _yw;
    Rgba*      _bufBase;
    Rgba*      _buf1[N + 2];
    Rgba*      _buf2[3];
    Rgba*      _tmpBuf;
    Rgba*      _fbBase;
    size_t     _fbXStride;
    size_t     _fbYStride;
};

RgbaInputFile::FromYca::FromYca (
    InputPart& inputFile, RgbaChannels rgbaChannels)
    : _inputPart (inputFile)
{
    _readC = (rgbaChannels & WRITE_C) ? true : false;

    const Box2i dw = _inputPart.header ().dataWindow ();

    _xMin            = dw.min.x;
    _yMin            = dw.min.y;
    _yMax            = dw.max.y;
    _width           = dw.max.x - dw.min.x + 1;
    _height          = dw.max.y - dw.min.y + 1;
    _currentScanLine = dw.min.y - N - 2;
    _lineOrder       = _inputPart.header ().lineOrder ();
    _yw              = ywFromHeader (_inputPart.header ());

    ptrdiff_t pad = cachePadding (_width * sizeof (Rgba)) / sizeof (Rgba);

    _bufBase = new Rgba[(_width + pad) * (N + 2 + 3)];

    for (int i = 0; i < N + 2; ++i)
        _buf1[i] = _bufBase + (i * (_width + pad));

    for (int i = 0; i < 3; ++i)
        _buf2[i] = _bufBase + ((i + N + 2) * (_width + pad));

    _tmpBuf = new Rgba[_width + N - 1];

    _fbBase    = 0;
    _fbXStride = 0;
    _fbYStride = 0;
}

RgbaInputFile::FromYca::~FromYca ()
{
    delete[] _bufBase;
    delete[] _tmpBuf;
}

void
RgbaInputFile::FromYca::setFrameBuffer (
    Rgba* base, size_t xStride, size_t yStride, const string& channelNamePrefix)
{
    if (_fbBase == 0)
    {
        FrameBuffer fb;

        fb.insert (
            channelNamePrefix + "Y",
            Slice (
                HALF,                           // type
                (char*) &_tmpBuf[N2 - _xMin].g, // base
                sizeof (Rgba),                  // xStride
                0,                              // yStride
                1,                              // xSampling
                1,                              // ySampling
                0.5));                          // fillValue

        if (_readC)
        {
            fb.insert (
                channelNamePrefix + "RY",
                Slice (
                    HALF,                           // type
                    (char*) &_tmpBuf[N2 - _xMin].r, // base
                    sizeof (Rgba) * 2,              // xStride
                    0,                              // yStride
                    2,                              // xSampling
                    2,                              // ySampling
                    0.0));                          // fillValue

            fb.insert (
                channelNamePrefix + "BY",
                Slice (
                    HALF,                           // type
                    (char*) &_tmpBuf[N2 - _xMin].b, // base
                    sizeof (Rgba) * 2,              // xStride
                    0,                              // yStride
                    2,                              // xSampling
                    2,                              // ySampling
                    0.0));                          // fillValue
        }

        fb.insert (
            channelNamePrefix + "A",
            Slice (
                HALF,                           // type
                (char*) &_tmpBuf[N2 - _xMin].a, // base
                sizeof (Rgba),                  // xStride
                0,                              // yStride
                1,                              // xSampling
                1,                              // ySampling
                1.0));                          // fillValue

        _inputPart.setFrameBuffer (fb);
    }

    _fbBase    = base;
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
        THROW (
            IEX_NAMESPACE::ArgExc,
            "No frame buffer was specified as the "
            "pixel data destination for image file "
            "\"" << _inputPart.fileName ()
                 << "\".");
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

    if (abs (dy) < N + 2) rotateBuf1 (dy);

    if (abs (dy) < 3) rotateBuf2 (dy);

    if (dy < 0)
    {
        {
            int n    = min (-dy, N + 2);
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
            int n    = min (dy, N + 2);
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

    intptr_t base = reinterpret_cast<intptr_t> (_fbBase);
    for (int i = 0; i < _width; ++i)
    {
        Rgba* ptr = reinterpret_cast<Rgba*> (
            base +
            sizeof (Rgba) * (_fbYStride * scanLine + _fbXStride * (i + _xMin)));
        *ptr = _tmpBuf[i];
    }
    _currentScanLine = scanLine;
}

void
RgbaInputFile::FromYca::rotateBuf1 (int d)
{
    d = modp (d, N + 2);

    Rgba* tmp[N + 2];

    for (int i = 0; i < N + 2; ++i)
        tmp[i] = _buf1[i];

    for (int i = 0; i < N + 2; ++i)
        _buf1[i] = tmp[(i + d) % (N + 2)];
}

void
RgbaInputFile::FromYca::rotateBuf2 (int d)
{
    d = modp (d, 3);

    Rgba* tmp[3];

    for (int i = 0; i < 3; ++i)
        tmp[i] = _buf2[i];

    for (int i = 0; i < 3; ++i)
        _buf2[i] = tmp[(i + d) % 3];
}

void
RgbaInputFile::FromYca::readYCAScanLine (int y, Rgba* buf)
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

    _inputPart.readPixels (y);

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

    if (y & 1) { memcpy (buf, _tmpBuf + N2, _width * sizeof (Rgba)); }
    else
    {
        padTmpBuf ();
        reconstructChromaHoriz (_width, _tmpBuf, buf);
    }
}

void
RgbaInputFile::FromYca::padTmpBuf ()
{
    for (int i = 0; i < N2; ++i)
    {
        _tmpBuf[i]               = _tmpBuf[N2];
        _tmpBuf[_width + N2 + i] = _tmpBuf[_width + N2 - 2];
    }
}

RgbaInputFile::RgbaInputFile (const char name[], int numThreads)
    : RgbaInputFile (0, name, numThreads)
{}

RgbaInputFile::RgbaInputFile (
    OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is, int numThreads)
    : RgbaInputFile (0, is, numThreads)
{}

RgbaInputFile::RgbaInputFile (
    const char name[], const string& layerName, int numThreads)
    : RgbaInputFile (0, name, layerName, numThreads)
{}

RgbaInputFile::RgbaInputFile (
    OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is,
    const string&                            layerName,
    int                                      numThreads)
    : RgbaInputFile (0, is, layerName, numThreads)
{}

RgbaInputFile::RgbaInputFile (int partNumber, const char name[], int numThreads)
    : _multiPartFile (new MultiPartInputFile (name, numThreads))
    , _inputPart (nullptr)
    , _fromYca (nullptr)
    , _channelNamePrefix ("")
{
    try
    {
        _inputPart                = new InputPart (*_multiPartFile, partNumber);
        RgbaChannels rgbaChannels = channels ();
        if (rgbaChannels & WRITE_C)
            _fromYca = new FromYca (*_inputPart, rgbaChannels);
    }
    catch (...)
    {
        if (_inputPart) { delete _inputPart; }
        delete _multiPartFile;
        throw;
    }
}

RgbaInputFile::RgbaInputFile (
    int partNumber, OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is, int numThreads)
    : _multiPartFile (new MultiPartInputFile (is, numThreads))
    , _inputPart (nullptr)
    , _fromYca (nullptr)
    , _channelNamePrefix ("")
{
    try
    {
        _inputPart                = new InputPart (*_multiPartFile, partNumber);
        RgbaChannels rgbaChannels = channels ();
        if (rgbaChannels & WRITE_C)
            _fromYca = new FromYca (*_inputPart, rgbaChannels);
    }
    catch (...)
    {
        if (_inputPart) { delete _inputPart; }
        delete _multiPartFile;
        throw;
    }
}

RgbaInputFile::RgbaInputFile (
    int partNumber, const char name[], const string& layerName, int numThreads)
    : _multiPartFile (new MultiPartInputFile (name, numThreads))
    , _inputPart (nullptr)
    , _fromYca (0)
{
    try
    {
        _inputPart = new InputPart (*_multiPartFile, partNumber);
        _channelNamePrefix =
            prefixFromLayerName (layerName, _inputPart->header ());
        RgbaChannels rgbaChannels = channels ();

        if (rgbaChannels & WRITE_C)
            _fromYca = new FromYca (*_inputPart, rgbaChannels);
    }
    catch (...)
    {
        if (_inputPart) { delete _inputPart; }
        delete _multiPartFile;
        throw;
    }
}

RgbaInputFile::RgbaInputFile (
    int                                      partNumber,
    OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is,
    const string&                            layerName,
    int                                      numThreads)
    : _multiPartFile (new MultiPartInputFile (is, numThreads))
    , _inputPart (nullptr)
    , _fromYca (0)
{
    try
    {
        _inputPart = new InputPart (*_multiPartFile, partNumber);
        _channelNamePrefix =
            prefixFromLayerName (layerName, _inputPart->header ());
        RgbaChannels rgbaChannels = channels ();

        if (rgbaChannels & WRITE_C)
            _fromYca = new FromYca (*_inputPart, rgbaChannels);
    }
    catch (...)
    {
        if (_inputPart) { delete _inputPart; }
        delete _multiPartFile;
        throw;
    }
}

RgbaInputFile::~RgbaInputFile ()
{
    if (_inputPart) { delete _inputPart; }
    if (_multiPartFile) { delete _multiPartFile; }
    delete _fromYca;
}

void
RgbaInputFile::setFrameBuffer (Rgba* base, size_t xStride, size_t yStride)
{
    if (_fromYca)
    {
        std::lock_guard<std::mutex> lock (*_fromYca);
        _fromYca->setFrameBuffer (base, xStride, yStride, _channelNamePrefix);
    }
    else
    {
        size_t xs = xStride * sizeof (Rgba);
        size_t ys = yStride * sizeof (Rgba);

        FrameBuffer fb;

        if (channels () & WRITE_Y)
        {
            fb.insert (
                _channelNamePrefix + "Y",
                Slice (
                    HALF,
                    (char*) &base[0].r,
                    xs,
                    ys,
                    1,
                    1,     // xSampling, ySampling
                    0.0)); // fillValue
        }
        else
        {

            fb.insert (
                _channelNamePrefix + "R",
                Slice (
                    HALF,
                    (char*) &base[0].r,
                    xs,
                    ys,
                    1,
                    1,     // xSampling, ySampling
                    0.0)); // fillValue

            fb.insert (
                _channelNamePrefix + "G",
                Slice (
                    HALF,
                    (char*) &base[0].g,
                    xs,
                    ys,
                    1,
                    1,     // xSampling, ySampling
                    0.0)); // fillValue

            fb.insert (
                _channelNamePrefix + "B",
                Slice (
                    HALF,
                    (char*) &base[0].b,
                    xs,
                    ys,
                    1,
                    1,     // xSampling, ySampling
                    0.0)); // fillValue
        }
        fb.insert (
            _channelNamePrefix + "A",
            Slice (
                HALF,
                (char*) &base[0].a,
                xs,
                ys,
                1,
                1,     // xSampling, ySampling
                1.0)); // fillValue

        _inputPart->setFrameBuffer (fb);
    }
}

void
RgbaInputFile::setLayerName (const string& layerName)
{
    delete _fromYca;
    _fromYca = nullptr;

    _channelNamePrefix = prefixFromLayerName (layerName, _inputPart->header ());

    RgbaChannels rgbaChannels = channels ();

    if (rgbaChannels & WRITE_C)
        _fromYca = new FromYca (*_inputPart, rgbaChannels);

    FrameBuffer fb;
    _inputPart->setFrameBuffer (fb);
}

void
RgbaInputFile::setPartAndLayer (int part, const string& layerName)
{
    delete _fromYca;
    _fromYca = nullptr;
    delete _inputPart;
    _inputPart = nullptr;

    _inputPart         = new InputPart (*_multiPartFile, part);
    _channelNamePrefix = prefixFromLayerName (layerName, _inputPart->header ());
    ;

    RgbaChannels rgbaChannels = channels ();

    if (rgbaChannels & WRITE_C)
        _fromYca = new FromYca (*_inputPart, rgbaChannels);

    FrameBuffer fb;
    _inputPart->setFrameBuffer (fb);
}

void
RgbaInputFile::setPart (int part)
{
    setPartAndLayer (part, "");
}

void
RgbaInputFile::readPixels (int scanLine1, int scanLine2)
{
    if (_fromYca)
    {
        std::lock_guard<std::mutex> lock (*_fromYca);
        _fromYca->readPixels (scanLine1, scanLine2);
    }
    else
    {
        _inputPart->readPixels (scanLine1, scanLine2);

        if (channels () & WRITE_Y)
        {
            //
            // Luma channel has been written into red channel
            // Duplicate into green and blue channel to create gray image
            //
            const Slice* s =
                _inputPart->frameBuffer ().findSlice (_channelNamePrefix + "Y");
            Box2i    dataWindow = _inputPart->header ().dataWindow ();
            intptr_t base       = reinterpret_cast<intptr_t> (s->base);

            for (int scanLine = scanLine1; scanLine <= scanLine2; scanLine++)
            {
                intptr_t rowBase = base + scanLine * s->yStride;
                for (int x = dataWindow.min.x; x <= dataWindow.max.x; ++x)
                {
                    Rgba* pixel =
                        reinterpret_cast<Rgba*> (rowBase + x * s->xStride);
                    pixel->g = pixel->r;
                    pixel->b = pixel->r;
                }
            }
        }
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
    for (int i = 0; i < _multiPartFile->parts (); ++i)
    {
        if (!_multiPartFile->partComplete (i)) { return false; }
    }
    return true;
}

const Header&
RgbaInputFile::header () const
{
    return _inputPart->header ();
}

int
RgbaInputFile::parts () const
{
    return _multiPartFile->parts ();
}

const char*
RgbaInputFile::fileName () const
{
    return _inputPart->fileName ();
}

const FrameBuffer&
RgbaInputFile::frameBuffer () const
{
    return _inputPart->frameBuffer ();
}

const IMATH_NAMESPACE::Box2i&
RgbaInputFile::displayWindow () const
{
    return _inputPart->header ().displayWindow ();
}

const IMATH_NAMESPACE::Box2i&
RgbaInputFile::dataWindow () const
{
    return _inputPart->header ().dataWindow ();
}

float
RgbaInputFile::pixelAspectRatio () const
{
    return _inputPart->header ().pixelAspectRatio ();
}

const IMATH_NAMESPACE::V2f
RgbaInputFile::screenWindowCenter () const
{
    return _inputPart->header ().screenWindowCenter ();
}

float
RgbaInputFile::screenWindowWidth () const
{
    return _inputPart->header ().screenWindowWidth ();
}

LineOrder
RgbaInputFile::lineOrder () const
{
    return _inputPart->header ().lineOrder ();
}

Compression
RgbaInputFile::compression () const
{
    return _inputPart->header ().compression ();
}

RgbaChannels
RgbaInputFile::channels () const
{
    return rgbaChannels (_inputPart->header ().channels (), _channelNamePrefix);
}

int
RgbaInputFile::version () const
{
    return _inputPart->version ();
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
