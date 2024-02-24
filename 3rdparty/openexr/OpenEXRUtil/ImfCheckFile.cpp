// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.

#include "ImfCheckFile.h"
#include "Iex.h"
#include "ImfArray.h"
#include "ImfChannelList.h"
#include "ImfCompositeDeepScanLine.h"
#include "ImfCompressor.h"
#include "ImfDeepFrameBuffer.h"
#include "ImfDeepScanLineInputFile.h"
#include "ImfDeepScanLineInputPart.h"
#include "ImfDeepTiledInputFile.h"
#include "ImfDeepTiledInputPart.h"
#include "ImfFrameBuffer.h"
#include "ImfInputFile.h"
#include "ImfInputPart.h"
#include "ImfMultiPartInputFile.h"
#include "ImfPartType.h"
#include "ImfRgbaFile.h"
#include "ImfStandardAttributes.h"
#include "ImfStdIO.h"
#include "ImfTiledInputFile.h"
#include "ImfTiledInputPart.h"
#include "ImfTiledMisc.h"

#include "openexr.h"

#include <algorithm>
#include <stdlib.h>
#include <vector>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

namespace
{

using IMATH_NAMESPACE::Box2i;
using std::max;
using std::vector;

//
// limits for reduceMemory mode
//
const uint64_t gMaxBytesPerScanline     = 8000000;
const uint64_t gMaxTileBytesPerScanline = 8000000;
const uint64_t gMaxTileBytes            = 1000 * 1000;
const uint64_t gMaxBytesPerDeepPixel    = 1000;
const uint64_t gMaxBytesPerDeepScanline = 1 << 12;

//
// limits for reduceTime mode
//
const int gTargetPixelsToRead = 1 << 28;
const int gMaxScanlinesToRead = 1 << 20;

//
// compute row stride appropriate to process files quickly
// only used for the 'Rgba' interfaces, which read potentially non-existent channels
//
//

int
getStep (const Box2i& dw, bool reduceTime)
{

    if (reduceTime)
    {
        size_t rowCount   = (dw.max.y - dw.min.y + 1);
        size_t pixelCount = rowCount * (dw.max.x - dw.min.x + 1);
        return max (
            1,
            max (
                static_cast<int> (pixelCount / gTargetPixelsToRead),
                static_cast<int> (rowCount / gMaxScanlinesToRead)));
    }
    else
    {
        return 1;
    }
}
//
// read image or part using the Rgba interface
//
bool
readRgba (RgbaInputFile& in, bool reduceMemory, bool reduceTime)
{

    bool threw = false;

    for (int part = 0; part < in.parts (); ++part)
    {
        in.setPart (part);
        try
        {
            const Box2i& dw = in.dataWindow ();

            uint64_t w = static_cast<uint64_t> (dw.max.x) -
                         static_cast<uint64_t> (dw.min.x) + 1;
            int      dx            = dw.min.x;
            uint64_t bytesPerPixel = calculateBytesPerPixel (in.header ());
            uint64_t numLines = numLinesInBuffer (in.header ().compression ());

            if (reduceMemory &&
                w * bytesPerPixel * numLines > gMaxBytesPerScanline)
            {
                return false;
            }

            Array<Rgba> pixels (w);
            intptr_t    base = reinterpret_cast<intptr_t> (&pixels[0]);
            in.setFrameBuffer (
                reinterpret_cast<Rgba*> (base - dx * sizeof (Rgba)), 1, 0);

            int step = getStep (dw, reduceTime);

            //
            // try reading scanlines. Continue reading scanlines
            // even if an exception is encountered
            //
            for (int y = dw.min.y; y <= dw.max.y; y += step)
            {
                try
                {
                    in.readPixels (y);
                }
                catch (...)
                {
                    threw = true;

                    //
                    // in reduceTime mode, fail immediately - the file is corrupt
                    //
                    if (reduceTime) { return threw; }
                }
            }
        }
        catch (...)
        {
            threw = true;
        }
    }
    return threw;
}

template <class T>
bool
readScanline (T& in, bool reduceMemory, bool reduceTime)
{

    bool threw = false;

    try
    {
        const Box2i& dw = in.header ().dataWindow ();

        uint64_t w = static_cast<uint64_t> (dw.max.x) -
                     static_cast<uint64_t> (dw.min.x) + 1;
        int      dx            = dw.min.x;
        uint64_t bytesPerPixel = calculateBytesPerPixel (in.header ());
        uint64_t numLines      = numLinesInBuffer (in.header ().compression ());

        if (reduceMemory && w * bytesPerPixel * numLines > gMaxBytesPerScanline)
        {
            return false;
        }

        FrameBuffer i;

        // read all channels present (later channels will overwrite earlier ones)
        vector<half>         halfChannels (w);
        vector<float>        floatChannels (w);
        vector<unsigned int> uintChannels (w);

        intptr_t halfData  = reinterpret_cast<intptr_t> (halfChannels.data ());
        intptr_t floatData = reinterpret_cast<intptr_t> (floatChannels.data ());
        intptr_t uintData  = reinterpret_cast<intptr_t> (uintChannels.data ());

        int                channelIndex = 0;
        const ChannelList& channelList  = in.header ().channels ();
        for (ChannelList::ConstIterator c = channelList.begin ();
             c != channelList.end ();
             ++c)
        {
            switch (channelIndex % 3)
            {
                case 0:
                    i.insert(c.name(),Slice(HALF, (char*) (halfData - sizeof(half)*(dx/c.channel().xSampling))  , sizeof(half) , 0 , c.channel().xSampling , c.channel().ySampling ));
                    break;
                case 1:
                    i.insert(c.name(),Slice(FLOAT, (char*) (floatData - sizeof(float)*(dx/c.channel().xSampling))  , sizeof(float) , 0 , c.channel().xSampling , c.channel().ySampling ));
                    break;
                case 2:
                    i.insert(c.name(),Slice(UINT, (char*) (uintData - sizeof(unsigned int)*(dx/c.channel().xSampling))  , sizeof(unsigned int) , 0 , c.channel().xSampling , c.channel().ySampling ));
                    break;
            }
            channelIndex++;
        }

        in.setFrameBuffer (i);

        int step = 1;

        //
        // try reading scanlines. Continue reading scanlines
        // even if an exception is encountered
        //
        for (int y = dw.min.y; y <= dw.max.y; y += step)
        {
            try
            {
                in.readPixels (y);
            }
            catch (...)
            {
                threw = true;

                //
                // in reduceTime mode, fail immediately - the file is corrupt
                //
                if (reduceTime) { return threw; }
            }
        }
    }
    catch (...)
    {
        threw = true;
    }

    return threw;
}

template <class T>
bool
readTileRgba (T& in, bool reduceMemory, bool reduceTime)
{
    try
    {
        const Box2i& dw = in.dataWindow ();

        int w     = dw.max.x - dw.min.x + 1;
        int h     = dw.max.y - dw.min.y + 1;
        int bytes = calculateBytesPerPixel (in.header ());

        if ((reduceMemory || reduceTime) && h * w * bytes > gMaxTileBytes)
        {
            return false;
        }

        int dwx = dw.min.x;
        int dwy = dw.min.y;

        Array2D<Rgba> pixels (h, w);
        in.setFrameBuffer (&pixels[-dwy][-dwx], 1, w);
        in.readTiles (0, in.numXTiles () - 1, 0, in.numYTiles () - 1);
    }
    catch (...)
    {
        return true;
    }

    return false;
}

// read image as ripmapped image
template <class T>
bool
readTile (T& in, bool reduceMemory, bool reduceTime)
{
    bool threw = false;
    try
    {
        const Box2i& dw = in.header ().dataWindow ();

        uint64_t w = static_cast<uint64_t> (dw.max.x) -
                     static_cast<uint64_t> (dw.min.x) + 1;
        int dwx        = dw.min.x;
        int numXLevels = in.numXLevels ();
        int numYLevels = in.numYLevels ();

        const TileDescription& td    = in.header ().tileDescription ();
        uint64_t               bytes = calculateBytesPerPixel (in.header ());

        if (reduceMemory && (w * bytes > gMaxBytesPerScanline ||
                             (td.xSize * td.ySize * bytes) > gMaxTileBytes))
        {
            return false;
        }

        FrameBuffer i;
        // read all channels present (later channels will overwrite earlier ones)
        vector<half>         halfChannels (w);
        vector<float>        floatChannels (w);
        vector<unsigned int> uintChannels (w);

        int                channelIndex = 0;
        const ChannelList& channelList  = in.header ().channels ();
        for (ChannelList::ConstIterator c = channelList.begin ();
             c != channelList.end ();
             ++c)
        {
            switch (channelIndex % 3)
            {
                case 0:
                    i.insert (
                        c.name (),
                        Slice (
                            HALF,
                            (char*) &halfChannels
                                [-dwx / c.channel ().xSampling],
                            sizeof (half),
                            0,
                            c.channel ().xSampling,
                            c.channel ().ySampling));
                    break;
                case 1:
                    i.insert (
                        c.name (),
                        Slice (
                            FLOAT,
                            (char*) &floatChannels
                                [-dwx / c.channel ().xSampling],
                            sizeof (float),
                            0,
                            c.channel ().xSampling,
                            c.channel ().ySampling));
                case 2:
                    i.insert (
                        c.name (),
                        Slice (
                            UINT,
                            (char*) &uintChannels
                                [-dwx / c.channel ().xSampling],
                            sizeof (unsigned int),
                            0,
                            c.channel ().xSampling,
                            c.channel ().ySampling));
                    break;
            }
            channelIndex++;
        }

        in.setFrameBuffer (i);

        size_t step = 1;

        size_t tileIndex = 0;
        bool   isRipMap  = td.mode == RIPMAP_LEVELS;

        //
        // read all tiles from all levels.
        //
        for (int ylevel = 0; ylevel < numYLevels; ++ylevel)
        {
            for (int xlevel = 0; xlevel < numXLevels; ++xlevel)
            {
                for (int y = 0; y < in.numYTiles (ylevel); ++y)
                {
                    for (int x = 0; x < in.numXTiles (xlevel); ++x)
                    {
                        if (tileIndex % step == 0)
                        {
                            try
                            {
                                in.readTile (x, y, xlevel, ylevel);
                            }
                            catch (...)
                            {
                                //
                                // for one level and mipmapped images,
                                // xlevel must match ylevel,
                                // otherwise an exception is thrown
                                // ignore that exception
                                //
                                if (isRipMap || xlevel == ylevel)
                                {
                                    threw = true;

                                    //
                                    // in reduceTime mode, fail immediately - the file is corrupt
                                    //
                                    if (reduceTime) { return threw; }
                                }
                            }
                        }
                        tileIndex++;
                    }
                }
            }
        }
    }
    catch (...)
    {
        threw = true;
    }

    return threw;
}

template <class T>
bool
readDeepScanLine (T& in, bool reduceMemory, bool reduceTime)
{

    bool threw = false;
    try
    {
        const Header& fileHeader = in.header ();
        const Box2i&  dw         = fileHeader.dataWindow ();

        uint64_t w = static_cast<uint64_t> (dw.max.x) -
                     static_cast<uint64_t> (dw.min.x) + 1;
        int dwx = dw.min.x;

        uint64_t bytesPerSample = calculateBytesPerPixel (in.header ());

        //
        // in reduce memory mode, check size required by sampleCount table
        //
        if (reduceMemory && w * 4 > gMaxBytesPerScanline) { return false; }

        int channelCount = 0;
        for (ChannelList::ConstIterator i = fileHeader.channels ().begin ();
             i != fileHeader.channels ().end ();
             ++i, ++channelCount)
            ;

        Array<unsigned int> localSampleCount;
        localSampleCount.resizeErase (w);
        Array<Array<void*>> data (channelCount);

        for (int i = 0; i < channelCount; i++)
        {
            data[i].resizeErase (w);
        }

        DeepFrameBuffer frameBuffer;

        frameBuffer.insertSampleCountSlice (Slice (
            UINT, (char*) (&localSampleCount[-dwx]), sizeof (unsigned int), 0));

        int channel = 0;
        for (ChannelList::ConstIterator i = fileHeader.channels ().begin ();
             i != fileHeader.channels ().end ();
             ++i, ++channel)
        {
            PixelType type = FLOAT;

            int sampleSize = sizeof (float);

            int pointerSize = sizeof (char*);

            frameBuffer.insert (
                i.name (),
                DeepSlice (
                    type,
                    (char*) (&data[channel][-dwx]),
                    pointerSize,
                    0,
                    sampleSize));
        }

        in.setFrameBuffer (frameBuffer);

        int step = 1;

        vector<float> pixelBuffer;

        for (int y = dw.min.y; y <= dw.max.y; y += step)
        {
            in.readPixelSampleCounts (y);

            //
            // count how many samples are required to store this scanline
            // in reduceMemory mode, pixels with large sample counts are not read,
            // but the library needs to allocate memory for them internally
            // - bufferSize is how much memory this function will allocate
            // - fileBufferSize tracks how much decompressed data the library will require
            //
            size_t bufferSize = 0;
            size_t fileBufferSize = 0;
            for (uint64_t j = 0; j < w; j++)
            {
                for (int k = 0; k < channelCount; k++)
                {
                    fileBufferSize += localSampleCount[j];
                    //
                    // don't read samples which require a lot of memory in reduceMemory mode
                    //

                    if (!reduceMemory || localSampleCount[j] * bytesPerSample <=
                                             gMaxBytesPerDeepPixel)
                    {
                        bufferSize += localSampleCount[j];
                    }
                }
            }

            //
            // limit total number of samples read in reduceMemory mode
            //
            if (!reduceMemory || fileBufferSize + bufferSize < gMaxBytesPerDeepScanline)
            {
                //
                // allocate sample buffer and set per-pixel pointers into buffer
                //
                pixelBuffer.resize (bufferSize);

                size_t bufferIndex = 0;
                for (uint64_t j = 0; j < w; j++)
                {
                    for (int k = 0; k < channelCount; k++)
                    {

                        if (localSampleCount[j] == 0 ||
                            (reduceMemory &&
                             localSampleCount[j] * bytesPerSample >
                                 gMaxBytesPerDeepPixel))
                        {
                            data[k][j] = nullptr;
                        }
                        else
                        {
                            data[k][j] = &pixelBuffer[bufferIndex];
                            bufferIndex += localSampleCount[j];
                        }
                    }
                }

                try
                {
                    in.readPixels (y);
                }
                catch (...)
                {
                    threw = true;
                    //
                    // in reduceTime mode, fail immediately - the file is corrupt
                    //
                    if (reduceTime) { return threw; }
                }
            }
        }
    }
    catch (...)
    {
        threw = true;
    }
    return threw;
}

//
// read a deep tiled image, tile by tile, using the 'tile relative' mode
//
template <class T>
bool
readDeepTile (T& in, bool reduceMemory, bool reduceTime)
{
    bool threw = false;
    try
    {
        const Header& fileHeader = in.header ();

        Array2D<unsigned int> localSampleCount;

        int      bytesPerSample = calculateBytesPerPixel (in.header ());

        const TileDescription& td         = in.header ().tileDescription ();
        int                    tileWidth  = td.xSize;
        int                    tileHeight = td.ySize;
        int                    numYLevels = in.numYLevels ();
        int                    numXLevels = in.numXLevels ();

        localSampleCount.resizeErase (tileHeight, tileWidth);

        int channelCount = 0;
        for (ChannelList::ConstIterator i = fileHeader.channels ().begin ();
             i != fileHeader.channels ().end ();
             ++i, channelCount++)
            ;

        Array<Array2D<float*>> data (channelCount);

        for (int i = 0; i < channelCount; i++)
        {
            data[i].resizeErase (tileHeight, tileWidth);
        }

        DeepFrameBuffer frameBuffer;

        //
        // Use integer arithmetic instead of pointer arithmetic to compute offset into array.
        // if memOffset is larger than base, then the computed pointer is negative, which is reported as undefined behavior
        // Instead, integers are used for computation which behaves as expected an all known architectures
        //

        frameBuffer.insertSampleCountSlice (Slice (
            UINT,
            reinterpret_cast<char*> (&localSampleCount[0][0]),
            sizeof (unsigned int) * 1,
            sizeof (unsigned int) * tileWidth,
            1,
            1,    // x/ysampling
            0.0,  // fill
            true, // relative x
            true  // relative y
            ));

        int channel = 0;
        for (ChannelList::ConstIterator i = fileHeader.channels ().begin ();
             i != fileHeader.channels ().end ();
             ++i, ++channel)
        {
            int sampleSize = sizeof (float);

            int pointerSize = sizeof (char*);

            frameBuffer.insert (
                i.name (),
                DeepSlice (
                    FLOAT,
                    reinterpret_cast<char*> (&data[channel][0][0]),
                    pointerSize * 1,
                    pointerSize * tileWidth,
                    sampleSize,
                    1,
                    1,
                    0.0,
                    true,
                    true));
        }

        in.setFrameBuffer (frameBuffer);
        size_t step = 1;

        int  tileIndex = 0;
        bool isRipMap  = td.mode == RIPMAP_LEVELS;

        vector<float> pixelBuffer;

        //
        // read all tiles from all levels.
        //
        for (int ylevel = 0; ylevel < numYLevels; ++ylevel)
        {
            for (int xlevel = 0; xlevel < numXLevels; ++xlevel)
            {
                for (int y = 0; y < in.numYTiles (ylevel); ++y)
                {
                    for (int x = 0; x < in.numXTiles (xlevel); ++x)
                    {
                        if (tileIndex % step == 0)
                        {
                            try
                            {

                                in.readPixelSampleCounts (
                                    x, y, x, y, xlevel, ylevel);

                                size_t bufferSize = 0;
                                size_t fileBufferSize = 0;

                                for (int ty = 0; ty < tileHeight; ++ty)
                                {
                                    for (int tx = 0; tx < tileWidth; ++tx)
                                    {
                                        fileBufferSize += channelCount *
                                                localSampleCount[ty][tx];

                                        if (!reduceMemory ||
                                            localSampleCount[ty][tx] *
                                                    bytesPerSample <
                                                gMaxBytesPerDeepScanline)
                                        {
                                            bufferSize +=
                                                channelCount *
                                                localSampleCount[ty][tx];
                                        }
                                    }
                                }

                                // skip reading if no data to read, or limiting memory and tile is too large
                                if (bufferSize > 0 &&
                                    (!reduceMemory ||
                                     (fileBufferSize + bufferSize) * bytesPerSample <
                                         gMaxBytesPerDeepPixel))
                                {

                                    pixelBuffer.resize (bufferSize);
                                    size_t bufferIndex = 0;

                                    for (int ty = 0; ty < tileHeight; ++ty)
                                    {
                                        for (int tx = 0; tx < tileWidth; ++tx)
                                        {
                                            if (!reduceMemory ||
                                                localSampleCount[ty][tx] *
                                                        bytesPerSample <
                                                    gMaxBytesPerDeepPixel)
                                            {
                                                for (int k = 0;
                                                     k < channelCount;
                                                     ++k)
                                                {
                                                    data[k][ty][tx] =
                                                        &pixelBuffer
                                                            [bufferIndex];
                                                    bufferIndex +=
                                                        localSampleCount[ty]
                                                                        [tx];
                                                }
                                            }
                                            else
                                            {
                                                for (int k = 0;
                                                     k < channelCount;
                                                     ++k)
                                                {
                                                    data[k][ty][tx] = nullptr;
                                                }
                                            }
                                        }
                                    }

                                    in.readTile (x, y, xlevel, ylevel);
                                }
                            }

                            catch (...)
                            {
                                //
                                // for one level and mipmapped images,
                                // xlevel must match ylevel,
                                // otherwise an exception is thrown
                                // ignore that exception
                                //
                                if (isRipMap || xlevel == ylevel)
                                {
                                    threw = true;
                                    //
                                    // in reduceTime mode, fail immediately - the file is corrupt
                                    //
                                    if (reduceTime) { return threw; }
                                }
                            }
                        }
                        tileIndex++;
                    }
                }
            }
        }
    }
    catch (...)
    {
        threw = true;
    }
    return threw;
}

//
// EXR will read files that have out-of-range values in certain enum attributes, to allow
// values to be added in the future. This function returns 'false' if any such enum attributes
// have unknown values
//
// (implementation node: it is undefined behavior to set an enum variable to an invalid value
//  this code circumvents that by casting the enums to integers and checking them that way)
//
bool
enumsValid (const Header& hdr)
{
    if (hasEnvmap (hdr))
    {

        const Envmap& typeInFile = envmap (hdr);
        if (typeInFile != ENVMAP_LATLONG && typeInFile != ENVMAP_CUBE)
        {
            return false;
        }
    }

    if (hasDeepImageState (hdr))
    {
        const DeepImageState& typeInFile = deepImageState (hdr);
        if (typeInFile < 0 || typeInFile >= DIS_NUMSTATES) { return false; }
    }

    return true;
}

bool
readMultiPart (MultiPartInputFile& in, bool reduceMemory, bool reduceTime)
{
    bool threw = false;
    for (int part = 0; part < in.parts (); ++part)
    {

        if (!enumsValid (in.header (part))) { threw = true; }

        bool     widePart      = false;
        bool     largeTiles    = false;
        Box2i    b             = in.header (part).dataWindow ();
        int      bytesPerPixel = calculateBytesPerPixel (in.header (part));
        uint64_t imageWidth    = static_cast<uint64_t> (b.max.x) -
                              static_cast<uint64_t> (b.min.x) + 1ll;
        uint64_t scanlinesInBuffer =
            numLinesInBuffer (in.header (part).compression ());

        //
        // very wide scanline parts take excessive memory to read.
        // compute memory required to store a group of scanlines
        // so tests can be skipped when reduceMemory is set
        //

        if (imageWidth * bytesPerPixel * scanlinesInBuffer >
            gMaxBytesPerScanline)
        {
            widePart = true;
        }
        //
        // significant memory is also required to read a tiled part
        // using the scanline interface with tall tiles - the scanlineAPI
        // needs to allocate memory to store an entire row of tiles
        //
        if (isTiled (in.header (part).type ()))
        {
            const TileDescription& tileDescription =
                in.header (part).tileDescription ();

            uint64_t tilesPerScanline =
                (imageWidth + tileDescription.xSize - 1ll) /
                tileDescription.xSize;
            uint64_t tileSize = static_cast<uint64_t> (tileDescription.xSize) *
                                static_cast<uint64_t> (tileDescription.ySize);

            if (tileSize * tilesPerScanline * bytesPerPixel >
                gMaxTileBytesPerScanline)
            {
                widePart = true;
            }
            if (tileSize * bytesPerPixel > gMaxTileBytes) { largeTiles = true; }
        }

        if (!reduceMemory || !widePart)
        {
            bool gotThrow = false;
            try
            {
                InputPart pt (in, part);
                gotThrow = readScanline (pt, reduceMemory, reduceTime);
            }
            catch (...)
            {
                gotThrow = true;
            }
            // only 'DeepTiled' parts are expected to throw
            // all others are an error
            if (gotThrow && in.header (part).type () != DEEPTILE)
            {
                threw = true;
            }
        }

        if (!reduceMemory || !largeTiles)
        {
            bool gotThrow = false;

            try
            {
                in.flushPartCache ();
                TiledInputPart pt (in, part);
                gotThrow = readTile (pt, reduceMemory, reduceTime);
            }
            catch (...)
            {
                gotThrow = true;
            }

            if (gotThrow && in.header (part).type () == TILEDIMAGE)
            {
                threw = true;
            }
        }

        if (!reduceMemory || !widePart)
        {
            bool gotThrow = false;

            try
            {
                in.flushPartCache ();
                DeepScanLineInputPart pt (in, part);
                gotThrow = readDeepScanLine (pt, reduceMemory, reduceTime);
            }
            catch (...)
            {
                gotThrow = true;
            }

            if (gotThrow && in.header (part).type () == DEEPSCANLINE)
            {
                threw = true;
            }
        }

        if (!reduceMemory || !largeTiles)
        {
            bool gotThrow = false;

            try
            {
                in.flushPartCache ();
                DeepTiledInputPart pt (in, part);
                gotThrow = readDeepTile (pt, reduceMemory, reduceTime);
            }
            catch (...)
            {
                gotThrow = true;
            }

            if (gotThrow && in.header (part).type () == DEEPTILE)
            {
                threw = true;
            }
        }
    }

    return threw;
}

//------------------------------------------------
// class PtrIStream -- allow reading an EXR file from
// a pointer
//
//------------------------------------------------

class PtrIStream : public IStream
{
public:
    PtrIStream (const char* data, size_t nBytes)
        : IStream ("none"), base (data), current (data), end (data + nBytes)
    {}

    virtual bool isMemoryMapped () const { return false; }

    virtual char* readMemoryMapped (int n)
    {

        if (n + current > end)
        {
            THROW (
                IEX_NAMESPACE::InputExc,
                "Early end of file: requesting "
                    << end - (n + current) << " extra bytes after file\n");
        }
        const char* value = current;
        current += n;

        return const_cast<char*> (value);
    }

    virtual bool read (char c[/*n*/], int n)
    {
        if (n < 0)
        {
            THROW (
                IEX_NAMESPACE::InputExc, n << " bytes requested from stream");
        }

        if (n + current > end)
        {
            THROW (
                IEX_NAMESPACE::InputExc,
                "Early end of file: requesting "
                    << end - (n + current) << " extra bytes after file\n");
        }
        memcpy (c, current, n);
        current += n;

        return (current != end);
    }

    virtual uint64_t tellg () { return (current - base); }
    virtual void     seekg (uint64_t pos)
    {

        if (pos < 0)
        {
            THROW (
                IEX_NAMESPACE::InputExc,
                "internal error: seek to " << pos << " requested");
        }

        const char* newcurrent = base + pos;

        if (newcurrent < base || newcurrent > end)
        {
            THROW (IEX_NAMESPACE::InputExc, "Out of range seek requested\n");
        }

        current = newcurrent;
    }

private:
    const char* base;
    const char* current;
    const char* end;
};

void
resetInput (const char* /*fileName*/)
{
    // do nothing: filename doesn't need to be 'reset' between calls
}

void
resetInput (PtrIStream& stream)
{
    // return stream to beginning to prepare reading with a different API
    stream.seekg (0);
}

template <class T>
bool
runChecks (T& source, bool reduceMemory, bool reduceTime)
{

    //
    // in reduceMemory/reduceTime mode, limit image size, tile size, and maximum deep samples
    //

    uint64_t oldMaxSampleCount = CompositeDeepScanLine::getMaximumSampleCount();

    int maxImageWidth , maxImageHeight;
    Header::getMaxImageSize(maxImageWidth,maxImageHeight);

    int maxTileWidth , maxTileHeight;
    Header::getMaxImageSize(maxTileWidth,maxTileHeight);


    if( reduceMemory || reduceTime)
    {
        CompositeDeepScanLine::setMaximumSampleCount(1<<20);
        Header::setMaxImageSize(2048,2048);
        Header::setMaxTileSize(512,512);
    }




    //
    // multipart test: also grab the type of the first part to
    // check which other tests are expected to fail
    // check the image width for the first part - significant memory
    // is required to process wide parts
    //

    string firstPartType;

    //
    // scanline images with very wide parts and tiled images with large tiles
    // take excessive memory to read.
    // Assume the first part requires excessive memory until the header of the first part is checked
    // so the single part input APIs can be skipped.
    //
    // If the MultiPartInputFile constructor throws an exception, the first part
    // will assumed to be a wide image
    //
    bool largeTiles    = true;

    bool threw = false;
    {
        try
        {
            MultiPartInputFile multi (source);

            //
            // significant memory is also required to read a tiled file
            // using the scanline interface with tall tiles - the scanlineAPI
            // needs to allocate memory to store an entire row of tiles
            //

            firstPartType = multi.header (0).type ();
            if (isTiled (firstPartType))
            {
                const TileDescription& tileDescription =
                    multi.header (0).tileDescription ();
                uint64_t tileSize =
                    static_cast<uint64_t> (tileDescription.xSize) *
                    static_cast<uint64_t> (tileDescription.ySize);
                int bytesPerPixel = calculateBytesPerPixel (multi.header (0));

                if (tileSize * bytesPerPixel <= gMaxTileBytes)
                {
                    largeTiles = false;
                }
            }
            else
            {
                // file is not tiled, so can't contain large tiles
                // setting largeTiles false here causes the Tile and DeepTile API
                // tests to run on non-tiled files, which should cause exceptions to be thrown
                largeTiles = false;
            }

            threw = readMultiPart (multi, reduceMemory, reduceTime);
        }
        catch (...)
        {
            threw = true;
        }
    }

    // read using both scanline interfaces (unless the image is wide and reduce memory enabled)
    if (!reduceMemory)
    {
        {
            bool gotThrow = false;
            resetInput (source);
            try
            {
                RgbaInputFile rgba (source);
                gotThrow = readRgba (rgba, reduceMemory, reduceTime);
            }
            catch (...)
            {
                gotThrow = true;
            }
            if (gotThrow && firstPartType != DEEPTILE) { threw = true; }
        }
        {
            bool gotThrow = false;
            resetInput (source);
            try
            {
                InputFile rgba (source);
                gotThrow = readScanline (rgba, reduceMemory, reduceTime);
            }
            catch (...)
            {
                gotThrow = true;
            }
            if (gotThrow && firstPartType != DEEPTILE) { threw = true; }
        }
    }

    if (!reduceMemory || !largeTiles)
    {
        bool gotThrow = false;
        resetInput (source);
        try
        {
            TiledInputFile rgba (source);
            gotThrow = readTile (rgba, reduceMemory, reduceTime);
        }
        catch (...)
        {
            gotThrow = true;
        }
        if (gotThrow && firstPartType == TILEDIMAGE) { threw = true; }
    }

    if (!reduceMemory)
    {
        bool gotThrow = false;
        resetInput (source);
        try
        {
            DeepScanLineInputFile rgba (source);
            gotThrow = readDeepScanLine (rgba, reduceMemory, reduceTime);
        }
        catch (...)
        {
            gotThrow = true;
        }
        if (gotThrow && firstPartType == DEEPSCANLINE) { threw = true; }
    }

    if (!reduceMemory || !largeTiles)
    {
        bool gotThrow = false;
        resetInput (source);
        try
        {
            DeepTiledInputFile rgba (source);
            gotThrow = readDeepTile (rgba, reduceMemory, reduceTime);
        }
        catch (...)
        {
            gotThrow = true;
        }
        if (gotThrow && firstPartType == DEEPTILE) { threw = true; }
    }



    CompositeDeepScanLine::setMaximumSampleCount(oldMaxSampleCount);
    Header::setMaxImageSize(maxImageWidth,maxImageHeight);
    Header::setMaxTileSize(maxTileWidth,maxTileHeight);

    return threw;
}

// This is not entirely needed in that the chunk info has the
// total unpacked_size field which can be used for allocation
// but this adds an additional point to use when debugging issues.
static exr_result_t
realloc_deepdata(exr_decode_pipeline_t* decode)
{
    int32_t w = decode->chunk.width;
    int32_t h = decode->chunk.height;
    uint64_t totsamps = 0, bytes = 0;
    const int32_t *sampbuffer = decode->sample_count_table;
    std::vector<uint8_t>* ud = static_cast<std::vector<uint8_t>*>(
        decode->decoding_user_data);

    if ( ! ud )
    {
        for (int c = 0; c < decode->channel_count; c++)
        {
            exr_coding_channel_info_t& outc = decode->channels[c];
            outc.decode_to_ptr              = NULL;
            outc.user_pixel_stride          = outc.user_bytes_per_element;
            outc.user_line_stride           = 0;
        }
        return EXR_ERR_SUCCESS;
    }

    if ((decode->decode_flags &
         EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL))
    {
        for (int32_t y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
                totsamps += sampbuffer[x];
            sampbuffer += w;
        }
    }
    else
    {
        for (int32_t y = 0; y < h; ++y)
            totsamps += sampbuffer[y*w + w - 1];
    }

    for (int c = 0; c < decode->channel_count; c++)
    {
        exr_coding_channel_info_t& outc = decode->channels[c];
        bytes += totsamps * outc.user_bytes_per_element;
    }

    if (bytes >= gMaxBytesPerDeepScanline)
    {
        for (int c = 0; c < decode->channel_count; c++)
        {
            exr_coding_channel_info_t& outc = decode->channels[c];
            outc.decode_to_ptr              = NULL;
            outc.user_pixel_stride          = outc.user_bytes_per_element;
            outc.user_line_stride           = 0;
        }
        return EXR_ERR_SUCCESS;
    }

    if (ud->size () < bytes)
        ud->resize (bytes);

    uint8_t* dptr = &((*ud)[0]);
    for (int c = 0; c < decode->channel_count; c++)
    {
        exr_coding_channel_info_t& outc = decode->channels[c];
        outc.decode_to_ptr              = dptr;
        outc.user_pixel_stride          = outc.user_bytes_per_element;
        outc.user_line_stride           = 0;

        dptr += totsamps * (uint64_t) outc.user_bytes_per_element;
    }
    return EXR_ERR_SUCCESS;
}

////////////////////////////////////////

bool
readCoreScanlinePart (
    exr_context_t f, int part, bool reduceMemory, bool reduceTime)
{
    exr_result_t     rv, frv;
    exr_attr_box2i_t datawin;
    rv = exr_get_data_window (f, part, &datawin);
    if (rv != EXR_ERR_SUCCESS) return true;

    uint64_t width =
        (uint64_t) ((int64_t) datawin.max.x - (int64_t) datawin.min.x + 1);
    uint64_t height =
        (uint64_t) ((int64_t) datawin.max.y - (int64_t) datawin.min.y + 1);

    std::vector<uint8_t>  imgdata;
    bool                  doread = false;
    exr_decode_pipeline_t decoder = EXR_DECODE_PIPELINE_INITIALIZER;

    int32_t lines_per_chunk;
    rv = exr_get_scanlines_per_chunk (f, part, &lines_per_chunk);
    if (rv != EXR_ERR_SUCCESS) return true;

    frv = rv;

    for (uint64_t chunk = 0; chunk < height; chunk += lines_per_chunk)
    {
        exr_chunk_info_t cinfo = {0};
        int              y     = ((int) chunk) + datawin.min.y;

        rv = exr_read_scanline_chunk_info (f, part, y, &cinfo);
        if (rv != EXR_ERR_SUCCESS)
        {
            frv = rv;
            if (reduceTime) break;
            continue;
        }

        if (decoder.channels == NULL)
        {
            rv = exr_decoding_initialize (f, part, &cinfo, &decoder);
            if (rv != EXR_ERR_SUCCESS) break;

            uint64_t bytes = 0;
            for (int c = 0; c < decoder.channel_count; c++)
            {
                exr_coding_channel_info_t& outc = decoder.channels[c];
                // fake addr for default routines
                outc.decode_to_ptr     = (uint8_t*) 0x1000;
                outc.user_pixel_stride = outc.user_bytes_per_element;
                outc.user_line_stride  = outc.user_pixel_stride * width;
                bytes += width * (uint64_t) outc.user_bytes_per_element *
                         (uint64_t) lines_per_chunk;
            }

            doread = true;
            if (reduceMemory && bytes >= gMaxBytesPerScanline)
                doread = false;

            if (cinfo.type == EXR_STORAGE_DEEP_SCANLINE)
            {
                decoder.decoding_user_data       = &imgdata;
                decoder.realloc_nonimage_data_fn = &realloc_deepdata;
            }
            else
            {
                if (doread) imgdata.resize (bytes);
            }
            rv = exr_decoding_choose_default_routines (f, part, &decoder);
            if (rv != EXR_ERR_SUCCESS)
            {
                frv = rv;
                break;
            }
        }
        else
        {
            rv = exr_decoding_update (f, part, &cinfo, &decoder);
            if (rv != EXR_ERR_SUCCESS)
            {
                frv = rv;
                if (reduceTime) break;
                continue;
            }
        }

        if (doread)
        {
            if (cinfo.type != EXR_STORAGE_DEEP_SCANLINE)
            {
                uint8_t* dptr = &(imgdata[0]);
                for (int c = 0; c < decoder.channel_count; c++)
                {
                    exr_coding_channel_info_t& outc = decoder.channels[c];
                    outc.decode_to_ptr              = dptr;
                    outc.user_pixel_stride          = outc.user_bytes_per_element;
                    outc.user_line_stride           = outc.user_pixel_stride * width;

                    dptr += width * (uint64_t) outc.user_bytes_per_element *
                        (uint64_t) lines_per_chunk;
                }
            }

            rv = exr_decoding_run (f, part, &decoder);
            if (rv != EXR_ERR_SUCCESS)
            {
                frv = rv;
                if (reduceTime) break;
            }
        }
    }

    exr_decoding_destroy (f, &decoder);

    return (frv != EXR_ERR_SUCCESS);
}

////////////////////////////////////////

bool
readCoreTiledPart (
    exr_context_t f, int part, bool reduceMemory, bool reduceTime)
{
    exr_result_t rv, frv;

    exr_attr_box2i_t datawin;
    rv = exr_get_data_window (f, part, &datawin);
    if (rv != EXR_ERR_SUCCESS) return true;

    uint32_t              txsz, tysz;
    exr_tile_level_mode_t levelmode;
    exr_tile_round_mode_t roundingmode;

    rv = exr_get_tile_descriptor (
        f, part, &txsz, &tysz, &levelmode, &roundingmode);
    if (rv != EXR_ERR_SUCCESS) return true;

    int32_t levelsx, levelsy;
    rv = exr_get_tile_levels (f, part, &levelsx, &levelsy);
    if (rv != EXR_ERR_SUCCESS) return true;

    frv = rv;
    bool keepgoing = true;
    for (int32_t ylevel = 0; keepgoing && ylevel < levelsy; ++ylevel)
    {
        for (int32_t xlevel = 0; keepgoing && xlevel < levelsx; ++xlevel)
        {
            int32_t levw, levh;
            rv = exr_get_level_sizes (f, part, xlevel, ylevel, &levw, &levh);
            if (rv != EXR_ERR_SUCCESS)
            {
                frv = rv;
                if (reduceTime)
                {
                    keepgoing = false;
                    break;
                }
                continue;
            }

            int32_t curtw, curth;
            rv = exr_get_tile_sizes (f, part, xlevel, ylevel, &curtw, &curth);
            if (rv != EXR_ERR_SUCCESS)
            {
                frv = rv;
                if (reduceTime)
                {
                    keepgoing = false;
                    break;
                }
                continue;
            }

            // we could make this over all levels but then would have to
            // re-check the allocation size, let's leave it here to check when
            // tile size is < full / top level tile size
            std::vector<uint8_t>  tiledata;
            bool                  doread = false;
            exr_chunk_info_t      cinfo;
            exr_decode_pipeline_t decoder = EXR_DECODE_PIPELINE_INITIALIZER;

            int tx, ty;
            ty = 0;
            for (int64_t cury = 0; keepgoing && cury < levh;
                 cury += curth, ++ty)
            {
                tx = 0;
                for (int64_t curx = 0; keepgoing && curx < levw;
                     curx += curtw, ++tx)
                {
                    rv = exr_read_tile_chunk_info (
                        f, part, tx, ty, xlevel, ylevel, &cinfo);
                    if (rv != EXR_ERR_SUCCESS)
                    {
                        frv = rv;
                        if (reduceTime)
                        {
                            keepgoing = false;
                            break;
                        }
                        continue;
                    }

                    if (decoder.channels == NULL)
                    {
                        rv =
                            exr_decoding_initialize (f, part, &cinfo, &decoder);
                        if (rv != EXR_ERR_SUCCESS)
                        {
                            frv = rv;
                            keepgoing = false;
                            break;
                        }

                        uint64_t bytes = 0;
                        for (int c = 0; c < decoder.channel_count; c++)
                        {
                            exr_coding_channel_info_t& outc =
                                decoder.channels[c];
                            // fake addr for default routines
                            outc.decode_to_ptr = (uint8_t*) 0x1000 + bytes;
                            outc.user_pixel_stride =
                                outc.user_bytes_per_element;
                            outc.user_line_stride =
                                outc.user_pixel_stride * curtw;
                            bytes += (uint64_t) curtw *
                                     (uint64_t) outc.user_bytes_per_element *
                                     (uint64_t) curth;
                        }

                        doread = true;
                        if (reduceMemory && bytes >= gMaxTileBytes)
                            doread = false;

                        if (cinfo.type == EXR_STORAGE_DEEP_TILED)
                        {
                            decoder.decoding_user_data       = &tiledata;
                            decoder.realloc_nonimage_data_fn = &realloc_deepdata;
                        }
                        else
                        {
                            if (doread) tiledata.resize (bytes);
                        }
                        rv = exr_decoding_choose_default_routines (
                            f, part, &decoder);
                        if (rv != EXR_ERR_SUCCESS)
                        {
                            frv = rv;
                            keepgoing = false;
                            break;
                        }
                    }
                    else
                    {
                        rv = exr_decoding_update (f, part, &cinfo, &decoder);
                        if (rv != EXR_ERR_SUCCESS)
                        {
                            frv = rv;
                            if (reduceTime)
                            {
                                keepgoing = false;
                                break;
                            }
                            continue;
                        }
                    }

                    if (doread)
                    {
                        if (cinfo.type != EXR_STORAGE_DEEP_TILED)
                        {
                            uint8_t* dptr = &(tiledata[0]);
                            for (int c = 0; c < decoder.channel_count; c++)
                            {
                                exr_coding_channel_info_t& outc =
                                    decoder.channels[c];
                                outc.decode_to_ptr = dptr;
                                outc.user_pixel_stride =
                                    outc.user_bytes_per_element;
                                outc.user_line_stride =
                                    outc.user_pixel_stride * curtw;
                                dptr += (uint64_t) curtw *
                                    (uint64_t) outc.user_bytes_per_element *
                                    (uint64_t) curth;
                            }
                        }

                        rv = exr_decoding_run (f, part, &decoder);
                        if (rv != EXR_ERR_SUCCESS)
                        {
                            frv = rv;
                            if (reduceTime)
                            {
                                keepgoing = false;
                                break;
                            }
                        }
                    }
                }
            }

            exr_decoding_destroy (f, &decoder);
        }
    }

    return (rv != EXR_ERR_SUCCESS);
}

////////////////////////////////////////

bool
checkCoreFile (exr_context_t f, bool reduceMemory, bool reduceTime)
{
    exr_result_t rv;
    int          numparts;

    rv = exr_get_count (f, &numparts);
    if (rv != EXR_ERR_SUCCESS) return true;

    for (int p = 0; p < numparts; ++p)
    {
        exr_storage_t store;
        rv = exr_get_storage (f, p, &store);
        if (rv != EXR_ERR_SUCCESS) return true;

        if (store == EXR_STORAGE_SCANLINE ||
            store == EXR_STORAGE_DEEP_SCANLINE)
        {
            if (readCoreScanlinePart (f, p, reduceMemory, reduceTime))
                return true;
        }
        else if (store == EXR_STORAGE_TILED ||
                 store == EXR_STORAGE_DEEP_TILED)
        {
            if (readCoreTiledPart (f, p, reduceMemory, reduceTime)) return true;
        }
    }

    return false;
}

////////////////////////////////////////

static void
core_error_handler_cb (exr_const_context_t f, int code, const char* msg)
{
    if (getenv ("EXR_CHECK_ENABLE_PRINTS") != NULL)
    {
        const char* fn;
        if (EXR_ERR_SUCCESS != exr_get_file_name (f, &fn)) fn = "<error>";
        fprintf (
            stderr,
            "ERROR '%s' (%s): %s\n",
            fn,
            exr_get_error_code_as_string (code),
            msg);
    }
}

////////////////////////////////////////

bool
runCoreChecks (const char* filename, bool reduceMemory, bool reduceTime)
{
    exr_result_t              rv;
    bool                      hadfail = false;
    exr_context_t             f;
    exr_context_initializer_t cinit = EXR_DEFAULT_CONTEXT_INITIALIZER;

    cinit.error_handler_fn = &core_error_handler_cb;

    if (reduceMemory || reduceTime)
    {
        /* could use set_default functions for this, but those just
         * initialize the context, doing it in the initializer is mt
         * safe...
         * exr_set_default_maximum_image_size (2048, 2048);
         * exr_set_default_maximum_tile_size (512, 512);
         */
        cinit.max_image_width = 2048;
        cinit.max_image_height = 2048;
        cinit.max_tile_width = 512;
        cinit.max_tile_height = 512;
    }

    rv = exr_start_read (&f, filename, &cinit);
    if (rv != EXR_ERR_SUCCESS) return true;

    hadfail = checkCoreFile (f, reduceMemory, reduceTime);

    exr_finish (&f);

    return hadfail;
}

////////////////////////////////////////

struct memdata
{
    const char* data;
    size_t      bytes;
};

static int64_t
memstream_read (
    exr_const_context_t         f,
    void*                       userdata,
    void*                       buffer,
    uint64_t                    sz,
    uint64_t                    offset,
    exr_stream_error_func_ptr_t errcb)
{
    int64_t rdsz = -1;
    if (userdata)
    {
        memdata* md   = static_cast<memdata*> (userdata);
        uint64_t left = sz;
        if (offset > md->bytes ||  sz > md->bytes || offset+sz > md->bytes)
            left = (offset < md->bytes) ? md->bytes - offset : 0;
        if (left > 0) memcpy (buffer, md->data + offset, left);
        rdsz = static_cast<int64_t> (left);
    }

    return rdsz;
}

static int64_t
memstream_size (exr_const_context_t ctxt, void* userdata)
{
    if (userdata)
    {
        memdata* md = static_cast<memdata*> (userdata);
        return static_cast<int64_t> (md->bytes);
    }
    return -1;
}

bool
runCoreChecks (
    const char* data, size_t numBytes, bool reduceMemory, bool reduceTime)
{
    bool                      hadfail = false;
    exr_result_t              rv;
    exr_context_t             f;
    exr_context_initializer_t cinit = EXR_DEFAULT_CONTEXT_INITIALIZER;
    memdata                   md;

    md.data  = data;
    md.bytes = numBytes;

    cinit.user_data        = &md;
    cinit.read_fn          = &memstream_read;
    cinit.size_fn          = &memstream_size;
    cinit.error_handler_fn = &core_error_handler_cb;

    rv = exr_start_read (&f, "<memstream>", &cinit);
    if (rv != EXR_ERR_SUCCESS) return true;

    hadfail = checkCoreFile (f, reduceMemory, reduceTime);

    exr_finish (&f);

    return hadfail;
}

} // namespace

bool
checkOpenEXRFile (
    const char* fileName,
    bool        reduceMemory,
    bool        reduceTime,
    bool        runCoreCheck)
{

    if (runCoreCheck)
    {
        return runCoreChecks (fileName, reduceMemory, reduceTime);
    }
    else
    {
        return runChecks (fileName, reduceMemory, reduceTime);
    }

}

bool
checkOpenEXRFile (
    const char* data,
    size_t      numBytes,
    bool        reduceMemory,
    bool        reduceTime,
    bool        runCoreCheck)
{


     if (runCoreCheck)
     {
        return runCoreChecks (data, numBytes, reduceMemory, reduceTime);
     }
     else
     {
        PtrIStream stream (data, numBytes);
        return runChecks (stream, reduceMemory, reduceTime);
    }


}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
