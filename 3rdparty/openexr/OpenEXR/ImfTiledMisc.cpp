//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	Miscellaneous stuff related to tiled files
//
//-----------------------------------------------------------------------------

#include <ImfTiledMisc.h>
#include "Iex.h"
#include <ImfMisc.h>
#include <ImfChannelList.h>
#include <ImfHeader.h>
#include <ImfTileDescription.h>
#include <algorithm>
#include <limits>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using IMATH_NAMESPACE::Box2i;
using IMATH_NAMESPACE::V2i;


int
levelSize (int min, int max, int l, LevelRoundingMode rmode)
{
    if (l < 0)
	throw IEX_NAMESPACE::ArgExc ("Argument not in valid range.");

    int a = max - min + 1;
    int b = (1 << l);
    int size = a / b;

    if (rmode == ROUND_UP && size * b < a)
	size += 1;

    return std::max (size, 1);
}


Box2i
dataWindowForLevel (const TileDescription &tileDesc,
		    int minX, int maxX,
		    int minY, int maxY,
		    int lx, int ly)
{
    V2i levelMin = V2i (minX, minY);

    V2i levelMax = levelMin +
		   V2i (levelSize (minX, maxX, lx, tileDesc.roundingMode) - 1,
			levelSize (minY, maxY, ly, tileDesc.roundingMode) - 1);

    return Box2i(levelMin, levelMax);
}


Box2i
dataWindowForTile (const TileDescription &tileDesc,
		   int minX, int maxX,
		   int minY, int maxY,
		   int dx, int dy,
		   int lx, int ly)
{
    V2i tileMin = V2i (minX + dx * tileDesc.xSize,
		       minY + dy * tileDesc.ySize);

    int64_t tileMaxX = int64_t(tileMin[0]) + tileDesc.xSize - 1;
    int64_t tileMaxY = int64_t(tileMin[1]) + tileDesc.ySize - 1;

    V2i levelMax = dataWindowForLevel
		       (tileDesc, minX, maxX, minY, maxY, lx, ly).max;

    V2i tileMax = V2i (std::min (tileMaxX, int64_t(levelMax[0])),
		   std::min (tileMaxY, int64_t(levelMax[1])));

    return Box2i (tileMin, tileMax);
}


size_t
calculateBytesPerPixel (const Header &header)
{
    const ChannelList &channels = header.channels();

    size_t bytesPerPixel = 0;

    for (ChannelList::ConstIterator c = channels.begin();
	 c != channels.end();
	 ++c)
    {
	bytesPerPixel += pixelTypeSize (c.channel().type);
    }

    return bytesPerPixel;
}


void
calculateBytesPerLine (const Header &header,
                       char* sampleCountBase,
                       int sampleCountXStride,
                       int sampleCountYStride,
                       int minX, int maxX,
                       int minY, int maxY,
                       std::vector<int>& xOffsets,
                       std::vector<int>& yOffsets,
                       std::vector<uint64_t>& bytesPerLine)
{
    const ChannelList &channels = header.channels();

    int pos = 0;
    for (ChannelList::ConstIterator c = channels.begin();
         c != channels.end();
         ++c, ++pos)
    {
        int xOffset = xOffsets[pos];
        int yOffset = yOffsets[pos];
        int i = 0;
        for (int y = minY - yOffset; y <= maxY - yOffset; y++, i++)
            for (int x = minX - xOffset; x <= maxX - xOffset; x++)
            {
                bytesPerLine[i] += sampleCount(sampleCountBase,
                                               sampleCountXStride,
                                               sampleCountYStride,
                                               x, y)
                                   * pixelTypeSize (c.channel().type);
            }
    }
}


namespace {

int
floorLog2 (int x)
{
    //
    // For x > 0, floorLog2(y) returns floor(log(x)/log(2)).
    //

    int y = 0;

    while (x > 1)
    {
	y +=  1;
	x >>= 1;
    }

    return y;
}


int
ceilLog2 (int x)
{
    //
    // For x > 0, ceilLog2(y) returns ceil(log(x)/log(2)).
    //

    int y = 0;
    int r = 0;

    while (x > 1)
    {
	if (x & 1)
	    r = 1;

	y +=  1;
	x >>= 1;
    }

    return y + r;
}


int
roundLog2 (int x, LevelRoundingMode rmode)
{
    return (rmode == ROUND_DOWN)? floorLog2 (x): ceilLog2 (x);
}


int
calculateNumXLevels (const TileDescription& tileDesc,
		     int minX, int maxX,
		     int minY, int maxY)
{
    int num = 0;

    switch (tileDesc.mode)
    {
      case ONE_LEVEL:

	num = 1;
	break;

      case MIPMAP_LEVELS:

	{
	  int w = maxX - minX + 1;
	  int h = maxY - minY + 1;
	  num = roundLog2 (std::max (w, h), tileDesc.roundingMode) + 1;
	}
        break;

      case RIPMAP_LEVELS:

	{
	  int w = maxX - minX + 1;
	  num = roundLog2 (w, tileDesc.roundingMode) + 1;
	}
	break;

      default:

	throw IEX_NAMESPACE::ArgExc ("Unknown LevelMode format.");
    }

    return num;
}


int
calculateNumYLevels (const TileDescription& tileDesc,
		     int minX, int maxX,
		     int minY, int maxY)
{
    int num = 0;

    switch (tileDesc.mode)
    {
      case ONE_LEVEL:

	num = 1;
	break;

      case MIPMAP_LEVELS:

	{
	  int w = maxX - minX + 1;
	  int h = maxY - minY + 1;
	  num = roundLog2 (std::max (w, h), tileDesc.roundingMode) + 1;
	}
        break;

      case RIPMAP_LEVELS:

	{
	  int h = maxY - minY + 1;
	  num = roundLog2 (h, tileDesc.roundingMode) + 1;
	}
	break;

      default:

	throw IEX_NAMESPACE::ArgExc ("Unknown LevelMode format.");
    }

    return num;
}


void
calculateNumTiles (int *numTiles,
		   int numLevels,
		   int min, int max,
		   int size,
		   LevelRoundingMode rmode)
{
    for (int i = 0; i < numLevels; i++)
    {
        // use 64 bits to avoid int overflow if size is large.
        uint64_t l = levelSize (min, max, i, rmode);
        numTiles[i] = (l + size - 1) / size;
    }
}

} // namespace


void
precalculateTileInfo (const TileDescription& tileDesc,
		      int minX, int maxX,
		      int minY, int maxY,
		      int *&numXTiles, int *&numYTiles,
		      int &numXLevels, int &numYLevels)
{
    numXLevels = calculateNumXLevels(tileDesc, minX, maxX, minY, maxY);
    numYLevels = calculateNumYLevels(tileDesc, minX, maxX, minY, maxY);
    
    numXTiles = new int[numXLevels];
    numYTiles = new int[numYLevels];

    calculateNumTiles (numXTiles,
		       numXLevels,
		       minX, maxX,
		       tileDesc.xSize,
		       tileDesc.roundingMode);

    calculateNumTiles (numYTiles,
		       numYLevels,
		       minY, maxY,
		       tileDesc.ySize,
		       tileDesc.roundingMode);
}


int
getTiledChunkOffsetTableSize(const Header& header)
{
    //
    // Save the dataWindow information
    //

    const Box2i &dataWindow = header.dataWindow();
    
    //
    // Precompute level and tile information.
    //

    int* numXTiles=nullptr;
    int* numYTiles=nullptr;
    int numXLevels;
    int numYLevels;
    try
    {
        precalculateTileInfo (header.tileDescription(),
                            dataWindow.min.x, dataWindow.max.x,
                            dataWindow.min.y, dataWindow.max.y,
                            numXTiles, numYTiles,
                            numXLevels, numYLevels);

        //
        // Calculate lineOffsetSize.
        //
        uint64_t lineOffsetSize = 0;
        const TileDescription &desc = header.tileDescription();
        switch (desc.mode)
        {
            case ONE_LEVEL:
            case MIPMAP_LEVELS:
                for (int i = 0; i < numXLevels; i++)
                {
                    lineOffsetSize += static_cast<uint64_t>(numXTiles[i]) * static_cast<uint64_t>(numYTiles[i]);
                    if ( lineOffsetSize > static_cast<uint64_t>(std::numeric_limits<int>::max()) )
                    {
                        throw IEX_NAMESPACE::LogicExc("Maximum number of tiles exceeded");
                    }
                }
            break;
            case RIPMAP_LEVELS:
                for (int i = 0; i < numXLevels; i++)
                {
                    for (int j = 0; j < numYLevels; j++)
                    {
                        lineOffsetSize += static_cast<uint64_t>(numXTiles[i]) * static_cast<uint64_t>(numYTiles[j]);
                        if ( lineOffsetSize > static_cast<uint64_t>(std::numeric_limits<int>::max()) )
                        {
                            throw IEX_NAMESPACE::LogicExc("Maximum number of tiles exceeded");
                        }
                    }
                }
            break;
            case NUM_LEVELMODES :
                throw IEX_NAMESPACE::LogicExc("Bad level mode getting chunk offset table size");
        }
        delete[] numXTiles;
        delete[] numYTiles;

        return static_cast<int>(lineOffsetSize);

    }
    catch(...)
    {
        delete[] numXTiles;
        delete[] numYTiles;

        throw;
    }

}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
