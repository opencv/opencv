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
//	Miscellaneous stuff related to tiled files
//
//-----------------------------------------------------------------------------

#include <ImfTiledMisc.h>
#include "Iex.h"
#include <ImfMisc.h>
#include <ImfChannelList.h>
#include <algorithm> // for std::max()


namespace Imf {

using Imath::Box2i;
using Imath::V2i;


int
levelSize (int min, int max, int l, LevelRoundingMode rmode)
{
    if (l < 0)
    throw Iex::ArgExc ("Argument not in valid range.");

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

    V2i tileMax = tileMin + V2i (tileDesc.xSize - 1, tileDesc.ySize - 1);

    V2i levelMax = dataWindowForLevel
               (tileDesc, minX, maxX, minY, maxY, lx, ly).max;

    tileMax = V2i (std::min (tileMax[0], levelMax[0]),
           std::min (tileMax[1], levelMax[1]));

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

    throw Iex::ArgExc ("Unknown LevelMode format.");
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

    throw Iex::ArgExc ("Unknown LevelMode format.");
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
    numTiles[i] = (levelSize (min, max, i, rmode) + size - 1) / size;
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


} // namespace Imf
