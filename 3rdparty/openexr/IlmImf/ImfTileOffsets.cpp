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
//	class TileOffsets
//
//-----------------------------------------------------------------------------

#include <ImfTileOffsets.h>
#include <ImfXdr.h>
#include <ImfIO.h>
#include "Iex.h"

namespace Imf {


TileOffsets::TileOffsets (LevelMode mode,
              int numXLevels, int numYLevels,
              const int *numXTiles, const int *numYTiles)
:
    _mode (mode),
    _numXLevels (numXLevels),
    _numYLevels (numYLevels)
{
    switch (_mode)
    {
      case ONE_LEVEL:
      case MIPMAP_LEVELS:

        _offsets.resize (_numXLevels);

        for (unsigned int l = 0; l < _offsets.size(); ++l)
        {
            _offsets[l].resize (numYTiles[l]);

            for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
        {
                _offsets[l][dy].resize (numXTiles[l]);
            }
        }
        break;

      case RIPMAP_LEVELS:

        _offsets.resize (_numXLevels * _numYLevels);

        for (unsigned int ly = 0; ly < _numYLevels; ++ly)
        {
            for (unsigned int lx = 0; lx < _numXLevels; ++lx)
            {
                int l = ly * _numXLevels + lx;
                _offsets[l].resize (numYTiles[ly]);

                for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
                {
                    _offsets[l][dy].resize (numXTiles[lx]);
                }
            }
        }
        break;
    }
}


bool
TileOffsets::anyOffsetsAreInvalid () const
{
    for (unsigned int l = 0; l < _offsets.size(); ++l)
    for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
        for (unsigned int dx = 0; dx < _offsets[l][dy].size(); ++dx)
        if (_offsets[l][dy][dx] <= 0)
            return true;

    return false;
}


void
TileOffsets::findTiles (IStream &is)
{
    for (unsigned int l = 0; l < _offsets.size(); ++l)
    {
    for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
    {
        for (unsigned int dx = 0; dx < _offsets[l][dy].size(); ++dx)
        {
        Int64 tileOffset = is.tellg();

        int tileX;
        Xdr::read <StreamIO> (is, tileX);

        int tileY;
        Xdr::read <StreamIO> (is, tileY);

        int levelX;
        Xdr::read <StreamIO> (is, levelX);

        int levelY;
        Xdr::read <StreamIO> (is, levelY);

        int dataSize;
        Xdr::read <StreamIO> (is, dataSize);

        Xdr::skip <StreamIO> (is, dataSize);

        if (!isValidTile(tileX, tileY, levelX, levelY))
            return;

        operator () (tileX, tileY, levelX, levelY) = tileOffset;
        }
    }
    }
}


void
TileOffsets::reconstructFromFile (IStream &is)
{
    //
    // Try to reconstruct a missing tile offset table by sequentially
    // scanning through the file, and recording the offsets in the file
    // of the tiles we find.
    //

    Int64 position = is.tellg();

    try
    {
    findTiles (is);
    }
    catch (...)
    {
        //
        // Suppress all exceptions.  This function is called only to
    // reconstruct the tile offset table for incomplete files,
    // and exceptions are likely.
        //
    }

    is.clear();
    is.seekg (position);
}


void
TileOffsets::readFrom (IStream &is, bool &complete)
{
    //
    // Read in the tile offsets from the file's tile offset table
    //

    for (unsigned int l = 0; l < _offsets.size(); ++l)
    for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
        for (unsigned int dx = 0; dx < _offsets[l][dy].size(); ++dx)
        Xdr::read <StreamIO> (is, _offsets[l][dy][dx]);

    //
    // Check if any tile offsets are invalid.
    //
    // Invalid offsets mean that the file is probably incomplete
    // (the offset table is the last thing written to the file).
    // Either some process is still busy writing the file, or
    // writing the file was aborted.
    //
    // We should still be able to read the existing parts of the
    // file.  In order to do this, we have to make a sequential
    // scan over the scan tile to reconstruct the tile offset
    // table.
    //

    if (anyOffsetsAreInvalid())
    {
    complete = false;
    reconstructFromFile (is);
    }
    else
    {
    complete = true;
    }

}


Int64
TileOffsets::writeTo (OStream &os) const
{
    //
    // Write the tile offset table to the file, and
    // return the position of the start of the table
    // in the file.
    //

    Int64 pos = os.tellp();

    if (pos == -1)
    Iex::throwErrnoExc ("Cannot determine current file position (%T).");

    for (unsigned int l = 0; l < _offsets.size(); ++l)
    for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
        for (unsigned int dx = 0; dx < _offsets[l][dy].size(); ++dx)
        Xdr::write <StreamIO> (os, _offsets[l][dy][dx]);

    return pos;
}


bool
TileOffsets::isEmpty () const
{
    for (unsigned int l = 0; l < _offsets.size(); ++l)
    for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
        for (unsigned int dx = 0; dx < _offsets[l][dy].size(); ++dx)
        if (_offsets[l][dy][dx] != 0)
            return false;
    return true;
}


bool
TileOffsets::isValidTile (int dx, int dy, int lx, int ly) const
{
    switch (_mode)
    {
      case ONE_LEVEL:

        if (lx == 0 &&
        ly == 0 &&
        _offsets.size() > 0 &&
            _offsets[0].size() > dy &&
            _offsets[0][dy].size() > dx)
    {
            return true;
    }

        break;

      case MIPMAP_LEVELS:

        if (lx < _numXLevels &&
        ly < _numYLevels &&
            _offsets.size() > lx &&
            _offsets[lx].size() > dy &&
            _offsets[lx][dy].size() > dx)
    {
            return true;
    }

        break;

      case RIPMAP_LEVELS:

        if (lx < _numXLevels &&
        ly < _numYLevels &&
            _offsets.size() > lx + ly * _numXLevels &&
            _offsets[lx + ly * _numXLevels].size() > dy &&
            _offsets[lx + ly * _numXLevels][dy].size() > dx)
    {
            return true;
    }

        break;

      default:

        return false;
    }

    return false;
}


Int64 &
TileOffsets::operator () (int dx, int dy, int lx, int ly)
{
    //
    // Looks up the value of the tile with tile coordinate (dx, dy)
    // and level number (lx, ly) in the _offsets array, and returns
    // the cooresponding offset.
    //

    switch (_mode)
    {
      case ONE_LEVEL:

        return _offsets[0][dy][dx];
        break;

      case MIPMAP_LEVELS:

        return _offsets[lx][dy][dx];
        break;

      case RIPMAP_LEVELS:

        return _offsets[lx + ly * _numXLevels][dy][dx];
        break;

      default:

        throw Iex::ArgExc ("Unknown LevelMode format.");
    }
}


Int64 &
TileOffsets::operator () (int dx, int dy, int l)
{
    return operator () (dx, dy, l, l);
}


const Int64 &
TileOffsets::operator () (int dx, int dy, int lx, int ly) const
{
    //
    // Looks up the value of the tile with tile coordinate (dx, dy)
    // and level number (lx, ly) in the _offsets array, and returns
    // the cooresponding offset.
    //

    switch (_mode)
    {
      case ONE_LEVEL:

        return _offsets[0][dy][dx];
        break;

      case MIPMAP_LEVELS:

        return _offsets[lx][dy][dx];
        break;

      case RIPMAP_LEVELS:

        return _offsets[lx + ly * _numXLevels][dy][dx];
        break;

      default:

        throw Iex::ArgExc ("Unknown LevelMode format.");
    }
}


const Int64 &
TileOffsets::operator () (int dx, int dy, int l) const
{
    return operator () (dx, dy, l, l);
}


} // namespace Imf
