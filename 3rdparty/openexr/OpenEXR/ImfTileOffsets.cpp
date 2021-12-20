//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class TileOffsets
//
//-----------------------------------------------------------------------------

#include <ImfTileOffsets.h>
#include <ImfXdr.h>
#include <ImfIO.h>
#include "Iex.h"
#include "ImfNamespace.h"
#include <algorithm>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


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

        for (int ly = 0; ly < _numYLevels; ++ly)
        {
            for (int lx = 0; lx < _numXLevels; ++lx)
            {
                int l = ly * _numXLevels + lx;
                _offsets[l].resize (numYTiles[ly]);

                for (size_t dy = 0; dy < _offsets[l].size(); ++dy)
                {
                    _offsets[l][dy].resize (numXTiles[lx]);
                }
            }
        }
        break;

      case NUM_LEVELMODES :
          throw IEX_NAMESPACE::ArgExc("Bad initialisation of TileOffsets object");
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
TileOffsets::findTiles (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, bool isMultiPartFile, bool isDeep, bool skipOnly)
{
    for (unsigned int l = 0; l < _offsets.size(); ++l)
    {
	for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
	{
	    for (unsigned int dx = 0; dx < _offsets[l][dy].size(); ++dx)
	    {
		uint64_t tileOffset = is.tellg();

		if (isMultiPartFile)
		{
		    int partNumber;
		    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, partNumber);
		}

		int tileX;
		OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, tileX);

		int tileY;
		OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, tileY);

		int levelX;
		OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, levelX);

		int levelY;
		OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, levelY);

                if(isDeep)
                {
                     uint64_t packed_offset_table_size;
                     uint64_t packed_sample_size;
                     
                     OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, packed_offset_table_size);
                     OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, packed_sample_size);
                     
                     // check for bad values to prevent overflow
                     if ( packed_offset_table_size < 0 ||
                          packed_sample_size < 0 ||
                         ( INT64_MAX -  packed_offset_table_size < packed_sample_size) ||
                         ( INT64_MAX - (packed_offset_table_size + packed_sample_size) ) < 8 )
                     {
                          throw IEX_NAMESPACE::IoExc("Invalid deep tile size");
                     }

                     // next uint64_t is unpacked sample size - skip that too
                     Xdr::skip <StreamIO> (is, packed_offset_table_size+packed_sample_size+8);
                    
                }
                else
                {
                     int dataSize;
		     OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, dataSize);

                     // check for bad values to prevent overflow
                     if ( dataSize < 0 )
                     {
                          throw IEX_NAMESPACE::IoExc("Invalid tile size");
                     }

		     Xdr::skip <StreamIO> (is, dataSize);
                }
		if (skipOnly) continue;

		if (!isValidTile(tileX, tileY, levelX, levelY))
		    return;

		operator () (tileX, tileY, levelX, levelY) = tileOffset;
	    }
	}
    }
}


void
TileOffsets::reconstructFromFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,bool isMultiPart,bool isDeep)
{
    //
    // Try to reconstruct a missing tile offset table by sequentially
    // scanning through the file, and recording the offsets in the file
    // of the tiles we find.
    //

    uint64_t position = is.tellg();

    try
    {
	findTiles (is,isMultiPart,isDeep,false);
    }
    catch (...) //NOSONAR - suppress vulnerability reports from SonarCloud.
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
TileOffsets::readFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, bool &complete,bool isMultiPartFile, bool isDeep)
{
    //
    // Read in the tile offsets from the file's tile offset table
    //

    for (unsigned int l = 0; l < _offsets.size(); ++l)
	for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
	    for (unsigned int dx = 0; dx < _offsets[l][dy].size(); ++dx)
		OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, _offsets[l][dy][dx]);

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
	reconstructFromFile (is,isMultiPartFile,isDeep);
    }
    else
    {
	complete = true;
    }

}


void
TileOffsets::readFrom (std::vector<uint64_t> chunkOffsets,bool &complete)
{
    size_t totalSize = 0;
 
    for (unsigned int l = 0; l < _offsets.size(); ++l)
        for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
            totalSize += _offsets[l][dy].size();

    if (chunkOffsets.size() != totalSize)
        throw IEX_NAMESPACE::ArgExc ("Wrong offset count, not able to read from this array");



    int pos = 0;
    for (size_t l = 0; l < _offsets.size(); ++l)
        for (size_t dy = 0; dy < _offsets[l].size(); ++dy)
            for (size_t dx = 0; dx < _offsets[l][dy].size(); ++dx)
            {
                _offsets[l][dy][dx] = chunkOffsets[pos];
                pos++;
            }

    complete = !anyOffsetsAreInvalid();

}


uint64_t
TileOffsets::writeTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os) const
{
    //
    // Write the tile offset table to the file, and
    // return the position of the start of the table
    // in the file.
    //
    
    uint64_t pos = os.tellp();

    if (pos == static_cast<uint64_t>(-1))
	IEX_NAMESPACE::throwErrnoExc ("Cannot determine current file position (%T).");

    for (unsigned int l = 0; l < _offsets.size(); ++l)
	for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
	    for (unsigned int dx = 0; dx < _offsets[l][dy].size(); ++dx)
		OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (os, _offsets[l][dy][dx]);

    return pos;
}

namespace {
struct tilepos{
    uint64_t filePos;
    int dx;
    int dy;
    int l;
    bool operator <(const tilepos & other) const
    {
        return filePos < other.filePos;
    }
};
}
//-------------------------------------
// fill array with tile coordinates in the order they appear in the file
//
// each input array must be of size (totalTiles)
// 
//
// if the tile order is not RANDOM_Y, it is more efficient to compute the
// tile ordering rather than using this function
//
//-------------------------------------
void TileOffsets::getTileOrder(int dx_table[],int dy_table[],int lx_table[],int ly_table[]) const
{
    // 
    // helper class
    // 

    // how many entries?
    size_t entries=0;
    for (unsigned int l = 0; l < _offsets.size(); ++l)
        for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
           entries+=_offsets[l][dy].size();
        
    std::vector<struct tilepos> table(entries);
    
    size_t i = 0;
    for (unsigned int l = 0; l < _offsets.size(); ++l)
        for (unsigned int dy = 0; dy < _offsets[l].size(); ++dy)
            for (unsigned int dx = 0; dx < _offsets[l][dy].size(); ++dx)
            {
                table[i].filePos = _offsets[l][dy][dx];
                table[i].dx = dx;
                table[i].dy = dy;
                table[i].l = l;

                ++i;
                
            }
              
    std::sort(table.begin(),table.end());
    
    //
    // write out the values
    //
    
    // pass 1: write out dx and dy, since these are independent of level mode
    
    for(size_t i=0;i<entries;i++)
    {
        dx_table[i] = table[i].dx;
        dy_table[i] = table[i].dy;
    }

    // now write out the levels, which depend on the level mode
    
    switch (_mode)
    {
        case ONE_LEVEL:
        {
            for(size_t i=0;i<entries;i++)
            {
                lx_table[i] = 0;
                ly_table[i] = 0;               
            }
            break;            
        }
        case MIPMAP_LEVELS:
        {
            for(size_t i=0;i<entries;i++)
            {
                lx_table[i]= table[i].l;
                ly_table[i] =table[i].l;               
                
            }
            break;
        }
            
        case RIPMAP_LEVELS:
        {
            for(size_t i=0;i<entries;i++)
            {
                lx_table[i]= table[i].l % _numXLevels;
                ly_table[i] = table[i].l / _numXLevels; 
                
            }
            break;
        }
        case NUM_LEVELMODES :
            throw IEX_NAMESPACE::LogicExc("Bad level mode getting tile order");
    }
    
    
    
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
    if(lx<0 || ly < 0 || dx<0 || dy < 0) return false;
    switch (_mode)
    {
      case ONE_LEVEL:

        if (lx == 0 &&
	    ly == 0 &&
	    _offsets.size() > 0 &&
            int(_offsets[0].size()) > dy &&
            int(_offsets[0][dy].size()) > dx)
	{
            return true;
	}

        break;

      case MIPMAP_LEVELS:

        if (lx < _numXLevels &&
	    ly < _numYLevels &&
            int(_offsets.size()) > lx &&
            int(_offsets[lx].size()) > dy &&
            int(_offsets[lx][dy].size()) > dx)
	{
            return true;
	}

        break;

      case RIPMAP_LEVELS:

        if (lx < _numXLevels &&
	    ly < _numYLevels &&
	    (_offsets.size() > (size_t) lx+  ly *  (size_t) _numXLevels) &&
            int(_offsets[lx + ly * _numXLevels].size()) > dy &&
            int(_offsets[lx + ly * _numXLevels][dy].size()) > dx)
	{
            return true;
	}

        break;

      default:

        return false;
    }
    
    return false;
}


uint64_t &
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

      case MIPMAP_LEVELS:

        return _offsets[lx][dy][dx];

      case RIPMAP_LEVELS:

        return _offsets[lx + ly * _numXLevels][dy][dx];

      default:

        throw IEX_NAMESPACE::ArgExc ("Unknown LevelMode format.");
    }
}


uint64_t &
TileOffsets::operator () (int dx, int dy, int l)
{
    return operator () (dx, dy, l, l);
}


const uint64_t &
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

      case MIPMAP_LEVELS:

        return _offsets[lx][dy][dx];

      case RIPMAP_LEVELS:

        return _offsets[lx + ly * _numXLevels][dy][dx];

      default:

        throw IEX_NAMESPACE::ArgExc ("Unknown LevelMode format.");
    }
}


const uint64_t &
TileOffsets::operator () (int dx, int dy, int l) const
{
    return operator () (dx, dy, l, l);
}

const std::vector<std::vector<std::vector <uint64_t> > >&
TileOffsets::getOffsets() const
{
    return _offsets;
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
