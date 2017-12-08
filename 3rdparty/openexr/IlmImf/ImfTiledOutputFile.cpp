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
//	class TiledOutputFile
//
//-----------------------------------------------------------------------------

#include <ImfTiledOutputFile.h>
#include <ImfTiledInputFile.h>
#include <ImfTiledInputPart.h>
#include <ImfInputFile.h>
#include <ImfInputPart.h>
#include <ImfTileDescriptionAttribute.h>
#include <ImfPreviewImageAttribute.h>
#include <ImfChannelList.h>
#include <ImfMisc.h>
#include <ImfTiledMisc.h>
#include <ImfStdIO.h>
#include <ImfCompressor.h>
#include "ImathBox.h"
#include <ImfArray.h>
#include <ImfXdr.h>
#include <ImfVersion.h>
#include <ImfTileOffsets.h>
#include <ImfThreading.h>
#include <ImfPartType.h>
#include "IlmThreadPool.h"
#include "IlmThreadSemaphore.h"
#include "IlmThreadMutex.h"
#include "ImfOutputStreamMutex.h"
#include "ImfOutputPartData.h"
#include "Iex.h"
#include <string>
#include <vector>
#include <fstream>
#include <assert.h>
#include <map>
#include <algorithm>

#include "ImfNamespace.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using IMATH_NAMESPACE::Box2i;
using IMATH_NAMESPACE::V2i;
using std::string;
using std::vector;
using std::ofstream;
using std::map;
using std::min;
using std::max;
using std::swap;
using ILMTHREAD_NAMESPACE::Mutex;
using ILMTHREAD_NAMESPACE::Lock;
using ILMTHREAD_NAMESPACE::Semaphore;
using ILMTHREAD_NAMESPACE::Task;
using ILMTHREAD_NAMESPACE::TaskGroup;
using ILMTHREAD_NAMESPACE::ThreadPool;

namespace {

struct TOutSliceInfo
{
    PixelType		type;
    const char *	base;
    size_t		xStride;
    size_t		yStride;
    bool		zero;
    int                 xTileCoords;
    int                 yTileCoords;

    TOutSliceInfo (PixelType type = HALF,
	           const char *base = 0,
	           size_t xStride = 0,
	           size_t yStride = 0,
	           bool zero = false,
                   int xTileCoords = 0,
                   int yTileCoords = 0);
};


TOutSliceInfo::TOutSliceInfo (PixelType t,
		              const char *b,
			      size_t xs, size_t ys,
			      bool z,
                              int xtc,
                              int ytc)
:
    type (t),
    base (b),
    xStride (xs),
    yStride (ys),
    zero (z),
    xTileCoords (xtc),
    yTileCoords (ytc)
{
    // empty
}


struct TileCoord
{
    int		dx;
    int		dy;
    int		lx;
    int		ly;
    

    TileCoord (int xTile = 0, int yTile = 0,
	       int xLevel = 0, int yLevel = 0)
    :
        dx (xTile),  dy (yTile),
	lx (xLevel), ly (yLevel)
    {
        // empty
    }
    

    bool
    operator < (const TileCoord &other) const
    {
        return (ly < other.ly) ||
	       (ly == other.ly && lx < other.lx) ||
	       ((ly == other.ly && lx == other.lx) &&
		    ((dy < other.dy) || (dy == other.dy && dx < other.dx)));
    }


    bool
    operator == (const TileCoord &other) const
    {
        return lx == other.lx &&
	       ly == other.ly &&
	       dx == other.dx &&
	       dy == other.dy;
    }
};


struct BufferedTile
{
    char *	pixelData;
    int		pixelDataSize;

    BufferedTile (const char *data, int size):
	pixelData (0),
	pixelDataSize(size)
    {
	pixelData = new char[pixelDataSize];
	memcpy (pixelData, data, pixelDataSize);
    }

    ~BufferedTile()
    {
	delete [] pixelData;
    }
};


typedef map <TileCoord, BufferedTile *> TileMap;


struct TileBuffer
{
    Array<char>		buffer;
    const char *	dataPtr;
    int			dataSize;
    Compressor *	compressor;
    TileCoord		tileCoord;
    bool		hasException;
    string		exception;

     TileBuffer (Compressor *comp);
    ~TileBuffer ();

    inline void		wait () {_sem.wait();}
    inline void		post () {_sem.post();}

  protected:

    Semaphore		_sem;
};


TileBuffer::TileBuffer (Compressor *comp):
    dataPtr (0),
    dataSize (0),
    compressor (comp),
    hasException (false),
    exception (),
    _sem (1)
{
    // empty
}


TileBuffer::~TileBuffer ()
{
    delete compressor;
}


} // namespace


struct TiledOutputFile::Data
{
    Header		header;			// the image header
    int			version;		// file format version
    bool                multipart;              // part came from a multipart file
    TileDescription	tileDesc;		// describes the tile layout
    FrameBuffer		frameBuffer;		// framebuffer to write into
    Int64		previewPosition;
    LineOrder		lineOrder;		// the file's lineorder
    int			minX;			// data window's min x coord
    int			maxX;			// data window's max x coord
    int			minY;			// data window's min y coord
    int			maxY;			// data window's max x coord

    int			numXLevels;		// number of x levels
    int			numYLevels;		// number of y levels
    int *		numXTiles;		// number of x tiles at a level
    int *		numYTiles;		// number of y tiles at a level

    TileOffsets		tileOffsets;		// stores offsets in file for
						// each tile

    Compressor::Format	format;			// compressor's data format
    vector<TOutSliceInfo> slices;		// info about channels in file

    size_t		maxBytesPerTileLine;	// combined size of a tile line
						// over all channels

    
    vector<TileBuffer*> tileBuffers;
    size_t		tileBufferSize;         // size of a tile buffer

    Int64		tileOffsetsPosition;	// position of the tile index
    
    TileMap		tileMap;
    TileCoord		nextTileToWrite;

    int                 partNumber;             // the output part number

     Data (int numThreads);
    ~Data ();
    
    inline TileBuffer *	getTileBuffer (int number);
    						// hash function from tile
						// buffer coords into our
						// vector of tile buffers
    
    TileCoord		nextTileCoord (const TileCoord &a);
};


TiledOutputFile::Data::Data (int numThreads):
    multipart(false),
    numXTiles(0),
    numYTiles(0),
    tileOffsetsPosition (0),
    partNumber(-1)
{
    //
    // We need at least one tileBuffer, but if threading is used,
    // to keep n threads busy we need 2*n tileBuffers
    //

    tileBuffers.resize (max (1, 2 * numThreads));
}


TiledOutputFile::Data::~Data ()
{
    delete [] numXTiles;
    delete [] numYTiles;

    //
    // Delete all the tile buffers, if any still happen to exist
    //
    
    for (TileMap::iterator i = tileMap.begin(); i != tileMap.end(); ++i)
	delete i->second;

    for (size_t i = 0; i < tileBuffers.size(); i++)
        delete tileBuffers[i];
}


TileBuffer*
TiledOutputFile::Data::getTileBuffer (int number)
{
    return tileBuffers[number % tileBuffers.size()];
}


TileCoord
TiledOutputFile::Data::nextTileCoord (const TileCoord &a)
{
    TileCoord b = a;
    
    if (lineOrder == INCREASING_Y)
    {
        b.dx++;

        if (b.dx >= numXTiles[b.lx])
        {
            b.dx = 0;
            b.dy++;

            if (b.dy >= numYTiles[b.ly])
            {
		//
		// the next tile is in the next level
		//

                b.dy = 0;

                switch (tileDesc.mode)
                {
                  case ONE_LEVEL:
                  case MIPMAP_LEVELS:

                    b.lx++;
                    b.ly++;
                    break;

                  case RIPMAP_LEVELS:

                    b.lx++;

                    if (b.lx >= numXLevels)
                    {
                        b.lx = 0;
                        b.ly++;

			#ifdef DEBUG
			    assert (b.ly <= numYLevels);
			#endif
                    }
                    break;
                  case  NUM_LEVELMODES:
                      throw(IEX_NAMESPACE::ArgExc("Invalid tile description"));
                      
                }
            }
        }
    }
    else if (lineOrder == DECREASING_Y)
    {
        b.dx++;

        if (b.dx >= numXTiles[b.lx])
        {
            b.dx = 0;
            b.dy--;

            if (b.dy < 0)
            {
		//
		// the next tile is in the next level
		//

                switch (tileDesc.mode)
                {
                  case ONE_LEVEL:
                  case MIPMAP_LEVELS:

                    b.lx++;
                    b.ly++;
                    break;

                  case RIPMAP_LEVELS:

                    b.lx++;

                    if (b.lx >= numXLevels)
                    {
                        b.lx = 0;
                        b.ly++;

			#ifdef DEBUG
			    assert (b.ly <= numYLevels);
			#endif
                    }
                    break;
                  case  NUM_LEVELMODES:
                      throw(IEX_NAMESPACE::ArgExc("Invalid tile description"));
                      
                }

		if (b.ly < numYLevels)
		    b.dy = numYTiles[b.ly] - 1;
            }
        }
    }
    
    return b;   
}


namespace {

void
writeTileData (OutputStreamMutex *streamData,
               TiledOutputFile::Data *ofd,
               int dx, int dy,
	       int lx, int ly, 
               const char pixelData[],
               int pixelDataSize)
{
    //
    // Store a block of pixel data in the output file, and try
    // to keep track of the current writing position the file,
    // without calling tellp() (tellp() can be fairly expensive).
    //

    Int64 currentPosition = streamData->currentPosition;
    streamData->currentPosition = 0;

    if (currentPosition == 0)
        currentPosition = streamData->os->tellp();

    ofd->tileOffsets (dx, dy, lx, ly) = currentPosition;

    #ifdef DEBUG
	assert (streamData->os->tellp() == currentPosition);
    #endif

    //
    // Write the tile header.
    //

    if (ofd->multipart)
    {
        Xdr::write <StreamIO> (*streamData->os, ofd->partNumber);
    }
    Xdr::write <StreamIO> (*streamData->os, dx);
    Xdr::write <StreamIO> (*streamData->os, dy);
    Xdr::write <StreamIO> (*streamData->os, lx);
    Xdr::write <StreamIO> (*streamData->os, ly);
    Xdr::write <StreamIO> (*streamData->os, pixelDataSize);

    streamData->os->write (pixelData, pixelDataSize);

    //
    // Keep current position in the file so that we can avoid 
    // redundant seekg() operations (seekg() can be fairly expensive).
    //

    streamData->currentPosition = currentPosition +
                           5 * Xdr::size<int>() +
                           pixelDataSize;

    if (ofd->multipart)
    {
        streamData->currentPosition += Xdr::size<int>();
    }
}



void
bufferedTileWrite (OutputStreamMutex *streamData,
                   TiledOutputFile::Data *ofd,
                   int dx, int dy,
		   int lx, int ly, 
                   const char pixelData[],
                   int pixelDataSize)
{
    //
    // Check if a tile with coordinates (dx,dy,lx,ly) has already been written.
    //

    if (ofd->tileOffsets (dx, dy, lx, ly))
    {
	THROW (IEX_NAMESPACE::ArgExc,
	       "Attempt to write tile "
	       "(" << dx << ", " << dy << ", " << lx << ", " << ly << ") "
	       "more than once.");
    }

    //
    // If tiles can be written in random order, then don't buffer anything.
    //
    
    if (ofd->lineOrder == RANDOM_Y)
    {
        writeTileData (streamData, ofd, dx, dy, lx, ly, pixelData, pixelDataSize);
        return;
    }
    
    //
    // If the tiles cannot be written in random order, then check if a
    // tile with coordinates (dx,dy,lx,ly) has already been buffered.
    //

    TileCoord currentTile = TileCoord(dx, dy, lx, ly);

    if (ofd->tileMap.find (currentTile) != ofd->tileMap.end())
    {
	THROW (IEX_NAMESPACE::ArgExc,
	       "Attempt to write tile "
	       "(" << dx << ", " << dy << ", " << lx << ", " << ly << ") "
	       "more than once.");
    }

    //
    // If all the tiles before this one have already been written to the file,
    // then write this tile immediately and check if we have buffered tiles
    // that can be written after this tile.
    //
    // Otherwise, buffer the tile so it can be written to file later.
    //
    
    if (ofd->nextTileToWrite == currentTile)
    {
        writeTileData (streamData, ofd, dx, dy, lx, ly, pixelData, pixelDataSize);
        ofd->nextTileToWrite = ofd->nextTileCoord (ofd->nextTileToWrite);

        TileMap::iterator i = ofd->tileMap.find (ofd->nextTileToWrite);
        
        //
        // Step through the tiles and write all successive buffered tiles after
        // the current one.
        //
        
        while(i != ofd->tileMap.end())
        {
            //
            // Write the tile, and then delete the tile's buffered data
            //

            writeTileData (streamData,
                           ofd,
			   i->first.dx, i->first.dy,
			   i->first.lx, i->first.ly,
			   i->second->pixelData,
			   i->second->pixelDataSize);

            delete i->second;
            ofd->tileMap.erase (i);
            
            //
            // Proceed to the next tile
            //
            
            ofd->nextTileToWrite = ofd->nextTileCoord (ofd->nextTileToWrite);
            i = ofd->tileMap.find (ofd->nextTileToWrite);
        }
    }
    else
    {
        //
        // Create a new BufferedTile, copy the pixelData into it, and
        // insert it into the tileMap.
        //

	ofd->tileMap[currentTile] =
	    new BufferedTile ((const char *)pixelData, pixelDataSize);
    }
}


void
convertToXdr (TiledOutputFile::Data *ofd,
              Array<char>& tileBuffer,
	      int numScanLines,
	      int numPixelsPerScanLine)
{
    //
    // Convert the contents of a TiledOutputFile's tileBuffer from the 
    // machine's native representation to Xdr format. This function is called
    // by writeTile(), below, if the compressor wanted its input pixel data
    // in the machine's native format, but then failed to compress the data
    // (most compressors will expand rather than compress random input data).
    //
    // Note that this routine assumes that the machine's native representation
    // of the pixel data has the same size as the Xdr representation.  This
    // makes it possible to convert the pixel data in place, without an
    // intermediate temporary buffer.
    //

    //
    // Set these to point to the start of the tile.
    // We will write to toPtr, and read from fromPtr.
    //

    char *writePtr = tileBuffer;
    const char *readPtr = writePtr;

    //
    // Iterate over all scan lines in the tile.
    //

    for (int y = 0; y < numScanLines; ++y)
    {
	//
	// Iterate over all slices in the file.
	//

	for (unsigned int i = 0; i < ofd->slices.size(); ++i)
	{
	    const TOutSliceInfo &slice = ofd->slices[i];

	    //
	    // Convert the samples in place.
	    //
            
            convertInPlace (writePtr, readPtr, slice.type,
                            numPixelsPerScanLine);
	}
    }

    #ifdef DEBUG

	assert (writePtr == readPtr);

    #endif
}


//
// A TileBufferTask encapsulates the task of copying a tile from
// the user's framebuffer into a LineBuffer and compressing the data
// if necessary.
//

class TileBufferTask: public Task
{
  public:
                    
    TileBufferTask (TaskGroup *group,
                    TiledOutputFile::Data *ofd,
                    int number,
		    int dx, int dy,
		    int lx, int ly);
                    
    virtual ~TileBufferTask ();

    virtual void		execute ();
    
  private:

    TiledOutputFile::Data *	_ofd;
    TileBuffer *		_tileBuffer;
};


TileBufferTask::TileBufferTask
    (TaskGroup *group,
     TiledOutputFile::Data *ofd,
     int number,
     int dx, int dy,
     int lx, int ly)
:
    Task (group),
    _ofd (ofd),
    _tileBuffer (_ofd->getTileBuffer (number))
{
    //
    // Wait for the tileBuffer to become available
    //

    _tileBuffer->wait ();
    _tileBuffer->tileCoord = TileCoord (dx, dy, lx, ly);
}


TileBufferTask::~TileBufferTask ()
{
    //
    // Signal that the tile buffer is now free
    //

    _tileBuffer->post ();
}


void
TileBufferTask::execute ()
{
    try
    {
        //
        // First copy the pixel data from the frame buffer
	// into the tile buffer
        //
        // Convert one tile's worth of pixel data to
        // a machine-independent representation, and store
        // the result in _tileBuffer->buffer.
        //
    
        char *writePtr = _tileBuffer->buffer;
    
        Box2i tileRange = dataWindowForTile (_ofd->tileDesc,
                                                  _ofd->minX, _ofd->maxX,
                                                  _ofd->minY, _ofd->maxY,
                                                  _tileBuffer->tileCoord.dx,
                                                  _tileBuffer->tileCoord.dy,
                                                  _tileBuffer->tileCoord.lx,
                                                  _tileBuffer->tileCoord.ly);
    
        int numScanLines = tileRange.max.y - tileRange.min.y + 1;
        int numPixelsPerScanLine = tileRange.max.x - tileRange.min.x + 1;
    
        //
        // Iterate over the scan lines in the tile.
        //
        
        for (int y = tileRange.min.y; y <= tileRange.max.y; ++y)
        {
            //
            // Iterate over all image channels.
            //
    
            for (unsigned int i = 0; i < _ofd->slices.size(); ++i)
            {
                const TOutSliceInfo &slice = _ofd->slices[i];
    
                //
                // These offsets are used to facilitate both absolute
                // and tile-relative pixel coordinates.
                //
            
                int xOffset = slice.xTileCoords * tileRange.min.x;
                int yOffset = slice.yTileCoords * tileRange.min.y;
    
		//
		// Fill the tile buffer with pixel data.
		//

                if (slice.zero)
                {
                    //
                    // The frame buffer contains no data for this channel.
                    // Store zeroes in _data->tileBuffer.
                    //
                    
                    fillChannelWithZeroes (writePtr, _ofd->format, slice.type,
                                           numPixelsPerScanLine);
                }
                else
                {
                    //
                    // The frame buffer contains data for this channel.
                    //
    
                    const char *readPtr = slice.base +
                                          (y - yOffset) * slice.yStride +
                                          (tileRange.min.x - xOffset) *
                                          slice.xStride;

                    const char *endPtr  = readPtr +
                                          (numPixelsPerScanLine - 1) *
                                          slice.xStride;
                                        
                    copyFromFrameBuffer (writePtr, readPtr, endPtr,
                                         slice.xStride, _ofd->format,
                                         slice.type);
                }
            }
        }
        
        //
        // Compress the contents of the tileBuffer, 
        // and store the compressed data in the output file.
        //
    
        _tileBuffer->dataSize = writePtr - _tileBuffer->buffer;
        _tileBuffer->dataPtr = _tileBuffer->buffer;
    
        if (_tileBuffer->compressor)
        {
            const char *compPtr;

            int compSize = _tileBuffer->compressor->compressTile
                                                (_tileBuffer->dataPtr,
                                                 _tileBuffer->dataSize,
                                                 tileRange, compPtr);
    
            if (compSize < _tileBuffer->dataSize)
            {
                _tileBuffer->dataSize = compSize;
                _tileBuffer->dataPtr = compPtr;
            }
            else if (_ofd->format == Compressor::NATIVE)
            {
                //
                // The data did not shrink during compression, but
                // we cannot write to the file using native format,
                // so we need to convert the lineBuffer to Xdr.
                //
    
                convertToXdr (_ofd, _tileBuffer->buffer, numScanLines,
                              numPixelsPerScanLine);
            }
        }
    }
    catch (std::exception &e)
    {
        if (!_tileBuffer->hasException)
        {
            _tileBuffer->exception = e.what ();
            _tileBuffer->hasException = true;
        }
    }
    catch (...)
    {
        if (!_tileBuffer->hasException)
        {
            _tileBuffer->exception = "unrecognized exception";
            _tileBuffer->hasException = true;
        }
    }
}

} // namespace


TiledOutputFile::TiledOutputFile
    (const char fileName[],
     const Header &header,
     int numThreads)
:
    _data (new Data (numThreads)),
    _streamData (new OutputStreamMutex()),
    _deleteStream (true)
{
    try
    {
	header.sanityCheck (true);
	_streamData->os = new StdOFStream (fileName);
        _data->multipart=false; // since we opened with one header we can't be multipart        
	initialize (header);
	_streamData->currentPosition = _streamData->os->tellp();

	// Write header and empty offset table to the file.
        writeMagicNumberAndVersionField(*_streamData->os, _data->header);
	_data->previewPosition = _data->header.writeTo (*_streamData->os, true);
        _data->tileOffsetsPosition = _data->tileOffsets.writeTo (*_streamData->os);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        delete _streamData;
	delete _data;

	REPLACE_EXC (e, "Cannot open image file "
			"\"" << fileName << "\". " << e);
	throw;
    }
    catch (...)
    {
        delete _streamData;
	delete _data;
        throw;
    }
}


TiledOutputFile::TiledOutputFile
    (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os,
     const Header &header,
     int numThreads)
:
    _data (new Data (numThreads)),
    _streamData (new OutputStreamMutex()),
    _deleteStream (false)
{
    try
    {
	header.sanityCheck(true);
	_streamData->os = &os;
        _data->multipart=false; // since we opened with one header we can't be multipart
	initialize (header);
	_streamData->currentPosition = _streamData->os->tellp();

	// Write header and empty offset table to the file.
	writeMagicNumberAndVersionField(*_streamData->os, _data->header);
	_data->previewPosition = _data->header.writeTo (*_streamData->os, true);
        _data->tileOffsetsPosition = _data->tileOffsets.writeTo (*_streamData->os);
	
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        delete _streamData;
	delete _data;

	REPLACE_EXC (e, "Cannot open image file "
			"\"" << os.fileName() << "\". " << e);
	throw;
    }
    catch (...)
    {
        delete _streamData;
	delete _data;
        throw;
    }
}

TiledOutputFile::TiledOutputFile(const OutputPartData* part) :
    _deleteStream (false)
{
    try
    {
        if (part->header.type() != TILEDIMAGE)
            throw IEX_NAMESPACE::ArgExc("Can't build a TiledOutputFile from a type-mismatched part.");

        _streamData = part->mutex;
        _data = new Data(part->numThreads);
        _data->multipart=part->multipart;
        initialize(part->header);
        _data->partNumber = part->partNumber;
        _data->tileOffsetsPosition = part->chunkOffsetTablePosition;
        _data->previewPosition = part->previewPosition;
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        delete _data;

        REPLACE_EXC (e, "Cannot initialize output part "
                        "\"" << part->partNumber << "\". " << e);
        throw;
    }
    catch (...)
    {
        delete _data;
        throw;
    }
}

void
TiledOutputFile::initialize (const Header &header)
{
    _data->header = header;
    _data->lineOrder = _data->header.lineOrder();

    
    
    //
    // Check that the file is indeed tiled
    //

    _data->tileDesc = _data->header.tileDescription();

    
    //
    // 'Fix' the type attribute if it exists but is incorrectly set
    // (attribute is optional, but ensure it is correct if it exists)
    //
    if(_data->header.hasType())
    {
        _data->header.setType(TILEDIMAGE);
    }

    
    //
    // Save the dataWindow information
    //

    const Box2i &dataWindow = _data->header.dataWindow();
    _data->minX = dataWindow.min.x;
    _data->maxX = dataWindow.max.x;
    _data->minY = dataWindow.min.y;
    _data->maxY = dataWindow.max.y;

    //
    // Precompute level and tile information to speed up utility functions
    //

    precalculateTileInfo (_data->tileDesc,
			  _data->minX, _data->maxX,
			  _data->minY, _data->maxY,
			  _data->numXTiles, _data->numYTiles,
			  _data->numXLevels, _data->numYLevels);       
    
    //
    // Determine the first tile coordinate that we will be writing
    // if the file is not RANDOM_Y.
    //
    
    _data->nextTileToWrite = (_data->lineOrder == INCREASING_Y)?
			       TileCoord (0, 0, 0, 0):
			       TileCoord (0, _data->numYTiles[0] - 1, 0, 0);

    _data->maxBytesPerTileLine =
	    calculateBytesPerPixel (_data->header) * _data->tileDesc.xSize;

    _data->tileBufferSize = _data->maxBytesPerTileLine * _data->tileDesc.ySize;
     
    //
    // Create all the TileBuffers and allocate their internal buffers
    //

    for (size_t i = 0; i < _data->tileBuffers.size(); i++)
    {
        _data->tileBuffers[i] = new TileBuffer (newTileCompressor
						  (_data->header.compression(),
						   _data->maxBytesPerTileLine,
						   _data->tileDesc.ySize,
						   _data->header));

        _data->tileBuffers[i]->buffer.resizeErase(_data->tileBufferSize);
    }

    _data->format = defaultFormat (_data->tileBuffers[0]->compressor);

    _data->tileOffsets = TileOffsets (_data->tileDesc.mode,
				      _data->numXLevels,
				      _data->numYLevels,
				      _data->numXTiles,
				      _data->numYTiles);
}


TiledOutputFile::~TiledOutputFile ()
{
    if (_data)
    {
        {
            Lock lock(*_streamData);
            Int64 originalPosition = _streamData->os->tellp();

            if (_data->tileOffsetsPosition > 0)
            {
                try
                {
                    _streamData->os->seekp (_data->tileOffsetsPosition);
                    _data->tileOffsets.writeTo (*_streamData->os);

                    //
                    // Restore the original position.
                    //
                    _streamData->os->seekp (originalPosition);
                }
                catch (...)
                {
                    //
                    // We cannot safely throw any exceptions from here.
                    // This destructor may have been called because the
                    // stack is currently being unwound for another
                    // exception.
                    //
                }
            }
        }
        
        if (_deleteStream && _streamData)
            delete _streamData->os;

        if (_data->partNumber == -1)
            delete _streamData;

        delete _data;
    }
}


const char *
TiledOutputFile::fileName () const
{
    return _streamData->os->fileName();
}


const Header &
TiledOutputFile::header () const
{
    return _data->header;
}


void	
TiledOutputFile::setFrameBuffer (const FrameBuffer &frameBuffer)
{
    Lock lock (*_streamData);

    //
    // Check if the new frame buffer descriptor
    // is compatible with the image file header.
    //

    const ChannelList &channels = _data->header.channels();

    for (ChannelList::ConstIterator i = channels.begin();
	 i != channels.end();
	 ++i)
    {
	FrameBuffer::ConstIterator j = frameBuffer.find (i.name());

	if (j == frameBuffer.end())
	    continue;

	if (i.channel().type != j.slice().type)
	    THROW (IEX_NAMESPACE::ArgExc, "Pixel type of \"" << i.name() << "\" channel "
				"of output file \"" << fileName() << "\" is "
				"not compatible with the frame buffer's "
				"pixel type.");

	if (j.slice().xSampling != 1 || j.slice().ySampling != 1)
	    THROW (IEX_NAMESPACE::ArgExc, "All channels in a tiled file must have"
				"sampling (1,1).");
    }
    
    //
    // Initialize slice table for writePixels().
    //

    vector<TOutSliceInfo> slices;

    for (ChannelList::ConstIterator i = channels.begin();
	 i != channels.end();
	 ++i)
    {
	FrameBuffer::ConstIterator j = frameBuffer.find (i.name());

	if (j == frameBuffer.end())
	{
	    //
	    // Channel i is not present in the frame buffer.
	    // In the file, channel i will contain only zeroes.
	    //

	    slices.push_back (TOutSliceInfo (i.channel().type,
					     0, // base
					     0, // xStride,
					     0, // yStride,
					     true)); // zero
	}
	else
	{
	    //
	    // Channel i is present in the frame buffer.
	    //

	    slices.push_back (TOutSliceInfo (j.slice().type,
					     j.slice().base,
					     j.slice().xStride,
					     j.slice().yStride,
					     false, // zero
                                             (j.slice().xTileCoords)? 1: 0,
                                             (j.slice().yTileCoords)? 1: 0));
	}
    }

    //
    // Store the new frame buffer.
    //

    _data->frameBuffer = frameBuffer;
    _data->slices = slices;
}


const FrameBuffer &
TiledOutputFile::frameBuffer () const
{
    Lock lock (*_streamData);
    return _data->frameBuffer;
}


void	
TiledOutputFile::writeTiles (int dx1, int dx2, int dy1, int dy2,
                             int lx, int ly)
{
    try
    {
        Lock lock (*_streamData);

        if (_data->slices.size() == 0)
	    throw IEX_NAMESPACE::ArgExc ("No frame buffer specified "
			       "as pixel data source.");

	if (!isValidTile (dx1, dy1, lx, ly) || !isValidTile (dx2, dy2, lx, ly))
	    throw IEX_NAMESPACE::ArgExc ("Tile coordinates are invalid.");

	if (!isValidLevel (lx, ly))
	    THROW (IEX_NAMESPACE::ArgExc,
                   "Level coordinate "
                   "(" << lx << ", " << ly << ") "
                   "is invalid.");
        //
        // Determine the first and last tile coordinates in both dimensions
        // based on the file's lineOrder
        //
                               
        if (dx1 > dx2)
            swap (dx1, dx2);
        
        if (dy1 > dy2)
            swap (dy1, dy2);
        
        int dyStart = dy1;
	int dyStop  = dy2 + 1;
	int dY      = 1;
    
        if (_data->lineOrder == DECREASING_Y)
        {
            dyStart = dy2;
            dyStop  = dy1 - 1;
            dY      = -1;
        }
        
        int numTiles = (dx2 - dx1 + 1) * (dy2 - dy1 + 1);
        int numTasks = min ((int)_data->tileBuffers.size(), numTiles);

        //
        // Create a task group for all tile buffer tasks.  When the
	// task group goes out of scope, the destructor waits until
	// all tasks are complete.
        //

        {
            TaskGroup taskGroup;
    
            //
            // Add in the initial compression tasks to the thread pool
            //
    
            int nextCompBuffer = 0;
	    int dxComp         = dx1;
	    int dyComp         = dyStart;

            while (nextCompBuffer < numTasks)
            {
                ThreadPool::addGlobalTask (new TileBufferTask (&taskGroup,
                                                               _data,
                                                               nextCompBuffer++,
                                                               dxComp, dyComp,
                                                               lx, ly));
                dxComp++;

                if (dxComp > dx2)
                {
                    dxComp = dx1;
                    dyComp += dY;
                }
            }
            
            //
            // Write the compressed buffers and add in more compression
	    // tasks until done
            //
    
            int nextWriteBuffer = 0;
	    int dxWrite         = dx1;
	    int dyWrite         = dyStart;

            while (nextWriteBuffer < numTiles)
            {
		//
                // Wait until the nextWriteBuffer is ready to be written
		//

                TileBuffer* writeBuffer =
                                    _data->getTileBuffer (nextWriteBuffer);

                writeBuffer->wait();
    
		//
                // Write the tilebuffer
		//

                bufferedTileWrite (_streamData, _data, dxWrite, dyWrite, lx, ly,
                                   writeBuffer->dataPtr,
                                   writeBuffer->dataSize);
                
		//
                // Release the lock on nextWriteBuffer
		//

                writeBuffer->post();
                
		//
                // If there are no more tileBuffers to compress, then
		// only continue to write out remaining tileBuffers,
		// otherwise keep adding compression tasks.
		//

                if (nextCompBuffer < numTiles)
                {
		    //
                    // add nextCompBuffer as a compression Task
		    //

                    ThreadPool::addGlobalTask
			(new TileBufferTask (&taskGroup,
					     _data,
					     nextCompBuffer,
                                             dxComp, dyComp,
					     lx, ly));
                }
    
                nextWriteBuffer++;
                dxWrite++;

                if (dxWrite > dx2)
                {
                    dxWrite = dx1;
                    dyWrite += dY;
                }
                    
                nextCompBuffer++;
                dxComp++;

                if (dxComp > dx2)
                {
                    dxComp = dx1;
                    dyComp += dY;
                }
            }

	    //
            // finish all tasks
	    //
        }

	//
	// Exeption handling:
	//
	// TileBufferTask::execute() may have encountered exceptions, but
	// those exceptions occurred in another thread, not in the thread
	// that is executing this call to TiledOutputFile::writeTiles().
	// TileBufferTask::execute() has caught all exceptions and stored
	// the exceptions' what() strings in the tile buffers.
	// Now we check if any tile buffer contains a stored exception; if
	// this is the case then we re-throw the exception in this thread.
	// (It is possible that multiple tile buffers contain stored
	// exceptions.  We re-throw the first exception we find and
	// ignore all others.)
	//

	const string *exception = 0;

        for (size_t i = 0; i < _data->tileBuffers.size(); ++i)
	{
            TileBuffer *tileBuffer = _data->tileBuffers[i];

	    if (tileBuffer->hasException && !exception)
		exception = &tileBuffer->exception;

	    tileBuffer->hasException = false;
	}

	if (exception)
	    throw IEX_NAMESPACE::IoExc (*exception);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        REPLACE_EXC (e, "Failed to write pixel data to image "
                        "file \"" << fileName() << "\". " << e);
        throw;
    }
}


void	
TiledOutputFile::writeTiles (int dx1, int dxMax, int dyMin, int dyMax, int l)
{
    writeTiles (dx1, dxMax, dyMin, dyMax, l, l);
}


void	
TiledOutputFile::writeTile (int dx, int dy, int lx, int ly)
{
    writeTiles (dx, dx, dy, dy, lx, ly);
}


void
TiledOutputFile::writeTile (int dx, int dy, int l)
{
    writeTile(dx, dy, l, l);
}


void	
TiledOutputFile::copyPixels (TiledInputFile &in)
{
    Lock lock (*_streamData);

    //
    // Check if this file's and and the InputFile's
    // headers are compatible.
    //

    const Header &hdr = _data->header;
    const Header &inHdr = in.header(); 

    if (!hdr.hasTileDescription() || !inHdr.hasTileDescription())
        THROW (IEX_NAMESPACE::ArgExc, "Cannot perform a quick pixel copy from image "
			    "file \"" << in.fileName() << "\" to image "
			    "file \"" << fileName() << "\".  The "
                            "output file is tiled, but the input file is not.  "
                            "Try using OutputFile::copyPixels() instead.");

    if (!(hdr.tileDescription() == inHdr.tileDescription()))
        THROW (IEX_NAMESPACE::ArgExc, "Quick pixel copy from image "
			    "file \"" << in.fileName() << "\" to image "
			    "file \"" << fileName() << "\" failed. "
			    "The files have different tile descriptions.");

    if (!(hdr.dataWindow() == inHdr.dataWindow()))
        THROW (IEX_NAMESPACE::ArgExc, "Cannot copy pixels from image "
			    "file \"" << in.fileName() << "\" to image "
			    "file \"" << fileName() << "\". The "
                            "files have different data windows.");

    if (!(hdr.lineOrder() == inHdr.lineOrder()))
        THROW (IEX_NAMESPACE::ArgExc, "Quick pixel copy from image "
			    "file \"" << in.fileName() << "\" to image "
			    "file \"" << fileName() << "\" failed. "
			    "The files have different line orders.");

    if (!(hdr.compression() == inHdr.compression()))
        THROW (IEX_NAMESPACE::ArgExc, "Quick pixel copy from image "
			    "file \"" << in.fileName() << "\" to image "
			    "file \"" << fileName() << "\" failed. "
			    "The files use different compression methods.");

    if (!(hdr.channels() == inHdr.channels()))
        THROW (IEX_NAMESPACE::ArgExc, "Quick pixel copy from image "
			     "file \"" << in.fileName() << "\" to image "
			     "file \"" << fileName() << "\" "
                             "failed.  The files have different channel "
                             "lists.");

    //
    // Verify that no pixel data have been written to this file yet.
    //

    if (!_data->tileOffsets.isEmpty())
        THROW (IEX_NAMESPACE::LogicExc, "Quick pixel copy from image "
			      "file \"" << in.fileName() << "\" to image "
			      "file \"" << _streamData->os->fileName() << "\" "
                              "failed. \"" << fileName() << "\" "
                              "already contains pixel data.");

    //
    // Calculate the total number of tiles in the file
    //

    int numAllTiles = 0;

    switch (levelMode ())
    {
      case ONE_LEVEL:
      case MIPMAP_LEVELS:

        for (int i_l = 0; i_l < numLevels (); ++i_l)
            numAllTiles += numXTiles (i_l) * numYTiles (i_l);

        break;

      case RIPMAP_LEVELS:

        for (int i_ly = 0; i_ly < numYLevels (); ++i_ly)
            for (int i_lx = 0; i_lx < numXLevels (); ++i_lx)
                numAllTiles += numXTiles (i_lx) * numYTiles (i_ly);

        break;

      default:

        throw IEX_NAMESPACE::ArgExc ("Unknown LevelMode format.");
    }

     bool random_y = _data->lineOrder==RANDOM_Y;

    std::vector<int> dx_table(random_y ? numAllTiles : 1);
    std::vector<int> dy_table(random_y ? numAllTiles : 1);
    std::vector<int> lx_table(random_y ? numAllTiles : 1);
    std::vector<int> ly_table(random_y ? numAllTiles : 1);
    
    if(random_y)
    {
        in.tileOrder(&dx_table[0],&dy_table[0],&lx_table[0],&ly_table[0]);
        _data->nextTileToWrite.dx=dx_table[0];
        _data->nextTileToWrite.dy=dy_table[0];
        _data->nextTileToWrite.lx=lx_table[0];
        _data->nextTileToWrite.ly=ly_table[0];
    }

    for (int i = 0; i < numAllTiles; ++i)
    {
        const char *pixelData;
        int pixelDataSize;
        
        int dx = _data->nextTileToWrite.dx;
        int dy = _data->nextTileToWrite.dy;
        int lx = _data->nextTileToWrite.lx;
        int ly = _data->nextTileToWrite.ly;

        
        in.rawTileData (dx, dy, lx, ly, pixelData, pixelDataSize);
        writeTileData (_streamData, _data, dx, dy, lx, ly, pixelData, pixelDataSize);
        
        if(random_y)
        {
            if(i<numAllTiles-1)
            {
               _data->nextTileToWrite.dx=dx_table[i+1];
               _data->nextTileToWrite.dy=dy_table[i+1];
               _data->nextTileToWrite.lx=lx_table[i+1];
               _data->nextTileToWrite.ly=ly_table[i+1];
            }
        }else{
            _data->nextTileToWrite=_data->nextTileCoord(_data->nextTileToWrite);
        }
    }
}


void	
TiledOutputFile::copyPixels (InputFile &in)
{
    copyPixels (*in.tFile());
}


void    
TiledOutputFile::copyPixels (InputPart &in)
{
    copyPixels (*in.file);
}

void    
TiledOutputFile::copyPixels (TiledInputPart &in)
{
    copyPixels (*in.file);
}



unsigned int
TiledOutputFile::tileXSize () const
{
    return _data->tileDesc.xSize;
}


unsigned int
TiledOutputFile::tileYSize () const
{
    return _data->tileDesc.ySize;
}


LevelMode
TiledOutputFile::levelMode () const
{
    return _data->tileDesc.mode;
}


LevelRoundingMode
TiledOutputFile::levelRoundingMode () const
{
    return _data->tileDesc.roundingMode;
}


int
TiledOutputFile::numLevels () const
{
    if (levelMode() == RIPMAP_LEVELS)
	THROW (IEX_NAMESPACE::LogicExc, "Error calling numLevels() on image "
			      "file \"" << fileName() << "\" "
			      "(numLevels() is not defined for RIPMAPs).");
    return _data->numXLevels;
}


int
TiledOutputFile::numXLevels () const
{
    return _data->numXLevels;
}


int
TiledOutputFile::numYLevels () const
{
    return _data->numYLevels;
}


bool	
TiledOutputFile::isValidLevel (int lx, int ly) const
{
    if (lx < 0 || ly < 0)
	return false;

    if (levelMode() == MIPMAP_LEVELS && lx != ly)
	return false;

    if (lx >= numXLevels() || ly >= numYLevels())
	return false;

    return true;
}


int
TiledOutputFile::levelWidth (int lx) const
{
    try
    {
	int retVal = levelSize (_data->minX, _data->maxX, lx,
			        _data->tileDesc.roundingMode);
        
        return retVal;
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error calling levelWidth() on image "
			"file \"" << fileName() << "\". " << e);
	throw;
    }
}


int
TiledOutputFile::levelHeight (int ly) const
{
    try
    {
	return levelSize (_data->minY, _data->maxY, ly,
			  _data->tileDesc.roundingMode);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error calling levelHeight() on image "
			"file \"" << fileName() << "\". " << e);
	throw;
    }
}


int
TiledOutputFile::numXTiles (int lx) const
{
    if (lx < 0 || lx >= _data->numXLevels)
	THROW (IEX_NAMESPACE::LogicExc, "Error calling numXTiles() on image "
			      "file \"" << _streamData->os->fileName() << "\" "
			      "(Argument is not in valid range).");

    return _data->numXTiles[lx];
}


int
TiledOutputFile::numYTiles (int ly) const
{
   if (ly < 0 || ly >= _data->numYLevels)
	THROW (IEX_NAMESPACE::LogicExc, "Error calling numXTiles() on image "
			      "file \"" << _streamData->os->fileName() << "\" "
			      "(Argument is not in valid range).");

    return _data->numYTiles[ly];
}


Box2i
TiledOutputFile::dataWindowForLevel (int l) const
{
    return dataWindowForLevel (l, l);
}


Box2i
TiledOutputFile::dataWindowForLevel (int lx, int ly) const
{
    try
    {
	return OPENEXR_IMF_INTERNAL_NAMESPACE::dataWindowForLevel (
	        _data->tileDesc,
	        _data->minX, _data->maxX,
	        _data->minY, _data->maxY,
	        lx, ly);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error calling dataWindowForLevel() on image "
			"file \"" << fileName() << "\". " << e);
	throw;
    }
}


Box2i
TiledOutputFile::dataWindowForTile (int dx, int dy, int l) const
{
    return dataWindowForTile (dx, dy, l, l);
}


Box2i
TiledOutputFile::dataWindowForTile (int dx, int dy, int lx, int ly) const
{
    try
    {
	if (!isValidTile (dx, dy, lx, ly))
	    throw IEX_NAMESPACE::ArgExc ("Arguments not in valid range.");

	return OPENEXR_IMF_INTERNAL_NAMESPACE::dataWindowForTile (
	        _data->tileDesc,
	        _data->minX, _data->maxX,
	        _data->minY, _data->maxY,
	        dx, dy,
	        lx, ly);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error calling dataWindowForTile() on image "
			"file \"" << fileName() << "\". " << e);
	throw;
    }
}


bool
TiledOutputFile::isValidTile (int dx, int dy, int lx, int ly) const
{
    return ((lx < _data->numXLevels && lx >= 0) &&
	    (ly < _data->numYLevels && ly >= 0) &&
	    (dx < _data->numXTiles[lx] && dx >= 0) &&
	    (dy < _data->numYTiles[ly] && dy >= 0));
}


void
TiledOutputFile::updatePreviewImage (const PreviewRgba newPixels[])
{
    Lock lock (*_streamData);

    if (_data->previewPosition <= 0)
	THROW (IEX_NAMESPACE::LogicExc, "Cannot update preview image pixels. "
			      "File \"" << fileName() << "\" does not "
			      "contain a preview image.");

    //
    // Store the new pixels in the header's preview image attribute.
    //

    PreviewImageAttribute &pia =
	_data->header.typedAttribute <PreviewImageAttribute> ("preview");

    PreviewImage &pi = pia.value();
    PreviewRgba *pixels = pi.pixels();
    int numPixels = pi.width() * pi.height();

    for (int i = 0; i < numPixels; ++i)
	pixels[i] = newPixels[i];

    //
    // Save the current file position, jump to the position in
    // the file where the preview image starts, store the new
    // preview image, and jump back to the saved file position.
    //

    Int64 savedPosition = _streamData->os->tellp();

    try
    {
        _streamData->os->seekp (_data->previewPosition);
	pia.writeValueTo (*_streamData->os, _data->version);
	_streamData->os->seekp (savedPosition);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Cannot update preview image pixels for "
			"file \"" << fileName() << "\". " << e);
	throw;
    }
}


void
TiledOutputFile::breakTile 
    (int dx, int dy,
     int lx, int ly,
     int offset,
     int length,
     char c)
{
    Lock lock (*_streamData);

    Int64 position = _data->tileOffsets (dx, dy, lx, ly);

    if (!position)
	THROW (IEX_NAMESPACE::ArgExc,
	       "Cannot overwrite tile "
	       "(" << dx << ", " << dy << ", " << lx << "," << ly << "). "
	       "The tile has not yet been stored in "
	       "file \"" << fileName() << "\".");

    _streamData->currentPosition = 0;
    _streamData->os->seekp (position + offset);

    for (int i = 0; i < length; ++i)
        _streamData->os->write (&c, 1);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
