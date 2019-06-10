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
//	class TiledInputFile
//
//-----------------------------------------------------------------------------

#include "ImfTiledInputFile.h"
#include "ImfTileDescriptionAttribute.h"
#include "ImfChannelList.h"
#include "ImfMisc.h"
#include "ImfTiledMisc.h"
#include "ImfStdIO.h"
#include "ImfCompressor.h"
#include "ImfXdr.h"
#include "ImfConvert.h"
#include "ImfVersion.h"
#include "ImfTileOffsets.h"
#include "ImfThreading.h"
#include "ImfPartType.h"
#include "ImfMultiPartInputFile.h"
#include "ImfInputStreamMutex.h"
#include "IlmThreadPool.h"
#include "IlmThreadSemaphore.h"
#include "IlmThreadMutex.h"
#include "ImathVec.h"
#include "Iex.h"
#include <string>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "ImfInputPartData.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using IMATH_NAMESPACE::Box2i;
using IMATH_NAMESPACE::V2i;
using std::string;
using std::vector;
using std::min;
using std::max;
using ILMTHREAD_NAMESPACE::Mutex;
using ILMTHREAD_NAMESPACE::Lock;
using ILMTHREAD_NAMESPACE::Semaphore;
using ILMTHREAD_NAMESPACE::Task;
using ILMTHREAD_NAMESPACE::TaskGroup;
using ILMTHREAD_NAMESPACE::ThreadPool;

namespace {

struct TInSliceInfo
{
    PixelType   typeInFrameBuffer;
    PixelType   typeInFile;
    char *      base;
    size_t      xStride;
    size_t      yStride;
    bool        fill;
    bool        skip;
    double      fillValue;
    int         xTileCoords;
    int         yTileCoords;

    TInSliceInfo (PixelType typeInFrameBuffer = HALF,
                  PixelType typeInFile = HALF,
                  char *base = 0,
                  size_t xStride = 0,
                  size_t yStride = 0,
                  bool fill = false,
                  bool skip = false,
                  double fillValue = 0.0,
                  int xTileCoords = 0,
                  int yTileCoords = 0);
};


TInSliceInfo::TInSliceInfo (PixelType tifb,
                            PixelType tifl,
                            char *b,
                            size_t xs, size_t ys,
                            bool f, bool s,
                            double fv,
                            int xtc,
                            int ytc)
:
    typeInFrameBuffer (tifb),
    typeInFile (tifl),
    base (b),
    xStride (xs),
    yStride (ys),
    fill (f),
    skip (s),
    fillValue (fv),
    xTileCoords (xtc),
    yTileCoords (ytc)
{
    // empty
}


struct TileBuffer
{
    const char *	uncompressedData;
    char *		buffer;
    int			dataSize;
    Compressor *	compressor;
    Compressor::Format	format;
    int			dx;
    int			dy;
    int			lx;
    int			ly;
    bool		hasException;
    string		exception;

     TileBuffer (Compressor * const comp);
    ~TileBuffer ();

    inline void		wait () {_sem.wait();}
    inline void		post () {_sem.post();}

 protected:

    Semaphore _sem;
};


TileBuffer::TileBuffer (Compressor *comp):
    uncompressedData (0),
    buffer (0),
    dataSize (0),
    compressor (comp),
    format (defaultFormat (compressor)),
    dx (-1),
    dy (-1),
    lx (-1),
    ly (-1),
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


class MultiPartInputFile;


//
// struct TiledInputFile::Data stores things that will be
// needed between calls to readTile()
//

struct TiledInputFile::Data: public Mutex
{
    Header	    header;	        	    // the image header
    TileDescription tileDesc;		            // describes the tile layout
    int		    version;		            // file's version
    FrameBuffer	    frameBuffer;	            // framebuffer to write into
    LineOrder	    lineOrder;		            // the file's lineorder
    int		    minX;		            // data window's min x coord
    int		    maxX;		            // data window's max x coord
    int		    minY;		            // data window's min y coord
    int		    maxY;		            // data window's max x coord

    int		    numXLevels;		            // number of x levels
    int		    numYLevels;		            // number of y levels
    int *	    numXTiles;		            // number of x tiles at a level
    int *	    numYTiles;		            // number of y tiles at a level

    TileOffsets	    tileOffsets;	            // stores offsets in file for
    // each tile

    bool	    fileIsComplete;	            // True if no tiles are missing
                                                    // in the file

    vector<TInSliceInfo> slices;        	    // info about channels in file

    size_t	    bytesPerPixel;                  // size of an uncompressed pixel

    size_t	    maxBytesPerTileLine;            // combined size of a line
                                                    // over all channels

    int             partNumber;                     // part number

    bool            multiPartBackwardSupport;       // if we are reading a multipart file
                                                    // using OpenEXR 1.7 API

    int             numThreads;                     // number of threads

    MultiPartInputFile* multiPartFile;              // the MultiPartInputFile used to
                                                    // support backward compatibility
    
    vector<TileBuffer*> tileBuffers;                // each holds a single tile
    size_t          tileBufferSize;	            // size of the tile buffers

    bool            memoryMapped;                   // if the stream is memory mapped

    InputStreamMutex * _streamData;
    bool                _deleteStream;

     Data (int numThreads);
    ~Data ();

    inline TileBuffer * getTileBuffer (int number);
					    // hash function from tile indices
					    // into our vector of tile buffers
};


TiledInputFile::Data::Data (int numThreads):
    numXTiles (0),
    numYTiles (0),
    partNumber (-1),
    multiPartBackwardSupport(false),
    numThreads(numThreads),
    memoryMapped(false),
    _streamData(NULL),
    _deleteStream(false)
{
    //
    // We need at least one tileBuffer, but if threading is used,
    // to keep n threads busy we need 2*n tileBuffers
    //

    tileBuffers.resize (max (1, 2 * numThreads));
}


TiledInputFile::Data::~Data ()
{
    delete [] numXTiles;
    delete [] numYTiles;

    for (size_t i = 0; i < tileBuffers.size(); i++)
        delete tileBuffers[i];

    if (multiPartBackwardSupport)
        delete multiPartFile;
}


TileBuffer*
TiledInputFile::Data::getTileBuffer (int number)
{
    return tileBuffers[number % tileBuffers.size()];
}


namespace {

void
readTileData (InputStreamMutex *streamData,
              TiledInputFile::Data *ifd,
	      int dx, int dy,
	      int lx, int ly,
              char *&buffer,
              int &dataSize)
{
    //
    // Read a single tile block from the file and into the array pointed
    // to by buffer.  If the file is memory-mapped, then we change where
    // buffer points instead of writing into the array (hence buffer needs
    // to be a reference to a char *).
    //

    //
    // Look up the location for this tile in the Index and
    // seek to that position if necessary
    //
    
    Int64 tileOffset = ifd->tileOffsets (dx, dy, lx, ly);

    if (tileOffset == 0)
    {
        THROW (IEX_NAMESPACE::InputExc, "Tile (" << dx << ", " << dy << ", " <<
			      lx << ", " << ly << ") is missing.");
    }


    //
    // In a multi-part file, the next chunk does not need to
    // belong to the same part, so we have to compare the
    // offset here.
    //

    if (!isMultiPart(ifd->version))
    {
        if (streamData->currentPosition != tileOffset)
            streamData->is->seekg (tileOffset);
    }
    else
    {
        //
        // In a multi-part file, the file pointer may be moved by other
        // parts, so we have to ask tellg() where we are.
        //
        if (streamData->is->tellg() != tileOffset)
            streamData->is->seekg (tileOffset);
    }

    //
    // Read the first few bytes of the tile (the header).
    // Verify that the tile coordinates and the level number
    // are correct.
    //
    
    int tileXCoord, tileYCoord, levelX, levelY;

    if (isMultiPart(ifd->version))
    {
        int partNumber;
        Xdr::read <StreamIO> (*streamData->is, partNumber);
        if (partNumber != ifd->partNumber)
        {
            THROW (IEX_NAMESPACE::ArgExc, "Unexpected part number " << partNumber
                   << ", should be " << ifd->partNumber << ".");
        }
    }

    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, tileXCoord);
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, tileYCoord);
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, levelX);
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, levelY);
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, dataSize);

    if (tileXCoord != dx)
        throw IEX_NAMESPACE::InputExc ("Unexpected tile x coordinate.");

    if (tileYCoord != dy)
        throw IEX_NAMESPACE::InputExc ("Unexpected tile y coordinate.");

    if (levelX != lx)
        throw IEX_NAMESPACE::InputExc ("Unexpected tile x level number coordinate.");

    if (levelY != ly)
        throw IEX_NAMESPACE::InputExc ("Unexpected tile y level number coordinate.");

    if (dataSize > (int) ifd->tileBufferSize)
        throw IEX_NAMESPACE::InputExc ("Unexpected tile block length.");

    //
    // Read the pixel data.
    //

    if (streamData->is->isMemoryMapped ())
        buffer = streamData->is->readMemoryMapped (dataSize);
    else
        streamData->is->read (buffer, dataSize);

    //
    // Keep track of which tile is the next one in
    // the file, so that we can avoid redundant seekg()
    // operations (seekg() can be fairly expensive).
    //
    
    streamData->currentPosition = tileOffset + 5 * Xdr::size<int>() + dataSize;
}


void
readNextTileData (InputStreamMutex *streamData,
                  TiledInputFile::Data *ifd,
		  int &dx, int &dy,
		  int &lx, int &ly,
                  char * & buffer,
		  int &dataSize)
{
    //
    // Read the next tile block from the file
    //

    if(isMultiPart(ifd->version))
    {
        int part;
        Xdr::read <StreamIO> (*streamData->is, part);
        if(part!=ifd->partNumber)
        {
           throw IEX_NAMESPACE::InputExc("Unexpected part number in readNextTileData");
        }
    }

    //
    // Read the first few bytes of the tile (the header).
    //

    Xdr::read <StreamIO> (*streamData->is, dx);
    Xdr::read <StreamIO> (*streamData->is, dy);
    Xdr::read <StreamIO> (*streamData->is, lx);
    Xdr::read <StreamIO> (*streamData->is, ly);
    Xdr::read <StreamIO> (*streamData->is, dataSize);

    if (dataSize > (int) ifd->tileBufferSize)
        throw IEX_NAMESPACE::InputExc ("Unexpected tile block length.");
    
    //
    // Read the pixel data.
    //

    streamData->is->read (buffer, dataSize);
    
    //
    // Keep track of which tile is the next one in
    // the file, so that we can avoid redundant seekg()
    // operations (seekg() can be fairly expensive).
    //

    streamData->currentPosition += 5 * Xdr::size<int>() + dataSize;
}


//
// A TileBufferTask encapsulates the task of uncompressing
// a single tile and copying it into the frame buffer.
//

class TileBufferTask : public Task
{
  public:

    TileBufferTask (TaskGroup *group,
                    TiledInputFile::Data *ifd,
		    TileBuffer *tileBuffer);
                    
    virtual ~TileBufferTask ();

    virtual void		execute ();
    
  private:

    TiledInputFile::Data *	_ifd;
    TileBuffer *		_tileBuffer;
};


TileBufferTask::TileBufferTask
    (TaskGroup *group,
     TiledInputFile::Data *ifd,
     TileBuffer *tileBuffer)
:
    Task (group),
    _ifd (ifd),
    _tileBuffer (tileBuffer)
{
    // empty
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
        // Calculate information about the tile
        //
    
        Box2i tileRange =  OPENEXR_IMF_INTERNAL_NAMESPACE::dataWindowForTile (
                _ifd->tileDesc,
                _ifd->minX, _ifd->maxX,
                _ifd->minY, _ifd->maxY,
                _tileBuffer->dx,
                _tileBuffer->dy,
                _tileBuffer->lx,
                _tileBuffer->ly);

        int numPixelsPerScanLine = tileRange.max.x - tileRange.min.x + 1;
    
        int numPixelsInTile = numPixelsPerScanLine *
                            (tileRange.max.y - tileRange.min.y + 1);
    
        int sizeOfTile = _ifd->bytesPerPixel * numPixelsInTile;
    
    
        //
        // Uncompress the data, if necessary
        //
    
        if (_tileBuffer->compressor && _tileBuffer->dataSize < sizeOfTile)
        {
            _tileBuffer->format = _tileBuffer->compressor->format();

            _tileBuffer->dataSize = _tileBuffer->compressor->uncompressTile
		(_tileBuffer->buffer, _tileBuffer->dataSize,
		 tileRange, _tileBuffer->uncompressedData);
        }
        else
        {
            //
            // If the line is uncompressed, it's in XDR format,
            // regardless of the compressor's output format.
            //
    
            _tileBuffer->format = Compressor::XDR;
            _tileBuffer->uncompressedData = _tileBuffer->buffer;
        }
    
        //
        // Convert the tile of pixel data back from the machine-independent
	// representation, and store the result in the frame buffer.
        //
    
        const char *readPtr = _tileBuffer->uncompressedData;
                                                        // points to where we
                                                        // read from in the
                                                        // tile block
        
        //
        // Iterate over the scan lines in the tile.
        //
    
        for (int y = tileRange.min.y; y <= tileRange.max.y; ++y)
        {
            //
            // Iterate over all image channels.
            //
            
            for (unsigned int i = 0; i < _ifd->slices.size(); ++i)
            {
                const TInSliceInfo &slice = _ifd->slices[i];
    
                //
                // These offsets are used to facilitate both
                // absolute and tile-relative pixel coordinates.
                //
            
                int xOffset = slice.xTileCoords * tileRange.min.x;
                int yOffset = slice.yTileCoords * tileRange.min.y;
    
                //
                // Fill the frame buffer with pixel data.
                //
    
                if (slice.skip)
                {
                    //
                    // The file contains data for this channel, but
                    // the frame buffer contains no slice for this channel.
                    //
    
                    skipChannel (readPtr, slice.typeInFile,
                                 numPixelsPerScanLine);
                }
                else
                {
                    //
                    // The frame buffer contains a slice for this channel.
                    //
    
                    char *writePtr = slice.base +
                                     (y - yOffset) * slice.yStride +
                                     (tileRange.min.x - xOffset) *
                                     slice.xStride;

                    char *endPtr = writePtr +
                                   (numPixelsPerScanLine - 1) * slice.xStride;
                                    
                    copyIntoFrameBuffer (readPtr, writePtr, endPtr,
                                         slice.xStride,
                                         slice.fill, slice.fillValue,
                                         _tileBuffer->format,
                                         slice.typeInFrameBuffer,
                                         slice.typeInFile);
                }
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


TileBufferTask *
newTileBufferTask
    (TaskGroup *group,
     InputStreamMutex *streamData,
     TiledInputFile::Data *ifd,
     int number,
     int dx, int dy,
     int lx, int ly)
{
    //
    // Wait for a tile buffer to become available,
    // fill the buffer with raw data from the file,
    // and create a new TileBufferTask whose execute()
    // method will uncompress the tile and copy the
    // tile's pixels into the frame buffer.
    //

    TileBuffer *tileBuffer = ifd->getTileBuffer (number);

    try
    {
	tileBuffer->wait();
	
	tileBuffer->dx = dx;
	tileBuffer->dy = dy;
	tileBuffer->lx = lx;
	tileBuffer->ly = ly;

	tileBuffer->uncompressedData = 0;

	readTileData (streamData, ifd, dx, dy, lx, ly,
		      tileBuffer->buffer,
		      tileBuffer->dataSize);
    }
    catch (...)
    {
	//
	// Reading from the file caused an exception.
	// Signal that the tile buffer is free, and
	// re-throw the exception.
	//

	tileBuffer->post();
	throw;
    }

    return new TileBufferTask (group, ifd, tileBuffer);
}


} // namespace


TiledInputFile::TiledInputFile (const char fileName[], int numThreads):
    _data (new Data (numThreads))
{
    _data->_streamData=NULL;
    _data->_deleteStream=true;
    
    //
    // This constructor is called when a user
    // explicitly wants to read a tiled file.
    //


    IStream* is = 0;
    try
    {
        is = new StdIFStream (fileName);
	readMagicNumberAndVersionField(*is, _data->version);

	//
        // Backward compatibility to read multpart file.
        //
	if (isMultiPart(_data->version))
	{
	    compatibilityInitialize(*is);
	    return;
	}

	_data->_streamData = new InputStreamMutex();
	_data->_streamData->is = is;
	_data->header.readFrom (*_data->_streamData->is, _data->version);
	initialize();
        //read tile offsets - we are not multipart or deep
        _data->tileOffsets.readFrom (*(_data->_streamData->is), _data->fileIsComplete,false,false);
	_data->_streamData->currentPosition = _data->_streamData->is->tellg();
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        if (_data->_streamData != 0)
        {
            if (_data->_streamData->is != 0)
            {
                delete _data->_streamData->is;
                _data->_streamData->is = is = 0;
            }

            delete _data->_streamData;
        }

        if (is != 0)
            delete is;

	REPLACE_EXC (e, "Cannot open image file "
                 "\"" << fileName << "\". " << e.what());
	throw;
    }
    catch (...)
    {
        if ( _data->_streamData != 0)
        {
            if ( _data->_streamData->is != 0)
            {
                delete _data->_streamData->is;
                _data->_streamData->is = is = 0;
            }

            delete _data->_streamData;
        }

        if (is != 0)
            delete is;
        throw;
    }
}


TiledInputFile::TiledInputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int numThreads):
    _data (new Data (numThreads))
{
    _data->_deleteStream=false;
    //
    // This constructor is called when a user
    // explicitly wants to read a tiled file.
    //

    bool streamDataCreated = false;

    try
    {
	readMagicNumberAndVersionField(is, _data->version);

	//
	// Backward compatibility to read multpart file.
	//
	if (isMultiPart(_data->version))
        {
	    compatibilityInitialize(is);
            return;
        }

	streamDataCreated = true;
	_data->_streamData = new InputStreamMutex();
	_data->_streamData->is = &is;
	_data->header.readFrom (*_data->_streamData->is, _data->version);
	initialize();
        // file is guaranteed to be single part, regular image
        _data->tileOffsets.readFrom (*(_data->_streamData->is), _data->fileIsComplete,false,false);
	_data->memoryMapped = _data->_streamData->is->isMemoryMapped();
	_data->_streamData->currentPosition = _data->_streamData->is->tellg();
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        if (streamDataCreated) delete _data->_streamData;
	delete _data;

	REPLACE_EXC (e, "Cannot open image file "
                 "\"" << is.fileName() << "\". " << e.what());
	throw;
    }
    catch (...)
    {
        if (streamDataCreated) delete _data->_streamData;
	delete _data;
        throw;
    }
}


TiledInputFile::TiledInputFile (const Header &header,
                                OPENEXR_IMF_INTERNAL_NAMESPACE::IStream *is,
                                int version,
                                int numThreads) :
    _data (new Data (numThreads))
{
    _data->_deleteStream=false;
    _data->_streamData = new InputStreamMutex();
    //
    // This constructor called by class Imf::InputFile
    // when a user wants to just read an image file, and
    // doesn't care or know if the file is tiled.
    // No need to have backward compatibility here, because
    // we have somehow got the header.
    //

    _data->_streamData->is = is;
    _data->header = header;
    _data->version = version;
    initialize();
    _data->tileOffsets.readFrom (*(_data->_streamData->is),_data->fileIsComplete,false,false);
    _data->memoryMapped = is->isMemoryMapped();
    _data->_streamData->currentPosition = _data->_streamData->is->tellg();
}


TiledInputFile::TiledInputFile (InputPartData* part) 
{
    _data = new Data (part->numThreads);
    _data->_deleteStream=false;
    multiPartInitialize(part);
}


void
TiledInputFile::compatibilityInitialize(OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is)
{
    is.seekg(0);
    //
    // Construct a MultiPartInputFile, initialize TiledInputFile
    // with the part 0 data.
    // (TODO) maybe change the third parameter of the constructor of MultiPartInputFile later.
    //
    _data->multiPartBackwardSupport = true;
    _data->multiPartFile = new MultiPartInputFile(is, _data->numThreads);
    InputPartData* part = _data->multiPartFile->getPart(0);

    multiPartInitialize(part);
}


void
TiledInputFile::multiPartInitialize(InputPartData* part)
{
    if (part->header.type() != TILEDIMAGE)
        throw IEX_NAMESPACE::ArgExc("Can't build a TiledInputFile from a type-mismatched part.");

    _data->_streamData = part->mutex;
    _data->header = part->header;
    _data->version = part->version;
    _data->partNumber = part->partNumber;
    _data->memoryMapped = _data->_streamData->is->isMemoryMapped();
    initialize();
    _data->tileOffsets.readFrom(part->chunkOffsets,_data->fileIsComplete);
    _data->_streamData->currentPosition = _data->_streamData->is->tellg();
}


void
TiledInputFile::initialize ()
{
    // fix bad types in header (arises when a tool built against an older version of
    // OpenEXR converts a scanline image to tiled)
    // only applies when file is a single part, regular image, tiled file
    //
    if(!isMultiPart(_data->version) &&
       !isNonImage(_data->version) && 
       isTiled(_data->version) && 
       _data->header.hasType() )
    {
        _data->header.setType(TILEDIMAGE);
    }
    
    if (_data->partNumber == -1)
    {
        if (!isTiled (_data->version))
            throw IEX_NAMESPACE::ArgExc ("Expected a tiled file but the file is not tiled.");
        
    }
    else
    {
        if(_data->header.hasType() && _data->header.type()!=TILEDIMAGE)
        {
            throw IEX_NAMESPACE::ArgExc ("TiledInputFile used for non-tiledimage part.");
        }
    }
    
    _data->header.sanityCheck (true);

    _data->tileDesc = _data->header.tileDescription();
    _data->lineOrder = _data->header.lineOrder();

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

    _data->bytesPerPixel = calculateBytesPerPixel (_data->header);

    _data->maxBytesPerTileLine = _data->bytesPerPixel * _data->tileDesc.xSize;

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

        if (!_data->_streamData->is->isMemoryMapped ())
            _data->tileBuffers[i]->buffer = new char [_data->tileBufferSize];
    }

    _data->tileOffsets = TileOffsets (_data->tileDesc.mode,
				      _data->numXLevels,
				      _data->numYLevels,
				      _data->numXTiles,
				      _data->numYTiles);
}


TiledInputFile::~TiledInputFile ()
{
    if (!_data->memoryMapped)
        for (size_t i = 0; i < _data->tileBuffers.size(); i++)
            delete [] _data->tileBuffers[i]->buffer;

    if (_data->_deleteStream)
        delete _data->_streamData->is;

    if (_data->partNumber == -1)
        delete _data->_streamData;

    delete _data;
}


const char *
TiledInputFile::fileName () const
{
    return _data->_streamData->is->fileName();
}


const Header &
TiledInputFile::header () const
{
    return _data->header;
}


int
TiledInputFile::version () const
{
    return _data->version;
}


void	
TiledInputFile::setFrameBuffer (const FrameBuffer &frameBuffer)
{
    Lock lock (*_data->_streamData);

    //
    // Set the frame buffer
    //

    //
    // Check if the new frame buffer descriptor is
    // compatible with the image file header.
    //

    const ChannelList &channels = _data->header.channels();

    for (FrameBuffer::ConstIterator j = frameBuffer.begin();
         j != frameBuffer.end();
         ++j)
    {
        ChannelList::ConstIterator i = channels.find (j.name());

        if (i == channels.end())
            continue;

        if (i.channel().xSampling != j.slice().xSampling ||
            i.channel().ySampling != j.slice().ySampling)
            THROW (IEX_NAMESPACE::ArgExc, "X and/or y subsampling factors "
				"of \"" << i.name() << "\" channel "
				"of input file \"" << fileName() << "\" are "
				"not compatible with the frame buffer's "
				"subsampling factors.");
    }

    //
    // Initialize the slice table for readPixels().
    //

    vector<TInSliceInfo> slices;
    ChannelList::ConstIterator i = channels.begin();

    for (FrameBuffer::ConstIterator j = frameBuffer.begin();
         j != frameBuffer.end();
         ++j)
    {
        while (i != channels.end() && strcmp (i.name(), j.name()) < 0)
        {
            //
            // Channel i is present in the file but not
            // in the frame buffer; data for channel i
            // will be skipped during readPixels().
            //

            slices.push_back (TInSliceInfo (i.channel().type,
					    i.channel().type,
					    0,      // base
					    0,      // xStride
					    0,      // yStride
					    false,  // fill
					    true,   // skip
					    0.0));  // fillValue
            ++i;
        }

        bool fill = false;

        if (i == channels.end() || strcmp (i.name(), j.name()) > 0)
        {
            //
            // Channel i is present in the frame buffer, but not in the file.
            // In the frame buffer, slice j will be filled with a default value.
            //

            fill = true;
        }

        slices.push_back (TInSliceInfo (j.slice().type,
                                        fill? j.slice().type: i.channel().type,
                                        j.slice().base,
                                        j.slice().xStride,
                                        j.slice().yStride,
                                        fill,
                                        false, // skip
                                        j.slice().fillValue,
                                        (j.slice().xTileCoords)? 1: 0,
                                        (j.slice().yTileCoords)? 1: 0));

        if (i != channels.end() && !fill)
            ++i;
    }

    while (i != channels.end())
    {
	//
	// Channel i is present in the file but not
	// in the frame buffer; data for channel i
	// will be skipped during readPixels().
	//

	slices.push_back (TInSliceInfo (i.channel().type,
					i.channel().type,
					0, // base
					0, // xStride
					0, // yStride
					false,  // fill
					true, // skip
					0.0)); // fillValue
	++i;
    }

    //
    // Store the new frame buffer.
    //

    _data->frameBuffer = frameBuffer;
    _data->slices = slices;
}


const FrameBuffer &
TiledInputFile::frameBuffer () const
{
    Lock lock (*_data->_streamData);
    return _data->frameBuffer;
}


bool
TiledInputFile::isComplete () const
{
    return _data->fileIsComplete;
}


void
TiledInputFile::readTiles (int dx1, int dx2, int dy1, int dy2, int lx, int ly)
{
    //
    // Read a range of tiles from the file into the framebuffer
    //

    try
    {
        Lock lock (*_data->_streamData);

        if (_data->slices.size() == 0)
            throw IEX_NAMESPACE::ArgExc ("No frame buffer specified "
			       "as pixel data destination.");
        
        if (!isValidLevel (lx, ly))
            THROW (IEX_NAMESPACE::ArgExc,
                   "Level coordinate "
                   "(" << lx << ", " << ly << ") "
                   "is invalid.");

        //
        // Determine the first and last tile coordinates in both dimensions.
        // We always attempt to read the range of tiles in the order that
        // they are stored in the file.
        //
                               
        if (dx1 > dx2)
            std::swap (dx1, dx2);
        
        if (dy1 > dy2)
            std::swap (dy1, dy2);
        
        int dyStart = dy1;
	int dyStop  = dy2 + 1;
	int dY      = 1;

        if (_data->lineOrder == DECREASING_Y)
        {
            dyStart = dy2;
            dyStop  = dy1 - 1;
            dY      = -1;
        }

        //
        // Create a task group for all tile buffer tasks.  When the
	// task group goes out of scope, the destructor waits until
	// all tasks are complete.
        //
        
        {
            TaskGroup taskGroup;
            int tileNumber = 0;
    
            for (int dy = dyStart; dy != dyStop; dy += dY)
            {
                for (int dx = dx1; dx <= dx2; dx++)
                {
                    if (!isValidTile (dx, dy, lx, ly))
                        THROW (IEX_NAMESPACE::ArgExc,
			       "Tile (" << dx << ", " << dy << ", " <<
			       lx << "," << ly << ") is not a valid tile.");
                    
                    ThreadPool::addGlobalTask (newTileBufferTask (&taskGroup,
                                                                  _data->_streamData,
                                                                  _data,
                                                                  tileNumber++,
                                                                  dx, dy,
                                                                  lx, ly));
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
	// that is executing this call to TiledInputFile::readTiles().
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
        REPLACE_EXC (e, "Error reading pixel data from image "
                     "file \"" << fileName() << "\". " << e.what());
        throw;
    }
}


void	
TiledInputFile::readTiles (int dx1, int dx2, int dy1, int dy2, int l)
{
    readTiles (dx1, dx2, dy1, dy2, l, l);
}


void	
TiledInputFile::readTile (int dx, int dy, int lx, int ly)
{
    readTiles (dx, dx, dy, dy, lx, ly);
}


void	
TiledInputFile::readTile (int dx, int dy, int l)
{
    readTile (dx, dy, l, l);
}


void
TiledInputFile::rawTileData (int &dx, int &dy,
			     int &lx, int &ly,
                             const char *&pixelData,
			     int &pixelDataSize)
{
    try
    {
        Lock lock (*_data->_streamData);

        if (!isValidTile (dx, dy, lx, ly))
            throw IEX_NAMESPACE::ArgExc ("Tried to read a tile outside "
			       "the image file's data window.");

        TileBuffer *tileBuffer = _data->getTileBuffer (0);

        //
        // if file is a multipart file, we have to seek to the required tile
        // since we don't know where the file pointer is
        //
        int old_dx=dx;
        int old_dy=dy;
        int old_lx=lx;
        int old_ly=ly;
        if(isMultiPart(version()))
        {
            _data->_streamData->is->seekg(_data->tileOffsets(dx,dy,lx,ly));
        }
        readNextTileData (_data->_streamData, _data, dx, dy, lx, ly,
			  tileBuffer->buffer,
                          pixelDataSize);
        if(isMultiPart(version()))
        {
            if (old_dx!=dx || old_dy !=dy || old_lx!=lx || old_ly!=ly)
            {
                throw IEX_NAMESPACE::ArgExc ("rawTileData read the wrong tile");
            }
        }
        pixelData = tileBuffer->buffer;
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        REPLACE_EXC (e, "Error reading pixel data from image "
                     "file \"" << fileName() << "\". " << e.what());
        throw;
    }
}


unsigned int
TiledInputFile::tileXSize () const
{
    return _data->tileDesc.xSize;
}


unsigned int
TiledInputFile::tileYSize () const
{
    return _data->tileDesc.ySize;
}


LevelMode
TiledInputFile::levelMode () const
{
    return _data->tileDesc.mode;
}


LevelRoundingMode
TiledInputFile::levelRoundingMode () const
{
    return _data->tileDesc.roundingMode;
}


int
TiledInputFile::numLevels () const
{
    if (levelMode() == RIPMAP_LEVELS)
	THROW (IEX_NAMESPACE::LogicExc, "Error calling numLevels() on image "
			      "file \"" << fileName() << "\" "
			      "(numLevels() is not defined for files "
			      "with RIPMAP level mode).");

    return _data->numXLevels;
}


int
TiledInputFile::numXLevels () const
{
    return _data->numXLevels;
}


int
TiledInputFile::numYLevels () const
{
    return _data->numYLevels;
}


bool	
TiledInputFile::isValidLevel (int lx, int ly) const
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
TiledInputFile::levelWidth (int lx) const
{
    try
    {
        return levelSize (_data->minX, _data->maxX, lx,
			  _data->tileDesc.roundingMode);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error calling levelWidth() on image "
                 "file \"" << fileName() << "\". " << e.what());
	throw;
    }
}


int
TiledInputFile::levelHeight (int ly) const
{
    try
    {
        return levelSize (_data->minY, _data->maxY, ly,
                          _data->tileDesc.roundingMode);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error calling levelHeight() on image "
                 "file \"" << fileName() << "\". " << e.what());
	throw;
    }
}


int
TiledInputFile::numXTiles (int lx) const
{
    if (lx < 0 || lx >= _data->numXLevels)
    {
        THROW (IEX_NAMESPACE::ArgExc, "Error calling numXTiles() on image "
			    "file \"" << _data->_streamData->is->fileName() << "\" "
			    "(Argument is not in valid range).");

    }
    
    return _data->numXTiles[lx];
}


int
TiledInputFile::numYTiles (int ly) const
{
    if (ly < 0 || ly >= _data->numYLevels)
    {
        THROW (IEX_NAMESPACE::ArgExc, "Error calling numYTiles() on image "
			    "file \"" << _data->_streamData->is->fileName() << "\" "
			    "(Argument is not in valid range).");
    }
    
    return _data->numYTiles[ly];
}


Box2i
TiledInputFile::dataWindowForLevel (int l) const
{
    return dataWindowForLevel (l, l);
}


Box2i
TiledInputFile::dataWindowForLevel (int lx, int ly) const
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
                 "file \"" << fileName() << "\". " << e.what());
	throw;
    }
}


Box2i
TiledInputFile::dataWindowForTile (int dx, int dy, int l) const
{
    return dataWindowForTile (dx, dy, l, l);
}


Box2i
TiledInputFile::dataWindowForTile (int dx, int dy, int lx, int ly) const
{
    try
    {
	if (!isValidTile (dx, dy, lx, ly))
	    throw IEX_NAMESPACE::ArgExc ("Arguments not in valid range.");

        return OPENEXR_IMF_INTERNAL_NAMESPACE::dataWindowForTile (
                _data->tileDesc,
                _data->minX, _data->maxX,
                _data->minY, _data->maxY,
                dx, dy, lx, ly);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error calling dataWindowForTile() on image "
                 "file \"" << fileName() << "\". " << e.what());
	throw;
    }
}


bool
TiledInputFile::isValidTile (int dx, int dy, int lx, int ly) const
{
    return ((lx < _data->numXLevels && lx >= 0) &&
            (ly < _data->numYLevels && ly >= 0) &&
            (dx < _data->numXTiles[lx] && dx >= 0) &&
            (dy < _data->numYTiles[ly] && dy >= 0));
}

void TiledInputFile::tileOrder(int dx[], int dy[], int lx[], int ly[]) const
{
   return _data->tileOffsets.getTileOrder(dx,dy,lx,ly);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
