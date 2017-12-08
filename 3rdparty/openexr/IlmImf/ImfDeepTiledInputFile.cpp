///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2011, Industrial Light & Magic, a division of Lucas
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
//      class DeepTiledInputFile
//
//-----------------------------------------------------------------------------

#include <ImfDeepTiledInputFile.h>
#include <ImfTileDescriptionAttribute.h>
#include <ImfChannelList.h>
#include <ImfMisc.h>
#include <ImfTiledMisc.h>
#include <ImfStdIO.h>
#include <ImfCompressor.h>
#include "ImathBox.h"
#include <ImfXdr.h>
#include <ImfConvert.h>
#include <ImfVersion.h>
#include <ImfTileOffsets.h>
#include <ImfThreading.h>
#include <ImfPartType.h>
#include <ImfMultiPartInputFile.h>
#include "IlmThreadPool.h"
#include "IlmThreadSemaphore.h"
#include "IlmThreadMutex.h"
#include "ImfInputStreamMutex.h"
#include "ImfInputPartData.h"
#include "ImathVec.h"
#include "Iex.h"
#include <string>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <limits>

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
    PixelType           typeInFrameBuffer;
    PixelType           typeInFile;
    char*               pointerArrayBase;
    size_t              xStride;
    size_t              yStride;
    ptrdiff_t           sampleStride;
    bool                fill;
    bool                skip;
    double              fillValue;
    int                 xTileCoords;
    int                 yTileCoords;

    TInSliceInfo (PixelType typeInFrameBuffer = HALF,
                  char * base = NULL,
                  PixelType typeInFile = HALF,
                  size_t xStride = 0,
                  size_t yStride = 0,
                  ptrdiff_t sampleStride = 0,
                  bool fill = false,
                  bool skip = false,
                  double fillValue = 0.0,
                  int xTileCoords = 0,
                  int yTileCoords = 0);
};


TInSliceInfo::TInSliceInfo (PixelType tifb,
                            char * b,
                            PixelType tifl,
                            size_t xs, size_t ys,
                            ptrdiff_t spst,
                            bool f, bool s,
                            double fv,
                            int xtc,
                            int ytc)
:
    typeInFrameBuffer (tifb),
    typeInFile (tifl),
    pointerArrayBase (b),
    xStride (xs),
    yStride (ys),
    sampleStride (spst),
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
    Array2D<unsigned int>       sampleCount;
    const char *                uncompressedData;
    char *                      buffer;
    Int64                         dataSize;
    Int64                         uncompressedDataSize;
    Compressor *                compressor;
    Compressor::Format          format;
    int                         dx;
    int                         dy;
    int                         lx;
    int                         ly;
    bool                        hasException;
    string                      exception;

     TileBuffer ();
    ~TileBuffer ();

    inline void         wait () {_sem.wait();}
    inline void         post () {_sem.post();}

 protected:

    Semaphore _sem;
};


TileBuffer::TileBuffer ():
    uncompressedData (0),
    buffer (0),
    dataSize (0),
    compressor (0),
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

struct DeepTiledInputFile::Data: public Mutex
{
    Header          header;                         // the image header
    TileDescription tileDesc;                       // describes the tile layout
    int             version;                        // file's version
    DeepFrameBuffer frameBuffer;                    // framebuffer to write into
    LineOrder       lineOrder;                      // the file's lineorder
    int             minX;                           // data window's min x coord
    int             maxX;                           // data window's max x coord
    int             minY;                           // data window's min y coord
    int             maxY;                           // data window's max x coord

    int             numXLevels;                     // number of x levels
    int             numYLevels;                     // number of y levels
    int *           numXTiles;                      // number of x tiles at a level
    int *           numYTiles;                      // number of y tiles at a level

    TileOffsets     tileOffsets;                    // stores offsets in file for
    // each tile

    bool            fileIsComplete;                 // True if no tiles are missing
                                                    // in the file

    vector<TInSliceInfo*> slices;                    // info about channels in file

                                                    // ourselves? or does someone
                                                    // else do it?

    int             partNumber;                     // part number

    bool            multiPartBackwardSupport;       // if we are reading a multipart file
                                                    // using OpenEXR 1.7 API

    int             numThreads;                     // number of threads

    MultiPartInputFile* multiPartFile;              // the MultiPartInputFile used to
                                                    // support backward compatibility

    vector<TileBuffer*> tileBuffers;                // each holds a single tile

    bool            memoryMapped;                   // if the stream is memory mapped

    char*           sampleCountSliceBase;           // pointer to the start of
                                                    // the sample count array
    ptrdiff_t       sampleCountXStride;             // x stride of the sample count array
    ptrdiff_t       sampleCountYStride;             // y stride of the sample count array

    int             sampleCountXTileCoords;         // the value of xTileCoords from the
                                                    // sample count slice
    int             sampleCountYTileCoords;         // the value of yTileCoords from the
                                                    // sample count slice

    Array<char>     sampleCountTableBuffer;         // the buffer for sample count table

    Compressor*     sampleCountTableComp;           // the decompressor for sample count table

    Int64           maxSampleCountTableSize;        // the max size in bytes for a pixel
                                                    // sample count table
    int             combinedSampleSize;             // total size of all channels combined to check sampletable size
                                                    
    InputStreamMutex *  _streamData;
    bool                _deleteStream; // should we delete the stream
     Data (int numThreads);
    ~Data ();

    inline TileBuffer * getTileBuffer (int number);
                                                    // hash function from tile indices
                                                    // into our vector of tile buffers

    int&                getSampleCount(int x, int y);
                                                    // get the number of samples
                                                    // in each pixel
};


DeepTiledInputFile::Data::Data (int numThreads):
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


DeepTiledInputFile::Data::~Data ()
{
    delete [] numXTiles;
    delete [] numYTiles;

    for (size_t i = 0; i < tileBuffers.size(); i++)
        delete tileBuffers[i];

    if (multiPartBackwardSupport)
        delete multiPartFile;

    for (size_t i = 0; i < slices.size(); i++)
        delete slices[i];
}


TileBuffer*
DeepTiledInputFile::Data::getTileBuffer (int number)
{
    return tileBuffers[number % tileBuffers.size()];
}


int&
DeepTiledInputFile::Data::getSampleCount(int x, int y)
{
    return sampleCount(sampleCountSliceBase,
                       sampleCountXStride,
                       sampleCountYStride,
                       x, y);
}


namespace {

void
readTileData (InputStreamMutex *streamData,
              DeepTiledInputFile::Data *ifd,
              int dx, int dy,
              int lx, int ly,
              char *&buffer,
              Int64 &dataSize,
              Int64 &unpackedDataSize)
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

    if ( !isMultiPart(ifd->version) )
    {
        if (streamData->currentPosition != tileOffset)
            streamData->is->seekg(tileOffset);
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

    Xdr::read <StreamIO> (*streamData->is, tileXCoord);
    Xdr::read <StreamIO> (*streamData->is, tileYCoord);
    Xdr::read <StreamIO> (*streamData->is, levelX);
    Xdr::read <StreamIO> (*streamData->is, levelY);

    Int64 tableSize;
    Xdr::read <StreamIO> (*streamData->is, tableSize);

    Xdr::read <StreamIO> (*streamData->is, dataSize);
    Xdr::read <StreamIO> (*streamData->is, unpackedDataSize);


    //
    // Skip the pixel sample count table because we have read this data.
    //

    Xdr::skip <StreamIO> (*streamData->is, tableSize);


    if (tileXCoord != dx)
        throw IEX_NAMESPACE::InputExc ("Unexpected tile x coordinate.");

    if (tileYCoord != dy)
        throw IEX_NAMESPACE::InputExc ("Unexpected tile y coordinate.");

    if (levelX != lx)
        throw IEX_NAMESPACE::InputExc ("Unexpected tile x level number coordinate.");

    if (levelY != ly)
        throw IEX_NAMESPACE::InputExc ("Unexpected tile y level number coordinate.");

    //
    // Read the pixel data.
    //

    if (streamData->is->isMemoryMapped ())
        buffer = streamData->is->readMemoryMapped (dataSize);
    else
    {
        // (TODO) check if the packed data size is too big?
        // (TODO) better memory management here. Don't delete buffer everytime.
        if (buffer != 0) delete[] buffer;
        buffer = new char[dataSize];
        streamData->is->read (buffer, dataSize);
    }

    //
    // Keep track of which tile is the next one in
    // the file, so that we can avoid redundant seekg()
    // operations (seekg() can be fairly expensive).
    //

    streamData->currentPosition = tileOffset + 4 * Xdr::size<int>() +
                                  3 * Xdr::size<Int64>()            +
                                  tableSize                         +
                                  dataSize;
}


void
readNextTileData (InputStreamMutex *streamData,
                  DeepTiledInputFile::Data *ifd,
                  int &dx, int &dy,
                  int &lx, int &ly,
                  char * & buffer,
                  Int64 &dataSize,
                  Int64 &unpackedDataSize)
{
    //
    // Read the next tile block from the file
    //

    //
    // Read the first few bytes of the tile (the header).
    //

    Xdr::read <StreamIO> (*streamData->is, dx);
    Xdr::read <StreamIO> (*streamData->is, dy);
    Xdr::read <StreamIO> (*streamData->is, lx);
    Xdr::read <StreamIO> (*streamData->is, ly);

    Int64 tableSize;
    Xdr::read <StreamIO> (*streamData->is, tableSize);

    Xdr::read <StreamIO> (*streamData->is, dataSize);
    Xdr::read <StreamIO> (*streamData->is, unpackedDataSize);

    //
    // Skip the pixel sample count table because we have read this data.
    //

    Xdr::skip <StreamIO> (*streamData->is, tableSize);

    //
    // Read the pixel data.
    //

    streamData->is->read (buffer, dataSize);

    //
    // Keep track of which tile is the next one in
    // the file, so that we can avoid redundant seekg()
    // operations (seekg() can be fairly expensive).
    //

    streamData->currentPosition += 4 * Xdr::size<int>()   +
                                   3 * Xdr::size<Int64>() +
                                   tableSize              +
                                   dataSize;
}


//
// A TileBufferTask encapsulates the task of uncompressing
// a single tile and copying it into the frame buffer.
//

class TileBufferTask : public Task
{
  public:

    TileBufferTask (TaskGroup *group,
                    DeepTiledInputFile::Data *ifd,
                    TileBuffer *tileBuffer);

    virtual ~TileBufferTask ();

    virtual void                execute ();

  private:

    DeepTiledInputFile::Data *      _ifd;
    TileBuffer *                _tileBuffer;
};


TileBufferTask::TileBufferTask
    (TaskGroup *group,
     DeepTiledInputFile::Data *ifd,
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

        Box2i tileRange = OPENEXR_IMF_INTERNAL_NAMESPACE::dataWindowForTile (
                _ifd->tileDesc,
                _ifd->minX, _ifd->maxX,
                _ifd->minY, _ifd->maxY,
                _tileBuffer->dx,
                _tileBuffer->dy,
                _tileBuffer->lx,
                _tileBuffer->ly);

        //
        // Get the size of the tile.
        //

        Array<unsigned int> numPixelsPerScanLine;
        numPixelsPerScanLine.resizeErase(tileRange.max.y - tileRange.min.y + 1);

        int sizeOfTile = 0;
        int maxBytesPerTileLine = 0;

        for (int y = tileRange.min.y; y <= tileRange.max.y; y++)
        {
            numPixelsPerScanLine[y - tileRange.min.y] = 0;

            int bytesPerLine = 0;

            for (int x = tileRange.min.x; x <= tileRange.max.x; x++)
            {
                int xOffset = _ifd->sampleCountXTileCoords * tileRange.min.x;
                int yOffset = _ifd->sampleCountYTileCoords * tileRange.min.y;

                int count = _ifd->getSampleCount(x - xOffset, y - yOffset);
                for (unsigned int c = 0; c < _ifd->slices.size(); ++c)
                {
                    sizeOfTile += count * pixelTypeSize(_ifd->slices[c]->typeInFile);
                    bytesPerLine += count * pixelTypeSize(_ifd->slices[c]->typeInFile);
                }
                numPixelsPerScanLine[y - tileRange.min.y] += count;
            }

            if (bytesPerLine > maxBytesPerTileLine)
                maxBytesPerTileLine = bytesPerLine;
        }

        // (TODO) don't do this every time.
        if (_tileBuffer->compressor != 0)
            delete _tileBuffer->compressor;
        _tileBuffer->compressor = newTileCompressor
                                  (_ifd->header.compression(),
                                   maxBytesPerTileLine,
                                   _ifd->tileDesc.ySize,
                                   _ifd->header);

        //
        // Uncompress the data, if necessary
        //

        if (_tileBuffer->compressor && _tileBuffer->dataSize < Int64(sizeOfTile))
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
                TInSliceInfo &slice = *_ifd->slices[i];

                //
                // These offsets are used to facilitate both
                // absolute and tile-relative pixel coordinates.
                //

                int xOffsetForData = (slice.xTileCoords == 0) ? 0 : tileRange.min.x;
                int yOffsetForData = (slice.yTileCoords == 0) ? 0 : tileRange.min.y;
                int xOffsetForSampleCount =
                        (_ifd->sampleCountXTileCoords == 0) ? 0 : tileRange.min.x;
                int yOffsetForSampleCount =
                        (_ifd->sampleCountYTileCoords == 0) ? 0 : tileRange.min.y;

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
                                 numPixelsPerScanLine[y - tileRange.min.y]);
                }
                else
                {
                    //
                    // The frame buffer contains a slice for this channel.
                    //

                    copyIntoDeepFrameBuffer (readPtr, slice.pointerArrayBase,
                                             _ifd->sampleCountSliceBase,
                                             _ifd->sampleCountXStride,
                                             _ifd->sampleCountYStride,
                                             y,
                                             tileRange.min.x,
                                             tileRange.max.x,
                                             xOffsetForSampleCount, yOffsetForSampleCount,
                                             xOffsetForData, yOffsetForData,
                                             slice.sampleStride, 
                                             slice.xStride,
                                             slice.yStride,
                                             slice.fill,
                                             slice.fillValue, _tileBuffer->format,
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
     DeepTiledInputFile::Data *ifd,
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

        readTileData (ifd->_streamData, ifd, dx, dy, lx, ly,
                      tileBuffer->buffer,
                      tileBuffer->dataSize,
                      tileBuffer->uncompressedDataSize);
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


DeepTiledInputFile::DeepTiledInputFile (const char fileName[], int numThreads):
    _data (new Data (numThreads))
{
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
        // Compatibility to read multpart file.
        //
        if (isMultiPart(_data->version))
        {
            compatibilityInitialize(*is);
        }
        else
        {
            _data->_streamData = new InputStreamMutex();
            _data->_streamData->is = is;
            _data->header.readFrom (*_data->_streamData->is, _data->version);
            initialize();
            _data->tileOffsets.readFrom (*(_data->_streamData->is), _data->fileIsComplete,false,true);
            _data->_streamData->currentPosition = _data->_streamData->is->tellg();
        }
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        if (is)          delete is;
        if (_data && !_data->multiPartBackwardSupport && _data->_streamData) delete _data->_streamData;
        if (_data)       delete _data;

        REPLACE_EXC (e, "Cannot open image file "
                        "\"" << fileName << "\". " << e);
        throw;
    }
    catch (...)
    {
        if (is)          delete is;
        if (_data && !_data->multiPartBackwardSupport && _data->_streamData) delete _data->_streamData;
        if (_data)       delete _data;

        throw;
    }
}


DeepTiledInputFile::DeepTiledInputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int numThreads):
    _data (new Data (numThreads))
{
    _data->_streamData=0;
    _data->_deleteStream=false;
    
    //
    // This constructor is called when a user
    // explicitly wants to read a tiled file.
    //

    try
    {
        readMagicNumberAndVersionField(is, _data->version);

        //
        // Backward compatibility to read multpart file.
        //
        if (isMultiPart(_data->version))
        {
            compatibilityInitialize(is);
        }
        else
        {
            _data->_streamData = new InputStreamMutex();
            _data->_streamData->is = &is;
            _data->header.readFrom (*_data->_streamData->is, _data->version);
            initialize();
            // file is guaranteed not to be multipart, but is deep
            _data->tileOffsets.readFrom (*(_data->_streamData->is), _data->fileIsComplete, false,true);
            _data->memoryMapped = _data->_streamData->is->isMemoryMapped();
            _data->_streamData->currentPosition = _data->_streamData->is->tellg();
        }
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        if (_data && !_data->multiPartBackwardSupport && _data->_streamData) delete _data->_streamData;
        if (_data)       delete _data;

        REPLACE_EXC (e, "Cannot open image file "
                        "\"" << is.fileName() << "\". " << e);
        throw;
    }
    catch (...)
    {
        if (_data && !_data->multiPartBackwardSupport && _data->_streamData) delete _data->_streamData;
        if (_data)       delete _data;

        throw;
    }
}


DeepTiledInputFile::DeepTiledInputFile (const Header &header,
                                        OPENEXR_IMF_INTERNAL_NAMESPACE::IStream *is,
                                        int version,
                                        int numThreads) :
    _data (new Data (numThreads))
    
{
    _data->_streamData->is = is;
    _data->_deleteStream=false;
    
    //
    // This constructor called by class Imf::InputFile
    // when a user wants to just read an image file, and
    // doesn't care or know if the file is tiled.
    // No need to have backward compatibility here, because
    // we have the header.
    //

    _data->header = header;
    _data->version = version;
    initialize();
    _data->tileOffsets.readFrom (*(_data->_streamData->is), _data->fileIsComplete,false,true);
    _data->memoryMapped = is->isMemoryMapped();
    _data->_streamData->currentPosition = _data->_streamData->is->tellg();
}


DeepTiledInputFile::DeepTiledInputFile (InputPartData* part) :
    _data (new Data (part->numThreads))
{
    _data->_deleteStream=false;
    multiPartInitialize(part);
}


void
DeepTiledInputFile::compatibilityInitialize(OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is)
{
    is.seekg(0);
    //
    // Construct a MultiPartInputFile, initialize TiledInputFile
    // with the part 0 data.
    // (TODO) maybe change the third parameter of the constructor of MultiPartInputFile later.
    //
    _data->multiPartFile = new MultiPartInputFile(is, _data->numThreads);
    _data->multiPartBackwardSupport = true;
    InputPartData* part = _data->multiPartFile->getPart(0);

    multiPartInitialize(part);
}


void
DeepTiledInputFile::multiPartInitialize(InputPartData* part)
{
    if (isTiled(part->header.type()) == false)
        THROW (IEX_NAMESPACE::ArgExc, "Can't build a DeepTiledInputFile from a part of type " << part->header.type());

    _data->_streamData = part->mutex;
    _data->header = part->header;
    _data->version = part->version;
    _data->partNumber = part->partNumber;
    _data->memoryMapped = _data->_streamData->is->isMemoryMapped();
    initialize();
    _data->tileOffsets.readFrom(part->chunkOffsets , _data->fileIsComplete);
    _data->_streamData->currentPosition = _data->_streamData->is->tellg();
}


void
DeepTiledInputFile::initialize ()
{
    if (_data->partNumber == -1)
        if (_data->header.type() != DEEPTILE)
            throw IEX_NAMESPACE::ArgExc ("Expected a deep tiled file but the file is not deep tiled.");
   if(_data->header.version()!=1)
   {
       THROW(IEX_NAMESPACE::ArgExc, "Version " << _data->header.version() << " not supported for deeptiled images in this version of the library");
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

    //
    // Create all the TileBuffers and allocate their internal buffers
    //

    _data->tileOffsets = TileOffsets (_data->tileDesc.mode,
                                      _data->numXLevels,
                                      _data->numYLevels,
                                      _data->numXTiles,
                                      _data->numYTiles);

    for (size_t i = 0; i < _data->tileBuffers.size(); i++)
        _data->tileBuffers[i] = new TileBuffer ();

    _data->maxSampleCountTableSize = _data->tileDesc.ySize *
                                     _data->tileDesc.xSize *
                                     sizeof(int);

    _data->sampleCountTableBuffer.resizeErase(_data->maxSampleCountTableSize);

    _data->sampleCountTableComp = newCompressor(_data->header.compression(),
                                                _data->maxSampleCountTableSize,
                                                _data->header);
                                                
                                                
    const ChannelList & c=_data->header.channels();
    _data->combinedSampleSize=0;
    for(ChannelList::ConstIterator i=c.begin();i!=c.end();i++)
    {
        switch( i.channel().type )
        {
            case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF  :
                _data->combinedSampleSize+=Xdr::size<half>();
                break;
            case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT :
                _data->combinedSampleSize+=Xdr::size<float>();
                break;
            case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT  :
                _data->combinedSampleSize+=Xdr::size<unsigned int>();
                break;
            default :
                THROW(IEX_NAMESPACE::ArgExc, "Bad type for channel " << i.name() << " initializing deepscanline reader");
        }
    }
                                                  
}


DeepTiledInputFile::~DeepTiledInputFile ()
{
    if (!_data->memoryMapped)
        for (size_t i = 0; i < _data->tileBuffers.size(); i++)
            if (_data->tileBuffers[i]->buffer != 0)
                delete [] _data->tileBuffers[i]->buffer;

    if (_data->_deleteStream)
        delete _data->_streamData->is;

    //
    // (TODO) we should have a way to tell if the stream data is owned by this file or
    // by a parent multipart file.
    //

    if (_data->partNumber == -1)
        delete _data->_streamData;

    delete _data;
}


const char *
DeepTiledInputFile::fileName () const
{
    return _data->_streamData->is->fileName();
}


const Header &
DeepTiledInputFile::header () const
{
    return _data->header;
}


int
DeepTiledInputFile::version () const
{
    return _data->version;
}


void
DeepTiledInputFile::setFrameBuffer (const DeepFrameBuffer &frameBuffer)
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

    for (DeepFrameBuffer::ConstIterator j = frameBuffer.begin();
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
    // Store the pixel sample count table.
    // (TODO) Support for different sampling rates?
    //

    const Slice& sampleCountSlice = frameBuffer.getSampleCountSlice();
    if (sampleCountSlice.base == 0)
    {
        throw IEX_NAMESPACE::ArgExc ("Invalid base pointer, please set a proper sample count slice.");
    }
    else
    {
        _data->sampleCountSliceBase = sampleCountSlice.base;
        _data->sampleCountXStride = sampleCountSlice.xStride;
        _data->sampleCountYStride = sampleCountSlice.yStride;
        _data->sampleCountXTileCoords = sampleCountSlice.xTileCoords;
        _data->sampleCountYTileCoords = sampleCountSlice.yTileCoords;
    }

    //
    // Initialize the slice table for readPixels().
    //

    vector<TInSliceInfo*> slices;
    ChannelList::ConstIterator i = channels.begin();

    for (DeepFrameBuffer::ConstIterator j = frameBuffer.begin();
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

            slices.push_back (new TInSliceInfo (i.channel().type,
                                                NULL,
                                                i.channel().type,
                                                0,      // xStride
                                                0,      // yStride
                                                0,      // sampleStride
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

        slices.push_back (new TInSliceInfo (j.slice().type,
                                            j.slice().base,
                                            fill? j.slice().type: i.channel().type,
                                            j.slice().xStride,
                                            j.slice().yStride,
                                            j.slice().sampleStride,
                                            fill,
                                            false, // skip
                                            j.slice().fillValue,
                                            (j.slice().xTileCoords)? 1: 0,
                                            (j.slice().yTileCoords)? 1: 0));


        if (i != channels.end() && !fill)
            ++i;
    }

    // (TODO) inspect the following code. It's additional to the scanline input file.
    // Is this needed?

    while (i != channels.end())
    {
        //
        // Channel i is present in the file but not
        // in the frame buffer; data for channel i
        // will be skipped during readPixels().
        //

        slices.push_back (new TInSliceInfo (i.channel().type,
                                            NULL,
                                            i.channel().type,
                                            0, // xStride
                                            0, // yStride
                                            0, // sampleStride
                                            false,  // fill
                                            true, // skip
                                            0.0)); // fillValue
        ++i;
    }

    //
    // Store the new frame buffer.
    //

    _data->frameBuffer = frameBuffer;

    for (size_t i = 0; i < _data->slices.size(); i++)
        delete _data->slices[i];
    _data->slices = slices;
}


const DeepFrameBuffer &
DeepTiledInputFile::frameBuffer () const
{
    Lock lock (*_data->_streamData);
    return _data->frameBuffer;
}


bool
DeepTiledInputFile::isComplete () const
{
    return _data->fileIsComplete;
}


void
DeepTiledInputFile::readTiles (int dx1, int dx2, int dy1, int dy2, int lx, int ly)
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
                        "file \"" << fileName() << "\". " << e);
        throw;
    }
}


void
DeepTiledInputFile::readTiles (int dx1, int dx2, int dy1, int dy2, int l)
{
    readTiles (dx1, dx2, dy1, dy2, l, l);
}


void
DeepTiledInputFile::readTile (int dx, int dy, int lx, int ly)
{
    readTiles (dx, dx, dy, dy, lx, ly);
}


void
DeepTiledInputFile::readTile (int dx, int dy, int l)
{
    readTile (dx, dy, l, l);
}


void
DeepTiledInputFile::rawTileData (int &dx, int &dy,
                             int &lx, int &ly,
                             char * pixelData,
                             Int64 &pixelDataSize) const
{
     if (!isValidTile (dx, dy, lx, ly))
               throw IEX_NAMESPACE::ArgExc ("Tried to read a tile outside "
                                   "the image file's data window.");
    
     Int64 tileOffset = _data->tileOffsets (dx, dy, lx, ly);
                                   
     if(tileOffset == 0)
     {
        THROW (IEX_NAMESPACE::InputExc, "Tile (" << dx << ", " << dy << ", " <<
        lx << ", " << ly << ") is missing.");
     }
     
     Lock lock(*_data->_streamData);
                                   
     if (_data->_streamData->is->tellg() != tileOffset)
                                          _data->_streamData->is->seekg (tileOffset);
                                   
     
     //
     // Read the first few bytes of the tile (the header).
     // Verify that the tile coordinates and the level number
     // are correct.
     //
     
     int tileXCoord, tileYCoord, levelX, levelY;
     
     if (isMultiPart(_data->version))
     {
         int partNumber;
         Xdr::read <StreamIO> (*_data->_streamData->is, partNumber);
         if (partNumber != _data->partNumber)
         {
             THROW (IEX_NAMESPACE::ArgExc, "Unexpected part number " << partNumber
             << ", should be " << _data->partNumber << ".");
         }
     }
     
     Xdr::read <StreamIO> (*_data->_streamData->is, tileXCoord);
     Xdr::read <StreamIO> (*_data->_streamData->is, tileYCoord);
     Xdr::read <StreamIO> (*_data->_streamData->is, levelX);
     Xdr::read <StreamIO> (*_data->_streamData->is, levelY);
     
     Int64 sampleCountTableSize;
     Int64 packedDataSize;
     Xdr::read <StreamIO> (*_data->_streamData->is, sampleCountTableSize);
     
     Xdr::read <StreamIO> (*_data->_streamData->is, packedDataSize);
     
          
     
     if (tileXCoord != dx)
         throw IEX_NAMESPACE::InputExc ("Unexpected tile x coordinate.");
     
     if (tileYCoord != dy)
         throw IEX_NAMESPACE::InputExc ("Unexpected tile y coordinate.");
     
     if (levelX != lx)
         throw IEX_NAMESPACE::InputExc ("Unexpected tile x level number coordinate.");
     
     if (levelY != ly)
         throw IEX_NAMESPACE::InputExc ("Unexpected tile y level number coordinate.");
     
     
     // total requirement for reading all the data
     
     Int64 totalSizeRequired=40+sampleCountTableSize+packedDataSize;
     
     bool big_enough = totalSizeRequired<=pixelDataSize;
     
     pixelDataSize = totalSizeRequired;
     
     // was the block we were given big enough?
     if(!big_enough || pixelData==NULL)
     {        
         // special case: seek stream back to start if we are at the beginning (regular reading pixels assumes it doesn't need to seek
         // in single part files)
         if(!isMultiPart(_data->version))
         {
             _data->_streamData->is->seekg(_data->_streamData->currentPosition); 
         }
         // leave lock here - bail before reading more data
         return;
     }
     
     // copy the values we have read into the output block
     *(int *) (pixelData+0) = dx;
     *(int *) (pixelData+4) = dy;
     *(int *) (pixelData+8) = levelX;
     *(int *) (pixelData+12) = levelY;
     *(Int64 *) (pixelData+16) =sampleCountTableSize;
     *(Int64 *) (pixelData+24) = packedDataSize;
     
     // didn't read the unpackedsize - do that now
     Xdr::read<StreamIO> (*_data->_streamData->is, *(Int64 *) (pixelData+32));
     
     // read the actual data
     _data->_streamData->is->read(pixelData+40, sampleCountTableSize+packedDataSize);
     
     
     if(!isMultiPart(_data->version))
     {
         _data->_streamData->currentPosition+=sampleCountTableSize+packedDataSize+40;
     }
     
     // leave lock here
     
     
}


unsigned int
DeepTiledInputFile::tileXSize () const
{
    return _data->tileDesc.xSize;
}


unsigned int
DeepTiledInputFile::tileYSize () const
{
    return _data->tileDesc.ySize;
}


LevelMode
DeepTiledInputFile::levelMode () const
{
    return _data->tileDesc.mode;
}


LevelRoundingMode
DeepTiledInputFile::levelRoundingMode () const
{
    return _data->tileDesc.roundingMode;
}


int
DeepTiledInputFile::numLevels () const
{
    if (levelMode() == RIPMAP_LEVELS)
        THROW (IEX_NAMESPACE::LogicExc, "Error calling numLevels() on image "
                              "file \"" << fileName() << "\" "
                              "(numLevels() is not defined for files "
                              "with RIPMAP level mode).");

    return _data->numXLevels;
}


int
DeepTiledInputFile::numXLevels () const
{
    return _data->numXLevels;
}


int
DeepTiledInputFile::numYLevels () const
{
    return _data->numYLevels;
}


bool
DeepTiledInputFile::isValidLevel (int lx, int ly) const
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
DeepTiledInputFile::levelWidth (int lx) const
{
    try
    {
        return levelSize (_data->minX, _data->maxX, lx,
                          _data->tileDesc.roundingMode);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        REPLACE_EXC (e, "Error calling levelWidth() on image "
                        "file \"" << fileName() << "\". " << e);
        throw;
    }
}


int
DeepTiledInputFile::levelHeight (int ly) const
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
DeepTiledInputFile::numXTiles (int lx) const
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
DeepTiledInputFile::numYTiles (int ly) const
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
DeepTiledInputFile::dataWindowForLevel (int l) const
{
    return dataWindowForLevel (l, l);
}


Box2i
DeepTiledInputFile::dataWindowForLevel (int lx, int ly) const
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
DeepTiledInputFile::dataWindowForTile (int dx, int dy, int l) const
{
    return dataWindowForTile (dx, dy, l, l);
}


Box2i
DeepTiledInputFile::dataWindowForTile (int dx, int dy, int lx, int ly) const
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
                        "file \"" << fileName() << "\". " << e);
        throw;
    }
}


bool
DeepTiledInputFile::isValidTile (int dx, int dy, int lx, int ly) const
{
    return ((lx < _data->numXLevels && lx >= 0) &&
            (ly < _data->numYLevels && ly >= 0) &&
            (dx < _data->numXTiles[lx] && dx >= 0) &&
            (dy < _data->numYTiles[ly] && dy >= 0));
}


void
DeepTiledInputFile::readPixelSampleCounts (int dx1, int dx2,
                                           int dy1, int dy2,
                                           int lx,  int ly)
{
    Int64 savedFilePos = 0;

    try
    {
        Lock lock (*_data->_streamData);

        savedFilePos = _data->_streamData->is->tellg();

        
        if (!isValidLevel (lx, ly))
        {
            THROW (IEX_NAMESPACE::ArgExc,
                   "Level coordinate "
                   "(" << lx << ", " << ly << ") "
                   "is invalid.");
        }
        
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

        // (TODO) Check if we have read the sample counts for those tiles,
        // if we have, no need to read again.
        for (int dy = dyStart; dy != dyStop; dy += dY)
        {
            for (int dx = dx1; dx <= dx2; dx++)
            {
                
                if (!isValidTile (dx, dy, lx, ly))
                {
                    THROW (IEX_NAMESPACE::ArgExc,
                           "Tile (" << dx << ", " << dy << ", " <<
                           lx << "," << ly << ") is not a valid tile.");
                }
                
                Box2i tileRange = OPENEXR_IMF_INTERNAL_NAMESPACE::dataWindowForTile (
                        _data->tileDesc,
                        _data->minX, _data->maxX,
                        _data->minY, _data->maxY,
                        dx, dy, lx, ly);

                int xOffset = _data->sampleCountXTileCoords * tileRange.min.x;
                int yOffset = _data->sampleCountYTileCoords * tileRange.min.y;

                //
                // Skip and check the tile coordinates.
                //

                _data->_streamData->is->seekg(_data->tileOffsets(dx, dy, lx, ly));

                if (isMultiPart(_data->version))
                {
                    int partNumber;
                    Xdr::read <StreamIO> (*_data->_streamData->is, partNumber);

                    if (partNumber != _data->partNumber)
                        throw IEX_NAMESPACE::InputExc ("Unexpected part number.");
                }

                int xInFile, yInFile, lxInFile, lyInFile;
                Xdr::read <StreamIO> (*_data->_streamData->is, xInFile);
                Xdr::read <StreamIO> (*_data->_streamData->is, yInFile);
                Xdr::read <StreamIO> (*_data->_streamData->is, lxInFile);
                Xdr::read <StreamIO> (*_data->_streamData->is, lyInFile);

                if (xInFile != dx)
                    throw IEX_NAMESPACE::InputExc ("Unexpected tile x coordinate.");

                if (yInFile != dy)
                    throw IEX_NAMESPACE::InputExc ("Unexpected tile y coordinate.");

                if (lxInFile != lx)
                    throw IEX_NAMESPACE::InputExc ("Unexpected tile x level number coordinate.");

                if (lyInFile != ly)
                    throw IEX_NAMESPACE::InputExc ("Unexpected tile y level number coordinate.");

                Int64 tableSize, dataSize, unpackedDataSize;
                Xdr::read <StreamIO> (*_data->_streamData->is, tableSize);
                Xdr::read <StreamIO> (*_data->_streamData->is, dataSize);
                Xdr::read <StreamIO> (*_data->_streamData->is, unpackedDataSize);

                
                if(tableSize>_data->maxSampleCountTableSize)
                {
                    THROW (IEX_NAMESPACE::ArgExc, "Bad sampleCountTableDataSize read from tile "<< dx << ',' << dy << ',' << lx << ',' << ly << ": expected " << _data->maxSampleCountTableSize << " or less, got "<< tableSize);
                }
                    
                
                //
                // We make a check on the data size requirements here.
                // Whilst we wish to store 64bit sizes on disk, not all the compressors
                // have been made to work with such data sizes and are still limited to
                // using signed 32 bit (int) for the data size. As such, this version
                // insists that we validate that the data size does not exceed the data
                // type max limit.
                // @TODO refactor the compressor code to ensure full 64-bit support.
                //

                Int64 compressorMaxDataSize = Int64(std::numeric_limits<int>::max());
                if (dataSize         > compressorMaxDataSize ||
                    unpackedDataSize > compressorMaxDataSize ||
                    tableSize        > compressorMaxDataSize)
                {
                    THROW (IEX_NAMESPACE::ArgExc, "This version of the library does not"
                          << "support the allocation of data with size  > "
                          << compressorMaxDataSize
                          << " file table size    :" << tableSize
                          << " file unpacked size :" << unpackedDataSize
                          << " file packed size   :" << dataSize << ".\n");
                }

                //
                // Read and uncompress the pixel sample count table.
                //

                _data->_streamData->is->read(_data->sampleCountTableBuffer, tableSize);

                const char* readPtr;

                if (tableSize < _data->maxSampleCountTableSize)
                {
                    if(!_data->sampleCountTableComp)
                    {
                        THROW(IEX_NAMESPACE::ArgExc,"Deep scanline data corrupt at tile " << dx << ',' << dy << ',' << lx << ',' <<  ly << " (sampleCountTableDataSize error)");
                    }
                    _data->sampleCountTableComp->uncompress(_data->sampleCountTableBuffer,
                                                            tableSize,
                                                            tileRange.min.y,
                                                            readPtr);
                }
                else
                    readPtr = _data->sampleCountTableBuffer;

                size_t cumulative_total_samples =0;
                int lastAccumulatedCount;
                for (int j = tileRange.min.y; j <= tileRange.max.y; j++)
                {
                    lastAccumulatedCount = 0;
                    for (int i = tileRange.min.x; i <= tileRange.max.x; i++)
                    {
                        int accumulatedCount;
                        Xdr::read <CharPtrIO> (readPtr, accumulatedCount);
                        
                        if (accumulatedCount < lastAccumulatedCount)
                        {
                            THROW(IEX_NAMESPACE::ArgExc,"Deep tile sampleCount data corrupt at tile " 
                                  << dx << ',' << dy << ',' << lx << ',' <<  ly << " (negative sample count detected)");
                        }

                        int count = accumulatedCount - lastAccumulatedCount;
                        lastAccumulatedCount = accumulatedCount;
                        
                        _data->getSampleCount(i - xOffset, j - yOffset) =count;
                    }
                    cumulative_total_samples += lastAccumulatedCount;
                }
                
                if(cumulative_total_samples * _data->combinedSampleSize > unpackedDataSize)
                {
                    THROW(IEX_NAMESPACE::ArgExc,"Deep scanline sampleCount data corrupt at tile " 
                                                << dx << ',' << dy << ',' << lx << ',' <<  ly 
                                                << ": pixel data only contains " << unpackedDataSize 
                                                << " bytes of data but table references at least " 
                                                << cumulative_total_samples*_data->combinedSampleSize << " bytes of sample data" );            
                }
                    
            }
        }

        _data->_streamData->is->seekg(savedFilePos);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        REPLACE_EXC (e, "Error reading sample count data from image "
                        "file \"" << fileName() << "\". " << e);

         _data->_streamData->is->seekg(savedFilePos);

        throw;
    }
}


void
DeepTiledInputFile::readPixelSampleCount (int dx, int dy, int l)
{
    readPixelSampleCount (dx, dy, l, l);
}


void
DeepTiledInputFile::readPixelSampleCount (int dx, int dy, int lx, int ly)
{
    readPixelSampleCounts (dx, dx, dy, dy, lx, ly);
}


void
DeepTiledInputFile::readPixelSampleCounts (int dx1, int dx2,
                                           int dy1, int dy2,
                                           int l)
{
    readPixelSampleCounts (dx1, dx2, dy1, dy2, l, l);
}


size_t
DeepTiledInputFile::totalTiles() const
{
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
    return numAllTiles;
}




void 
DeepTiledInputFile::getTileOrder(int dx[],int dy[],int lx[],int ly[]) const
{
  return _data->tileOffsets.getTileOrder(dx,dy,lx,ly);
  
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
