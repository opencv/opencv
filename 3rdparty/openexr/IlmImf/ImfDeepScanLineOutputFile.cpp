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
//      class DeepScanLineOutputFile
//
//-----------------------------------------------------------------------------

#include <ImfDeepScanLineOutputFile.h>
#include <ImfDeepScanLineInputFile.h>
#include <ImfDeepScanLineInputPart.h>
#include <ImfChannelList.h>
#include <ImfMisc.h>
#include <ImfStdIO.h>
#include <ImfCompressor.h>
#include "ImathBox.h"
#include "ImathFun.h"
#include <ImfArray.h>
#include <ImfXdr.h>
#include <ImfPreviewImageAttribute.h>
#include <ImfPartType.h>
#include "ImfDeepFrameBuffer.h"
#include "ImfOutputStreamMutex.h"
#include "ImfOutputPartData.h"

#include "IlmThreadPool.h"
#include "IlmThreadSemaphore.h"
#include "IlmThreadMutex.h"
#include "Iex.h"
#include <string>
#include <vector>
#include <fstream>
#include <assert.h>
#include <algorithm>

#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using IMATH_NAMESPACE::Box2i;
using IMATH_NAMESPACE::divp;
using IMATH_NAMESPACE::modp;
using std::string;
using std::vector;
using std::ofstream;
using std::min;
using std::max;
using ILMTHREAD_NAMESPACE::Mutex;
using ILMTHREAD_NAMESPACE::Lock;
using ILMTHREAD_NAMESPACE::Semaphore;
using ILMTHREAD_NAMESPACE::Task;
using ILMTHREAD_NAMESPACE::TaskGroup;
using ILMTHREAD_NAMESPACE::ThreadPool;

namespace {


struct OutSliceInfo
{
    PixelType                    type;
    const char *                 base;
    ptrdiff_t                    sampleStride;
    ptrdiff_t                    xStride;
    ptrdiff_t                    yStride;
    int                          xSampling;
    int                          ySampling;
    bool                         zero;

    OutSliceInfo (PixelType type = HALF,
                  const char * base =NULL,
                  ptrdiff_t sampleStride = 0,
                  ptrdiff_t xStride = 0,
                  ptrdiff_t yStride =0,
                  int xSampling = 1,
                  int ySampling = 1,
                  bool zero = false);
};


OutSliceInfo::OutSliceInfo (PixelType t,
                            const char * base,
                            ptrdiff_t spstride,
                            ptrdiff_t xst,
                            ptrdiff_t yst,
                            int xsm, int ysm,
                            bool z)
:
    type (t),
    base (base),
    sampleStride (spstride),
    xStride(xst),
    yStride(yst),
    xSampling (xsm),
    ySampling (ysm),
    zero (z)
{
    // empty
}


struct LineBuffer
{
    Array< Array<char> >  buffer;
    Array<char>           consecutiveBuffer;
    const char *          dataPtr;
    Int64                 uncompressedDataSize;
    Int64                 dataSize;
    Array<char>           sampleCountTableBuffer;
    const char *          sampleCountTablePtr;
    Int64                 sampleCountTableSize;
    Compressor*           sampleCountTableCompressor;
    int                   minY;                 // the min y scanline stored
    int                   maxY;                 // the max y scanline stored
    int                   scanLineMin;          // the min y scanline writing out
    int                   scanLineMax;          // the max y scanline writing out
    Compressor *          compressor;
    bool                  partiallyFull;        // has incomplete data
    bool                  hasException;
    string                exception;

    LineBuffer (int linesInBuffer);
    ~LineBuffer ();

    void                  wait () {_sem.wait();}
    void                  post () {_sem.post();}

  private:

    Semaphore             _sem;
};


LineBuffer::LineBuffer (int linesInBuffer) :
    dataPtr (0),
    dataSize (0),
    sampleCountTablePtr (0),
    sampleCountTableCompressor (0),
    compressor (0),
    partiallyFull (false),
    hasException (false),
    exception (),
    _sem (1)
{
    buffer.resizeErase(linesInBuffer);
}


LineBuffer::~LineBuffer ()
{
    if (compressor != 0)
        delete compressor;

    if (sampleCountTableCompressor != 0)
        delete sampleCountTableCompressor;
}

} // namespace


struct DeepScanLineOutputFile::Data
{
    Header                      header;                // the image header
    int                         version;               // file format version
    bool                        multipart;             // from a multipart file
    Int64                       previewPosition;       // file position for preview
    DeepFrameBuffer             frameBuffer;           // framebuffer to write into
    int                         currentScanLine;       // next scanline to be written
    int                         missingScanLines;      // number of lines to write
    LineOrder                   lineOrder;             // the file's lineorder
    int                         minX;                  // data window's min x coord
    int                         maxX;                  // data window's max x coord
    int                         minY;                  // data window's min y coord
    int                         maxY;                  // data window's max x coord
    vector<Int64>               lineOffsets;           // stores offsets in file for
                                                       // each scanline
    vector<size_t>              bytesPerLine;          // combined size of a line over
                                                       // all channels
    Compressor::Format          format;                // compressor's data format
    vector<OutSliceInfo*>       slices;                // info about channels in file
    Int64                       lineOffsetsPosition;   // file position for line
                                                       // offset table

    vector<LineBuffer*>         lineBuffers;           // each holds one line buffer
    int                         linesInBuffer;         // number of scanlines each
                                                       // buffer holds
    int                         partNumber;            // the output part number

    char*                       sampleCountSliceBase;  // the pointer to the number
                                                       // of samples in each pixel
    int                         sampleCountXStride;    // the x stride for sampleCountSliceBase
    int                         sampleCountYStride;    // the y stride for sampleCountSliceBase

    Array<unsigned int>         lineSampleCount;       // the number of samples
                                                       // in each line

    Int64                       maxSampleCountTableSize;
                                                       // the max size in bytes for a pixel
                                                       // sample count table
    OutputStreamMutex*  _streamData;
    bool                _deleteStream;

    Data (int numThreads);
    ~Data ();


    inline LineBuffer *         getLineBuffer (int number);// hash function from line
                                                           // buffer indices into our
                                                           // vector of line buffers

    inline int&                 getSampleCount(int x, int y); // get the number of samples
                                                              // in each pixel
};


DeepScanLineOutputFile::Data::Data (int numThreads):
    lineOffsetsPosition (0),
    partNumber (-1) ,
    _streamData(NULL),
    _deleteStream(false)
{
    //
    // We need at least one lineBuffer, but if threading is used,
    // to keep n threads busy we need 2*n lineBuffers.
    //

    lineBuffers.resize (max (1, 2 * numThreads));
    for (size_t i = 0; i < lineBuffers.size(); i++)
        lineBuffers[i] = 0;
}


DeepScanLineOutputFile::Data::~Data ()
{
    for (size_t i = 0; i < lineBuffers.size(); i++)
        if (lineBuffers[i] != 0)
            delete lineBuffers[i];

    for (size_t i = 0; i < slices.size(); i++)
        delete slices[i];
}


int&
DeepScanLineOutputFile::Data::getSampleCount(int x, int y)
{
    return sampleCount(sampleCountSliceBase,
                       sampleCountXStride,
                       sampleCountYStride,
                       x, y);
}


LineBuffer*
DeepScanLineOutputFile::Data::getLineBuffer (int number)
{
    return lineBuffers[number % lineBuffers.size()];
}


namespace {

Int64
writeLineOffsets (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, const vector<Int64> &lineOffsets)
{
    Int64 pos = os.tellp();

    if (pos == -1)
        IEX_NAMESPACE::throwErrnoExc ("Cannot determine current file position (%T).");

    for (unsigned int i = 0; i < lineOffsets.size(); i++)
        OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (os, lineOffsets[i]);

    return pos;
}


void
writePixelData (OutputStreamMutex *filedata,
                DeepScanLineOutputFile::Data *partdata,
                int lineBufferMinY,
                const char pixelData[],
                Int64 packedDataSize,
                Int64 unpackedDataSize,
                const char sampleCountTableData[],
                Int64 sampleCountTableSize)
{
    //
    // Store a block of pixel data in the output file, and try
    // to keep track of the current writing position the file
    // without calling tellp() (tellp() can be fairly expensive).
    //

    Int64 currentPosition = filedata->currentPosition;
    filedata->currentPosition = 0;

    if (currentPosition == 0)
        currentPosition = filedata->os->tellp();

    partdata->lineOffsets[(partdata->currentScanLine - partdata->minY) / partdata->linesInBuffer] =
        currentPosition;

    #ifdef DEBUG

        assert (filedata->os->tellp() == currentPosition);

    #endif

    //
    // Write the optional part number.
    //

    if (partdata->multipart)
    {
        OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*filedata->os, partdata->partNumber);
    }

    //
    // Write the y coordinate of the first scanline in the chunk.
    //

    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*filedata->os, lineBufferMinY);

    //
    // Write the packed size of the pixel sample count table.
    //

    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*filedata->os, sampleCountTableSize);

    //
    // Write the packed pixel data size.
    //

    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*filedata->os, packedDataSize);

    //
    // Write the unpacked pixel data size.
    //

    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*filedata->os, unpackedDataSize);

    //
    // Write the packed pixel sample count table.
    //

    filedata->os->write (sampleCountTableData, sampleCountTableSize);

    //
    // Write the compressed data.
    //

    filedata->os->write (pixelData, packedDataSize);

    //
    // Update stream position.
    //

    filedata->currentPosition = currentPosition      +
                                Xdr::size<int>()     +  // y coordinate
                                Xdr::size<Int64>()   +  // packed sample count table size
                                Xdr::size<Int64>()   +  // packed data size
                                Xdr::size<Int64>()   +  // unpacked data size
                                sampleCountTableSize +  // pixel sample count table
                                packedDataSize;         // pixel data

    if (partdata->multipart)
    {
        filedata->currentPosition += Xdr::size<int>();  // optional part number
    }
}


inline void
writePixelData (OutputStreamMutex* filedata,
                DeepScanLineOutputFile::Data *partdata,
                const LineBuffer *lineBuffer)
{
    writePixelData (filedata, partdata,
                    lineBuffer->minY,
                    lineBuffer->dataPtr,
                    lineBuffer->dataSize,
                    lineBuffer->uncompressedDataSize,
                    lineBuffer->sampleCountTablePtr,
                    lineBuffer->sampleCountTableSize);
}


void
convertToXdr (DeepScanLineOutputFile::Data *ofd,
              Array<char> &lineBuffer,
              int lineBufferMinY,
              int lineBufferMaxY,
              int inSize)
{
    //
    // Convert the contents of a lineBuffer from the machine's native
    // representation to Xdr format.  This function is called by
    // CompressLineBuffer::execute(), below, if the compressor wanted
    // its input pixel data in the machine's native format, but then
    // failed to compress the data (most compressors will expand rather
    // than compress random input data).
    //
    // Note that this routine assumes that the machine's native
    // representation of the pixel data has the same size as the
    // Xdr representation.  This makes it possible to convert the
    // pixel data in place, without an intermediate temporary buffer.
    //

    //
    // Iterate over all scanlines in the lineBuffer to convert.
    //

    char* writePtr = &lineBuffer[0];
    for (int y = lineBufferMinY; y <= lineBufferMaxY; y++)
    {
        //
        // Set these to point to the start of line y.
        // We will write to writePtr from readPtr.
        //

        const char *readPtr = writePtr;

        //
        // Iterate over all slices in the file.
        //

        for (unsigned int i = 0; i < ofd->slices.size(); ++i)
        {
            //
            // Test if scan line y of this channel is
            // contains any data (the scan line contains
            // data only if y % ySampling == 0).
            //

            const OutSliceInfo &slice = *ofd->slices[i];

            if (modp (y, slice.ySampling) != 0)
                continue;

            //
            // Find the number of sampled pixels, dMaxX-dMinX+1, for
            // slice i in scan line y (i.e. pixels within the data window
            // for which x % xSampling == 0).
            //

            int xSampleCount = ofd->lineSampleCount[y - ofd->minY];

            //
            // Convert the samples in place.
            //

            convertInPlace (writePtr, readPtr, slice.type, xSampleCount);
        }
    }
}


//
// A LineBufferTask encapsulates the task of copying a set of scanlines
// from the user's frame buffer into a LineBuffer object, compressing
// the data if necessary.
//

class LineBufferTask: public Task
{
  public:

    LineBufferTask (TaskGroup *group,
                    DeepScanLineOutputFile::Data *ofd,
                    int number,
                    int scanLineMin,
                    int scanLineMax);

    virtual ~LineBufferTask ();

    virtual void        execute ();

  private:

    DeepScanLineOutputFile::Data *  _ofd;
    LineBuffer *        _lineBuffer;
};


LineBufferTask::LineBufferTask
    (TaskGroup *group,
     DeepScanLineOutputFile::Data *ofd,
     int number,
     int scanLineMin,
     int scanLineMax)
:
    Task (group),
    _ofd (ofd),
    _lineBuffer (_ofd->getLineBuffer(number))
{
    //
    // Wait for the lineBuffer to become available
    //

    _lineBuffer->wait ();

    //
    // Initialize the lineBuffer data if necessary
    //

    if (!_lineBuffer->partiallyFull)
    {

        _lineBuffer->minY = _ofd->minY + number * _ofd->linesInBuffer;

        _lineBuffer->maxY = min (_lineBuffer->minY + _ofd->linesInBuffer - 1,
                                 _ofd->maxY);

        _lineBuffer->partiallyFull = true;
    }

    _lineBuffer->scanLineMin = max (_lineBuffer->minY, scanLineMin);
    _lineBuffer->scanLineMax = min (_lineBuffer->maxY, scanLineMax);
}


LineBufferTask::~LineBufferTask ()
{
    //
    // Signal that the line buffer is now free
    //

    _lineBuffer->post ();
}


void
LineBufferTask::execute ()
{
    try
    {
        //
        // First copy the pixel data from the
        // frame buffer into the line buffer
        //

        int yStart, yStop, dy;

        if (_ofd->lineOrder == INCREASING_Y)
        {
            yStart = _lineBuffer->scanLineMin;
            yStop = _lineBuffer->scanLineMax + 1;
            dy = 1;
        }
        else
        {
            yStart = _lineBuffer->scanLineMax;
            yStop = _lineBuffer->scanLineMin - 1;
            dy = -1;
        }

        //
        // Allocate buffers for scanlines.
        // And calculate the sample counts for each line.
        //

        bytesPerDeepLineTable (_ofd->header,
                               _lineBuffer->scanLineMin,
                               _lineBuffer->scanLineMax,
                               _ofd->sampleCountSliceBase,
                               _ofd->sampleCountXStride,
                               _ofd->sampleCountYStride,
                               _ofd->bytesPerLine);
        for (int i = _lineBuffer->scanLineMin; i <= _lineBuffer->scanLineMax; i++)
        {
            // (TODO) don't do this all the time.
            _lineBuffer->buffer[i - _lineBuffer->minY].resizeErase(
                            _ofd->bytesPerLine[i - _ofd->minY]);

            for (int j = _ofd->minX; j <= _ofd->maxX; j++)
                _ofd->lineSampleCount[i - _ofd->minY] += _ofd->getSampleCount(j, i);
        }

        //
        // Copy data from frame buffer to line buffer.
        //

        int y;

        for (y = yStart; y != yStop; y += dy)
        {
            //
            // Gather one scan line's worth of pixel data and store
            // them in _ofd->lineBuffer.
            //

            char *writePtr = &_lineBuffer->buffer[y - _lineBuffer->minY][0];
            //
            // Iterate over all image channels.
            //

            for (unsigned int i = 0; i < _ofd->slices.size(); ++i)
            {
                //
                // Test if scan line y of this channel contains any data
                // (the scan line contains data only if y % ySampling == 0).
                //

                const OutSliceInfo &slice = *_ofd->slices[i];

                if (modp (y, slice.ySampling) != 0)
                    continue;

                //
                // Fill the line buffer with with pixel data.
                //

                if (slice.zero)
                {
                    //
                    // The frame buffer contains no data for this channel.
                    // Store zeroes in _lineBuffer->buffer.
                    //

                    fillChannelWithZeroes (writePtr, _ofd->format, slice.type,
                                           _ofd->lineSampleCount[y - _ofd->minY]);
                }
                else
                {

                    copyFromDeepFrameBuffer (writePtr, slice.base,
                                             _ofd->sampleCountSliceBase,
                                             _ofd->sampleCountXStride,
                                             _ofd->sampleCountYStride,
                                             y, _ofd->minX, _ofd->maxX,
                                             0, 0,//offsets for samplecount
                                             0, 0,//offsets for data
                                             slice.sampleStride, 
                                             slice.xStride,
                                             slice.yStride,
                                             _ofd->format,
                                             slice.type);
                }
            }
        }

        //
        // If the next scanline isn't past the bounds of the lineBuffer
        // then we have partially filled the linebuffer,
        // otherwise the whole linebuffer is filled and then
        // we compress the linebuffer and write it out.
        //

        if (y >= _lineBuffer->minY && y <= _lineBuffer->maxY)
            return;

        //
        // Copy all data into a consecutive buffer.
        //

        Int64 totalBytes = 0;
        Int64 maxBytesPerLine = 0;
        for (int i = 0; i < _lineBuffer->maxY - _lineBuffer->minY + 1; i++)
        {
            totalBytes += _lineBuffer->buffer[i].size();
            if (Int64(_lineBuffer->buffer[i].size()) > maxBytesPerLine)
                maxBytesPerLine = _lineBuffer->buffer[i].size();
        }
        _lineBuffer->consecutiveBuffer.resizeErase(totalBytes);

        int pos = 0;
        for (int i = 0; i < _lineBuffer->maxY - _lineBuffer->minY + 1; i++)
        {
            memcpy(_lineBuffer->consecutiveBuffer + pos,
                   &_lineBuffer->buffer[i][0],
                   _lineBuffer->buffer[i].size());
            pos += _lineBuffer->buffer[i].size();
        }

        _lineBuffer->dataPtr = _lineBuffer->consecutiveBuffer;

        _lineBuffer->dataSize = totalBytes;
        _lineBuffer->uncompressedDataSize = _lineBuffer->dataSize;

        //
        // Compress the pixel sample count table.
        //

        char* ptr = _lineBuffer->sampleCountTableBuffer;
        Int64 tableDataSize = 0;
        for (int i = _lineBuffer->minY; i <= _lineBuffer->maxY; i++)
        {
            int count = 0;
            for (int j = _ofd->minX; j <= _ofd->maxX; j++)
            {
                count += _ofd->getSampleCount(j, i);
                Xdr::write <CharPtrIO> (ptr, count);
                tableDataSize += sizeof (int);
            }
        }

       if(_lineBuffer->sampleCountTableCompressor)
       {
          _lineBuffer->sampleCountTableSize =
                  _lineBuffer->sampleCountTableCompressor->compress (
                                                      _lineBuffer->sampleCountTableBuffer,
                                                      tableDataSize,
                                                      _lineBuffer->minY,
                                                      _lineBuffer->sampleCountTablePtr);
       }

        //
        // If we can't make data shrink (or we weren't compressing), then just use the raw data.
        //

        if (!_lineBuffer->sampleCountTableCompressor || 
            _lineBuffer->sampleCountTableSize >= tableDataSize)
        {
            _lineBuffer->sampleCountTableSize = tableDataSize;
            _lineBuffer->sampleCountTablePtr = _lineBuffer->sampleCountTableBuffer;
        }

        //
        // Compress the sample data
        //

        // (TODO) don't do this all the time.
        if (_lineBuffer->compressor != 0)
            delete _lineBuffer->compressor;
        _lineBuffer->compressor = newCompressor (_ofd->header.compression(),
                                                 maxBytesPerLine,
                                                 _ofd->header);

        Compressor *compressor = _lineBuffer->compressor;

        if (compressor)
        {
            const char *compPtr;

            Int64 compSize = compressor->compress (_lineBuffer->dataPtr,
                                                 _lineBuffer->dataSize,
                                                 _lineBuffer->minY, compPtr);

            if (compSize < _lineBuffer->dataSize)
            {
                _lineBuffer->dataSize = compSize;
                _lineBuffer->dataPtr = compPtr;
            }
            else if (_ofd->format == Compressor::NATIVE)
            {
                //
                // The data did not shrink during compression, but
                // we cannot write to the file using the machine's
                // native format, so we need to convert the lineBuffer
                // to Xdr.
                //

                convertToXdr (_ofd, _lineBuffer->consecutiveBuffer, _lineBuffer->minY,
                              _lineBuffer->maxY, _lineBuffer->dataSize);
            }
        }

        _lineBuffer->partiallyFull = false;
    }
    catch (std::exception &e)
    {
        if (!_lineBuffer->hasException)
        {
            _lineBuffer->exception = e.what ();
            _lineBuffer->hasException = true;
        }
    }
    catch (...)
    {
        if (!_lineBuffer->hasException)
        {
            _lineBuffer->exception = "unrecognized exception";
            _lineBuffer->hasException = true;
        }
    }
}

} // namespace


DeepScanLineOutputFile::DeepScanLineOutputFile
    (const char fileName[],
     const Header &header,
     int numThreads)
:
    _data (new Data (numThreads))
{
    _data->_streamData=new OutputStreamMutex ();
    _data->_deleteStream=true;
    try
    {
        header.sanityCheck();
        _data->_streamData->os = new StdOFStream (fileName);
        initialize (header);
        _data->_streamData->currentPosition = _data->_streamData->os->tellp();

        // Write header and empty offset table to the file.
        writeMagicNumberAndVersionField(*_data->_streamData->os, _data->header);
        _data->previewPosition =
                _data->header.writeTo (*_data->_streamData->os);
        _data->lineOffsetsPosition =
                writeLineOffsets (*_data->_streamData->os, _data->lineOffsets);
        _data->multipart=false;// not multipart; only one header
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        delete _data->_streamData->os;
        delete _data->_streamData;
        delete _data;

        REPLACE_EXC (e, "Cannot open image file "
                     "\"" << fileName << "\". " << e.what());
        throw;
    }
    catch (...)
    {
        delete _data->_streamData->os;
        delete _data->_streamData;
        delete _data;
        throw;
    }
}


DeepScanLineOutputFile::DeepScanLineOutputFile
    (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os,
     const Header &header,
     int numThreads)
:
    _data (new Data (numThreads))
    
{
    _data->_streamData   = new OutputStreamMutex ();
    _data->_deleteStream = false;
    try
    {
        header.sanityCheck();
        _data->_streamData->os = &os;
        initialize (header);
        _data->_streamData->currentPosition = _data->_streamData->os->tellp();

        // Write header and empty offset table to the file.
        writeMagicNumberAndVersionField(*_data->_streamData->os, _data->header);
        _data->previewPosition =
                _data->header.writeTo (*_data->_streamData->os);
        _data->lineOffsetsPosition =
                writeLineOffsets (*_data->_streamData->os, _data->lineOffsets);
	_data->multipart=false;
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        delete _data->_streamData;
        delete _data;

        REPLACE_EXC (e, "Cannot open image file "
                     "\"" << os.fileName() << "\". " << e.what());
        throw;
    }
    catch (...)
    {
        delete _data->_streamData;
        delete _data;
        throw;
    }
}

DeepScanLineOutputFile::DeepScanLineOutputFile(const OutputPartData* part)
{
    try
    {
        if (part->header.type() != DEEPSCANLINE)
            throw IEX_NAMESPACE::ArgExc("Can't build a DeepScanLineOutputFile from a type-mismatched part.");

        _data = new Data (part->numThreads);
        _data->_streamData = part->mutex;
        _data->_deleteStream=false;
        initialize (part->header);
        _data->partNumber = part->partNumber;
        _data->lineOffsetsPosition = part->chunkOffsetTablePosition;
        _data->previewPosition = part->previewPosition;
	_data->multipart=part->multipart;
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        delete _data;

        REPLACE_EXC (e, "Cannot initialize output part "
                     "\"" << part->partNumber << "\". " << e.what());
        throw;
    }
    catch (...)
    {
        delete _data;
        throw;
    }
}

void
DeepScanLineOutputFile::initialize (const Header &header)
{
    _data->header = header;

    _data->header.setType(DEEPSCANLINE);
    
    const Box2i &dataWindow = header.dataWindow();

    _data->currentScanLine = (header.lineOrder() == INCREASING_Y)?
                                 dataWindow.min.y: dataWindow.max.y;

    _data->missingScanLines = dataWindow.max.y - dataWindow.min.y + 1;
    _data->lineOrder = header.lineOrder();
    _data->minX = dataWindow.min.x;
    _data->maxX = dataWindow.max.x;
    _data->minY = dataWindow.min.y;
    _data->maxY = dataWindow.max.y;

    _data->lineSampleCount.resizeErase(_data->maxY - _data->minY + 1);

    Compressor* compressor = newCompressor (_data->header.compression(),
                                            0,
                                            _data->header);
    _data->format = defaultFormat (compressor);
    _data->linesInBuffer = numLinesInBuffer (compressor);
    if (compressor != 0)
        delete compressor;

    int lineOffsetSize = (_data->maxY - _data->minY +
                          _data->linesInBuffer) / _data->linesInBuffer;


    _data->header.setChunkCount(lineOffsetSize);

    _data->lineOffsets.resize (lineOffsetSize);

    _data->bytesPerLine.resize (_data->maxY - _data->minY + 1);

    _data->maxSampleCountTableSize = min(_data->linesInBuffer, _data->maxY - _data->minY + 1) *
                                     (_data->maxX - _data->minX + 1) *
                                     sizeof(unsigned int);

    for (size_t i = 0; i < _data->lineBuffers.size(); ++i)
    {
        _data->lineBuffers[i] = new LineBuffer (_data->linesInBuffer);
        _data->lineBuffers[i]->sampleCountTableBuffer.resizeErase(_data->maxSampleCountTableSize);

        _data->lineBuffers[i]->sampleCountTableCompressor =
        newCompressor (_data->header.compression(),
                               _data->maxSampleCountTableSize,
                               _data->header);
    }
}


DeepScanLineOutputFile::~DeepScanLineOutputFile ()
{
    {
        Lock lock(*_data->_streamData);
        Int64 originalPosition = _data->_streamData->os->tellp();

        if (_data->lineOffsetsPosition > 0)
        {
            try
            {
                _data->_streamData->os->seekp (_data->lineOffsetsPosition);
                writeLineOffsets (*_data->_streamData->os, _data->lineOffsets);

                //
                // Restore the original position.
                //
                _data->_streamData->os->seekp (originalPosition);
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

    if (_data->_deleteStream)
        delete _data->_streamData->os;

    //
    // (TODO) we should have a way to tell if the stream data is owned by this file or
    // by a parent multipart file.
    //

    if (_data->partNumber == -1)
        delete _data->_streamData;

    delete _data;
}


const char *
DeepScanLineOutputFile::fileName () const
{
    return _data->_streamData->os->fileName();
}


const Header &
DeepScanLineOutputFile::header () const
{
    return _data->header;
}


void
DeepScanLineOutputFile::setFrameBuffer (const DeepFrameBuffer &frameBuffer)
{
    Lock lock (*_data->_streamData);

    //
    // Check if the new frame buffer descriptor
    // is compatible with the image file header.
    //

    const ChannelList &channels = _data->header.channels();

    for (ChannelList::ConstIterator i = channels.begin();
         i != channels.end();
         ++i)
    {
        DeepFrameBuffer::ConstIterator j = frameBuffer.find (i.name());

        if (j == frameBuffer.end())
            continue;

        if (i.channel().type != j.slice().type)
        {
            THROW (IEX_NAMESPACE::ArgExc, "Pixel type of \"" << i.name() << "\" channel "
                                "of output file \"" << fileName() << "\" is "
                                "not compatible with the frame buffer's "
                                "pixel type.");
        }

        if (i.channel().xSampling != j.slice().xSampling ||
            i.channel().ySampling != j.slice().ySampling)
        {
            THROW (IEX_NAMESPACE::ArgExc, "X and/or y subsampling factors "
                                "of \"" << i.name() << "\" channel "
                                "of output file \"" << fileName() << "\" are "
                                "not compatible with the frame buffer's "
                                "subsampling factors.");
        }
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
    }

    //
    // Initialize slice table for writePixels().
    // Pixel sample count slice is not presented in the header,
    // so it wouldn't be added here.
    // Store the pixel base pointer table.
    // (TODO) Support for different sampling rates?
    //

    vector<OutSliceInfo*> slices;

    for (ChannelList::ConstIterator i = channels.begin();
         i != channels.end();
         ++i)
    {
        DeepFrameBuffer::ConstIterator j = frameBuffer.find (i.name());

        if (j == frameBuffer.end())
        {
            //
            // Channel i is not present in the frame buffer.
            // In the file, channel i will contain only zeroes.
            //

            slices.push_back (new OutSliceInfo (i.channel().type,
                                                NULL,// base
                                                0,// sampleStride,
                                                0,// xStride
                                                0,// yStride
                                                i.channel().xSampling,
                                                i.channel().ySampling,
                                                true)); // zero
        }
        else
        {
            //
            // Channel i is present in the frame buffer.
            //

            slices.push_back (new OutSliceInfo (j.slice().type,
                                                j.slice().base,                      
                                                j.slice().sampleStride,
                                                j.slice().xStride,
                                                j.slice().yStride,
                                                j.slice().xSampling,
                                                j.slice().ySampling,
                                                false)); // zero

        }
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
DeepScanLineOutputFile::frameBuffer () const
{
    Lock lock (*_data->_streamData);
    return _data->frameBuffer;
}


void
DeepScanLineOutputFile::writePixels (int numScanLines)
{
    try
    {
        Lock lock (*_data->_streamData);

        if (_data->slices.size() == 0)
            throw IEX_NAMESPACE::ArgExc ("No frame buffer specified "
                               "as pixel data source.");

        //
        // Maintain two iterators:
        //     nextWriteBuffer: next linebuffer to be written to the file
        //     nextCompressBuffer: next linebuffer to compress
        //

        int first = (_data->currentScanLine - _data->minY) /
                         _data->linesInBuffer;

        int nextWriteBuffer = first;
        int nextCompressBuffer;
        int stop;
        int step;
        int scanLineMin;
        int scanLineMax;

        {
            //
            // Create a task group for all line buffer tasks. When the
            // taskgroup goes out of scope, the destructor waits until
            // all tasks are complete.
            //

            TaskGroup taskGroup;

            //
            // Determine the range of lineBuffers that intersect the scan
            // line range.  Then add the initial compression tasks to the
            // thread pool.  We always add in at least one task but the
            // individual task might not do anything if numScanLines == 0.
            //

            if (_data->lineOrder == INCREASING_Y)
            {
                int last = (_data->currentScanLine + (numScanLines - 1) -
                            _data->minY) / _data->linesInBuffer;

                scanLineMin = _data->currentScanLine;
                scanLineMax = _data->currentScanLine + numScanLines - 1;

                int numTasks = max (min ((int)_data->lineBuffers.size(),
                                         last - first + 1),
                                    1);

                for (int i = 0; i < numTasks; i++)
                {
                    ThreadPool::addGlobalTask
                        (new LineBufferTask (&taskGroup, _data, first + i,
                                             scanLineMin, scanLineMax));
                }

                nextCompressBuffer = first + numTasks;
                stop = last + 1;
                step = 1;
            }
            else
            {
                int last = (_data->currentScanLine - (numScanLines - 1) -
                            _data->minY) / _data->linesInBuffer;

                scanLineMax = _data->currentScanLine;
                scanLineMin = _data->currentScanLine - numScanLines + 1;

                int numTasks = max (min ((int)_data->lineBuffers.size(),
                                         first - last + 1),
                                    1);

                for (int i = 0; i < numTasks; i++)
                {
                    ThreadPool::addGlobalTask
                        (new LineBufferTask (&taskGroup, _data, first - i,
                                             scanLineMin, scanLineMax));
                }

                nextCompressBuffer = first - numTasks;
                stop = last - 1;
                step = -1;
            }

            while (true)
            {
                if (_data->missingScanLines <= 0)
                {
                    throw IEX_NAMESPACE::ArgExc ("Tried to write more scan lines "
                                       "than specified by the data window.");
                }

                //
                // Wait until the next line buffer is ready to be written
                //

                LineBuffer *writeBuffer =
                    _data->getLineBuffer (nextWriteBuffer);

                writeBuffer->wait();

                int numLines = writeBuffer->scanLineMax -
                               writeBuffer->scanLineMin + 1;

                _data->missingScanLines -= numLines;

                //
                // If the line buffer is only partially full, then it is
                // not complete and we cannot write it to disk yet.
                //

                if (writeBuffer->partiallyFull)
                {
                    _data->currentScanLine = _data->currentScanLine +
                                             step * numLines;
                    writeBuffer->post();

                    return;
                }

                //
                // Write the line buffer
                //

                writePixelData (_data->_streamData, _data, writeBuffer);
                nextWriteBuffer += step;

                _data->currentScanLine = _data->currentScanLine +
                                         step * numLines;

                #ifdef DEBUG

                    assert (_data->currentScanLine ==
                            ((_data->lineOrder == INCREASING_Y) ?
                             writeBuffer->scanLineMax + 1:
                             writeBuffer->scanLineMin - 1));

                #endif

                //
                // Release the lock on the line buffer
                //

                writeBuffer->post();

                //
                // If this was the last line buffer in the scanline range
                //

                if (nextWriteBuffer == stop)
                    break;

                //
                // If there are no more line buffers to compress,
                // then only continue to write out remaining lineBuffers
                //

                if (nextCompressBuffer == stop)
                    continue;

                //
                // Add nextCompressBuffer as a compression task
                //

                ThreadPool::addGlobalTask
                    (new LineBufferTask (&taskGroup, _data, nextCompressBuffer,
                                         scanLineMin, scanLineMax));

                //
                // Update the next line buffer we need to compress
                //

                nextCompressBuffer += step;
            }

            //
            // Finish all tasks
            //
        }

        //
        // Exeption handling:
        //
        // LineBufferTask::execute() may have encountered exceptions, but
        // those exceptions occurred in another thread, not in the thread
        // that is executing this call to OutputFile::writePixels().
        // LineBufferTask::execute() has caught all exceptions and stored
        // the exceptions' what() strings in the line buffers.
        // Now we check if any line buffer contains a stored exception; if
        // this is the case then we re-throw the exception in this thread.
        // (It is possible that multiple line buffers contain stored
        // exceptions.  We re-throw the first exception we find and
        // ignore all others.)
        //

        const string *exception = 0;

        for (size_t i = 0; i < _data->lineBuffers.size(); ++i)
        {
            LineBuffer *lineBuffer = _data->lineBuffers[i];

            if (lineBuffer->hasException && !exception)
                exception = &lineBuffer->exception;

            lineBuffer->hasException = false;
        }

        if (exception)
            throw IEX_NAMESPACE::IoExc (*exception);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        REPLACE_EXC (e, "Failed to write pixel data to image "
                     "file \"" << fileName() << "\". " << e.what());
        throw;
    }
}


int
DeepScanLineOutputFile::currentScanLine () const
{
    Lock lock (*_data->_streamData);
    return _data->currentScanLine;
}


void
DeepScanLineOutputFile::copyPixels (DeepScanLineInputPart &in)
{
    copyPixels(*in.file);
}

void
DeepScanLineOutputFile::copyPixels (DeepScanLineInputFile &in)
{

    Lock lock (*_data->_streamData);

    //
    // Check if this file's and and the InputFile's
    // headers are compatible.
    //

    const Header &hdr = _data->header;
    const Header &inHdr = in.header();

    if(!inHdr.hasType() || inHdr.type()!=DEEPSCANLINE)
    {
        THROW (IEX_NAMESPACE::ArgExc, "Cannot copy pixels from image "
                            "file \"" << in.fileName() << "\" to image "
                            "file \"" << fileName() << "\": the input needs to be a deep scanline image");
    }
    if (!(hdr.dataWindow() == inHdr.dataWindow()))
        THROW (IEX_NAMESPACE::ArgExc, "Cannot copy pixels from image "
                            "file \"" << in.fileName() << "\" to image "
                            "file \"" << fileName() << "\". "
                            "The files have different data windows.");

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
                            "file \"" << fileName() << "\" failed.  "
                            "The files have different channel lists.");

    //
    // Verify that no pixel data have been written to this file yet.
    //

    const Box2i &dataWindow = hdr.dataWindow();

    if (_data->missingScanLines != dataWindow.max.y - dataWindow.min.y + 1)
        THROW (IEX_NAMESPACE::LogicExc, "Quick pixel copy from image "
                              "file \"" << in.fileName() << "\" to image "
                              "file \"" << fileName() << "\" failed. "
                              "\"" << fileName() << "\" already contains "
                              "pixel data.");

    //
    // Copy the pixel data.
    //

    vector<char> data(4096);
    
    while (_data->missingScanLines > 0)
    {
        Int64 dataSize = (Int64) data.size();
        in.rawPixelData(_data->currentScanLine, &data[0], dataSize);
        if(dataSize > data.size())
        {
            // block wasn't big enough - go again with enough memory this time
            data.resize(dataSize);
            in.rawPixelData(_data->currentScanLine, &data[0], dataSize);
        }

        // extract header from block to pass to writePixelData
        
        Int64 packedSampleCountSize = *(Int64 *) (&data[4]);
        Int64 packedDataSize = *(Int64 *) (&data[12]);
        Int64 unpackedDataSize = *(Int64 *) (&data[20]);
        const char * sampleCountTable = &data[0]+28;
        const char * pixelData = sampleCountTable + packedSampleCountSize;
        
        
        writePixelData (_data->_streamData, _data, lineBufferMinY (_data->currentScanLine,
                                               _data->minY,
                                               _data->linesInBuffer),
                        pixelData, packedDataSize, unpackedDataSize,sampleCountTable,packedSampleCountSize);

        _data->currentScanLine += (_data->lineOrder == INCREASING_Y)?
                                   _data->linesInBuffer: -_data->linesInBuffer;

        _data->missingScanLines -= _data->linesInBuffer;
    }
}


void
DeepScanLineOutputFile::updatePreviewImage (const PreviewRgba newPixels[])
{
    Lock lock (*_data->_streamData);

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

    Int64 savedPosition = _data->_streamData->os->tellp();

    try
    {
        _data->_streamData->os->seekp (_data->previewPosition);
        pia.writeValueTo (*_data->_streamData->os, _data->version);
        _data->_streamData->os->seekp (savedPosition);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        REPLACE_EXC (e, "Cannot update preview image pixels for "
                     "file \"" << fileName() << "\". " << e.what());
        throw;
    }
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
