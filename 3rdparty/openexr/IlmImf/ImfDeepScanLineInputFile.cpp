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
//      class DeepScanLineInputFile
//
//-----------------------------------------------------------------------------

#include <ImfDeepScanLineInputFile.h>
#include <ImfChannelList.h>
#include <ImfMisc.h>
#include <ImfStdIO.h>
#include <ImfCompressor.h>
#include <ImfXdr.h>
#include <ImfConvert.h>
#include <ImfThreading.h>
#include <ImfPartType.h>
#include <ImfVersion.h>
#include "ImfMultiPartInputFile.h"
#include "ImfDeepFrameBuffer.h"
#include "ImfInputStreamMutex.h"
#include "ImfInputPartData.h"


#include "ImathBox.h"
#include "ImathFun.h"


#include "IlmThreadPool.h"
#include "IlmThreadSemaphore.h"
#include "IlmThreadMutex.h"

#include "Iex.h"

#include <string>
#include <vector>
#include <assert.h>
#include <limits>
#include <algorithm>


#include "ImfNamespace.h"
OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using IMATH_NAMESPACE::Box2i;
using IMATH_NAMESPACE::divp;
using IMATH_NAMESPACE::modp;
using std::string;
using std::vector;
using std::ifstream;
using std::min;
using std::max;
using ILMTHREAD_NAMESPACE::Mutex;
using ILMTHREAD_NAMESPACE::Lock;
using ILMTHREAD_NAMESPACE::Semaphore;
using ILMTHREAD_NAMESPACE::Task;
using ILMTHREAD_NAMESPACE::TaskGroup;
using ILMTHREAD_NAMESPACE::ThreadPool;

namespace {

struct InSliceInfo
{
    PixelType           typeInFrameBuffer;
    PixelType           typeInFile;
    char *              base;
    char*               pointerArrayBase;
    size_t              xPointerStride;
    size_t              yPointerStride;
    size_t              sampleStride;
    int                 xSampling;
    int                 ySampling;
    bool                fill;
    bool                skip;
    double              fillValue;

    InSliceInfo (PixelType typeInFrameBuffer = HALF,
                 char * base = NULL,
                 PixelType typeInFile = HALF,
                 size_t xPointerStride = 0,
                 size_t yPointerStride = 0,
                 size_t sampleStride = 0,
                 int xSampling = 1,
                 int ySampling = 1,
                 bool fill = false,
                 bool skip = false,
                 double fillValue = 0.0);
};


InSliceInfo::InSliceInfo (PixelType tifb,
                          char * b,
                          PixelType tifl,
                          size_t xpst,
                          size_t ypst,
                          size_t spst,
                          int xsm, int ysm,
                          bool f, bool s,
                          double fv)
:
    typeInFrameBuffer (tifb),
    typeInFile (tifl),
    base(b),
    xPointerStride (xpst),
    yPointerStride (ypst),
    sampleStride (spst),
    xSampling (xsm),
    ySampling (ysm),
    fill (f),
    skip (s),
    fillValue (fv)
{
    // empty
}


struct LineBuffer
{
    const char *        uncompressedData;
    char *              buffer;
    Int64               packedDataSize;
    Int64               unpackedDataSize;

    int                 minY;
    int                 maxY;
    Compressor *        compressor;
    Compressor::Format  format;
    int                 number;
    bool                hasException;
    string              exception;

    LineBuffer ();
    ~LineBuffer ();

    inline void         wait () {_sem.wait();}
    inline void         post () {_sem.post();}

  private:

    Semaphore           _sem;
};


LineBuffer::LineBuffer ():
    uncompressedData (0),
    buffer (0),
    packedDataSize (0),
    compressor (0),
    format (defaultFormat(compressor)),
    number (-1),
    hasException (false),
    exception (),
    _sem (1)
{
    // empty
}


LineBuffer::~LineBuffer ()
{
    if (compressor != 0)
        delete compressor;
}

} // namespace


struct DeepScanLineInputFile::Data: public Mutex
{
    Header                      header;             // the image header
    int                         version;            // file's version
    DeepFrameBuffer             frameBuffer;        // framebuffer to write into
    LineOrder                   lineOrder;          // order of the scanlines in file
    int                         minX;               // data window's min x coord
    int                         maxX;               // data window's max x coord
    int                         minY;               // data window's min y coord
    int                         maxY;               // data window's max x coord
    vector<Int64>               lineOffsets;        // stores offsets in file for
                                                    // each line
    bool                        fileIsComplete;     // True if no scanlines are missing
                                                    // in the file
    int                         nextLineBufferMinY; // minimum y of the next linebuffer
    vector<size_t>              bytesPerLine;       // combined size of a line over all
                                                    // channels
    vector<size_t>              offsetInLineBuffer; // offset for each scanline in its
                                                    // linebuffer
    vector<InSliceInfo*>        slices;             // info about channels in file

    vector<LineBuffer*>         lineBuffers;        // each holds one line buffer
    int                         linesInBuffer;      // number of scanlines each buffer
                                                    // holds
    int                         partNumber;         // part number
    int                         numThreads;         // number of threads
    
    bool                        multiPartBackwardSupport;       // if we are reading a multipart file using single file API
    MultiPartInputFile*         multiPartFile;      // for multipart files opened as single part
    bool                        memoryMapped;       // if the stream is memory mapped

    Array2D<unsigned int>       sampleCount;        // the number of samples
                                                    // in each pixel

    Array<unsigned int>         lineSampleCount;    // the number of samples
                                                    // in each line

    Array<bool>                 gotSampleCount;     // for each scanline, indicating if
                                                    // we have got its sample count table

    char*                       sampleCountSliceBase; // pointer to the start of
                                                      // the sample count array
    int                         sampleCountXStride; // x stride of the sample count array
    int                         sampleCountYStride; // y stride of the sample count array
    bool                        frameBufferValid;   // set by setFrameBuffer: excepts if readPixelSampleCounts if false

    Array<char>                 sampleCountTableBuffer;
                                                    // the buffer for sample count table

    Compressor*                 sampleCountTableComp;
                                                    // the decompressor for sample count table

    int                         combinedSampleSize; // total size of all channels combined: used to sanity check sample table size

    int                         maxSampleCountTableSize;
                                                    // the max size in bytes for a pixel
                                                    // sample count table
    InputStreamMutex*   _streamData;
    bool                _deleteStream;
                                                    

    Data (int numThreads);
    ~Data ();

    inline LineBuffer * getLineBuffer (int number); // hash function from line
                                                    // buffer indices into our
                                                    // vector of line buffers
};


DeepScanLineInputFile::Data::Data (int numThreads):
        partNumber(-1),
        numThreads(numThreads),
        multiPartBackwardSupport(false),
        multiPartFile(NULL),
        memoryMapped(false),
        frameBufferValid(false),
        _streamData(NULL),
        _deleteStream(false)
{
    //
    // We need at least one lineBuffer, but if threading is used,
    // to keep n threads busy we need 2*n lineBuffers
    //

    lineBuffers.resize (max (1, 2 * numThreads));

    for (size_t i = 0; i < lineBuffers.size(); i++)
        lineBuffers[i] = 0;

    sampleCountTableComp = 0;
}


DeepScanLineInputFile::Data::~Data ()
{
    for (size_t i = 0; i < lineBuffers.size(); i++)
        if (lineBuffers[i] != 0)
            delete lineBuffers[i];

    for (size_t i = 0; i < slices.size(); i++)
        delete slices[i];

    if (sampleCountTableComp != 0)
        delete sampleCountTableComp;
    
    if (multiPartBackwardSupport)
        delete multiPartFile;
}


inline LineBuffer *
DeepScanLineInputFile::Data::getLineBuffer (int lineBufferNumber)
{
    return lineBuffers[lineBufferNumber % lineBuffers.size()];
}


namespace {


void
reconstructLineOffsets (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,
                        LineOrder lineOrder,
                        vector<Int64> &lineOffsets)
{
    Int64 position = is.tellg();

    try
    {
        for (unsigned int i = 0; i < lineOffsets.size(); i++)
        {
            Int64 lineOffset = is.tellg();

            int y;
            OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, y);
            
            Int64 packed_offset;
            Int64 packed_sample;
            OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, packed_offset);
            OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, packed_sample);
            //next is unpacked sample table size - skip this too
            Xdr::skip <StreamIO> (is, packed_offset+packed_sample+8);

            if (lineOrder == INCREASING_Y)
                lineOffsets[i] = lineOffset;
            else
                lineOffsets[lineOffsets.size() - i - 1] = lineOffset;
        }
    }
    catch (...)
    {
        //
        // Suppress all exceptions.  This functions is
        // called only to reconstruct the line offset
        // table for incomplete files, and exceptions
        // are likely.
        //
    }

    is.clear();
    is.seekg (position);
}


void
readLineOffsets (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,
                 LineOrder lineOrder,
                 vector<Int64> &lineOffsets,
                 bool &complete)
{
    for (unsigned int i = 0; i < lineOffsets.size(); i++)
    {
        OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, lineOffsets[i]);
    }

    complete = true;

    for (unsigned int i = 0; i < lineOffsets.size(); i++)
    {
        if (lineOffsets[i] <= 0)
        {
            //
            // Invalid data in the line offset table mean that
            // the file is probably incomplete (the table is
            // the last thing written to the file).  Either
            // some process is still busy writing the file,
            // or writing the file was aborted.
            //
            // We should still be able to read the existing
            // parts of the file.  In order to do this, we
            // have to make a sequential scan over the scan
            // line data to reconstruct the line offset table.
            //

            complete = false;
            reconstructLineOffsets (is, lineOrder, lineOffsets);
            break;
        }
    }
}


void
readPixelData (InputStreamMutex *streamData,
               DeepScanLineInputFile::Data *ifd,
               int minY,
               char *&buffer,
               Int64 &packedDataSize,
               Int64 &unpackedDataSize)
{
    //
    // Read a single line buffer from the input file.
    //
    // If the input file is not memory-mapped, we copy the pixel data into
    // into the array pointed to by buffer.  If the file is memory-mapped,
    // then we change where buffer points to instead of writing into the
    // array (hence buffer needs to be a reference to a char *).
    //

    int lineBufferNumber = (minY - ifd->minY) / ifd->linesInBuffer;

    Int64 lineOffset = ifd->lineOffsets[lineBufferNumber];

    if (lineOffset == 0)
        THROW (IEX_NAMESPACE::InputExc, "Scan line " << minY << " is missing.");

    //
    // Seek to the start of the scan line in the file,
    // if necessary.
    //

    if (!isMultiPart(ifd->version))
    {
        if (ifd->nextLineBufferMinY != minY)
            streamData->is->seekg (lineOffset);
    }
    else
    {
        //
        // In a multi-part file, the file pointer may have been moved by
        // other parts, so we have to ask tellg() where we are.
        //
        if (streamData->is->tellg() != ifd->lineOffsets[lineBufferNumber])
            streamData->is->seekg (lineOffset);
    }

    //
    // Read the data block's header.
    //

    int yInFile;

    //
    // Read the part number when we are dealing with a multi-part file.
    //

    if (isMultiPart(ifd->version))
    {
        int partNumber;
        OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, partNumber);
        if (partNumber != ifd->partNumber)
        {
            THROW (IEX_NAMESPACE::ArgExc, "Unexpected part number " << partNumber
                   << ", should be " << ifd->partNumber << ".");
        }
    }

    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, yInFile);

    if (yInFile != minY)
        throw IEX_NAMESPACE::InputExc ("Unexpected data block y coordinate.");

    Int64 sampleCountTableSize;
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, sampleCountTableSize);
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, packedDataSize);
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, unpackedDataSize);


    //
    // We make a check on the data size requirements here.
    // Whilst we wish to store 64bit sizes on disk, not all the compressors
    // have been made to work with such data sizes and are still limited to
    // using signed 32 bit (int) for the data size. As such, this version
    // insists that we validate that the data size does not exceed the data
    // type max limit.
    // @TODO refactor the compressor code to ensure full 64-bit support.
    //

    int compressorMaxDataSize = std::numeric_limits<int>::max();
    if (packedDataSize   > Int64(compressorMaxDataSize) ||
        unpackedDataSize > Int64(compressorMaxDataSize))
    {
        THROW (IEX_NAMESPACE::ArgExc, "This version of the library does not support "
              << "the allocation of data with size  > " << compressorMaxDataSize
              << " file unpacked size :" << unpackedDataSize
              << " file packed size   :" << packedDataSize << ".\n");
    }

    //
    // Skip the pixel sample count table because we have read this data.
    //

    Xdr::skip <StreamIO> (*streamData->is, sampleCountTableSize);

    //
    // Read the pixel data.
    //

    if (streamData->is->isMemoryMapped ())
        buffer = streamData->is->readMemoryMapped (packedDataSize);
    else
    {
        // (TODO) check if the packed data size is too big?
        // (TODO) better memory management. Don't delete buffer all the time.
        if (buffer != 0) delete[] buffer;
        buffer = new char[packedDataSize];
        streamData->is->read (buffer, packedDataSize);
    }

    //
    // Keep track of which scan line is the next one in
    // the file, so that we can avoid redundant seekg()
    // operations (seekg() can be fairly expensive).
    //

    if (ifd->lineOrder == INCREASING_Y)
        ifd->nextLineBufferMinY = minY + ifd->linesInBuffer;
    else
        ifd->nextLineBufferMinY = minY - ifd->linesInBuffer;
}

//
// A LineBufferTask encapsulates the task uncompressing a set of
// scanlines (line buffer) and copying them into the frame buffer.
//

class LineBufferTask : public Task
{
  public:

    LineBufferTask (TaskGroup *group,
                    DeepScanLineInputFile::Data *ifd,
                    LineBuffer *lineBuffer,
                    int scanLineMin,
                    int scanLineMax);

    virtual ~LineBufferTask ();

    virtual void                execute ();

  private:

    DeepScanLineInputFile::Data *   _ifd;
    LineBuffer *                _lineBuffer;
    int                         _scanLineMin;
    int                         _scanLineMax;
};


LineBufferTask::LineBufferTask
    (TaskGroup *group,
     DeepScanLineInputFile::Data *ifd,
     LineBuffer *lineBuffer,
     int scanLineMin,
     int scanLineMax)
:
    Task (group),
    _ifd (ifd),
    _lineBuffer (lineBuffer),
    _scanLineMin (scanLineMin),
    _scanLineMax (scanLineMax)
{
    // empty
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
        // Uncompress the data, if necessary
        //

        if (_lineBuffer->uncompressedData == 0)
        {
            Int64 uncompressedSize = 0;
            int maxY = min (_lineBuffer->maxY, _ifd->maxY);

            for (int i = _lineBuffer->minY - _ifd->minY;
                 i <= maxY - _ifd->minY;
                 ++i)
            {
                uncompressedSize += (int) _ifd->bytesPerLine[i];
            }

            //
            // Create the compressor everytime when we want to use it,
            // because we don't know maxBytesPerLine beforehand.
            // (TODO) optimize this. don't do this every time.
            //

            if (_lineBuffer->compressor != 0)
                delete _lineBuffer->compressor;
            Int64 maxBytesPerLine = 0;
            for (int i = _lineBuffer->minY - _ifd->minY;
                 i <= maxY - _ifd->minY;
                 ++i)
            {
                if (_ifd->bytesPerLine[i] > maxBytesPerLine)
                    maxBytesPerLine = _ifd->bytesPerLine[i];
            }
            _lineBuffer->compressor = newCompressor(_ifd->header.compression(),
                                                    maxBytesPerLine,
                                                    _ifd->header);

            if (_lineBuffer->compressor &&
                _lineBuffer->packedDataSize < uncompressedSize)
            {
                _lineBuffer->format = _lineBuffer->compressor->format();

                _lineBuffer->packedDataSize = _lineBuffer->compressor->uncompress
                    (_lineBuffer->buffer, _lineBuffer->packedDataSize,
                     _lineBuffer->minY, _lineBuffer->uncompressedData);
            }
            else
            {
                //
                // If the line is uncompressed, it's in XDR format,
                // regardless of the compressor's output format.
                //

                _lineBuffer->format = Compressor::XDR;
                _lineBuffer->uncompressedData = _lineBuffer->buffer;
            }
        }

        int yStart, yStop, dy;

        if (_ifd->lineOrder == INCREASING_Y)
        {
            yStart = _scanLineMin;
            yStop = _scanLineMax + 1;
            dy = 1;
        }
        else
        {
            yStart = _scanLineMax;
            yStop = _scanLineMin - 1;
            dy = -1;
        }

        for (int y = yStart; y != yStop; y += dy)
        {
            //
            // Convert one scan line's worth of pixel data back
            // from the machine-independent representation, and
            // store the result in the frame buffer.
            //

            const char *readPtr = _lineBuffer->uncompressedData +
                                  _ifd->offsetInLineBuffer[y - _ifd->minY];

            //
            // Iterate over all image channels.
            //

            for (unsigned int i = 0; i < _ifd->slices.size(); ++i)
            {
                //
                // Test if scan line y of this channel contains any data
                // (the scan line contains data only if y % ySampling == 0).
                //

                InSliceInfo &slice = *_ifd->slices[i];

                if (modp (y, slice.ySampling) != 0)
                    continue;

                //
                // Find the x coordinates of the leftmost and rightmost
                // sampled pixels (i.e. pixels within the data window
                // for which x % xSampling == 0).
                //

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
                                 _ifd->lineSampleCount[y - _ifd->minY]);
                }
                else
                {
                    //
                    // The frame buffer contains a slice for this channel.
                    //

                    int width = (_ifd->maxX - _ifd->minX + 1);

                    copyIntoDeepFrameBuffer (readPtr, slice.base,
                                             (char*) (&_ifd->sampleCount[0][0]
                                                      - _ifd->minX
                                                      - _ifd->minY * width),
                                             sizeof(unsigned int) * 1,
                                             sizeof(unsigned int) * width,
                                             y, _ifd->minX, _ifd->maxX,
                                             0, 0,
                                             0, 0,
                                             slice.sampleStride, 
                                             slice.xPointerStride,
                                             slice.yPointerStride,
                                             slice.fill,
                                             slice.fillValue, _lineBuffer->format,
                                             slice.typeInFrameBuffer,
                                             slice.typeInFile);
                }
            }
        }
    }
    catch (std::exception &e)
    {
        if (!_lineBuffer->hasException)
        {
            _lineBuffer->exception = e.what();
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


LineBufferTask *
newLineBufferTask
    (TaskGroup *group,
     DeepScanLineInputFile::Data *ifd,
     int number,
     int scanLineMin,
     int scanLineMax)
{
    //
    // Wait for a line buffer to become available, fill the line
    // buffer with raw data from the file if necessary, and create
    // a new LineBufferTask whose execute() method will uncompress
    // the contents of the buffer and copy the pixels into the
    // frame buffer.
    //

    LineBuffer *lineBuffer = ifd->getLineBuffer (number);

    try
    {
        lineBuffer->wait ();

        if (lineBuffer->number != number)
        {
            lineBuffer->minY = ifd->minY + number * ifd->linesInBuffer;
            lineBuffer->maxY = lineBuffer->minY + ifd->linesInBuffer - 1;

            lineBuffer->number = number;
            lineBuffer->uncompressedData = 0;

            readPixelData (ifd->_streamData, ifd, lineBuffer->minY,
                           lineBuffer->buffer,
                           lineBuffer->packedDataSize,
                           lineBuffer->unpackedDataSize);
        }
    }
    catch (std::exception &e)
    {
        if (!lineBuffer->hasException)
        {
            lineBuffer->exception = e.what();
            lineBuffer->hasException = true;
        }
        lineBuffer->number = -1;
        lineBuffer->post();
        throw;
    }
    catch (...)
    {
        //
        // Reading from the file caused an exception.
        // Signal that the line buffer is free, and
        // re-throw the exception.
        //

        lineBuffer->exception = "unrecognized exception";
        lineBuffer->hasException = true;
        lineBuffer->number = -1;
        lineBuffer->post();
        throw;
    }

    scanLineMin = max (lineBuffer->minY, scanLineMin);
    scanLineMax = min (lineBuffer->maxY, scanLineMax);

    return new LineBufferTask (group, ifd, lineBuffer,
                               scanLineMin, scanLineMax);
}

} // namespace


void DeepScanLineInputFile::initialize(const Header& header)
{
    try
    {
        if (header.type() != DEEPSCANLINE)
            throw IEX_NAMESPACE::ArgExc("Can't build a DeepScanLineInputFile from "
            "a type-mismatched part.");
        
        if(header.version()!=1)
        {
            THROW(IEX_NAMESPACE::ArgExc, "Version " << header.version() << " not supported for deepscanline images in this version of the library");
        }
        
        _data->header = header;

        _data->lineOrder = _data->header.lineOrder();

        const Box2i &dataWindow = _data->header.dataWindow();

        _data->minX = dataWindow.min.x;
        _data->maxX = dataWindow.max.x;
        _data->minY = dataWindow.min.y;
        _data->maxY = dataWindow.max.y;

        _data->sampleCount.resizeErase(_data->maxY - _data->minY + 1,
                                       _data->maxX - _data->minX + 1);
        _data->lineSampleCount.resizeErase(_data->maxY - _data->minY + 1);

        Compressor* compressor = newCompressor(_data->header.compression(),
                                               0,
                                               _data->header);

        _data->linesInBuffer = numLinesInBuffer (compressor);

        delete compressor;

        _data->nextLineBufferMinY = _data->minY - 1;

        int lineOffsetSize = (dataWindow.max.y - dataWindow.min.y +
                              _data->linesInBuffer) / _data->linesInBuffer;

        _data->lineOffsets.resize (lineOffsetSize);

        for (size_t i = 0; i < _data->lineBuffers.size(); i++)
            _data->lineBuffers[i] = new LineBuffer ();

        _data->gotSampleCount.resizeErase(_data->maxY - _data->minY + 1);
        for (int i = 0; i < _data->maxY - _data->minY + 1; i++)
            _data->gotSampleCount[i] = false;

        _data->maxSampleCountTableSize = min(_data->linesInBuffer, _data->maxY - _data->minY + 1) *
                                        (_data->maxX - _data->minX + 1) *
                                        sizeof(unsigned int);

        _data->sampleCountTableBuffer.resizeErase(_data->maxSampleCountTableSize);

        _data->sampleCountTableComp = newCompressor(_data->header.compression(),
                                                    _data->maxSampleCountTableSize,
                                                    _data->header);

        _data->bytesPerLine.resize (_data->maxY - _data->minY + 1);
        
        const ChannelList & c=header.channels();
        
        _data->combinedSampleSize=0;
        for(ChannelList::ConstIterator i=c.begin();i!=c.end();i++)
        {
            switch(i.channel().type)
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
    catch (...)
    {
        delete _data;
        _data=NULL;
        throw;
    }
}


DeepScanLineInputFile::DeepScanLineInputFile(InputPartData* part)
    
{

    _data = new Data(part->numThreads);
    _data->_deleteStream=false;
    _data->_streamData = part->mutex;
    _data->memoryMapped = _data->_streamData->is->isMemoryMapped();
    _data->version = part->version;

    initialize(part->header);

    _data->lineOffsets = part->chunkOffsets;

    _data->partNumber = part->partNumber;
}


DeepScanLineInputFile::DeepScanLineInputFile
    (const char fileName[], int numThreads)
:
     _data (new Data (numThreads))
{
    _data->_streamData = new InputStreamMutex();
    _data->_deleteStream = true;
    OPENEXR_IMF_INTERNAL_NAMESPACE::IStream* is = 0;

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
        _data->_streamData->is = is;
        _data->memoryMapped = is->isMemoryMapped();
        _data->header.readFrom (*_data->_streamData->is, _data->version);
        _data->header.sanityCheck (isTiled (_data->version));

        initialize(_data->header);

        readLineOffsets (*_data->_streamData->is,
                         _data->lineOrder,
                         _data->lineOffsets,
                         _data->fileIsComplete);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        if (is)          delete is;
        if (_data && _data->_streamData) delete _data->_streamData;
        if (_data)       delete _data;

        REPLACE_EXC (e, "Cannot read image file "
                     "\"" << fileName << "\". " << e.what());
        throw;
    }
    catch (...)
    {
        if (is)          delete is;
        if (_data && _data->_streamData) delete _data->_streamData;
        if (_data)       delete _data;

        throw;
    }
}


DeepScanLineInputFile::DeepScanLineInputFile
    (const Header &header,
     OPENEXR_IMF_INTERNAL_NAMESPACE::IStream *is,
     int version,
     int numThreads)
:
    _data (new Data (numThreads))
{
    _data->_streamData=new InputStreamMutex();
    _data->_deleteStream=false;
    _data->_streamData->is = is;
    
    _data->memoryMapped = is->isMemoryMapped();

    _data->version =version;
    
    initialize (header);

    readLineOffsets (*_data->_streamData->is,
                     _data->lineOrder,
                     _data->lineOffsets,
                     _data->fileIsComplete);
}


DeepScanLineInputFile::~DeepScanLineInputFile ()
{
    if (_data->_deleteStream)
        delete _data->_streamData->is;

    if (_data)
    {
        if (!_data->memoryMapped)
            for (size_t i = 0; i < _data->lineBuffers.size(); i++)
                delete [] _data->lineBuffers[i]->buffer;

        //
        // Unless this file was opened via the multipart API, delete the streamdata
        // object too.
        // (TODO) it should be "isMultiPart(data->version)", but when there is only
        // single part,
        // (see the above constructor) the version field is not set.
        //
        // (TODO) we should have a way to tell if the stream data is owned by this
        // file or by a parent multipart file.
        //

        if (_data->partNumber == -1 && _data->_streamData)
            delete _data->_streamData;

        delete _data;
    }
}

void
DeepScanLineInputFile::compatibilityInitialize(OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is)
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

void DeepScanLineInputFile::multiPartInitialize(InputPartData* part)
{
    
    _data->_streamData = part->mutex;
    _data->memoryMapped = _data->_streamData->is->isMemoryMapped();
    _data->version = part->version;
    
    initialize(part->header);
    
    _data->lineOffsets = part->chunkOffsets;
    
    _data->partNumber = part->partNumber;
    
}


const char *
DeepScanLineInputFile::fileName () const
{
    return _data->_streamData->is->fileName();
}


const Header &
DeepScanLineInputFile::header () const
{
    return _data->header;
}


int
DeepScanLineInputFile::version () const
{
    return _data->version;
}


void
DeepScanLineInputFile::setFrameBuffer (const DeepFrameBuffer &frameBuffer)
{
    Lock lock (*_data->_streamData);

    
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
    }

    //
    // Initialize the slice table for readPixels().
    //

    vector<InSliceInfo*> slices;
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

            slices.push_back (new InSliceInfo (i.channel().type,
                                               NULL,
                                               i.channel().type,
                                               0,
                                               0,
                                               0, // sampleStride
                                               i.channel().xSampling,
                                               i.channel().ySampling,
                                               false,  // fill
                                               true, // skip
                                               0.0)); // fillValue
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

        slices.push_back (new InSliceInfo (j.slice().type,
                                           j.slice().base,
                                           fill? j.slice().type:
                                                 i.channel().type,
                                           j.slice().xStride,
                                           j.slice().yStride,
                                           j.slice().sampleStride,
                                           j.slice().xSampling,
                                           j.slice().ySampling,
                                           fill,
                                           false, // skip
                                           j.slice().fillValue));


        if (i != channels.end() && !fill)
            ++i;
    }

    //
    // Client may want data to be filled in multiple arrays,
    // so we reset gotSampleCount and bytesPerLine.
    //

    for (long i = 0; i < _data->gotSampleCount.size(); i++)
        _data->gotSampleCount[i] = false;
    for (size_t i = 0; i < _data->bytesPerLine.size(); i++)
        _data->bytesPerLine[i] = 0;

    //
    // Store the new frame buffer.
    //

    _data->frameBuffer = frameBuffer;

    for (size_t i = 0; i < _data->slices.size(); i++)
        delete _data->slices[i];
    _data->slices = slices;
    _data->frameBufferValid = true;
}


const DeepFrameBuffer &
DeepScanLineInputFile::frameBuffer () const
{
    Lock lock (*_data->_streamData);
    return _data->frameBuffer;
}


bool
DeepScanLineInputFile::isComplete () const
{
    return _data->fileIsComplete;
}


void
DeepScanLineInputFile::readPixels (int scanLine1, int scanLine2)
{
    try
    {
        Lock lock (*_data->_streamData);

        if (_data->slices.size() == 0)
            throw IEX_NAMESPACE::ArgExc ("No frame buffer specified "
                               "as pixel data destination.");

        int scanLineMin = min (scanLine1, scanLine2);
        int scanLineMax = max (scanLine1, scanLine2);

        if (scanLineMin < _data->minY || scanLineMax > _data->maxY)
            throw IEX_NAMESPACE::ArgExc ("Tried to read scan line outside "
                               "the image file's data window.");

        for (int i = scanLineMin; i <= scanLineMax; i++)
        {
            if (_data->gotSampleCount[i - _data->minY] == false)
                throw IEX_NAMESPACE::ArgExc ("Tried to read scan line without "
                                   "knowing the sample counts, please"
                                   "read the sample counts first.");
        }

 
        //
        // We impose a numbering scheme on the lineBuffers where the first
        // scanline is contained in lineBuffer 1.
        //
        // Determine the first and last lineBuffer numbers in this scanline
        // range. We always attempt to read the scanlines in the order that
        // they are stored in the file.
        //

        int start, stop, dl;

        if (_data->lineOrder == INCREASING_Y)
        {
            start = (scanLineMin - _data->minY) / _data->linesInBuffer;
            stop  = (scanLineMax - _data->minY) / _data->linesInBuffer + 1;
            dl = 1;
        }
        else
        {
            start = (scanLineMax - _data->minY) / _data->linesInBuffer;
            stop  = (scanLineMin - _data->minY) / _data->linesInBuffer - 1;
            dl = -1;
        }

        //
        // Create a task group for all line buffer tasks.  When the
        // task group goes out of scope, the destructor waits until
        // all tasks are complete.
        //

        {
            TaskGroup taskGroup;

            //
            // Add the line buffer tasks.
            //
            // The tasks will execute in the order that they are created
            // because we lock the line buffers during construction and the
            // constructors are called by the main thread.  Hence, in order
            // for a successive task to execute the previous task which
            // used that line buffer must have completed already.
            //

            for (int l = start; l != stop; l += dl)
            {
                ThreadPool::addGlobalTask (newLineBufferTask (&taskGroup,
                                                              _data, l,
                                                              scanLineMin,
                                                              scanLineMax));
            }

            //
            // finish all tasks
            //
        }

        //
        // Exeption handling:
        //
        // LineBufferTask::execute() may have encountered exceptions, but
        // those exceptions occurred in another thread, not in the thread
        // that is executing this call to ScanLineInputFile::readPixels().
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
        REPLACE_EXC (e, "Error reading pixel data from image "
                     "file \"" << fileName() << "\". " << e.what());
        throw;
    }
}


void
DeepScanLineInputFile::readPixels (int scanLine)
{
    readPixels (scanLine, scanLine);
}


void
DeepScanLineInputFile::rawPixelData (int firstScanLine,
                                     char *pixelData,
                                     Int64 &pixelDataSize)
{
   
    
    int minY = lineBufferMinY
    (firstScanLine, _data->minY, _data->linesInBuffer);
    int lineBufferNumber = (minY - _data->minY) / _data->linesInBuffer;
    
    Int64 lineOffset = _data->lineOffsets[lineBufferNumber];
    
    if (lineOffset == 0)
        THROW (IEX_NAMESPACE::InputExc, "Scan line " << minY << " is missing.");
    
    
    // enter the lock here - prevent another thread reseeking the file during read
    Lock lock (*_data->_streamData);
    
    //
    // Seek to the start of the scan line in the file,
    //
    
    if (_data->_streamData->is->tellg() != _data->lineOffsets[lineBufferNumber])
        _data->_streamData->is->seekg (lineOffset);
    
    //
    // Read the data block's header.
    //
    
    int yInFile;
    
    //
    // Read the part number when we are dealing with a multi-part file.
    //
    
    if (isMultiPart(_data->version))
    {
        int partNumber;
        OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*_data->_streamData->is, partNumber);
        if (partNumber != _data->partNumber)
        {
            THROW (IEX_NAMESPACE::ArgExc, "Unexpected part number " << partNumber
            << ", should be " << _data->partNumber << ".");
        }
    }
    
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*_data->_streamData->is, yInFile);
    
    if (yInFile != minY)
        throw IEX_NAMESPACE::InputExc ("Unexpected data block y coordinate.");
    
    Int64 sampleCountTableSize;
    Int64 packedDataSize;
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*_data->_streamData->is, sampleCountTableSize);
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*_data->_streamData->is, packedDataSize);
    
    // total requirement for reading all the data
    
    Int64 totalSizeRequired=28+sampleCountTableSize+packedDataSize;
    
    bool big_enough = totalSizeRequired<=pixelDataSize;
    
    pixelDataSize = totalSizeRequired;
    
    // was the block we were given big enough?
    if(!big_enough || pixelData==NULL)
    {        
        // special case: seek stream back to start if we are at the beginning (regular reading pixels assumes it doesn't need to seek
        // in single part files)
        if(!isMultiPart(_data->version))
        {
          if (_data->nextLineBufferMinY == minY)
              _data->_streamData->is->seekg (lineOffset);
        }
        // leave lock here - bail before reading more data
        return;
    }
    
    // copy the values we have read into the output block
    *(int *) pixelData = yInFile;
    *(Int64 *) (pixelData+4) =sampleCountTableSize;
    *(Int64 *) (pixelData+12) = packedDataSize;
    
    // didn't read the unpackedsize - do that now
    Xdr::read<StreamIO> (*_data->_streamData->is, *(Int64 *) (pixelData+20));
    
    // read the actual data
    _data->_streamData->is->read(pixelData+28, sampleCountTableSize+packedDataSize);
    
    // special case: seek stream back to start if we are at the beginning (regular reading pixels assumes it doesn't need to seek
    // in single part files)
    if(!isMultiPart(_data->version))
    {
        if (_data->nextLineBufferMinY == minY)
            _data->_streamData->is->seekg (lineOffset);
    }
    
    // leave lock here
    
}

void DeepScanLineInputFile::readPixels (const char* rawPixelData, 
                                        const DeepFrameBuffer& frameBuffer, 
                                        int scanLine1, 
                                        int scanLine2) const
{
    //
    // read header from block - already converted from Xdr to native format
    //
    int data_scanline = *(int *) rawPixelData;
    Int64 sampleCountTableDataSize=*(Int64 *) (rawPixelData+4);
    Int64 packedDataSize = *(Int64 *) (rawPixelData+12);
    Int64 unpackedDataSize = *(Int64 *) (rawPixelData+20);

    
    
    //
    // Uncompress the data, if necessary
    //
    
    
    Compressor * decomp = NULL;
    const char * uncompressed_data;
    Compressor::Format format = Compressor::XDR;
    if(packedDataSize <unpackedDataSize)
    {
        decomp = newCompressor(_data->header.compression(),
                                             unpackedDataSize,
                                             _data->header);
                                             
        decomp->uncompress(rawPixelData+28+sampleCountTableDataSize,
                           packedDataSize,
                           data_scanline, uncompressed_data);
        format = decomp->format();
    }
    else
    {
        //
        // If the line is uncompressed, it's in XDR format,
        // regardless of the compressor's output format.
        //
        
        format = Compressor::XDR;
        uncompressed_data = rawPixelData+28+sampleCountTableDataSize;
    }
  
    
    int yStart, yStop, dy;
    
    if (_data->lineOrder == INCREASING_Y)
    {
        yStart = scanLine1;
        yStop = scanLine2 + 1;
        dy = 1;
    }
    else
    {
        yStart = scanLine2;
        yStop = scanLine1 - 1;
        dy = -1;
    }
    
    
    
    const char* samplecount_base = frameBuffer.getSampleCountSlice().base;
    int samplecount_xstride = frameBuffer.getSampleCountSlice().xStride;
    int samplecount_ystride = frameBuffer.getSampleCountSlice().yStride;
    
    //
    // For each line within the block, get the count of bytes.
    //
    
    int minYInLineBuffer = data_scanline;
    int maxYInLineBuffer = min(minYInLineBuffer + _data->linesInBuffer - 1, _data->maxY);
    
    vector<size_t> bytesPerLine(1+_data->maxY-_data->minY);
    
    
    bytesPerDeepLineTable (_data->header,
                           minYInLineBuffer,
                           maxYInLineBuffer,
                           samplecount_base,
                           samplecount_xstride,
                           samplecount_ystride,
                           bytesPerLine);
                           
    //
    // For each scanline within the block, get the offset.
    //
      
    vector<size_t> offsetInLineBuffer;
    offsetInLineBufferTable (bytesPerLine,
                             minYInLineBuffer - _data->minY,
                             maxYInLineBuffer - _data->minY,
                             _data->linesInBuffer,
                             offsetInLineBuffer);
                             
                             
    const ChannelList & channels=header().channels();    
    
    
    for (int y = yStart; y != yStop; y += dy)
    {
        
        const char *readPtr =uncompressed_data +
        offsetInLineBuffer[y - _data->minY];

        //
        // need to know the total number of samples on a scanline to skip channels
        // compute on demand: -1 means uncomputed
        //
        int lineSampleCount = -1;
        
        
        //
        // Iterate over all image channels in frame buffer
        //
    
    
        ChannelList::ConstIterator i = channels.begin();
                             
        for (DeepFrameBuffer::ConstIterator j = frameBuffer.begin();
                                            j != frameBuffer.end();
             ++j)
        {
            while (i != channels.end() && strcmp (i.name(), j.name()) < 0)
            {
                //
                // Channel i is present in the file but not
                // in the frame buffer; skip
                    
                if(lineSampleCount==-1)
                {
                     lineSampleCount=0;
                     const char * ptr = (samplecount_base+y*samplecount_ystride + samplecount_xstride*_data->minX);
                     for(int x=_data->minX;x<=_data->maxX;x++)
                     { 
                         
                          lineSampleCount+=*(const unsigned int *) ptr;
                          ptr+=samplecount_xstride;
                     }
                }

               skipChannel (readPtr, i.channel().type, lineSampleCount );
        
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
            if (modp (y, i.channel().ySampling) == 0)
            {        
                
                copyIntoDeepFrameBuffer (readPtr, j.slice().base,
                                         samplecount_base,
                                         samplecount_xstride,
                                         samplecount_ystride,
                                         y, _data->minX, _data->maxX,
                                         0, 0,
                                         0, 0,
                                         j.slice().sampleStride, 
                                         j.slice().xStride,
                                         j.slice().yStride,
                                         fill,
                                         j.slice().fillValue, 
                                         format,
                                         j.slice().type,
                                         i.channel().type);
                                         
                ++i;
                                         
            }
        }//next slice in framebuffer
    }//next row in image
    
    //
    // clean up
    //
    
    delete decomp;    
}
        
      

void DeepScanLineInputFile::readPixelSampleCounts (const char* rawPixelData, 
                                                   const DeepFrameBuffer& frameBuffer, 
                                                   int scanLine1, 
                                                   int scanLine2) const
{
    //
    // read header from block - already converted from Xdr to native format
    //
    int data_scanline = *(int *) rawPixelData;
    Int64 sampleCountTableDataSize=*(Int64 *) (rawPixelData+4);
    
    
    int maxY;
    maxY = min(data_scanline + _data->linesInBuffer - 1, _data->maxY);
    
    if(scanLine1 != data_scanline)
    {
        THROW(IEX_NAMESPACE::ArgExc,"readPixelSampleCounts(rawPixelData,frameBuffer,"<< scanLine1 << ',' << scanLine2 << ") called with incorrect start scanline - should be " << data_scanline );
    }
    
    if(scanLine2 != maxY)
    {
        THROW(IEX_NAMESPACE::ArgExc,"readPixelSampleCounts(rawPixelData,frameBuffer,"<< scanLine1 << ',' << scanLine2 << ") called with incorrect end scanline - should be " << maxY );
    }
    
    
    //
    // If the sample count table is compressed, we'll uncompress it.
    //
    
    Int64 rawSampleCountTableSize = (maxY - data_scanline + 1) * (_data->maxX - _data->minX + 1) *
    Xdr::size <unsigned int> ();
    
    
    Compressor * decomp=NULL;
    const char* readPtr;
    if (sampleCountTableDataSize < rawSampleCountTableSize)
    {
        decomp = newCompressor(_data->header.compression(),
                               rawSampleCountTableSize,
                               _data->header);
                                                    
        decomp->uncompress(rawPixelData+28,
                                               sampleCountTableDataSize,
                                               data_scanline,
                                               readPtr);
    }
    else readPtr = rawPixelData+28;
    
    char* base = frameBuffer.getSampleCountSlice().base;
    int xStride = frameBuffer.getSampleCountSlice().xStride;
    int yStride = frameBuffer.getSampleCountSlice().yStride;
    
   
    
    for (int y = scanLine1; y <= scanLine2; y++)
    {
        int lastAccumulatedCount = 0;
        for (int x = _data->minX; x <= _data->maxX; x++)
        {
            int accumulatedCount, count;
            
            //
            // Read the sample count for pixel (x, y).
            //
            
            Xdr::read <CharPtrIO> (readPtr, accumulatedCount);
            if (x == _data->minX)
                count = accumulatedCount;
            else
                count = accumulatedCount - lastAccumulatedCount;
            lastAccumulatedCount = accumulatedCount;
            
            //
            // Store the data in both internal and external data structure.
            //
            
            sampleCount(base, xStride, yStride, x, y) = count;
        }
    }
    
    if(decomp)
    {
       delete decomp;
    }
}



namespace
{

void
readSampleCountForLineBlock(InputStreamMutex* streamData,
                            DeepScanLineInputFile::Data* data,
                            int lineBlockId)
{
    streamData->is->seekg(data->lineOffsets[lineBlockId]);

    if (isMultiPart(data->version))
    {
        int partNumber;
        OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, partNumber);

        if (partNumber != data->partNumber)
            throw IEX_NAMESPACE::ArgExc("Unexpected part number.");
    }

    int minY;
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, minY);

    //
    // Check the correctness of minY.
    //

    if (minY != data->minY + lineBlockId * data->linesInBuffer)
        throw IEX_NAMESPACE::ArgExc("Unexpected data block y coordinate.");

    int maxY;
    maxY = min(minY + data->linesInBuffer - 1, data->maxY);

    Int64 sampleCountTableDataSize;
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, sampleCountTableDataSize);

    
    
    if(sampleCountTableDataSize>data->maxSampleCountTableSize)
    {
        THROW (IEX_NAMESPACE::ArgExc, "Bad sampleCountTableDataSize read from chunk "<< lineBlockId << ": expected " << data->maxSampleCountTableSize << " or less, got "<< sampleCountTableDataSize);
    }
    
    Int64 packedDataSize;
    Int64 unpackedDataSize;
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, packedDataSize);
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, unpackedDataSize);

    
    
    //
    // We make a check on the data size requirements here.
    // Whilst we wish to store 64bit sizes on disk, not all the compressors
    // have been made to work with such data sizes and are still limited to
    // using signed 32 bit (int) for the data size. As such, this version
    // insists that we validate that the data size does not exceed the data
    // type max limit.
    // @TODO refactor the compressor code to ensure full 64-bit support.
    //

    int compressorMaxDataSize = std::numeric_limits<int>::max();
    if (sampleCountTableDataSize > Int64(compressorMaxDataSize))
    {
        THROW (IEX_NAMESPACE::ArgExc, "This version of the library does not "
              << "support the allocation of data with size  > "
              << compressorMaxDataSize
              << " file table size    :" << sampleCountTableDataSize << ".\n");
    }
    streamData->is->read(data->sampleCountTableBuffer, sampleCountTableDataSize);
    
    const char* readPtr;

    //
    // If the sample count table is compressed, we'll uncompress it.
    //


    if (sampleCountTableDataSize < data->maxSampleCountTableSize)
    {
        if(!data->sampleCountTableComp)
        {
            THROW(IEX_NAMESPACE::ArgExc,"Deep scanline data corrupt at chunk " << lineBlockId << " (sampleCountTableDataSize error)");
        }
        data->sampleCountTableComp->uncompress(data->sampleCountTableBuffer,
                                               sampleCountTableDataSize,
                                               minY,
                                               readPtr);
    }
    else readPtr = data->sampleCountTableBuffer;

    char* base = data->sampleCountSliceBase;
    int xStride = data->sampleCountXStride;
    int yStride = data->sampleCountYStride;

    // total number of samples in block: used to check samplecount table doesn't
    // reference more data than exists
    
    size_t cumulative_total_samples=0;
    
    for (int y = minY; y <= maxY; y++)
    {
        int yInDataWindow = y - data->minY;
        data->lineSampleCount[yInDataWindow] = 0;

        int lastAccumulatedCount = 0;
        for (int x = data->minX; x <= data->maxX; x++)
        {
            int accumulatedCount, count;

            //
            // Read the sample count for pixel (x, y).
            //

            Xdr::read <CharPtrIO> (readPtr, accumulatedCount);
            
            // sample count table should always contain monotonically
            // increasing values.
            if (accumulatedCount < lastAccumulatedCount)
            {
                THROW(IEX_NAMESPACE::ArgExc,"Deep scanline sampleCount data corrupt at chunk " << lineBlockId << " (negative sample count detected)");
            }

            count = accumulatedCount - lastAccumulatedCount;
            lastAccumulatedCount = accumulatedCount;

            //
            // Store the data in both internal and external data structure.
            //

            data->sampleCount[yInDataWindow][x - data->minX] = count;
            data->lineSampleCount[yInDataWindow] += count;
            sampleCount(base, xStride, yStride, x, y) = count;
        }
        cumulative_total_samples+=data->lineSampleCount[yInDataWindow];
        if(cumulative_total_samples*data->combinedSampleSize > unpackedDataSize)
        {
            THROW(IEX_NAMESPACE::ArgExc,"Deep scanline sampleCount data corrupt at chunk " << lineBlockId << ": pixel data only contains " << unpackedDataSize 
            << " bytes of data but table references at least " << cumulative_total_samples*data->combinedSampleSize << " bytes of sample data" );            
        }
        data->gotSampleCount[y - data->minY] = true;
    }
}


void
fillSampleCountFromCache(int y, DeepScanLineInputFile::Data* data)
{
    int yInDataWindow = y - data->minY;
    char* base = data->sampleCountSliceBase;
    int xStride = data->sampleCountXStride;
    int yStride = data->sampleCountYStride;
    
    for (int x = data->minX; x <= data->maxX; x++)
    {
        unsigned int count = data->sampleCount[yInDataWindow][x - data->minX];    
        sampleCount(base, xStride, yStride, x, y) = count;
    }
}

} // namespace

void
DeepScanLineInputFile::readPixelSampleCounts (int scanline1, int scanline2)
{
    Int64 savedFilePos = 0;

    if(!_data->frameBufferValid)
    {
        throw IEX_NAMESPACE::ArgExc("readPixelSampleCounts called with no valid frame buffer");
    }
    
    try
    {
        Lock lock (*_data->_streamData);

        savedFilePos = _data->_streamData->is->tellg();

        int scanLineMin = min (scanline1, scanline2);
        int scanLineMax = max (scanline1, scanline2);

        if (scanLineMin < _data->minY || scanLineMax > _data->maxY)
            throw IEX_NAMESPACE::ArgExc ("Tried to read scan line sample counts outside "
                               "the image file's data window.");

        for (int i = scanLineMin; i <= scanLineMax; i++)
        {
            //
            // if scanline is already read, it'll be in the cache
            // otherwise, read from file, store in cache and in caller's framebuffer
            //
            if (_data->gotSampleCount[i - _data->minY])
            {
                fillSampleCountFromCache(i,_data);
                                         
            }else{

                int lineBlockId = ( i - _data->minY ) / _data->linesInBuffer;

                readSampleCountForLineBlock ( _data->_streamData, _data, lineBlockId );

                int minYInLineBuffer = lineBlockId * _data->linesInBuffer + _data->minY;
                int maxYInLineBuffer = min ( minYInLineBuffer + _data->linesInBuffer - 1, _data->maxY );

                //
                // For each line within the block, get the count of bytes.
                //

                bytesPerDeepLineTable ( _data->header,
                                        minYInLineBuffer,
                                        maxYInLineBuffer,
                                        _data->sampleCountSliceBase,
                                        _data->sampleCountXStride,
                                        _data->sampleCountYStride,
                                        _data->bytesPerLine );

                //
                // For each scanline within the block, get the offset.
                //

                offsetInLineBufferTable ( _data->bytesPerLine,
                                          minYInLineBuffer - _data->minY,
                                          maxYInLineBuffer - _data->minY,
                                          _data->linesInBuffer,
                                          _data->offsetInLineBuffer );
            }
        }

        _data->_streamData->is->seekg(savedFilePos);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        REPLACE_EXC (e, "Error reading sample count data from image "
                     "file \"" << fileName() << "\". " << e.what());

        _data->_streamData->is->seekg(savedFilePos);

        throw;
    }
}

void
DeepScanLineInputFile::readPixelSampleCounts(int scanline)
{
    readPixelSampleCounts(scanline, scanline);
}

int 
DeepScanLineInputFile::firstScanLineInChunk(int y) const
{
    return int((y-_data->minY)/_data->linesInBuffer)*_data->linesInBuffer + _data->minY;
}

int
DeepScanLineInputFile::lastScanLineInChunk(int y) const
{
    int minY = firstScanLineInChunk(y);
    return min(minY+_data->linesInBuffer-1,_data->maxY);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
