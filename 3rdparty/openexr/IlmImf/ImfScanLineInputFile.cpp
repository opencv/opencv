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
//	class ScanLineInputFile
//
//-----------------------------------------------------------------------------

#include <ImfScanLineInputFile.h>
#include <ImfChannelList.h>
#include <ImfMisc.h>
#include <ImfStdIO.h>
#include <ImfCompressor.h>
#include "ImathBox.h"
#include "ImathFun.h"
#include <ImfXdr.h>
#include <ImfConvert.h>
#include <ImfThreading.h>
#include "IlmThreadPool.h"
#include "IlmThreadSemaphore.h"
#include "IlmThreadMutex.h"
#include "Iex.h"
#include <string>
#include <vector>
#include <assert.h>


namespace Imf {

using Imath::Box2i;
using Imath::divp;
using Imath::modp;
using std::string;
using std::vector;
using std::ifstream;
using std::min;
using std::max;
using IlmThread::Mutex;
using IlmThread::Lock;
using IlmThread::Semaphore;
using IlmThread::Task;
using IlmThread::TaskGroup;
using IlmThread::ThreadPool;

namespace {

struct InSliceInfo
{
    PixelType	typeInFrameBuffer;
    PixelType	typeInFile;
    char *	base;
    size_t	xStride;
    size_t	yStride;
    int		xSampling;
    int		ySampling;
    bool	fill;
    bool	skip;
    double	fillValue;

    InSliceInfo (PixelType typeInFrameBuffer = HALF,
		 PixelType typeInFile = HALF,
	         char *base = 0,
	         size_t xStride = 0,
	         size_t yStride = 0,
	         int xSampling = 1,
	         int ySampling = 1,
	         bool fill = false,
	         bool skip = false,
	         double fillValue = 0.0);
};


InSliceInfo::InSliceInfo (PixelType tifb,
			  PixelType tifl,
			  char *b,
			  size_t xs, size_t ys,
			  int xsm, int ysm,
			  bool f, bool s,
			  double fv)
:
    typeInFrameBuffer (tifb),
    typeInFile (tifl),
    base (b),
    xStride (xs),
    yStride (ys),
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
    const char *	uncompressedData;
    char *		buffer;
    int			dataSize;
    int			minY;
    int			maxY;
    Compressor *	compressor;
    Compressor::Format	format;
    int			number;
    bool		hasException;
    string		exception;

    LineBuffer (Compressor * const comp);
    ~LineBuffer ();

    inline void		wait () {_sem.wait();}
    inline void		post () {_sem.post();}

  private:

    Semaphore		_sem;
};


LineBuffer::LineBuffer (Compressor *comp):
    uncompressedData (0),
    buffer (0),
    dataSize (0),
    compressor (comp),
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
    delete compressor;
}

} // namespace


struct ScanLineInputFile::Data: public Mutex
{
    Header		header;		    // the image header
    int			version;            // file's version
    FrameBuffer		frameBuffer;	    // framebuffer to write into
    LineOrder		lineOrder;          // order of the scanlines in file
    int			minX;		    // data window's min x coord
    int			maxX;		    // data window's max x coord
    int			minY;		    // data window's min y coord
    int			maxY;		    // data window's max x coord
    vector<Int64>	lineOffsets;	    // stores offsets in file for
					    // each line
    bool		fileIsComplete;	    // True if no scanlines are missing
    					    // in the file
    int			nextLineBufferMinY; // minimum y of the next linebuffer
    vector<size_t>	bytesPerLine;       // combined size of a line over all
                                            // channels
    vector<size_t>	offsetInLineBuffer; // offset for each scanline in its
                                            // linebuffer
    vector<InSliceInfo>	slices;             // info about channels in file
    IStream *		is;                 // file stream to read from
    
    vector<LineBuffer*> lineBuffers;        // each holds one line buffer
    int			linesInBuffer;      // number of scanlines each buffer
                                            // holds
    size_t		lineBufferSize;     // size of the line buffer

     Data (IStream *is, int numThreads);
    ~Data ();
    
    inline LineBuffer * getLineBuffer (int number); // hash function from line
    						    // buffer indices into our
						    // vector of line buffers
};


ScanLineInputFile::Data::Data (IStream *is, int numThreads):
    is (is)
{
    //
    // We need at least one lineBuffer, but if threading is used,
    // to keep n threads busy we need 2*n lineBuffers
    //

    lineBuffers.resize (max (1, 2 * numThreads));
}


ScanLineInputFile::Data::~Data ()
{
    for (size_t i = 0; i < lineBuffers.size(); i++)
        delete lineBuffers[i];
}


inline LineBuffer *
ScanLineInputFile::Data::getLineBuffer (int lineBufferNumber)
{
    return lineBuffers[lineBufferNumber % lineBuffers.size()];
}


namespace {


void
reconstructLineOffsets (IStream &is,
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
	    Xdr::read <StreamIO> (is, y);

	    int dataSize;
	    Xdr::read <StreamIO> (is, dataSize);

	    Xdr::skip <StreamIO> (is, dataSize);

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
readLineOffsets (IStream &is,
		 LineOrder lineOrder,
		 vector<Int64> &lineOffsets,
		 bool &complete)
{
    for (unsigned int i = 0; i < lineOffsets.size(); i++)
    {
	Xdr::read <StreamIO> (is, lineOffsets[i]);
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
readPixelData (ScanLineInputFile::Data *ifd,
	       int minY,
	       char *&buffer,
	       int &dataSize)
{
    //
    // Read a single line buffer from the input file.
    //
    // If the input file is not memory-mapped, we copy the pixel data into
    // into the array pointed to by buffer.  If the file is memory-mapped,
    // then we change where buffer points to instead of writing into the
    // array (hence buffer needs to be a reference to a char *).
    //

    Int64 lineOffset =
	ifd->lineOffsets[(minY - ifd->minY) / ifd->linesInBuffer];

    if (lineOffset == 0)
	THROW (Iex::InputExc, "Scan line " << minY << " is missing.");

    //
    // Seek to the start of the scan line in the file,
    // if necessary.
    //

    if (ifd->nextLineBufferMinY != minY)
	ifd->is->seekg (lineOffset);

    //
    // Read the data block's header.
    //

    int yInFile;

    Xdr::read <StreamIO> (*ifd->is, yInFile);
    Xdr::read <StreamIO> (*ifd->is, dataSize);
    
    if (yInFile != minY)
        throw Iex::InputExc ("Unexpected data block y coordinate.");

    if (dataSize > (int) ifd->lineBufferSize)
	throw Iex::InputExc ("Unexpected data block length.");

    //
    // Read the pixel data.
    //

    if (ifd->is->isMemoryMapped ())
        buffer = ifd->is->readMemoryMapped (dataSize);
    else
        ifd->is->read (buffer, dataSize);

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
                    ScanLineInputFile::Data *ifd,
		    LineBuffer *lineBuffer,
                    int scanLineMin,
		    int scanLineMax);

    virtual ~LineBufferTask ();

    virtual void		execute ();

  private:

    ScanLineInputFile::Data *	_ifd;
    LineBuffer *		_lineBuffer;
    int				_scanLineMin;
    int				_scanLineMax;
};


LineBufferTask::LineBufferTask
    (TaskGroup *group,
     ScanLineInputFile::Data *ifd,
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
            int uncompressedSize = 0;
            int maxY = min (_lineBuffer->maxY, _ifd->maxY);
    
            for (int i = _lineBuffer->minY - _ifd->minY;
                 i <= maxY - _ifd->minY;
		 ++i)
	    {
                uncompressedSize += (int) _ifd->bytesPerLine[i];
	    }
    
            if (_lineBuffer->compressor &&
                _lineBuffer->dataSize < uncompressedSize)
            {
                _lineBuffer->format = _lineBuffer->compressor->format();

                _lineBuffer->dataSize = _lineBuffer->compressor->uncompress
                    (_lineBuffer->buffer, _lineBuffer->dataSize,
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
    
                const InSliceInfo &slice = _ifd->slices[i];
    
                if (modp (y, slice.ySampling) != 0)
                    continue;
    
                //
                // Find the x coordinates of the leftmost and rightmost
                // sampled pixels (i.e. pixels within the data window
                // for which x % xSampling == 0).
                //
    
                int dMinX = divp (_ifd->minX, slice.xSampling);
                int dMaxX = divp (_ifd->maxX, slice.xSampling);
    
                //
		// Fill the frame buffer with pixel data.
                //
    
                if (slice.skip)
                {
                    //
                    // The file contains data for this channel, but
                    // the frame buffer contains no slice for this channel.
                    //
    
                    skipChannel (readPtr, slice.typeInFile, dMaxX - dMinX + 1);
                }
                else
                {
                    //
                    // The frame buffer contains a slice for this channel.
                    //
    
                    char *linePtr  = slice.base +
                                        divp (y, slice.ySampling) *
                                        slice.yStride;
    
                    char *writePtr = linePtr + dMinX * slice.xStride;
                    char *endPtr   = linePtr + dMaxX * slice.xStride;
                    
                    copyIntoFrameBuffer (readPtr, writePtr, endPtr,
                                         slice.xStride, slice.fill,
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
     ScanLineInputFile::Data *ifd,
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

	    readPixelData (ifd, lineBuffer->minY,
			   lineBuffer->buffer,
			   lineBuffer->dataSize);
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
	lineBuffer->post();\
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


ScanLineInputFile::ScanLineInputFile
    (const Header &header,
     IStream *is,
     int numThreads)
:
    _data (new Data (is, numThreads))
{
    try
    {
	_data->header = header;

	_data->lineOrder = _data->header.lineOrder();

	const Box2i &dataWindow = _data->header.dataWindow();

	_data->minX = dataWindow.min.x;
	_data->maxX = dataWindow.max.x;
	_data->minY = dataWindow.min.y;
	_data->maxY = dataWindow.max.y;

	size_t maxBytesPerLine = bytesPerLineTable (_data->header,
                                                    _data->bytesPerLine);

        for (size_t i = 0; i < _data->lineBuffers.size(); i++)
        {
            _data->lineBuffers[i] = new LineBuffer (newCompressor
                                                (_data->header.compression(),
                                                 maxBytesPerLine,
                                                 _data->header));
        }

        _data->linesInBuffer =
	    numLinesInBuffer (_data->lineBuffers[0]->compressor);

        _data->lineBufferSize = maxBytesPerLine * _data->linesInBuffer;

        if (!_data->is->isMemoryMapped())
            for (size_t i = 0; i < _data->lineBuffers.size(); i++)
                _data->lineBuffers[i]->buffer = new char[_data->lineBufferSize];

	_data->nextLineBufferMinY = _data->minY - 1;

	offsetInLineBufferTable (_data->bytesPerLine,
				 _data->linesInBuffer,
				 _data->offsetInLineBuffer);

	int lineOffsetSize = (dataWindow.max.y - dataWindow.min.y +
			      _data->linesInBuffer) / _data->linesInBuffer;

	_data->lineOffsets.resize (lineOffsetSize);

	readLineOffsets (*_data->is,
			 _data->lineOrder,
			 _data->lineOffsets,
			 _data->fileIsComplete);
    }
    catch (...)
    {
	delete _data;
	throw;
    }
}


ScanLineInputFile::~ScanLineInputFile ()
{
    if (!_data->is->isMemoryMapped())
        for (size_t i = 0; i < _data->lineBuffers.size(); i++)
            delete [] _data->lineBuffers[i]->buffer;

    delete _data;
}


const char *
ScanLineInputFile::fileName () const
{
    return _data->is->fileName();
}


const Header &
ScanLineInputFile::header () const
{
    return _data->header;
}


int
ScanLineInputFile::version () const
{
    return _data->version;
}


void	
ScanLineInputFile::setFrameBuffer (const FrameBuffer &frameBuffer)
{
    Lock lock (*_data);

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
	    THROW (Iex::ArgExc, "X and/or y subsampling factors "
				"of \"" << i.name() << "\" channel "
				"of input file \"" << fileName() << "\" are "
				"not compatible with the frame buffer's "
				"subsampling factors.");
    }

    //
    // Initialize the slice table for readPixels().
    //

    vector<InSliceInfo> slices;
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

	    slices.push_back (InSliceInfo (i.channel().type,
					   i.channel().type,
					   0, // base
					   0, // xStride
					   0, // yStride
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

	slices.push_back (InSliceInfo (j.slice().type,
				       fill? j.slice().type:
				             i.channel().type,
				       j.slice().base,
				       j.slice().xStride,
				       j.slice().yStride,
				       j.slice().xSampling,
				       j.slice().ySampling,
				       fill,
				       false, // skip
				       j.slice().fillValue));

	if (i != channels.end() && !fill)
	    ++i;
    }

    //
    // Store the new frame buffer.
    //

    _data->frameBuffer = frameBuffer;
    _data->slices = slices;
}


const FrameBuffer &
ScanLineInputFile::frameBuffer () const
{
    Lock lock (*_data);
    return _data->frameBuffer;
}


bool
ScanLineInputFile::isComplete () const
{
    return _data->fileIsComplete;
}


void	
ScanLineInputFile::readPixels (int scanLine1, int scanLine2)
{
    try
    {
        Lock lock (*_data);

	if (_data->slices.size() == 0)
	    throw Iex::ArgExc ("No frame buffer specified "
			       "as pixel data destination.");

	int scanLineMin = min (scanLine1, scanLine2);
	int scanLineMax = max (scanLine1, scanLine2);

	if (scanLineMin < _data->minY || scanLineMax > _data->maxY)
	    throw Iex::ArgExc ("Tried to read scan line outside "
			       "the image file's data window.");

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

        for (int i = 0; i < _data->lineBuffers.size(); ++i)
	{
            LineBuffer *lineBuffer = _data->lineBuffers[i];

	    if (lineBuffer->hasException && !exception)
		exception = &lineBuffer->exception;

	    lineBuffer->hasException = false;
	}

	if (exception)
	    throw Iex::IoExc (*exception);
    }
    catch (Iex::BaseExc &e)
    {
	REPLACE_EXC (e, "Error reading pixel data from image "
		        "file \"" << fileName() << "\". " << e);
	throw;
    }
}


void	
ScanLineInputFile::readPixels (int scanLine)
{
    readPixels (scanLine, scanLine);
}


void
ScanLineInputFile::rawPixelData (int firstScanLine,
				 const char *&pixelData,
				 int &pixelDataSize)
{
    try
    {
        Lock lock (*_data);

	if (firstScanLine < _data->minY || firstScanLine > _data->maxY)
	{
	    throw Iex::ArgExc ("Tried to read scan line outside "
			       "the image file's data window.");
	}

        int minY = lineBufferMinY
	    (firstScanLine, _data->minY, _data->linesInBuffer);

	readPixelData
	    (_data, minY, _data->lineBuffers[0]->buffer, pixelDataSize);

	pixelData = _data->lineBuffers[0]->buffer;
    }
    catch (Iex::BaseExc &e)
    {
	REPLACE_EXC (e, "Error reading pixel data from image "
		        "file \"" << fileName() << "\". " << e);
	throw;
    }
}

} // namespace Imf
