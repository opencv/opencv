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

#include "ImfScanLineInputFile.h"
#include "ImfChannelList.h"
#include "ImfMisc.h"
#include "ImfStdIO.h"
#include "ImfCompressor.h"
#include "ImathBox.h"
#include "ImathFun.h"
#include <ImfXdr.h>
#include <ImfConvert.h>
#include <ImfThreading.h>
#include <ImfPartType.h>
#include "IlmThreadPool.h"
#include "IlmThreadSemaphore.h"
#include "IlmThreadMutex.h"
#include "Iex.h"
#include "ImfVersion.h"
#include "ImfOptimizedPixelReading.h"
#include "ImfNamespace.h"
#include "ImfStandardAttributes.h"

#include <algorithm>
#include <string>
#include <vector>
#include <assert.h>
#include <cstring>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


using IMATH_NAMESPACE::Box2i;
using IMATH_NAMESPACE::divp;
using IMATH_NAMESPACE::modp;
using std::string;
using std::vector;
using std::ifstream;
using std::min;
using std::max;
using std::sort;
using ILMTHREAD_NAMESPACE::Mutex;
using ILMTHREAD_NAMESPACE::Lock;
using ILMTHREAD_NAMESPACE::Semaphore;
using ILMTHREAD_NAMESPACE::Task;
using ILMTHREAD_NAMESPACE::TaskGroup;
using ILMTHREAD_NAMESPACE::ThreadPool;

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

/// helper struct used to detect the order that the channels are stored

struct sliceOptimizationData
{
    const char * base;   ///< pointer to pixel data 
    bool fill;           ///< is this channel being filled with constant, instead of read?
    half fillValue;      ///< if filling, the value to use
    size_t offset;       ///< position this channel will be in the read buffer, accounting for previous channels, as well as their type
    PixelType type;      ///< type of channel
    size_t xStride;      ///< x-stride of channel in buffer (must be set to cause channels to interleave)
    size_t yStride;      ///< y-stride of channel in buffer (must be same in all channels, else order will change, which is bad)
    int xSampling;       ///< channel x sampling
    int ySampling;       ///< channel y sampling
            
            
    /// we need to keep the list sorted in the order they'll be written to memory
    bool operator<(const sliceOptimizationData& other ) const
    {
        return base < other.base;
    }
};
    

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
    
    vector<LineBuffer*> lineBuffers;        // each holds one line buffer
    int			linesInBuffer;      // number of scanlines each buffer
                                            // holds
    size_t		lineBufferSize;     // size of the line buffer
    int                 partNumber;         // part number

    bool                memoryMapped;       // if the stream is memory mapped
    OptimizationMode    optimizationMode;   // optimizibility of the input file
    vector<sliceOptimizationData>  optimizationData; ///< channel ordering for optimized reading
    
    Data (int numThreads);
    ~Data ();
    
    inline LineBuffer * getLineBuffer (int number); // hash function from line
    						    // buffer indices into our
						    // vector of line buffers
    
    
};


ScanLineInputFile::Data::Data (int numThreads):
        partNumber(-1),
        memoryMapped(false)
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

	    int dataSize;
	    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, dataSize);

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
               ScanLineInputFile::Data *ifd,
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

    int lineBufferNumber = (minY - ifd->minY) / ifd->linesInBuffer;

    Int64 lineOffset = ifd->lineOffsets[lineBufferNumber];

    if (lineOffset == 0)
	THROW (IEX_NAMESPACE::InputExc, "Scan line " << minY << " is missing.");

    //
    // Seek to the start of the scan line in the file,
    // if necessary.
    //

    if ( !isMultiPart(ifd->version) )
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
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*streamData->is, dataSize);
    
    if (yInFile != minY)
        throw IEX_NAMESPACE::InputExc ("Unexpected data block y coordinate.");

    if (dataSize > (int) ifd->lineBufferSize)
	throw IEX_NAMESPACE::InputExc ("Unexpected data block length.");

    //
    // Read the pixel data.
    //

    if (streamData->is->isMemoryMapped ())
        buffer = streamData->is->readMemoryMapped (dataSize);
    else
        streamData->is->read (buffer, dataSize);

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
		    int scanLineMax,
                    OptimizationMode optimizationMode);

    virtual ~LineBufferTask ();

    virtual void		execute ();

  private:

    ScanLineInputFile::Data *	_ifd;
    LineBuffer *		_lineBuffer;
    int				_scanLineMin;
    int				_scanLineMax;
    OptimizationMode            _optimizationMode;
};


LineBufferTask::LineBufferTask
    (TaskGroup *group,
     ScanLineInputFile::Data *ifd,
     LineBuffer *lineBuffer,
     int scanLineMin,
     int scanLineMax,OptimizationMode optimizationMode)
:
    Task (group),
    _ifd (ifd),
    _lineBuffer (lineBuffer),
    _scanLineMin (scanLineMin),
    _scanLineMax (scanLineMax),
    _optimizationMode(optimizationMode)
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
                    (_lineBuffer->buffer,
                     _lineBuffer->dataSize,
		     _lineBuffer->minY,
                     _lineBuffer->uncompressedData);
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


#ifdef IMF_HAVE_SSE2
//
// IIF format is more restricted than a perfectly generic one,
// so it is possible to perform some optimizations.
//
class LineBufferTaskIIF : public Task
{
    public:
        
        LineBufferTaskIIF (TaskGroup *group,
                           ScanLineInputFile::Data *ifd,
                           LineBuffer *lineBuffer,
                           int scanLineMin,
                           int scanLineMax,
                           OptimizationMode optimizationMode);
                           
        virtual ~LineBufferTaskIIF ();
                           
        virtual void                execute ();
        
        template<typename TYPE>
        void getWritePointer (int y,
                              unsigned short*& pOutWritePointerRight,
                              size_t& outPixelsToCopySSE,
                              size_t& outPixelsToCopyNormal,int bank=0) const;
                              
        template<typename TYPE>
        void getWritePointerStereo (int y,
                                    unsigned short*& outWritePointerRight,
                                    unsigned short*& outWritePointerLeft,
                                    size_t& outPixelsToCopySSE,
                                    size_t& outPixelsToCopyNormal) const;

    private:
        
        ScanLineInputFile::Data *   _ifd;
        LineBuffer *                _lineBuffer;
        int                         _scanLineMin;
        int                         _scanLineMax;
        OptimizationMode            _optimizationMode;
  
};

LineBufferTaskIIF::LineBufferTaskIIF
    (TaskGroup *group,
     ScanLineInputFile::Data *ifd,
     LineBuffer *lineBuffer,
     int scanLineMin,
     int scanLineMax,
     OptimizationMode optimizationMode
    )
    :
     Task (group),
     _ifd (ifd),
     _lineBuffer (lineBuffer),
     _scanLineMin (scanLineMin),
     _scanLineMax (scanLineMax),
     _optimizationMode (optimizationMode)
{
     /*
     //
     // indicates the optimised path has been taken
     //
     static bool could_optimise=false;
     if(could_optimise==false)
     {
         std::cerr << " optimised path\n";
         could_optimise=true;
     }
     */
}
 
LineBufferTaskIIF::~LineBufferTaskIIF ()
{
     //
     // Signal that the line buffer is now free
     //
     
     _lineBuffer->post ();
}
 
// Return 0 if we are to skip because of sampling
// channelBank is 0 for the first group of channels, 1 for the second
template<typename TYPE>
void LineBufferTaskIIF::getWritePointer 
                            (int y,
                             unsigned short*& outWritePointerRight,
                             size_t& outPixelsToCopySSE,
                             size_t& outPixelsToCopyNormal,
                             int channelBank
                            ) const
{
      // Channels are saved alphabetically, so the order is B G R.
      // The last slice (R) will give us the location of our write pointer.
      // The only slice that we support skipping is alpha, i.e. the first one.  
      // This does not impact the write pointer or the pixels to copy at all.
      
      size_t nbSlicesInBank = _ifd->optimizationData.size();
      
      int sizeOfSingleValue = sizeof(TYPE);
      
      if(_ifd->optimizationData.size()>4)
      {
          // there are two banks - we only copy one at once
          nbSlicesInBank/=2;
      }

      
      size_t firstChannel = 0;
      if(channelBank==1)
      {
          firstChannel = _ifd->optimizationData.size()/2;
      }
      
       sliceOptimizationData& firstSlice = _ifd->optimizationData[firstChannel];
      
      if (modp (y, firstSlice.ySampling) != 0)
      {
          outPixelsToCopySSE    = 0;
          outPixelsToCopyNormal = 0;
          outWritePointerRight  = 0;
      }
      
      const char* linePtr1  = firstSlice.base +
      divp (y, firstSlice.ySampling) *
      firstSlice.yStride;
      
      int dMinX1 = divp (_ifd->minX, firstSlice.xSampling);
      int dMaxX1 = divp (_ifd->maxX, firstSlice.xSampling);
      
      // Construct the writePtr so that we start writing at
      // linePtr + Min offset in the line.
      outWritePointerRight =  (unsigned short*)(linePtr1 +
      dMinX1 * firstSlice.xStride );
      
      size_t bytesToCopy  = ((linePtr1 + dMaxX1 * firstSlice.xStride ) -
      (linePtr1 + dMinX1 * firstSlice.xStride )) + 2;
      size_t shortsToCopy = bytesToCopy / sizeOfSingleValue;
      size_t pixelsToCopy = (shortsToCopy / nbSlicesInBank ) + 1;
      
      // We only support writing to SSE if we have no pixels to copy normally
      outPixelsToCopySSE    = pixelsToCopy / 8;
      outPixelsToCopyNormal = pixelsToCopy % 8;
      
}


template<typename TYPE>
void LineBufferTaskIIF::getWritePointerStereo 
                          (int y,
                           unsigned short*& outWritePointerRight,
                           unsigned short*& outWritePointerLeft,
                           size_t& outPixelsToCopySSE,
                           size_t& outPixelsToCopyNormal) const
{
   getWritePointer<TYPE>(y,outWritePointerRight,outPixelsToCopySSE,outPixelsToCopyNormal,0);
   
   
   if(outWritePointerRight)
   {
       getWritePointer<TYPE>(y,outWritePointerLeft,outPixelsToCopySSE,outPixelsToCopyNormal,1);
   }
   
}

void
LineBufferTaskIIF::execute()
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
                
                _lineBuffer->dataSize =
                _lineBuffer->compressor->uncompress (_lineBuffer->buffer,
                                                     _lineBuffer->dataSize,
                                                     _lineBuffer->minY,
                                                     _lineBuffer->uncompressedData);
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
            if (modp (y, _optimizationMode._ySampling) != 0)
                continue;
            
            //
            // Convert one scan line's worth of pixel data back
            // from the machine-independent representation, and
            // store the result in the frame buffer.
            //
            
            // Set the readPtr to read at the start of uncompressedData
            // but with an offet based on calculated array.
            // _ifd->offsetInLineBuffer contains offsets based on which
            // line we are currently processing.
            // Stride will be taken into consideration later.
                
                
            const char* readPtr = _lineBuffer->uncompressedData +
            _ifd->offsetInLineBuffer[y - _ifd->minY];
            
            size_t pixelsToCopySSE = 0;
            size_t pixelsToCopyNormal = 0;
            
            unsigned short* writePtrLeft = 0;
            unsigned short* writePtrRight = 0;
            
            size_t channels = _ifd->optimizationData.size();
       
            if(channels>4)
            {
                getWritePointerStereo<half>(y, writePtrRight, writePtrLeft, pixelsToCopySSE, pixelsToCopyNormal);
            }
            else 
            {
                getWritePointer<half>(y, writePtrRight, pixelsToCopySSE, pixelsToCopyNormal);
            }
            
            if (writePtrRight == 0 && pixelsToCopySSE == 0 && pixelsToCopyNormal == 0)
            {
                continue;
            }
            
            
            //
            // support reading up to eight channels
            //
            unsigned short* readPointers[8];
            
            for (size_t i = 0; i < channels ; ++i)
            {
                readPointers[i] = (unsigned short*)readPtr + (_ifd->optimizationData[i].offset * (pixelsToCopySSE * 8 + pixelsToCopyNormal));
            }
            
            //RGB only
            if(channels==3 || channels == 6 )
            {
                    optimizedWriteToRGB(readPointers[0], readPointers[1], readPointers[2], writePtrRight, pixelsToCopySSE, pixelsToCopyNormal);
                  
                    //stereo RGB
                    if( channels == 6)
                    {
                        optimizedWriteToRGB(readPointers[3], readPointers[4], readPointers[5], writePtrLeft, pixelsToCopySSE, pixelsToCopyNormal);
                    }                
            //RGBA
            }else if(channels==4 || channels==8)
            {
                
                if(_ifd->optimizationData[3].fill)
                {
                    optimizedWriteToRGBAFillA(readPointers[0], readPointers[1], readPointers[2], _ifd->optimizationData[3].fillValue.bits() , writePtrRight, pixelsToCopySSE, pixelsToCopyNormal);
                }else{
                    optimizedWriteToRGBA(readPointers[0], readPointers[1], readPointers[2], readPointers[3] , writePtrRight, pixelsToCopySSE, pixelsToCopyNormal);
                }
                
                //stereo RGBA
                if( channels == 8)
                {
                    if(_ifd->optimizationData[7].fill)
                    {
                        optimizedWriteToRGBAFillA(readPointers[4], readPointers[5], readPointers[6], _ifd->optimizationData[7].fillValue.bits() , writePtrLeft, pixelsToCopySSE, pixelsToCopyNormal);
                    }else{
                        optimizedWriteToRGBA(readPointers[4], readPointers[5], readPointers[6], readPointers[7] , writePtrLeft, pixelsToCopySSE, pixelsToCopyNormal);
                    }
                }
            }
            else {
                throw(IEX_NAMESPACE::LogicExc("IIF mode called with incorrect channel pattern"));
            }
            
            // If we are in NO_OPTIMIZATION mode, this class will never
            // get instantiated, so no need to check for it and duplicate
            // the code.
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
#endif


Task *
newLineBufferTask (TaskGroup *group,
                   InputStreamMutex *streamData,
                   ScanLineInputFile::Data *ifd,
                   int number,
                   int scanLineMin,
                   int scanLineMax,
                   OptimizationMode optimizationMode)
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
             
             readPixelData (streamData, ifd, lineBuffer->minY,
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
     
     
     Task* retTask = 0;
     
#ifdef IMF_HAVE_SSE2     
     if (optimizationMode._optimizable)
     {
         
         retTask = new LineBufferTaskIIF (group, ifd, lineBuffer,
                                          scanLineMin, scanLineMax,
                                          optimizationMode);
      
     }
     else
#endif         
     {
         retTask = new LineBufferTask (group, ifd, lineBuffer,
                                       scanLineMin, scanLineMax,
                                       optimizationMode);
     }
     
     return retTask;
     
 }
 
  


} // namespace


void ScanLineInputFile::initialize(const Header& header)
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

        if (!_streamData->is->isMemoryMapped())
        {
            for (size_t i = 0; i < _data->lineBuffers.size(); i++)
            {
                _data->lineBuffers[i]->buffer = (char *) EXRAllocAligned(_data->lineBufferSize*sizeof(char),16);
            }
        }
        _data->nextLineBufferMinY = _data->minY - 1;

        offsetInLineBufferTable (_data->bytesPerLine,
                                 _data->linesInBuffer,
                                 _data->offsetInLineBuffer);

        int lineOffsetSize = (dataWindow.max.y - dataWindow.min.y +
                              _data->linesInBuffer) / _data->linesInBuffer;

        _data->lineOffsets.resize (lineOffsetSize);
    }
    catch (...)
    {
        delete _data;
        _data=NULL;
        throw;
    }
}


ScanLineInputFile::ScanLineInputFile(InputPartData* part)
{
    if (part->header.type() != SCANLINEIMAGE)
        throw IEX_NAMESPACE::ArgExc("Can't build a ScanLineInputFile from a type-mismatched part.");

    _data = new Data(part->numThreads);
    _streamData = part->mutex;
    _data->memoryMapped = _streamData->is->isMemoryMapped();

    _data->version = part->version;

    initialize(part->header);

    _data->lineOffsets = part->chunkOffsets;

    _data->partNumber = part->partNumber;
    //
    // (TODO) change this code later.
    // The completeness of the file should be detected in MultiPartInputFile.
    //
    _data->fileIsComplete = true;
}


ScanLineInputFile::ScanLineInputFile
    (const Header &header,
     OPENEXR_IMF_INTERNAL_NAMESPACE::IStream *is,
     int numThreads)
:
    _data (new Data (numThreads)),
    _streamData (new InputStreamMutex())
{
    _streamData->is = is;
    _data->memoryMapped = is->isMemoryMapped();

    initialize(header);
    
    //
    // (TODO) this is nasty - we need a better way of working out what type of file has been used.
    // in any case I believe this constructor only gets used with single part files
    // and 'version' currently only tracks multipart state, so setting to 0 (not multipart) works for us
    //
    
    _data->version=0;
    readLineOffsets (*_streamData->is,
                     _data->lineOrder,
                     _data->lineOffsets,
                     _data->fileIsComplete);
}


ScanLineInputFile::~ScanLineInputFile ()
{
    if (!_data->memoryMapped)
    {
        for (size_t i = 0; i < _data->lineBuffers.size(); i++)
        {
            EXRFreeAligned(_data->lineBuffers[i]->buffer);
        }
    }
            

    //
    // ScanLineInputFile should never delete the stream,
    // because it does not own the stream.
    // We just delete the Mutex here.
    //
    if (_data->partNumber == -1)
        delete _streamData;

    delete _data;
}


const char *
ScanLineInputFile::fileName () const
{
    return _streamData->is->fileName();
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


namespace
{
    
    
// returns the optimization state for the given arrangement of frame bufers
// this assumes:
//   both the file and framebuffer are half float data
//   both the file and framebuffer have xSampling and ySampling=1
//   entries in optData are sorted into their interleave order (i.e. by base address)
//   These tests are done by SetFrameBuffer as it is building optData
//  
OptimizationMode
detectOptimizationMode (const vector<sliceOptimizationData>& optData)
{
    OptimizationMode w;
    
    // need to be compiled with SSE optimisations: if not, just returns false
#if IMF_HAVE_SSE2
    
    
    // only handle reading 3,4,6 or 8 channels
    switch(optData.size())
    {
        case 3 : break;
        case 4 : break;
        case 6 : break;
        case 8 : break;
        default :
            return w;
    }
    
    //
    // the point at which data switches between the primary and secondary bank
    //
    size_t bankSize = optData.size()>4 ? optData.size()/2 : optData.size();
    
    for(size_t i=0;i<optData.size();i++)
    {
        const sliceOptimizationData& data = optData[i];
        // can't fill anything other than channel 3 or channel 7
        if(data.fill)
        {
            if(i!=3 && i!=7)
            {
                return w;
            }
        }
        
        // cannot have gaps in the channel layout, so the stride must be (number of channels written in the bank)*2
        if(data.xStride !=bankSize*2)
        {
            return w;
        }
        
        // each bank of channels must be channel interleaved: each channel base pointer must be (previous channel+2)
        // this also means channel sampling pattern must be consistent, as must yStride
        if(i!=0 && i!=bankSize)
        {
            if(data.base!=optData[i-1].base+2)
            {
                return w;
            }
        }
        if(i!=0)
        {
            
            if(data.yStride!=optData[i-1].yStride)
            {
                return w;
            }
        }
    }
    

    w._ySampling=optData[0].ySampling;
    w._optimizable=true;
    
#endif

    return w;
}


} // Anonymous namespace

void	
ScanLineInputFile::setFrameBuffer (const FrameBuffer &frameBuffer)
{
    Lock lock (*_streamData);

    
    
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

    // optimization is possible if this is a little endian system
    // and both inputs and outputs are half floats
    // 
    bool optimizationPossible = true;
    
    if (!GLOBAL_SYSTEM_LITTLE_ENDIAN)
    {
        optimizationPossible =false;
    }
    
    vector<sliceOptimizationData> optData;
    

    //
    // Initialize the slice table for readPixels().
    //

    vector<InSliceInfo> slices;
    ChannelList::ConstIterator i = channels.begin();
    
    // current offset of channel: pixel data starts at offset*width into the
    // decompressed scanline buffer
    size_t offset = 0;
    
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
	    
              switch(i.channel().type)
              {
                  case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF :
                      offset++;
                      break;
                  case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT :
                      offset+=2;
                      break;
                  case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT :
                      offset+=2;
                      break;
              }
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

          if(!fill && i.channel().type!=OPENEXR_IMF_INTERNAL_NAMESPACE::HALF)
          {
              optimizationPossible = false;
          }
          
          if(j.slice().type != OPENEXR_IMF_INTERNAL_NAMESPACE::HALF)
          {
              optimizationPossible = false;
          }
          if(j.slice().xSampling!=1 || j.slice().ySampling!=1)
          {
              optimizationPossible = false;
          }

          
          if(optimizationPossible)
          {
              sliceOptimizationData dat;
              dat.base = j.slice().base;
              dat.fill = fill;
              dat.fillValue = j.slice().fillValue;
              dat.offset = offset;
              dat.xStride = j.slice().xStride;
              dat.yStride = j.slice().yStride;
              dat.xSampling = j.slice().xSampling;
              dat.ySampling = j.slice().ySampling;
              optData.push_back(dat);
          }
          
          if(!fill)
          {
              switch(i.channel().type)
              {
                  case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF :
                      offset++;
                      break;
                  case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT :
                      offset+=2;
                      break;
                  case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT :
                      offset+=2;
                      break;
              }
          }
          

          
	if (i != channels.end() && !fill)
	    ++i;
    }

   
   if(optimizationPossible)
   {
       //
       // check optimisibility
       // based on channel ordering and fill channel positions
       //
       sort(optData.begin(),optData.end());
       _data->optimizationMode = detectOptimizationMode(optData);
   }
   
   if(!optimizationPossible || _data->optimizationMode._optimizable==false)
   {   
       optData = vector<sliceOptimizationData>();
       _data->optimizationMode._optimizable=false;
   }
    
    //
    // Store the new frame buffer.
    //

    _data->frameBuffer = frameBuffer;
    _data->slices = slices;
    _data->optimizationData = optData;
}


const FrameBuffer &
ScanLineInputFile::frameBuffer () const
{
    Lock lock (*_streamData);
    return _data->frameBuffer;
}


bool
ScanLineInputFile::isComplete () const
{
    return _data->fileIsComplete;
}

bool ScanLineInputFile::isOptimizationEnabled() const
{
    if (_data->slices.size() == 0)
        throw IEX_NAMESPACE::ArgExc ("No frame buffer specified "
        "as pixel data destination.");
    
    return _data->optimizationMode._optimizable;
}


void	
ScanLineInputFile::readPixels (int scanLine1, int scanLine2)
{
    try
    {
        Lock lock (*_streamData);

	if (_data->slices.size() == 0)
	    throw IEX_NAMESPACE::ArgExc ("No frame buffer specified "
			       "as pixel data destination.");

	int scanLineMin = min (scanLine1, scanLine2);
	int scanLineMax = max (scanLine1, scanLine2);

	if (scanLineMin < _data->minY || scanLineMax > _data->maxY)
	    throw IEX_NAMESPACE::ArgExc ("Tried to read scan line outside "
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
                                                              _streamData,
                                                              _data, l,
                                                              scanLineMin,
                                                              scanLineMax,
                                                              _data->optimizationMode));
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
        Lock lock (*_streamData);

	if (firstScanLine < _data->minY || firstScanLine > _data->maxY)
	{
	    throw IEX_NAMESPACE::ArgExc ("Tried to read scan line outside "
			       "the image file's data window.");
	}

        int minY = lineBufferMinY
	    (firstScanLine, _data->minY, _data->linesInBuffer);

	readPixelData
	    (_streamData, _data, minY, _data->lineBuffers[0]->buffer, pixelDataSize);

	pixelData = _data->lineBuffers[0]->buffer;
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error reading pixel data from image "
		        "file \"" << fileName() << "\". " << e);
	throw;
    }
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
