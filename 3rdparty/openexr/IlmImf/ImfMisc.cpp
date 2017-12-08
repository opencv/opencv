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
//	Miscellaneous helper functions for OpenEXR image file I/O
//
//-----------------------------------------------------------------------------

#include <ImfMisc.h>
#include <ImfHeader.h>
#include <ImfAttribute.h>
#include <ImfCompressor.h>
#include <ImfChannelList.h>
#include <ImfXdr.h>
#include <ImathFun.h>
#include <Iex.h>
#include <ImfStdIO.h>
#include <ImfConvert.h>
#include <ImfPartType.h>
#include <ImfTileDescription.h>
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using IMATH_NAMESPACE::Box2i;
using IMATH_NAMESPACE::divp;
using IMATH_NAMESPACE::modp;
using std::vector;

int
pixelTypeSize (PixelType type)
{
    int size;

    switch (type)
    {
      case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:
	
	size = Xdr::size <unsigned int> ();
	break;

      case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

	size = Xdr::size <half> ();
	break;

      case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

	size = Xdr::size <float> ();
	break;

      default:

	throw IEX_NAMESPACE::ArgExc ("Unknown pixel type.");
    }

    return size;
}


int
numSamples (int s, int a, int b)
{
    int a1 = divp (a, s);
    int b1 = divp (b, s);
    return  b1 - a1 + ((a1 * s < a)? 0: 1);
}


size_t
bytesPerLineTable (const Header &header,
		   vector<size_t> &bytesPerLine)
{
    const Box2i &dataWindow = header.dataWindow();
    const ChannelList &channels = header.channels();

    bytesPerLine.resize (dataWindow.max.y - dataWindow.min.y + 1);

    for (ChannelList::ConstIterator c = channels.begin();
	 c != channels.end();
	 ++c)
    {
	int nBytes = pixelTypeSize (c.channel().type) *
		     (dataWindow.max.x - dataWindow.min.x + 1) /
		     c.channel().xSampling;

	for (int y = dataWindow.min.y, i = 0; y <= dataWindow.max.y; ++y, ++i)
	    if (modp (y, c.channel().ySampling) == 0)
		bytesPerLine[i] += nBytes;
    }

    size_t maxBytesPerLine = 0;

    for (int y = dataWindow.min.y, i = 0; y <= dataWindow.max.y; ++y, ++i)
	if (maxBytesPerLine < bytesPerLine[i])
	    maxBytesPerLine = bytesPerLine[i];

    return maxBytesPerLine;
}


const int&
sampleCount(const char* base, int xStride, int yStride, int x, int y)
{
    const char* ptr = base + y * yStride + x * xStride;
    int* intPtr = (int*) ptr;

    return *intPtr;
}

int&
sampleCount(char* base, int xStride, int yStride, int x, int y)
{
    char* ptr = base + y * yStride + x * xStride;
    int* intPtr = (int*) ptr;
    
    return *intPtr;
}


size_t
bytesPerDeepLineTable (const Header &header,
                       int minY, int maxY,
                       const char* base,
                       int xStride,
                       int yStride,
                       vector<size_t> &bytesPerLine)
{
    const Box2i &dataWindow = header.dataWindow();
    const ChannelList &channels = header.channels();

    for (ChannelList::ConstIterator c = channels.begin();
         c != channels.end();
         ++c)
    {
        for (int y = minY; y <= maxY; ++y)
            if (modp (y, c.channel().ySampling) == 0)
            {
                int nBytes = 0;
                for (int x = dataWindow.min.x; x <= dataWindow.max.x; x++)
                {
                    if (modp (x, c.channel().xSampling) == 0)
                        nBytes += pixelTypeSize (c.channel().type) *
                                  sampleCount(base, xStride, yStride, x, y);
                }
                bytesPerLine[y - dataWindow.min.y] += nBytes;
            }
    }

    size_t maxBytesPerLine = 0;

    for (int y = minY; y <= maxY; ++y)
        if (maxBytesPerLine < bytesPerLine[y - dataWindow.min.y])
            maxBytesPerLine = bytesPerLine[y - dataWindow.min.y];

    return maxBytesPerLine;
}


size_t
bytesPerDeepLineTable (const Header &header,
                       char* base,
                       int xStride,
                       int yStride,
                       vector<size_t> &bytesPerLine)
{
    return bytesPerDeepLineTable(header,
                                 header.dataWindow().min.y,
                                 header.dataWindow().max.y,
                                 base,
                                 xStride,
                                 yStride,
                                 bytesPerLine);
}


void
offsetInLineBufferTable (const vector<size_t> &bytesPerLine,
                         int scanline1, int scanline2,
                         int linesInLineBuffer,
                         vector<size_t> &offsetInLineBuffer)
{
    offsetInLineBuffer.resize (bytesPerLine.size());

    size_t offset = 0;

    for (int i = scanline1; i <= scanline2; ++i)
    {
        if (i % linesInLineBuffer == 0)
            offset = 0;

        offsetInLineBuffer[i] = offset;
        offset += bytesPerLine[i];
    }
}


void
offsetInLineBufferTable (const vector<size_t> &bytesPerLine,
			 int linesInLineBuffer,
			 vector<size_t> &offsetInLineBuffer)
{
    offsetInLineBufferTable (bytesPerLine,
                             0, bytesPerLine.size() - 1,
                             linesInLineBuffer,
                             offsetInLineBuffer);
}


int
lineBufferMinY (int y, int minY, int linesInLineBuffer)
{
    return ((y - minY) / linesInLineBuffer) * linesInLineBuffer + minY;
}


int
lineBufferMaxY (int y, int minY, int linesInLineBuffer)
{
    return lineBufferMinY (y, minY, linesInLineBuffer) + linesInLineBuffer - 1;
}


Compressor::Format
defaultFormat (Compressor * compressor)
{
    return compressor? compressor->format(): Compressor::XDR;
}


int
numLinesInBuffer (Compressor * compressor)
{
    return compressor? compressor->numScanLines(): 1;
}


void
copyIntoFrameBuffer (const char *& readPtr,
		     char * writePtr,
		     char * endPtr,
                     size_t xStride,
		     bool fill,
		     double fillValue,
                     Compressor::Format format,
                     PixelType typeInFrameBuffer,
                     PixelType typeInFile)
{
    //
    // Copy a horizontal row of pixels from an input
    // file's line or tile buffer to a frame buffer.
    //

    if (fill)
    {
        //
        // The file contains no data for this channel.
        // Store a default value in the frame buffer.
        //

        switch (typeInFrameBuffer)
        {
	  case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:
            
            {
                unsigned int fillVal = (unsigned int) (fillValue);

                while (writePtr <= endPtr)
                {
                    *(unsigned int *) writePtr = fillVal;
                    writePtr += xStride;
                }
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            {
                half fillVal = half (fillValue);

                while (writePtr <= endPtr)
                {
                    *(half *) writePtr = fillVal;
                    writePtr += xStride;
                }
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            {
                float fillVal = float (fillValue);

                while (writePtr <= endPtr)
                {
                    *(float *) writePtr = fillVal;
                    writePtr += xStride;
                }
            }
            break;

          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
    else if (format == Compressor::XDR)
    {
        //
        // The the line or tile buffer is in XDR format.
        //
        // Convert the pixels from the file's machine-
        // independent representation, and store the
        // results in the frame buffer.
        //

        switch (typeInFrameBuffer)
        {
          case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:
    
            switch (typeInFile)
            {
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                while (writePtr <= endPtr)
                {
                    Xdr::read <CharPtrIO> (readPtr, *(unsigned int *) writePtr);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                while (writePtr <= endPtr)
                {
                    half h;
                    Xdr::read <CharPtrIO> (readPtr, h);
                    *(unsigned int *) writePtr = halfToUint (h);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                while (writePtr <= endPtr)
                {
                    float f;
                    Xdr::read <CharPtrIO> (readPtr, f);
                    *(unsigned int *)writePtr = floatToUint (f);
                    writePtr += xStride;
                }
                break;
                
              default:                  
                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            switch (typeInFile)
            {
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                while (writePtr <= endPtr)
                {
                    unsigned int ui;
                    Xdr::read <CharPtrIO> (readPtr, ui);
                    *(half *) writePtr = uintToHalf (ui);
                    writePtr += xStride;
                }
                break;
                
              case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                while (writePtr <= endPtr)
                {
                    Xdr::read <CharPtrIO> (readPtr, *(half *) writePtr);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                while (writePtr <= endPtr)
                {
                    float f;
                    Xdr::read <CharPtrIO> (readPtr, f);
                    *(half *) writePtr = floatToHalf (f);
                    writePtr += xStride;
                }
                break;
              default:
                  
                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            switch (typeInFile)
            {
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                while (writePtr <= endPtr)
                {
                    unsigned int ui;
                    Xdr::read <CharPtrIO> (readPtr, ui);
                    *(float *) writePtr = float (ui);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                while (writePtr <= endPtr)
                {
                    half h;
                    Xdr::read <CharPtrIO> (readPtr, h);
                    *(float *) writePtr = float (h);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                while (writePtr <= endPtr)
                {
                    Xdr::read <CharPtrIO> (readPtr, *(float *) writePtr);
                    writePtr += xStride;
                }
                break;
              default:
                  
                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }
            break;

          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
    else
    {
        //
        // The the line or tile buffer is in NATIVE format.
        // Copy the results into the frame buffer.
        //

        switch (typeInFrameBuffer)
        {
          case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:
    
            switch (typeInFile)
            {
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                while (writePtr <= endPtr)
                {
                    for (size_t i = 0; i < sizeof (unsigned int); ++i)
                        writePtr[i] = readPtr[i];

                    readPtr += sizeof (unsigned int);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                while (writePtr <= endPtr)
                {
                    half h = *(half *) readPtr;
                    *(unsigned int *) writePtr = halfToUint (h);
                    readPtr += sizeof (half);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                while (writePtr <= endPtr)
                {
                    float f;

                    for (size_t i = 0; i < sizeof (float); ++i)
                        ((char *)&f)[i] = readPtr[i];

                    *(unsigned int *)writePtr = floatToUint (f);
                    readPtr += sizeof (float);
                    writePtr += xStride;
                }
                break;
                
              default:
                  
                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            switch (typeInFile)
            {
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                while (writePtr <= endPtr)
                {
                    unsigned int ui;

                    for (size_t i = 0; i < sizeof (unsigned int); ++i)
                        ((char *)&ui)[i] = readPtr[i];

                    *(half *) writePtr = uintToHalf (ui);
                    readPtr += sizeof (unsigned int);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                // If we're tightly packed, just memcpy
                if (xStride == sizeof(half)) {
                    int numBytes = endPtr-writePtr+sizeof(half);
                    memcpy(writePtr, readPtr, numBytes);
                    readPtr  += numBytes;
                    writePtr += numBytes;                    
                } else {
                    while (writePtr <= endPtr)
                    {
                        *(half *) writePtr = *(half *)readPtr;
                        readPtr += sizeof (half);
                        writePtr += xStride;
                    }
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                while (writePtr <= endPtr)
                {
                    float f;

                    for (size_t i = 0; i < sizeof (float); ++i)
                        ((char *)&f)[i] = readPtr[i];

                    *(half *) writePtr = floatToHalf (f);
                    readPtr += sizeof (float);
                    writePtr += xStride;
                }
                break;
              default:
                  
                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            switch (typeInFile)
            {
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                while (writePtr <= endPtr)
                {
                    unsigned int ui;

                    for (size_t i = 0; i < sizeof (unsigned int); ++i)
                        ((char *)&ui)[i] = readPtr[i];

                    *(float *) writePtr = float (ui);
                    readPtr += sizeof (unsigned int);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                while (writePtr <= endPtr)
                {
                    half h = *(half *) readPtr;
                    *(float *) writePtr = float (h);
                    readPtr += sizeof (half);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                while (writePtr <= endPtr)
                {
                    for (size_t i = 0; i < sizeof (float); ++i)
                        writePtr[i] = readPtr[i];

                    readPtr += sizeof (float);
                    writePtr += xStride;
                }
                break;
              default:
                  
                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }
            break;

          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
}

void
copyIntoDeepFrameBuffer (const char *& readPtr,
                         char * base,
                         const char* sampleCountBase,
                         ptrdiff_t sampleCountXStride,
                         ptrdiff_t sampleCountYStride,
                         int y, int minX, int maxX,
                         int xOffsetForSampleCount,
                         int yOffsetForSampleCount,
                         int xOffsetForData,
                         int yOffsetForData,
                         ptrdiff_t sampleStride,
                         ptrdiff_t xPointerStride,
                         ptrdiff_t yPointerStride,
                         bool fill,
                         double fillValue,
                         Compressor::Format format,
                         PixelType typeInFrameBuffer,
                         PixelType typeInFile)
{
    //
    // Copy a horizontal row of pixels from an input
    // file's line or tile buffer to a frame buffer.
    //

    if (fill)
    {
        //
        // The file contains no data for this channel.
        // Store a default value in the frame buffer.
        //

        switch (typeInFrameBuffer)
        {
          case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

            {
                unsigned int fillVal = (unsigned int) (fillValue);

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    if(writePtr)
                    {
                        int count = sampleCount(sampleCountBase,
                                                sampleCountXStride,
                                                sampleCountYStride,
                                                x - xOffsetForSampleCount,
                                                y - yOffsetForSampleCount);
                        for (int i = 0; i < count; i++)
                        {
                            *(unsigned int *) writePtr = fillVal;
                            writePtr += sampleStride;
                        }
                    }
                }
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            {
                half fillVal = half (fillValue);

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    if(writePtr)
                    {                            
                        int count = sampleCount(sampleCountBase,
                                                sampleCountXStride,
                                                sampleCountYStride,
                                                x - xOffsetForSampleCount,
                                                y - yOffsetForSampleCount);
                        for (int i = 0; i < count; i++)
                        {
                            *(half *) writePtr = fillVal;
                           writePtr += sampleStride;
                       }
                    }
                }
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            {
                float fillVal = float (fillValue);

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    if(writePtr)
                    {
                        int count = sampleCount(sampleCountBase,
                                                sampleCountXStride,
                                                sampleCountYStride,
                                                x - xOffsetForSampleCount,
                                                y - yOffsetForSampleCount);
                        for (int i = 0; i < count; i++)
                        {
                            *(float *) writePtr = fillVal;
                            writePtr += sampleStride;
                        }
                    }
                }
            }
            break;

          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
    else if (format == Compressor::XDR)
    {
        //
        // The the line or tile buffer is in XDR format.
        //
        // Convert the pixels from the file's machine-
        // independent representation, and store the
        // results in the frame buffer.
        //

        switch (typeInFrameBuffer)
        {
          case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

            switch (typeInFile)
            {
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                    if(writePtr)
                    {
                   
                        for (int i = 0; i < count; i++)
                        {
                            Xdr::read <CharPtrIO> (readPtr, *(unsigned int *) writePtr);
                            writePtr += sampleStride;
                        }
                    }else{
                        Xdr::skip <CharPtrIO> (readPtr,count*Xdr::size<unsigned int>());
                    }
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                    if(writePtr)
                    {

                        for (int i = 0; i < count; i++)
                        {
                            half h;
                            Xdr::read <CharPtrIO> (readPtr, h);
                           *(unsigned int *) writePtr = halfToUint (h);
                           writePtr += sampleStride;
                       }
                    }else{
                       Xdr::skip <CharPtrIO> (readPtr,count*Xdr::size<half>());
                    }
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                                                                                        
                    if(writePtr)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            float f;
                            Xdr::read <CharPtrIO> (readPtr, f);
                            *(unsigned int *)writePtr = floatToUint (f);
                            writePtr += sampleStride;
                        } 
                     }else{
                       Xdr::skip <CharPtrIO> (readPtr,count*Xdr::size<float>());
                     }
                
                }
                break;
              default:
                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            switch (typeInFile)
            {
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                    if(writePtr)
                    {

                        for (int i = 0; i < count; i++)
                        {
                            unsigned int ui;
                            Xdr::read <CharPtrIO> (readPtr, ui);
                            *(half *) writePtr = uintToHalf (ui);
                            writePtr += sampleStride;
                        }
                    }else{
                        Xdr::skip <CharPtrIO> (readPtr,count*Xdr::size<unsigned int>());
                    }
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                    if(writePtr)
                    {
                    
                        for (int i = 0; i < count; i++)
                        {
                            Xdr::read <CharPtrIO> (readPtr, *(half *) writePtr);
                            writePtr += sampleStride;
                        }
                    }else{
                        Xdr::skip <CharPtrIO> (readPtr,count*Xdr::size<half>());
                    }
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **) (base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                    if(writePtr)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            float f;
                            Xdr::read <CharPtrIO> (readPtr, f);
                            *(half *) writePtr = floatToHalf (f);
                            writePtr += sampleStride;
                        }
                    }else{
                        Xdr::skip <CharPtrIO> (readPtr,count*Xdr::size<float>());
                    }
                }
                break;
              default:
                  
                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            switch (typeInFile)
            {
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                    if(writePtr)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            unsigned int ui;
                            Xdr::read <CharPtrIO> (readPtr, ui);
                            *(float *) writePtr = float (ui);
                            writePtr += sampleStride;
                        }
                    }else{
                        Xdr::skip <CharPtrIO> (readPtr,count*Xdr::size<unsigned int>());
                    }
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                    if(writePtr)
                    {

                        for (int i = 0; i < count; i++)
                        {
                            half h;
                            Xdr::read <CharPtrIO> (readPtr, h);
                            *(float *) writePtr = float (h);
                            writePtr += sampleStride;
                        }
                    
                   }else{
                      Xdr::skip <CharPtrIO> (readPtr,count*Xdr::size<half>());
                   }               
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                    if(writePtr)
                    {
                    
                        for (int i = 0; i < count; i++)
                        {
                            Xdr::read <CharPtrIO> (readPtr, *(float *) writePtr);
                            writePtr += sampleStride;
                        }
                    } else{
                        Xdr::skip <CharPtrIO> (readPtr,count*Xdr::size<float>());
                    }      
                    
                }
                break;
              default:
                  
                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }
            break;

          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
    else
    {
        //
        // The the line or tile buffer is in NATIVE format.
        // Copy the results into the frame buffer.
        //

        switch (typeInFrameBuffer)
        {
          case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

            switch (typeInFile)
            {
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                                            
                    if(writePtr)
                    {
                         for (int i = 0; i < count; i++)
                         {
                             for (size_t i = 0; i < sizeof (unsigned int); ++i)
                                 writePtr[i] = readPtr[i];

                             readPtr += sizeof (unsigned int);
                             writePtr += sampleStride;
                         }
                    }else{
                        readPtr+=sizeof(unsigned int)*count;
                    }
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                                            
                    if(writePtr)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            half h = *(half *) readPtr;
                            *(unsigned int *) writePtr = halfToUint (h);
                            readPtr += sizeof (half);
                            writePtr += sampleStride;
                        }
                    }else{
                        readPtr+=sizeof(half)*count;
                    }
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                                            
                    if(writePtr)
                    {
                    
                        for (int i = 0; i < count; i++)
                        {
                            float f;

                            for (size_t i = 0; i < sizeof (float); ++i)
                                ((char *)&f)[i] = readPtr[i];

                            *(unsigned int *)writePtr = floatToUint (f);
                            readPtr += sizeof (float);
                            writePtr += sampleStride;
                        }
                    }else{
                        readPtr+=sizeof(float)*count;
                    }
                }
                break;
              default:
                  
                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            switch (typeInFile)
            {
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                                            
                    if(writePtr)
                    {
                         for (int i = 0; i < count; i++)
                         {
                             unsigned int ui;
 
                             for (size_t i = 0; i < sizeof (unsigned int); ++i)
                                 ((char *)&ui)[i] = readPtr[i];
  
                             *(half *) writePtr = uintToHalf (ui);
                             readPtr += sizeof (unsigned int);
                             writePtr += sampleStride;
                         }
                    }else{
                        readPtr+=sizeof(unsigned int)*count;
                    }
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                                            
                    if(writePtr)
                    {
                         for (int i = 0; i < count; i++)
                         {
                             *(half *) writePtr = *(half *)readPtr;
                             readPtr += sizeof (half);
                             writePtr += sampleStride;
                         }
                    }else{
                        readPtr+=sizeof(half)*count;
                    }
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                                            
                    if(writePtr)
                    {
                         for (int i = 0; i < count; i++)
                         {
                            float f;

                             for (size_t i = 0; i < sizeof (float); ++i)
                                 ((char *)&f)[i] = readPtr[i];

                            *(half *) writePtr = floatToHalf (f);
                            readPtr += sizeof (float);
                            writePtr += sampleStride;
                         }
                    }else{
                        readPtr+=sizeof(float)*count;
                    }
                }
                break;
              default:
                  
                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            switch (typeInFile)
            {
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                                            
                    if(writePtr)
                    {
                         for (int i = 0; i < count; i++)
                         {
                              unsigned int ui;
 
                              for (size_t i = 0; i < sizeof (unsigned int); ++i)
                                  ((char *)&ui)[i] = readPtr[i];

                              *(float *) writePtr = float (ui);
                              readPtr += sizeof (unsigned int);
                              writePtr += sampleStride;
                         }
                    }else{
                        readPtr+=sizeof(unsigned int)*count;
                    }
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                                            
                    if(writePtr)
                    {
                         for (int i = 0; i < count; i++)
                         {
                             half h = *(half *) readPtr;
                             *(float *) writePtr = float (h);
                             readPtr += sizeof (half);
                             writePtr += sampleStride;
                         }
                    }else{
                        readPtr+=sizeof(half)*count;
                    }
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                for (int x = minX; x <= maxX; x++)
                {
                    char* writePtr = *(char **)(base+(y-yOffsetForData)*yPointerStride + (x-xOffsetForData)*xPointerStride);
                    
                    int count = sampleCount(sampleCountBase,
                                            sampleCountXStride,
                                            sampleCountYStride,
                                            x - xOffsetForSampleCount,
                                            y - yOffsetForSampleCount);
                                            
                    if(writePtr)
                    {
                         for (int i = 0; i < count; i++)
                         {
                              for (size_t i = 0; i < sizeof (float); ++i)
                                  writePtr[i] = readPtr[i];

                             readPtr += sizeof (float);
                             writePtr += sampleStride;
                         }
                    }else{
                        readPtr+=sizeof(float)*count;
                    }
                }
                break;
              default:
                  
                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }
            break;

          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
}


void
skipChannel (const char *& readPtr,
             PixelType typeInFile,
	     size_t xSize)
{
    switch (typeInFile)
    {
      case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:
        
        Xdr::skip <CharPtrIO> (readPtr, Xdr::size <unsigned int> () * xSize);
        break;

      case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

        Xdr::skip <CharPtrIO> (readPtr, Xdr::size <half> () * xSize);
        break;

      case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

        Xdr::skip <CharPtrIO> (readPtr, Xdr::size <float> () * xSize);
        break;

      default:

        throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
    }
}


void
convertInPlace (char *& writePtr,
                const char *& readPtr,
		PixelType type,
                size_t numPixels)
{
    switch (type)
    {
      case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:
    
        for (size_t j = 0; j < numPixels; ++j)
        {
            Xdr::write <CharPtrIO> (writePtr, *(const unsigned int *) readPtr);
            readPtr += sizeof(unsigned int);
        }
        break;
    
      case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:
    
        for (size_t j = 0; j < numPixels; ++j)
        {               
            Xdr::write <CharPtrIO> (writePtr, *(const half *) readPtr);
            readPtr += sizeof(half);
        }
        break;
    
      case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:
    
        for (size_t j = 0; j < numPixels; ++j)
        {
            Xdr::write <CharPtrIO> (writePtr, *(const float *) readPtr);
            readPtr += sizeof(float);
        }
        break;
    
      default:
    
        throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
    }
}


void
copyFromFrameBuffer (char *& writePtr,
		     const char *& readPtr,
                     const char * endPtr,
		     size_t xStride,
                     Compressor::Format format,
		     PixelType type)
{
    //
    // Copy a horizontal row of pixels from a frame
    // buffer to an output file's line or tile buffer.
    //

    if (format == Compressor::XDR)
    {
        //
        // The the line or tile buffer is in XDR format.
        //

        switch (type)
        {
          case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

            while (readPtr <= endPtr)
            {
                Xdr::write <CharPtrIO> (writePtr,
                                        *(const unsigned int *) readPtr);
                readPtr += xStride;
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            while (readPtr <= endPtr)
            {
                Xdr::write <CharPtrIO> (writePtr, *(const half *) readPtr);
                readPtr += xStride;
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            while (readPtr <= endPtr)
            {
                Xdr::write <CharPtrIO> (writePtr, *(const float *) readPtr);
                readPtr += xStride;
            }
            break;

          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
    else
    {
        //
        // The the line or tile buffer is in NATIVE format.
        //

        switch (type)
        {
          case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

            while (readPtr <= endPtr)
            {
                for (size_t i = 0; i < sizeof (unsigned int); ++i)
                    *writePtr++ = readPtr[i];

                readPtr += xStride;
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            while (readPtr <= endPtr)
            {
                *(half *) writePtr = *(const half *) readPtr;
                writePtr += sizeof (half);
                readPtr += xStride;
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            while (readPtr <= endPtr)
            {
                for (size_t i = 0; i < sizeof (float); ++i)
                    *writePtr++ = readPtr[i];

                readPtr += xStride;
            }
            break;
            
          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
}

void
copyFromDeepFrameBuffer (char *& writePtr,
                         const char * base,
                         char* sampleCountBase,
                         ptrdiff_t sampleCountXStride,
                         ptrdiff_t sampleCountYStride,
                         int y, int xMin, int xMax,
                         int xOffsetForSampleCount,
                         int yOffsetForSampleCount,
                         int xOffsetForData,
                         int yOffsetForData,
                         ptrdiff_t sampleStride,
                         ptrdiff_t dataXStride,
                         ptrdiff_t dataYStride,
                         Compressor::Format format,
                         PixelType type)
{
    //
    // Copy a horizontal row of pixels from a frame
    // buffer to an output file's line or tile buffer.
    //

    if (format == Compressor::XDR)
    {
        //
        // The the line or tile buffer is in XDR format.
        //

        switch (type)
        {
          case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

            for (int x = xMin; x <= xMax; x++)
            {
                unsigned int count =
                        sampleCount(sampleCountBase,
                                   sampleCountXStride,
                                   sampleCountYStride,
                                   x - xOffsetForSampleCount,
                                   y - yOffsetForSampleCount);
                const char* ptr = base + (y-yOffsetForData) * dataYStride + (x-xOffsetForData) * dataXStride;
                const char* readPtr = ((const char**) ptr)[0];
                for (unsigned int i = 0; i < count; i++)
                {
                    Xdr::write <CharPtrIO> (writePtr,
                                            *(const unsigned int *) readPtr);
                    readPtr += sampleStride;
                }
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            for (int x = xMin; x <= xMax; x++)
            {
                unsigned int count =
                        sampleCount(sampleCountBase,
                                   sampleCountXStride,
                                   sampleCountYStride,
                                   x - xOffsetForSampleCount,
                                   y - yOffsetForSampleCount);
                const char* ptr = base + (y-yOffsetForData) * dataYStride + (x-xOffsetForData) * dataXStride;
                const char* readPtr = ((const char**) ptr)[0];
                for (unsigned int i = 0; i < count; i++)
                {
                    Xdr::write <CharPtrIO> (writePtr, *(const half *) readPtr);
                    readPtr += sampleStride;
                }
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            for (int x = xMin; x <= xMax; x++)
            {
                unsigned int count =
                        sampleCount(sampleCountBase,
                                   sampleCountXStride,
                                   sampleCountYStride,
                                   x - xOffsetForSampleCount,
                                   y - yOffsetForSampleCount);
                const char* ptr = base + (y-yOffsetForData) * dataYStride + (x-xOffsetForData) * dataXStride;                                   
                                   
                const char* readPtr = ((const char**) ptr)[0];
                for (unsigned int i = 0; i < count; i++)
                {
                    Xdr::write <CharPtrIO> (writePtr, *(const float *) readPtr);
                    readPtr += sampleStride;
                }
            }
            break;

          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
    else
    {
        //
        // The the line or tile buffer is in NATIVE format.
        //

        switch (type)
        {
          case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

            for (int x = xMin; x <= xMax; x++)
            {
                unsigned int count =
                        sampleCount(sampleCountBase,
                                   sampleCountXStride,
                                   sampleCountYStride,
                                   x - xOffsetForSampleCount,
                                   y - yOffsetForSampleCount);
                                   
                const char* ptr = base + (y-yOffsetForData) * dataYStride + (x-xOffsetForData) * dataXStride;                                                                      
                const char* readPtr = ((const char**) ptr)[0];
                for (unsigned int i = 0; i < count; i++)
                {
                    for (size_t j = 0; j < sizeof (unsigned int); ++j)
                        *writePtr++ = readPtr[j];

                    readPtr += sampleStride;
                }
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            for (int x = xMin; x <= xMax; x++)
            {
                unsigned int count =
                        sampleCount(sampleCountBase,
                                   sampleCountXStride,
                                   sampleCountYStride,
                                   x - xOffsetForSampleCount,
                                   y - yOffsetForSampleCount);
                const char* ptr = base + (y-yOffsetForData) * dataYStride + (x-xOffsetForData) * dataXStride;                                   
                const char* readPtr = ((const char**) ptr)[0];
                for (unsigned int i = 0; i < count; i++)
                {
                    *(half *) writePtr = *(const half *) readPtr;
                    writePtr += sizeof (half);
                    readPtr += sampleStride;
                }
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            for (int x = xMin; x <= xMax; x++)
            {
                unsigned int count =
                        sampleCount(sampleCountBase,
                                   sampleCountXStride,
                                   sampleCountYStride,
                                   x - xOffsetForSampleCount,
                                   y - yOffsetForSampleCount);
                                   
                const char* ptr = base + (y-yOffsetForData) * dataYStride + (x-xOffsetForData) * dataXStride;                                   
                const char* readPtr = ((const char**) ptr)[0];
                for (unsigned int i = 0; i < count; i++)
                {
                    for (size_t j = 0; j < sizeof (float); ++j)
                        *writePtr++ = readPtr[j];

                    readPtr += sampleStride;
                }
            }
            break;

          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
}


void
fillChannelWithZeroes (char *& writePtr,
		       Compressor::Format format,
		       PixelType type,
		       size_t xSize)
{
    if (format == Compressor::XDR)
    {
        //
        // Fill with data in XDR format.
        //

        switch (type)
        {
          case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

            for (size_t j = 0; j < xSize; ++j)
                Xdr::write <CharPtrIO> (writePtr, (unsigned int) 0);

            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            for (size_t j = 0; j < xSize; ++j)
                Xdr::write <CharPtrIO> (writePtr, (half) 0);

            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            for (size_t j = 0; j < xSize; ++j)
                Xdr::write <CharPtrIO> (writePtr, (float) 0);

            break;
            
          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
    else
    {
        //
        // Fill with data in NATIVE format.
        //

        switch (type)
        {
          case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

            for (size_t j = 0; j < xSize; ++j)
            {
                static const unsigned int ui = 0;

                for (size_t i = 0; i < sizeof (ui); ++i)
                    *writePtr++ = ((char *) &ui)[i];
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            for (size_t j = 0; j < xSize; ++j)
            {
                *(half *) writePtr = half (0);
                writePtr += sizeof (half);
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            for (size_t j = 0; j < xSize; ++j)
            {
                static const float f = 0;

                for (size_t i = 0; i < sizeof (f); ++i)
                    *writePtr++ = ((char *) &f)[i];
            }
            break;
            
          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
}

bool
usesLongNames (const Header &header)
{
    //
    // If an OpenEXR file contains any attribute names, attribute type names
    // or channel names longer than 31 characters, then the file cannot be
    // read by older versions of the IlmImf library (up to OpenEXR 1.6.1).
    // Before writing the file header, we check if the header contains
    // any names longer than 31 characters; if it does, then we set the
    // LONG_NAMES_FLAG in the file version number.  Older versions of the
    // IlmImf library will refuse to read files that have the LONG_NAMES_FLAG
    // set.  Without the flag, older versions of the library would mis-
    // interpret the file as broken.
    //

    for (Header::ConstIterator i = header.begin();
         i != header.end();
         ++i)
    {
        if (strlen (i.name()) >= 32 || strlen (i.attribute().typeName()) >= 32)
            return true;
    }

    const ChannelList &channels = header.channels();

    for (ChannelList::ConstIterator i = channels.begin();
         i != channels.end();
         ++i)
    {
        if (strlen (i.name()) >= 32)
            return true;
    }

    return false;
}

int
getScanlineChunkOffsetTableSize(const Header& header)
{
    const Box2i &dataWindow = header.dataWindow();

    vector<size_t> bytesPerLine;
    size_t maxBytesPerLine = bytesPerLineTable (header,
                                                bytesPerLine);

    Compressor* compressor = newCompressor(header.compression(),
                                           maxBytesPerLine,
                                           header);

    int linesInBuffer = numLinesInBuffer (compressor);

    int lineOffsetSize = (dataWindow.max.y - dataWindow.min.y +
                          linesInBuffer) / linesInBuffer;

    delete compressor;

    return lineOffsetSize;
}

//
// Located in ImfTiledMisc.cpp
//
int
getTiledChunkOffsetTableSize(const Header& header);

int
getChunkOffsetTableSize(const Header& header,bool ignore_attribute)
{
    if(!ignore_attribute && header.hasChunkCount())
    {
        return header.chunkCount();
    }
    
    if(header.hasType()  && !isSupportedType(header.type()))
    {
        throw IEX_NAMESPACE::ArgExc ("unsupported header type to "
        "get chunk offset table size");
    }
    if (isTiled(header.type()) == false)
        return getScanlineChunkOffsetTableSize(header);
    else
        return getTiledChunkOffsetTableSize(header);
    
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
