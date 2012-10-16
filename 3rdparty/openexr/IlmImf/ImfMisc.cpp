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
#include <ImfCompressor.h>
#include <ImfChannelList.h>
#include <ImfXdr.h>
#include <ImathFun.h>
#include <Iex.h>
#include <ImfStdIO.h>
#include <ImfConvert.h>

namespace Imf {

using Imath::Box2i;
using Imath::divp;
using Imath::modp;
using std::vector;

int
pixelTypeSize (PixelType type)
{
    int size;

    switch (type)
    {
      case UINT:

    size = Xdr::size <unsigned int> ();
    break;

      case HALF:

    size = Xdr::size <half> ();
    break;

      case FLOAT:

    size = Xdr::size <float> ();
    break;

      default:

    throw Iex::ArgExc ("Unknown pixel type.");
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


void
offsetInLineBufferTable (const vector<size_t> &bytesPerLine,
             int linesInLineBuffer,
             vector<size_t> &offsetInLineBuffer)
{
    offsetInLineBuffer.resize (bytesPerLine.size());

    size_t offset = 0;

    for (int i = 0; i < bytesPerLine.size(); ++i)
    {
    if (i % linesInLineBuffer == 0)
        offset = 0;

    offsetInLineBuffer[i] = offset;
    offset += bytesPerLine[i];
    }
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
      case UINT:

            {
                unsigned int fillVal = (unsigned int) (fillValue);

                while (writePtr <= endPtr)
                {
                    *(unsigned int *) writePtr = fillVal;
                    writePtr += xStride;
                }
            }
            break;

          case HALF:

            {
                half fillVal = half (fillValue);

                while (writePtr <= endPtr)
                {
                    *(half *) writePtr = fillVal;
                    writePtr += xStride;
                }
            }
            break;

          case FLOAT:

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

            throw Iex::ArgExc ("Unknown pixel data type.");
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
          case UINT:

            switch (typeInFile)
            {
              case UINT:

                while (writePtr <= endPtr)
                {
                    Xdr::read <CharPtrIO> (readPtr, *(unsigned int *) writePtr);
                    writePtr += xStride;
                }
                break;

              case HALF:

                while (writePtr <= endPtr)
                {
                    half h;
                    Xdr::read <CharPtrIO> (readPtr, h);
                    *(unsigned int *) writePtr = halfToUint (h);
                    writePtr += xStride;
                }
                break;

              case FLOAT:

                while (writePtr <= endPtr)
                {
                    float f;
                    Xdr::read <CharPtrIO> (readPtr, f);
                    *(unsigned int *)writePtr = floatToUint (f);
                    writePtr += xStride;
                }
                break;
            }
            break;

          case HALF:

            switch (typeInFile)
            {
              case UINT:

                while (writePtr <= endPtr)
                {
                    unsigned int ui;
                    Xdr::read <CharPtrIO> (readPtr, ui);
                    *(half *) writePtr = uintToHalf (ui);
                    writePtr += xStride;
                }
                break;

              case HALF:

                while (writePtr <= endPtr)
                {
                    Xdr::read <CharPtrIO> (readPtr, *(half *) writePtr);
                    writePtr += xStride;
                }
                break;

              case FLOAT:

                while (writePtr <= endPtr)
                {
                    float f;
                    Xdr::read <CharPtrIO> (readPtr, f);
                    *(half *) writePtr = floatToHalf (f);
                    writePtr += xStride;
                }
                break;
            }
            break;

          case FLOAT:

            switch (typeInFile)
            {
              case UINT:

                while (writePtr <= endPtr)
                {
                    unsigned int ui;
                    Xdr::read <CharPtrIO> (readPtr, ui);
                    *(float *) writePtr = float (ui);
                    writePtr += xStride;
                }
                break;

              case HALF:

                while (writePtr <= endPtr)
                {
                    half h;
                    Xdr::read <CharPtrIO> (readPtr, h);
                    *(float *) writePtr = float (h);
                    writePtr += xStride;
                }
                break;

              case FLOAT:

                while (writePtr <= endPtr)
                {
                    Xdr::read <CharPtrIO> (readPtr, *(float *) writePtr);
                    writePtr += xStride;
                }
                break;
            }
            break;

          default:

            throw Iex::ArgExc ("Unknown pixel data type.");
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
          case UINT:

            switch (typeInFile)
            {
              case UINT:

                while (writePtr <= endPtr)
                {
                    for (size_t i = 0; i < sizeof (unsigned int); ++i)
                        writePtr[i] = readPtr[i];

                    readPtr += sizeof (unsigned int);
                    writePtr += xStride;
                }
                break;

              case HALF:

                while (writePtr <= endPtr)
                {
                    half h = *(half *) readPtr;
                    *(unsigned int *) writePtr = halfToUint (h);
                    readPtr += sizeof (half);
                    writePtr += xStride;
                }
                break;

              case FLOAT:

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
            }
            break;

          case HALF:

            switch (typeInFile)
            {
              case UINT:

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

              case HALF:

                while (writePtr <= endPtr)
                {
                    *(half *) writePtr = *(half *)readPtr;
                    readPtr += sizeof (half);
                    writePtr += xStride;
                }
                break;

              case FLOAT:

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
            }
            break;

          case FLOAT:

            switch (typeInFile)
            {
              case UINT:

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

              case HALF:

                while (writePtr <= endPtr)
                {
                    half h = *(half *) readPtr;
                    *(float *) writePtr = float (h);
                    readPtr += sizeof (half);
                    writePtr += xStride;
                }
                break;

              case FLOAT:

                while (writePtr <= endPtr)
                {
                    for (size_t i = 0; i < sizeof (float); ++i)
                        writePtr[i] = readPtr[i];

                    readPtr += sizeof (float);
                    writePtr += xStride;
                }
                break;
            }
            break;

          default:

            throw Iex::ArgExc ("Unknown pixel data type.");
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
      case UINT:

        Xdr::skip <CharPtrIO> (readPtr, Xdr::size <unsigned int> () * xSize);
        break;

      case HALF:

        Xdr::skip <CharPtrIO> (readPtr, Xdr::size <half> () * xSize);
        break;

      case FLOAT:

        Xdr::skip <CharPtrIO> (readPtr, Xdr::size <float> () * xSize);
        break;

      default:

        throw Iex::ArgExc ("Unknown pixel data type.");
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
      case UINT:

        for (int j = 0; j < numPixels; ++j)
        {
            Xdr::write <CharPtrIO> (writePtr, *(const unsigned int *) readPtr);
            readPtr += sizeof(unsigned int);
        }
        break;

      case HALF:

        for (int j = 0; j < numPixels; ++j)
        {
            Xdr::write <CharPtrIO> (writePtr, *(const half *) readPtr);
            readPtr += sizeof(half);
        }
        break;

      case FLOAT:

        for (int j = 0; j < numPixels; ++j)
        {
            Xdr::write <CharPtrIO> (writePtr, *(const float *) readPtr);
            readPtr += sizeof(float);
        }
        break;

      default:

        throw Iex::ArgExc ("Unknown pixel data type.");
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
          case UINT:

            while (readPtr <= endPtr)
            {
                Xdr::write <CharPtrIO> (writePtr,
                                        *(const unsigned int *) readPtr);
                readPtr += xStride;
            }
            break;

          case HALF:

            while (readPtr <= endPtr)
            {
                Xdr::write <CharPtrIO> (writePtr, *(const half *) readPtr);
                readPtr += xStride;
            }
            break;

          case FLOAT:

            while (readPtr <= endPtr)
            {
                Xdr::write <CharPtrIO> (writePtr, *(const float *) readPtr);
                readPtr += xStride;
            }
            break;

          default:

            throw Iex::ArgExc ("Unknown pixel data type.");
        }
    }
    else
    {
        //
        // The the line or tile buffer is in NATIVE format.
        //

        switch (type)
        {
          case UINT:

            while (readPtr <= endPtr)
            {
                for (size_t i = 0; i < sizeof (unsigned int); ++i)
                    *writePtr++ = readPtr[i];

                readPtr += xStride;
            }
            break;

          case HALF:

            while (readPtr <= endPtr)
            {
                *(half *) writePtr = *(const half *) readPtr;
                writePtr += sizeof (half);
                readPtr += xStride;
            }
            break;

          case FLOAT:

            while (readPtr <= endPtr)
            {
                for (size_t i = 0; i < sizeof (float); ++i)
                    *writePtr++ = readPtr[i];

                readPtr += xStride;
            }
            break;

          default:

            throw Iex::ArgExc ("Unknown pixel data type.");
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
          case UINT:

            for (int j = 0; j < xSize; ++j)
                Xdr::write <CharPtrIO> (writePtr, (unsigned int) 0);

            break;

          case HALF:

            for (int j = 0; j < xSize; ++j)
                Xdr::write <CharPtrIO> (writePtr, (half) 0);

            break;

          case FLOAT:

            for (int j = 0; j < xSize; ++j)
                Xdr::write <CharPtrIO> (writePtr, (float) 0);

            break;

          default:

            throw Iex::ArgExc ("Unknown pixel data type.");
        }
    }
    else
    {
        //
        // Fill with data in NATIVE format.
        //

        switch (type)
        {
          case UINT:

            for (int j = 0; j < xSize; ++j)
            {
                static const unsigned int ui = 0;

                for (size_t i = 0; i < sizeof (ui); ++i)
                    *writePtr++ = ((char *) &ui)[i];
            }
            break;

          case HALF:

            for (int j = 0; j < xSize; ++j)
            {
                *(half *) writePtr = half (0);
                writePtr += sizeof (half);
            }
            break;

          case FLOAT:

            for (int j = 0; j < xSize; ++j)
            {
                static const float f = 0;

                for (size_t i = 0; i < sizeof (f); ++i)
                    *writePtr++ = ((char *) &f)[i];
            }
            break;

          default:

            throw Iex::ArgExc ("Unknown pixel data type.");
        }
    }
}

} // namespace Imf
