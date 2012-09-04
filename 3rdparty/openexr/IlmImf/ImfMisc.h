///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IMF_MISC_H
#define INCLUDED_IMF_MISC_H

//-----------------------------------------------------------------------------
//
//	Miscellaneous helper functions for OpenEXR image file I/O
//
//-----------------------------------------------------------------------------

#include <ImfPixelType.h>
#include <vector>
#include <ImfCompressor.h>

namespace Imf {

class Header;

//
// Return the size of a single value of the indicated type,
// in the machine's native format.
//

int	pixelTypeSize (PixelType type);


//
// Return the number of samples a channel with subsampling rate
// s has in the interval [a, b].  For example, a channel with
// subsampling rate 2 (and samples at 0, 2, 4, 6, 8, etc.) has
// 2 samples in the interval [1, 5] and three samples in the
// interval [2, 6].
//

int	numSamples (int s, int a, int b);


//
// Build a table that lists, for each scanline in a file's
// data window, how many bytes are required to store all
// pixels in all channels in that scanline (assuming that
// the pixel data are tightly packed).
//

size_t	bytesPerLineTable (const Header &header,
		           std::vector<size_t> &bytesPerLine);

//
// For scanline-based files, pixels are read or written in
// in multi-scanline blocks.  Internally, class OutputFile
// and class ScanLineInputFile store a block of scan lines
// in a "line buffer".  Function offsetInLineBufferTable()
// builds a table that lists, for each scan line in a file's
// data window, the location of the pixel data for the scanline
// relative to the beginning of the line buffer.
//

void	offsetInLineBufferTable (const std::vector<size_t> &bytesPerLine,
				 int linesInLineBuffer,
				 std::vector<size_t> &offsetInLineBuffer);

//
// For a scanline-based file, compute the range of scanlines
// that occupy the same line buffer as a given scanline, y.
// (minY is the minimum y coordinate of the file's data window.)
//

int	lineBufferMinY (int y, int minY, int linesInLineBuffer);
int	lineBufferMaxY (int y, int minY, int linesInLineBuffer);


//
// Return a compressor's data format (Compressor::NATIVE or Compressor::XDR).
// If compressor is 0, return Compressor::XDR.
//

Compressor::Format defaultFormat (Compressor *compressor);


//
// Return the number of scan lines a compressor wants to compress
// or uncompress at once.  If compressor is 0, return 1.
//

int     numLinesInBuffer (Compressor *compressor);


//
// Copy a single channel of a horizontal row of pixels from an
// input file's internal line buffer or tile buffer into a
// frame buffer slice.  If necessary, perform on-the-fly data
// type conversion.
//
//    readPtr		initially points to the beginning of the
//			data in the line or tile buffer. readPtr
//			is advanced as the pixel data are copied;
//			when copyIntoFrameBuffer() returns,
//			readPtr points just past the end of the
//			copied data.
//
//    writePtr, endPtr	point to the lefmost and rightmost pixels
//			in the frame buffer slice
//
//    xStride		the xStride for the frame buffer slice
//
//    format		indicates if the line or tile buffer is
//			in NATIVE or XDR format.
//
//    typeInFrameBuffer the pixel data type of the frame buffer slice
//
//    typeInFile        the pixel data type in the input file's channel
//

void    copyIntoFrameBuffer (const char *&readPtr,
			     char *writePtr,
                             char *endPtr,
			     size_t xStride,
			     bool fill,
                             double fillValue,
			     Compressor::Format format,
                             PixelType typeInFrameBuffer,
                             PixelType typeInFile);

//
// Given a pointer into a an input file's line buffer or tile buffer,
// skip over the data for xSize pixels of type typeInFile.
// readPtr initially points to the beginning of the data to be skipped;
// when skipChannel() returns, readPtr points just past the end of the
// skipped data.
//

void    skipChannel (const char *&readPtr,
		     PixelType typeInFile,
		     size_t xSize);

//
// Convert an array of pixel data from the machine's native
// representation to XDR format.
//
//    toPtr, fromPtr	initially point to the beginning of the input
//			and output pixel data arrays; when convertInPlace()
//			returns, toPtr and fromPtr point just past the
//			end of the input and output arrays.
// 			If the native representation of the data has the
//			same size as the XDR data, then the conversion
//			can take in place, without an intermediate
//			temporary buffer (toPtr and fromPtr can point
//			to the same location).
//
//    type		the pixel data type
//
//    numPixels		number of pixels in the input and output arrays
// 

void    convertInPlace (char *&toPtr,
			const char *&fromPtr,
			PixelType type,
                        size_t numPixels);

//
// Copy a single channel of a horizontal row of pixels from a
// a frame buffer into an output file's internal line buffer or
// tile buffer.
//
//    writePtr		initially points to the beginning of the
//			data in the line or tile buffer. writePtr
//			is advanced as the pixel data are copied;
//			when copyFromFrameBuffer() returns,
//			writePtr points just past the end of the
//			copied data.
//
//    readPtr, endPtr	point to the lefmost and rightmost pixels
//			in the frame buffer slice
//
//    xStride		the xStride for the frame buffer slice
//
//    format		indicates if the line or tile buffer is
//			in NATIVE or XDR format.
//
//    type              the pixel data type in the frame buffer
//			and in the output file's channel (function
//			copyFromFrameBuffer() doesn't do on-the-fly
//			data type conversion)
//

void    copyFromFrameBuffer (char *&writePtr,
			     const char *&readPtr,
                             const char *endPtr,
			     size_t xStride,
                             Compressor::Format format,
			     PixelType type);

//
// Fill part of an output file's line buffer or tile buffer with
// zeroes.  This routine is called when an output file contains
// a channel for which the frame buffer contains no corresponding
// slice.
//
//    writePtr		initially points to the beginning of the
//			data in the line or tile buffer.  When
//			fillChannelWithZeroes() returns, writePtr
//			points just past the end of the zeroed
//			data.
//
//    format		indicates if the line or tile buffer is
//			in NATIVE or XDR format.
//
//    type              the pixel data type in the line or frame buffer.
//
//    xSize             number of pixels to be filled with zeroes.
//

void    fillChannelWithZeroes (char *&writePtr,
			       Compressor::Format format,
			       PixelType type,
			       size_t xSize);

} // namespace Imf

#endif
