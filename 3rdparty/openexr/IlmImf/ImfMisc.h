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

#include "ImfPixelType.h"
#include "ImfCompressor.h"
#include "ImfArray.h"
#include "ImfNamespace.h"
#include "ImfExport.h"
#include "ImfForward.h"

#include <cstddef>
#include <vector>


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


//
// Return the size of a single value of the indicated type,
// in the machine's native format.
//

IMF_EXPORT
int	pixelTypeSize (PixelType type);


//
// Return the number of samples a channel with subsampling rate
// s has in the interval [a, b].  For example, a channel with
// subsampling rate 2 (and samples at 0, 2, 4, 6, 8, etc.) has
// 2 samples in the interval [1, 5] and three samples in the
// interval [2, 6].
//

IMF_EXPORT
int	numSamples (int s, int a, int b);


//
// Build a table that lists, for each scanline in a file's
// data window, how many bytes are required to store all
// pixels in all channels in that scanline (assuming that
// the pixel data are tightly packed).
//

IMF_EXPORT
size_t	bytesPerLineTable (const Header &header,
		           std::vector<size_t> &bytesPerLine);


//
// Get the sample count for pixel (x, y) using the array base
// pointer, xStride and yStride.
//

IMF_EXPORT
int&
sampleCount(char* base, int xStride, int yStride, int x, int y);


IMF_EXPORT
const int&
sampleCount(const char* base, int xStride, int yStride, int x, int y);

//
// Build a table that lists, for each scanline in a DEEP file's
// data window, how many bytes are required to store all
// pixels in all channels in scanlines ranged in [minY, maxY]
// (assuming that the pixel data are tightly packed).
//

IMF_EXPORT
size_t bytesPerDeepLineTable (const Header &header,
                              int minY, int maxY,
                              const char* base,
                              int xStride,
                              int yStride,
                              std::vector<size_t> &bytesPerLine);


//
// Build a table that lists, for each scanline in a DEEP file's
// data window, how many bytes are required to store all
// pixels in all channels in every scanline (assuming that
// the pixel data are tightly packed).
//

IMF_EXPORT
size_t bytesPerDeepLineTable (const Header &header,
                              char* base,
                              int xStride,
                              int yStride,
                              std::vector<size_t> &bytesPerLine);


//
// For scanline-based files, pixels are read or written in
// in multi-scanline blocks.  Internally, class OutputFile
// and class ScanLineInputFile store a block of scan lines
// in a "line buffer".  Function offsetInLineBufferTable()
// builds a table that lists, scanlines within range
// [scanline1, scanline2], the location of the pixel data
// for the scanline relative to the beginning of the line buffer,
// where scanline1 = 0 represents the first line in the DATA WINDOW.
// The one without specifying the range will make scanline1 = 0
// and scanline2 = bytesPerLine.size().
//

IMF_EXPORT
void    offsetInLineBufferTable (const std::vector<size_t> &bytesPerLine,
                                 int scanline1, int scanline2,
                                 int linesInLineBuffer,
                                 std::vector<size_t> &offsetInLineBuffer);

IMF_EXPORT
void	offsetInLineBufferTable (const std::vector<size_t> &bytesPerLine,
				 int linesInLineBuffer,
				 std::vector<size_t> &offsetInLineBuffer);

//
// For a scanline-based file, compute the range of scanlines
// that occupy the same line buffer as a given scanline, y.
// (minY is the minimum y coordinate of the file's data window.)
//

IMF_EXPORT int	lineBufferMinY (int y, int minY, int linesInLineBuffer);
IMF_EXPORT int	lineBufferMaxY (int y, int minY, int linesInLineBuffer);


//
// Return a compressor's data format (Compressor::NATIVE or Compressor::XDR).
// If compressor is 0, return Compressor::XDR.
//

IMF_EXPORT
Compressor::Format defaultFormat (Compressor *compressor);


//
// Return the number of scan lines a compressor wants to compress
// or uncompress at once.  If compressor is 0, return 1.
//

IMF_EXPORT
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

IMF_EXPORT
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
// Copy a single channel of a horizontal row of pixels from an
// input file's internal line buffer or tile buffer into a
// frame buffer slice.  If necessary, perform on-the-fly data
// type conversion.
//
//    readPtr             initially points to the beginning of the
//                        data in the line or tile buffer. readPtr
//                        is advanced as the pixel data are copied;
//                        when copyIntoFrameBuffer() returns,
//                        readPtr points just past the end of the
//                        copied data.
//
//    base                point to each pixel in the framebuffer
//
//    sampleCountBase,    provide the number of samples in each pixel
//    sampleCountXStride,
//    sampleCountYStride
//
//    y                   the scanline to copy. The coordinate is
//                        relative to the datawindow.min.y.
//
//    minX, maxX          used to indicate which pixels in the scanline
//                        will be copied.
//
//    xOffsetForSampleCount,    used to offset the sample count array
//    yOffsetForSampleCount,    and the base array.
//    xOffsetForData,
//    yOffsetForData
//
//    xStride             the xStride for the frame buffer slice
//
//    format              indicates if the line or tile buffer is
//                        in NATIVE or XDR format.
//
//    typeInFrameBuffer   the pixel data type of the frame buffer slice
//
//    typeInFile          the pixel data type in the input file's channel
//

IMF_EXPORT
void    copyIntoDeepFrameBuffer (const char *& readPtr,
                                 char * base,
                                 const char* sampleCountBase,
                                 ptrdiff_t sampleCountXStride,
                                 ptrdiff_t sampleCountYStride,
                                 int y, int minX, int maxX,
                                 int xOffsetForSampleCount,
                                 int yOffsetForSampleCount,
                                 int xOffsetForData,
                                 int yOffsetForData,
                                 ptrdiff_t xStride,
                                 ptrdiff_t xPointerStride,
                                 ptrdiff_t yPointerStride,
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

IMF_EXPORT
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

IMF_EXPORT
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

IMF_EXPORT
void    copyFromFrameBuffer (char *&writePtr,
			     const char *&readPtr,
                             const char *endPtr,
			     size_t xStride,
                             Compressor::Format format,
			     PixelType type);

//
// Copy a single channel of a horizontal row of pixels from a
// a frame buffer in a deep data file into an output file's
// internal line buffer or tile buffer.
//
//    writePtr                  initially points to the beginning of the
//                              data in the line or tile buffer. writePtr
//                              is advanced as the pixel data are copied;
//                              when copyFromDeepFrameBuffer() returns,
//                              writePtr points just past the end of the
//                              copied data.
//
//    base                      the start pointer of each pixel in this channel.
//                              It points to the real data in FrameBuffer.
//                              It is different for different channels.
//                              dataWindowMinX and dataWindowMinY are involved in
//                              locating for base.
//
//    sampleCountBase,          used to locate the position to get
//    sampleCountXStride,       the number of samples for each pixel.
//    sampleCountYStride        Used to determine how far we should
//                              read based on the pointer provided by base.
//
//    y                         the scanline to copy. If we are dealing
//                              with a tiled deep file, then probably a portion
//                              of the scanline is copied.
//
//    xMin, xMax                used to indicate which pixels in the scanline
//                              will be copied.
//
//    xOffsetForSampleCount,    used to offset the sample count array
//    yOffsetForSampleCount,    and the base array.
//    xOffsetForData,
//    yOffsetForData
//
//    xStride                   the xStride for the frame buffer slice
//
//    format                    indicates if the line or tile buffer is
//                              in NATIVE or XDR format.
//
//    type                      the pixel data type in the frame buffer
//                              and in the output file's channel (function
//                              copyFromFrameBuffer() doesn't do on-the-fly
//                              data type conversion)
//

IMF_EXPORT
void    copyFromDeepFrameBuffer (char *& writePtr,
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
                                 ptrdiff_t xStrideForData,
                                 ptrdiff_t yStrideForData,
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

IMF_EXPORT
void    fillChannelWithZeroes (char *&writePtr,
			       Compressor::Format format,
			       PixelType type,
			       size_t xSize);

IMF_EXPORT
bool usesLongNames (const Header &header);


//
// compute size of chunk offset table - if ignore_attribute set to true
// will compute from the image size and layout, rather than the attribute
// The default behaviour is to read the attribute
//

IMF_EXPORT
int getChunkOffsetTableSize(const Header& header,bool ignore_attribute=false);

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif
