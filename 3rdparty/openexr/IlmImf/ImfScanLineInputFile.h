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


#ifndef INCLUDED_IMF_SCAN_LINE_INPUT_FILE_H
#define INCLUDED_IMF_SCAN_LINE_INPUT_FILE_H

//-----------------------------------------------------------------------------
//
//	class ScanLineInputFile
//
//-----------------------------------------------------------------------------

#include "ImfHeader.h"
#include "ImfFrameBuffer.h"
#include "ImfThreading.h"
#include "ImfInputStreamMutex.h"
#include "ImfInputPartData.h"
#include "ImfGenericInputFile.h"
#include "ImfExport.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


class IMF_EXPORT ScanLineInputFile : public GenericInputFile
{
  public:

    //------------
    // Constructor
    //------------

    ScanLineInputFile (const Header &header, OPENEXR_IMF_INTERNAL_NAMESPACE::IStream *is,
                       int numThreads = globalThreadCount());


    //-----------------------------------------
    // Destructor -- deallocates internal data
    // structures, but does not close the file.
    //-----------------------------------------

    virtual ~ScanLineInputFile ();


    //------------------------
    // Access to the file name
    //------------------------

    const char *	fileName () const;


    //--------------------------
    // Access to the file header
    //--------------------------

    const Header &	header () const;


    //----------------------------------
    // Access to the file format version
    //----------------------------------

    int			version () const;


    //-----------------------------------------------------------
    // Set the current frame buffer -- copies the FrameBuffer
    // object into the InputFile object.
    //
    // The current frame buffer is the destination for the pixel
    // data read from the file.  The current frame buffer must be
    // set at least once before readPixels() is called.
    // The current frame buffer can be changed after each call
    // to readPixels().
    //-----------------------------------------------------------

    void		setFrameBuffer (const FrameBuffer &frameBuffer);


    //-----------------------------------
    // Access to the current frame buffer
    //-----------------------------------

    const FrameBuffer &	frameBuffer () const;


    //---------------------------------------------------------------
    // Check if the file is complete:
    //
    // isComplete() returns true if all pixels in the data window are
    // present in the input file, or false if any pixels are missing.
    // (Another program may still be busy writing the file, or file
    // writing may have been aborted prematurely.)
    //---------------------------------------------------------------

    bool		isComplete () const;

    
    
    //---------------------------------------------------------------
    // Check if SSE optimisation is enabled
    //
    // Call after setFrameBuffer() to query whether optimised file decoding
    // is available - decode times will be faster if returns true
    //
    // Optimisation depends on the framebuffer channels and channel types
    // as well as the file/part channels and channel types, as well as
    // whether SSE2 instruction support was detected at compile time
    //
    // Calling before setFrameBuffer will throw an exception
    //
    //---------------------------------------------------------------
    
    bool                isOptimizationEnabled () const;
    
    
    

    //---------------------------------------------------------------
    // Read pixel data:
    //
    // readPixels(s1,s2) reads all scan lines with y coordinates
    // in the interval [min (s1, s2), max (s1, s2)] from the file,
    // and stores them in the current frame buffer.
    //
    // Both s1 and s2 must be within the interval
    // [header().dataWindow().min.y, header.dataWindow().max.y]
    //
    // The scan lines can be read from the file in random order, and
    // individual scan lines may be skipped or read multiple times.
    // For maximum efficiency, the scan lines should be read in the
    // order in which they were written to the file.
    //
    // readPixels(s) calls readPixels(s,s).
    //
    // If threading is enabled, readPixels (s1, s2) tries to perform
    // decopmression of multiple scanlines in parallel.
    //
    //---------------------------------------------------------------

    void		readPixels (int scanLine1, int scanLine2);
    void		readPixels (int scanLine);


    //----------------------------------------------
    // Read a block of raw pixel data from the file,
    // without uncompressing it (this function is
    // used to implement OutputFile::copyPixels()).
    //----------------------------------------------

    void		rawPixelData (int firstScanLine,
				      const char *&pixelData,
				      int &pixelDataSize);

    struct Data;

  private:

    Data *		_data;

    InputStreamMutex*   _streamData;

    ScanLineInputFile   (InputPartData* part);

    void                initialize(const Header& header);

    friend class MultiPartInputFile;
    friend class InputFile;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif
