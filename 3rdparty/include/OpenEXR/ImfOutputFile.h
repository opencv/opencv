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



#ifndef INCLUDED_IMF_OUTPUT_FILE_H
#define INCLUDED_IMF_OUTPUT_FILE_H

//-----------------------------------------------------------------------------
//
//	class OutputFile
//
//-----------------------------------------------------------------------------

#include <ImfHeader.h>
#include <ImfFrameBuffer.h>
#include <ImfThreading.h>

namespace Imf {

class InputFile;
struct PreviewRgba;


class OutputFile
{
  public:

    //-----------------------------------------------------------
    // Constructor -- opens the file and writes the file header.
    // The file header is also copied into the OutputFile object,
    // and can later be accessed via the header() method.
    // Destroying this OutputFile object automatically closes
    // the file.
    //
    // numThreads determines the number of threads that will be
    // used to write the file (see ImfThreading.h).
    //-----------------------------------------------------------

    OutputFile (const char fileName[], const Header &header,
                int numThreads = globalThreadCount());


    //------------------------------------------------------------
    // Constructor -- attaches the new OutputFile object to a file
    // that has already been opened, and writes the file header.
    // The file header is also copied into the OutputFile object,
    // and can later be accessed via the header() method.
    // Destroying this OutputFile object does not automatically
    // close the file.
    //
    // numThreads determines the number of threads that will be
    // used to write the file (see ImfThreading.h).
    //------------------------------------------------------------

    OutputFile (OStream &os, const Header &header,
                int numThreads = globalThreadCount());


    //-------------------------------------------------
    // Destructor
    //
    // Destroying the OutputFile object before writing
    // all scan lines within the data window results in
    // an incomplete file.
    //-------------------------------------------------

    virtual ~OutputFile ();


    //------------------------
    // Access to the file name
    //------------------------

    const char *	fileName () const;


    //--------------------------
    // Access to the file header
    //--------------------------

    const Header &	header () const;


    //-------------------------------------------------------
    // Set the current frame buffer -- copies the FrameBuffer
    // object into the OutputFile object.
    //
    // The current frame buffer is the source of the pixel
    // data written to the file.  The current frame buffer
    // must be set at least once before writePixels() is
    // called.  The current frame buffer can be changed
    // after each call to writePixels.
    //-------------------------------------------------------

    void		setFrameBuffer (const FrameBuffer &frameBuffer);


    //-----------------------------------
    // Access to the current frame buffer
    //-----------------------------------

    const FrameBuffer &	frameBuffer () const;


    //-------------------------------------------------------------------
    // Write pixel data:
    //
    // writePixels(n) retrieves the next n scan lines worth of data from
    // the current frame buffer, starting with the scan line indicated by
    // currentScanLine(), and stores the data in the output file, and
    // progressing in the direction indicated by header.lineOrder().
    //
    // To produce a complete and correct file, exactly m scan lines must
    // be written, where m is equal to
    // header().dataWindow().max.y - header().dataWindow().min.y + 1.
    //-------------------------------------------------------------------

    void		writePixels (int numScanLines = 1);


    //------------------------------------------------------------------
    // Access to the current scan line:
    //
    // currentScanLine() returns the y coordinate of the first scan line
    // that will be read from the current frame buffer during the next
    // call to writePixels().
    //
    // If header.lineOrder() == INCREASING_Y:
    //
    //	The current scan line before the first call to writePixels()
    //  is header().dataWindow().min.y.  After writing each scan line,
    //  the current scan line is incremented by 1.
    //
    // If header.lineOrder() == DECREASING_Y:
    //
    //	The current scan line before the first call to writePixels()
    //  is header().dataWindow().max.y.  After writing each scan line,
    //  the current scan line is decremented by 1.
    //
    //------------------------------------------------------------------

    int			currentScanLine () const;


    //--------------------------------------------------------------
    // Shortcut to copy all pixels from an InputFile into this file,
    // without uncompressing and then recompressing the pixel data.
    // This file's header must be compatible with the InputFile's
    // header:  The two header's "dataWindow", "compression",
    // "lineOrder" and "channels" attributes must be the same.
    //--------------------------------------------------------------

    void		copyPixels (InputFile &in);


    //--------------------------------------------------------------
    // Updating the preview image:
    //
    // updatePreviewImage() supplies a new set of pixels for the
    // preview image attribute in the file's header.  If the header
    // does not contain a preview image, updatePreviewImage() throws
    // an Iex::LogicExc.
    //
    // Note: updatePreviewImage() is necessary because images are
    // often stored in a file incrementally, a few scan lines at a
    // time, while the image is being generated.  Since the preview
    // image is an attribute in the file's header, it gets stored in
    // the file as soon as the file is opened, but we may not know
    // what the preview image should look like until we have written
    // the last scan line of the main image.
    //
    //--------------------------------------------------------------

    void		updatePreviewImage (const PreviewRgba newPixels[]);


    //---------------------------------------------------------
    // Break a scan line -- for testing and debugging only:
    // 
    // breakScanLine(y,p,n,c) introduces an error into the
    // output file by writing n copies of character c, starting
    // p bytes from the beginning of the pixel data block that
    // contains scan line y.
    //
    // Warning: Calling this function usually results in a
    // broken image file.  The file or parts of it may not
    // be readable, or the file may contain bad data.
    //
    //---------------------------------------------------------

    void		breakScanLine  (int y, int offset, int length, char c);


    struct Data;

  private:

    OutputFile (const OutputFile &);			// not implemented
    OutputFile & operator = (const OutputFile &);	// not implemented

    void		initialize (const Header &header);

    Data *		_data;
};


} // namespace Imf

#endif
