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



#ifndef INCLUDED_IMF_RGBA_FILE_H
#define INCLUDED_IMF_RGBA_FILE_H


//-----------------------------------------------------------------------------
//
//	Simplified RGBA image I/O
//
//	class RgbaOutputFile
//	class RgbaInputFile
//
//-----------------------------------------------------------------------------

#include <ImfHeader.h>
#include <ImfFrameBuffer.h>
#include <ImfRgba.h>
#include "ImathVec.h"
#include "ImathBox.h"
#include "half.h"
#include <ImfThreading.h>

namespace Imf {


class OutputFile;
class InputFile;
struct PreviewRgba;

//
// RGBA output file.
//

class RgbaOutputFile
{
  public:

    //---------------------------------------------------
    // Constructor -- header is constructed by the caller
    //---------------------------------------------------

    RgbaOutputFile (const char name[],
		    const Header &header,
		    RgbaChannels rgbaChannels = WRITE_RGBA,
                    int numThreads = globalThreadCount());


    //----------------------------------------------------
    // Constructor -- header is constructed by the caller,
    // file is opened by the caller, destructor will not
    // automatically close the file.
    //----------------------------------------------------

    RgbaOutputFile (OStream &os,
		    const Header &header,
		    RgbaChannels rgbaChannels = WRITE_RGBA,
                    int numThreads = globalThreadCount());


    //----------------------------------------------------------------
    // Constructor -- header data are explicitly specified as function
    // call arguments (empty dataWindow means "same as displayWindow")
    //----------------------------------------------------------------

    RgbaOutputFile (const char name[],
		    const Imath::Box2i &displayWindow,
		    const Imath::Box2i &dataWindow = Imath::Box2i(),
		    RgbaChannels rgbaChannels = WRITE_RGBA,
		    float pixelAspectRatio = 1,
		    const Imath::V2f screenWindowCenter = Imath::V2f (0, 0),
		    float screenWindowWidth = 1,
		    LineOrder lineOrder = INCREASING_Y,
		    Compression compression = PIZ_COMPRESSION,
                    int numThreads = globalThreadCount());


    //-----------------------------------------------
    // Constructor -- like the previous one, but both
    // the display window and the data window are
    // Box2i (V2i (0, 0), V2i (width - 1, height -1))
    //-----------------------------------------------

    RgbaOutputFile (const char name[],
		    int width,
		    int height,
		    RgbaChannels rgbaChannels = WRITE_RGBA,
		    float pixelAspectRatio = 1,
		    const Imath::V2f screenWindowCenter = Imath::V2f (0, 0),
		    float screenWindowWidth = 1,
		    LineOrder lineOrder = INCREASING_Y,
		    Compression compression = PIZ_COMPRESSION,
                    int numThreads = globalThreadCount());


    //-----------
    // Destructor
    //-----------

    virtual ~RgbaOutputFile ();


    //------------------------------------------------
    // Define a frame buffer as the pixel data source:
    // Pixel (x, y) is at address
    //
    //  base + x * xStride + y * yStride
    //
    //------------------------------------------------

    void			setFrameBuffer (const Rgba *base,
						size_t xStride,
						size_t yStride);


    //---------------------------------------------
    // Write pixel data (see class Imf::OutputFile)
    //---------------------------------------------

    void			writePixels (int numScanLines = 1);
    int				currentScanLine () const;


    //--------------------------
    // Access to the file header
    //--------------------------

    const Header &		header () const;
    const FrameBuffer &		frameBuffer () const;
    const Imath::Box2i &	displayWindow () const;
    const Imath::Box2i &	dataWindow () const;
    float			pixelAspectRatio () const;
    const Imath::V2f		screenWindowCenter () const;
    float			screenWindowWidth () const;
    LineOrder			lineOrder () const;
    Compression			compression () const;
    RgbaChannels		channels () const;


    // --------------------------------------------------------------------
    // Update the preview image (see Imf::OutputFile::updatePreviewImage())
    // --------------------------------------------------------------------

    void			updatePreviewImage (const PreviewRgba[]);


    //-----------------------------------------------------------------------
    // Rounding control for luminance/chroma images:
    //
    // If the output file contains luminance and chroma channels (WRITE_YC
    // or WRITE_YCA), then the the significands of the luminance and
    // chroma values are rounded to roundY and roundC bits respectively (see
    // function half::round()).  Rounding improves compression with minimal
    // image degradation, usually much less than the degradation caused by
    // chroma subsampling.  By default, roundY is 7, and roundC is 5.
    //
    // If the output file contains RGB channels or a luminance channel,
    // without chroma, then no rounding is performed.
    //-----------------------------------------------------------------------

    void			setYCRounding (unsigned int roundY,
					       unsigned int roundC);


    //----------------------------------------------------
    // Break a scan line -- for testing and debugging only
    // (see Imf::OutputFile::updatePreviewImage()
    //
    // Warning: Calling this function usually results in a
    // broken image file.  The file or parts of it may not
    // be readable, or the file may contain bad data.
    //
    //----------------------------------------------------

    void			breakScanLine  (int y,
						int offset,
						int length,
						char c);
  private:

    RgbaOutputFile (const RgbaOutputFile &);		  // not implemented
    RgbaOutputFile & operator = (const RgbaOutputFile &); // not implemented

    class ToYca;

    OutputFile *		_outputFile;
    ToYca *			_toYca;
};


//
// RGBA input file
//

class RgbaInputFile
{
  public:

    //-------------------------------------------------------
    // Constructor -- opens the file with the specified name,
    // destructor will automatically close the file.
    //-------------------------------------------------------

    RgbaInputFile (const char name[], int numThreads = globalThreadCount());


    //-----------------------------------------------------------
    // Constructor -- attaches the new RgbaInputFile object to a
    // file that has already been opened by the caller.
    // Destroying the RgbaInputFile object will not automatically
    // close the file.
    //-----------------------------------------------------------

    RgbaInputFile (IStream &is, int numThreads = globalThreadCount());


    //-----------
    // Destructor
    //-----------

    virtual ~RgbaInputFile ();

    //-----------------------------------------------------
    // Define a frame buffer as the pixel data destination:
    // Pixel (x, y) is at address
    //
    //  base + x * xStride + y * yStride
    //
    //-----------------------------------------------------

    void			setFrameBuffer (Rgba *base,
						size_t xStride,
						size_t yStride);


    //-------------------------------------------
    // Read pixel data (see class Imf::InputFile)
    //-------------------------------------------

    void			readPixels (int scanLine1, int scanLine2);
    void			readPixels (int scanLine);


    //--------------------------
    // Access to the file header
    //--------------------------

    const Header &		header () const;
    const FrameBuffer &		frameBuffer () const;
    const Imath::Box2i &	displayWindow () const;
    const Imath::Box2i &	dataWindow () const;
    float			pixelAspectRatio () const;
    const Imath::V2f		screenWindowCenter () const;
    float			screenWindowWidth () const;
    LineOrder			lineOrder () const;
    Compression			compression () const;
    RgbaChannels		channels () const;
    const char *                fileName () const;
    bool			isComplete () const;

    //----------------------------------
    // Access to the file format version
    //----------------------------------

    int				version () const;

  private:

    RgbaInputFile (const RgbaInputFile &);		  // not implemented
    RgbaInputFile & operator = (const RgbaInputFile &);   // not implemented

    class FromYca;

    InputFile *			_inputFile;
    FromYca *			_fromYca;
};


} // namespace Imf

#endif
