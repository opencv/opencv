///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2007, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IMF_ACES_FILE_H
#define INCLUDED_IMF_ACES_FILE_H


//-----------------------------------------------------------------------------
//
//	ACES image file I/O.
//	
//	This header file declares two classes that directly support
//	image file input and output according to the Academy Image
//	Interchange Framework.
//	
//	The Academy Image Interchange file format is a subset of OpenEXR:
//	
//	    - Images are stored as scanlines.  Tiles are not allowed.
//	
//	    - Images contain three color channels, either
//		    R, G, B (red, green, blue) or
//		    Y, RY, BY (luminance, sub-sampled chroma)
//	
//	    - Images may optionally contain an alpha channel.
//	
//	    - Only three compression types are allowed:
//		    - NO_COMPRESSION (file is not compressed)
//		    - PIZ_COMPRESSION (lossless)
//		    - B44A_COMPRESSION (lossy)
//	
//	    - The "chromaticities" header attribute must specify
//	      the ACES RGB primaries and white point.
//	
//	class AcesOutputFile writes an OpenEXR file, enforcing the
//	restrictions listed above.  Pixel data supplied by application
//	software must already be in the ACES RGB space.
//	
//	class AcesInputFile reads an OpenEXR file.  Pixel data delivered
//	to application software is guaranteed to be in the ACES RGB space.
//	If the RGB space of the file is not the same as the ACES space,
//	then the pixels are automatically converted: the pixels are
//	converted to CIE XYZ, a color adaptation transform shifts the
//	white point, and the result is converted to ACES RGB.
//
//-----------------------------------------------------------------------------

#include <ImfHeader.h>
#include <ImfRgba.h>
#include "ImathVec.h"
#include "ImathBox.h"
#include <ImfThreading.h>
#include <string>

namespace Imf {


class RgbaOutputFile;
class RgbaInputFile;
struct PreviewRgba;
struct Chromaticities;

//
// ACES red, green, blue and white-point chromaticities.
//

const Chromaticities &	acesChromaticities ();


//
// ACES output file.
//

class AcesOutputFile
{
  public:

    //---------------------------------------------------
    // Constructor -- header is constructed by the caller
    //---------------------------------------------------

    AcesOutputFile (const std::string &name,
		    const Header &header,
		    RgbaChannels rgbaChannels = WRITE_RGBA,
                    int numThreads = globalThreadCount());


    //----------------------------------------------------
    // Constructor -- header is constructed by the caller,
    // file is opened by the caller, destructor will not
    // automatically close the file.
    //----------------------------------------------------

    AcesOutputFile (OStream &os,
		    const Header &header,
		    RgbaChannels rgbaChannels = WRITE_RGBA,
                    int numThreads = globalThreadCount());


    //----------------------------------------------------------------
    // Constructor -- header data are explicitly specified as function
    // call arguments (empty dataWindow means "same as displayWindow")
    //----------------------------------------------------------------

    AcesOutputFile (const std::string &name,
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

    AcesOutputFile (const std::string &name,
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

    virtual ~AcesOutputFile ();


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


    //-------------------------------------------------
    // Write pixel data (see class Imf::OutputFile)
    // The pixels are assumed to contain ACES RGB data.
    //-------------------------------------------------

    void			writePixels (int numScanLines = 1);
    int				currentScanLine () const;


    //--------------------------
    // Access to the file header
    //--------------------------

    const Header &		header () const;
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


  private:

    AcesOutputFile (const AcesOutputFile &);		  // not implemented
    AcesOutputFile & operator = (const AcesOutputFile &); // not implemented

    class Data;

    Data *			_data;
};


//
// ACES input file
//

class AcesInputFile
{
  public:

    //-------------------------------------------------------
    // Constructor -- opens the file with the specified name,
    // destructor will automatically close the file.
    //-------------------------------------------------------

    AcesInputFile (const std::string &name,
		   int numThreads = globalThreadCount());


    //-----------------------------------------------------------
    // Constructor -- attaches the new AcesInputFile object to a
    // file that has already been opened by the caller.
    // Destroying the AcesInputFile object will not automatically
    // close the file.
    //-----------------------------------------------------------

    AcesInputFile (IStream &is,
		   int numThreads = globalThreadCount());


    //-----------
    // Destructor
    //-----------

    virtual ~AcesInputFile ();


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


    //--------------------------------------------
    // Read pixel data (see class Imf::InputFile)
    // Pixels returned will contain ACES RGB data.
    //--------------------------------------------

    void			readPixels (int scanLine1, int scanLine2);
    void			readPixels (int scanLine);


    //--------------------------
    // Access to the file header
    //--------------------------

    const Header &		header () const;
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

    AcesInputFile (const AcesInputFile &);		  // not implemented
    AcesInputFile & operator = (const AcesInputFile &);   // not implemented

    class Data;

    Data *			_data;
};


} // namespace Imf

#endif
