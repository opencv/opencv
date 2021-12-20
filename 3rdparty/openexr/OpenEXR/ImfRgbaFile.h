//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


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

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfHeader.h"
#include "ImfFrameBuffer.h"
#include "ImfRgba.h"

#include <ImathVec.h>
#include <ImathBox.h>
#include <half.h>
#include "ImfThreading.h"
#include <string>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//-------------------------------------------------------
// Utility to compute the origin-based pointer address
//
// With large offsets for the data window, the naive code
// can wrap around, especially on 32-bit machines.
// This can be used to avoid that
//-------------------------------------------------------

inline const Rgba *
ComputeBasePointer (
    const Rgba*                 ptr,
    const IMATH_NAMESPACE::V2i& origin,
    int64_t                     w,
    size_t                      xStride = 1,
    size_t                      yStride = 0)
{
    if (yStride == 0)
        yStride = w;
    int64_t offx = static_cast<int64_t> (origin.x);
    offx *= xStride;
    int64_t offy = static_cast<int64_t> (origin.y);
    offy *= yStride;
    return ptr - offx - offy;
}

inline const Rgba *
ComputeBasePointer (const Rgba* ptr, const IMATH_NAMESPACE::Box2i& dataWindow)
{
    return ComputeBasePointer (ptr, dataWindow.min,
                         static_cast<int64_t> (dataWindow.max.x) -
                          static_cast<int64_t> (dataWindow.min.x) + 1);
}

inline Rgba*
ComputeBasePointer (
    Rgba*                       ptr,
    const IMATH_NAMESPACE::V2i& origin,
    int64_t                     w,
    size_t                      xStride = 1,
    size_t                      yStride = 0)
{
    if (yStride == 0)
        yStride = w;
    int64_t offx = static_cast<int64_t> (origin.x);
    offx *= xStride;
    int64_t offy = static_cast<int64_t> (origin.y);
    offy *= yStride;
    return ptr - offx - offy;
}

inline Rgba*
ComputeBasePointer (Rgba* ptr, const IMATH_NAMESPACE::Box2i& dataWindow)
{
    return ComputeBasePointer (
        ptr,
        dataWindow.min,
        static_cast<int64_t> (dataWindow.max.x) -
            static_cast<int64_t> (dataWindow.min.x) + 1);
}

//
// RGBA output file.
//

class IMF_EXPORT_TYPE RgbaOutputFile
{
  public:

    //---------------------------------------------------
    // Constructor -- header is constructed by the caller
    //---------------------------------------------------

    IMF_EXPORT
    RgbaOutputFile (const char name[],
		    const Header &header,
		    RgbaChannels rgbaChannels = WRITE_RGBA,
                    int numThreads = globalThreadCount());


    //----------------------------------------------------
    // Constructor -- header is constructed by the caller,
    // file is opened by the caller, destructor will not
    // automatically close the file.
    //----------------------------------------------------

    IMF_EXPORT
    RgbaOutputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os,
		    const Header &header,
		    RgbaChannels rgbaChannels = WRITE_RGBA,
                    int numThreads = globalThreadCount());


    //----------------------------------------------------------------
    // Constructor -- header data are explicitly specified as function
    // call arguments (empty dataWindow means "same as displayWindow")
    //----------------------------------------------------------------

    IMF_EXPORT
    RgbaOutputFile (const char name[],
		    const IMATH_NAMESPACE::Box2i &displayWindow,
		    const IMATH_NAMESPACE::Box2i &dataWindow = IMATH_NAMESPACE::Box2i(),
		    RgbaChannels rgbaChannels = WRITE_RGBA,
		    float pixelAspectRatio = 1,
		    const IMATH_NAMESPACE::V2f screenWindowCenter = IMATH_NAMESPACE::V2f (0, 0),
		    float screenWindowWidth = 1,
		    LineOrder lineOrder = INCREASING_Y,
		    Compression compression = PIZ_COMPRESSION,
                    int numThreads = globalThreadCount());


    //-----------------------------------------------
    // Constructor -- like the previous one, but both
    // the display window and the data window are
    // Box2i (V2i (0, 0), V2i (width - 1, height -1))
    //-----------------------------------------------

    IMF_EXPORT
    RgbaOutputFile (const char name[],
		    int width,
		    int height,
		    RgbaChannels rgbaChannels = WRITE_RGBA,
		    float pixelAspectRatio = 1,
		    const IMATH_NAMESPACE::V2f screenWindowCenter = IMATH_NAMESPACE::V2f (0, 0),
		    float screenWindowWidth = 1,
		    LineOrder lineOrder = INCREASING_Y,
		    Compression compression = PIZ_COMPRESSION,
                    int numThreads = globalThreadCount());


    //-----------
    // Destructor
    //-----------

    IMF_EXPORT
    virtual ~RgbaOutputFile ();


    //------------------------------------------------
    // Define a frame buffer as the pixel data source:
    // Pixel (x, y) is at address
    //
    //  base + x * xStride + y * yStride
    //
    //------------------------------------------------

    IMF_EXPORT
    void			setFrameBuffer (const Rgba *base,
						size_t xStride,
						size_t yStride);


    //---------------------------------------------
    // Write pixel data (see class Imf::OutputFile)
    //---------------------------------------------

    IMF_EXPORT
    void			writePixels (int numScanLines = 1);
    IMF_EXPORT
    int				currentScanLine () const;


    //--------------------------
    // Access to the file header
    //--------------------------

    IMF_EXPORT
    const Header &		header () const;
    IMF_EXPORT
    const FrameBuffer &		frameBuffer () const;
    IMF_EXPORT
    const IMATH_NAMESPACE::Box2i &	displayWindow () const;
    IMF_EXPORT
    const IMATH_NAMESPACE::Box2i &	dataWindow () const;
    IMF_EXPORT
    float			pixelAspectRatio () const;
    IMF_EXPORT
    const IMATH_NAMESPACE::V2f		screenWindowCenter () const;
    IMF_EXPORT
    float			screenWindowWidth () const;
    IMF_EXPORT
    LineOrder			lineOrder () const;
    IMF_EXPORT
    Compression			compression () const;
    IMF_EXPORT
    RgbaChannels		channels () const;


    // --------------------------------------------------------------------
    // Update the preview image (see Imf::OutputFile::updatePreviewImage())
    // --------------------------------------------------------------------

    IMF_EXPORT
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

    IMF_EXPORT
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

    IMF_EXPORT
    void			breakScanLine  (int y,
						int offset,
						int length,
						char c);
  private:

    RgbaOutputFile (const RgbaOutputFile &) = delete;
    RgbaOutputFile & operator = (const RgbaOutputFile &) = delete;
    RgbaOutputFile (RgbaOutputFile &&) = delete;
    RgbaOutputFile & operator = (RgbaOutputFile &&) = delete;

    class IMF_HIDDEN ToYca;

    OutputFile *		_outputFile;
    ToYca *			_toYca;
};


//
// RGBA input file
//

class IMF_EXPORT_TYPE RgbaInputFile
{
  public:

    //-------------------------------------------------------
    // Constructor -- opens the file with the specified name,
    // destructor will automatically close the file.
    //-------------------------------------------------------

    IMF_EXPORT
    RgbaInputFile (const char name[], int numThreads = globalThreadCount());


    //-----------------------------------------------------------
    // Constructor -- attaches the new RgbaInputFile object to a
    // file that has already been opened by the caller.
    // Destroying the RgbaInputFile object will not automatically
    // close the file.
    //-----------------------------------------------------------

    IMF_EXPORT
    RgbaInputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int numThreads = globalThreadCount());


    //--------------------------------------------------------------
    // Constructors -- the same as the previous two, but the names
    // of the red, green, blue, alpha, luminance and chroma channels
    // are expected to be layerName.R, layerName.G, etc.
    //--------------------------------------------------------------

    IMF_EXPORT
    RgbaInputFile (const char name[],
		   const std::string &layerName,
		   int numThreads = globalThreadCount());

    IMF_EXPORT
    RgbaInputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,
		   const std::string &layerName,
		   int numThreads = globalThreadCount());


    //-----------
    // Destructor
    //-----------

    IMF_EXPORT
    virtual ~RgbaInputFile ();


    //-----------------------------------------------------
    // Define a frame buffer as the pixel data destination:
    // Pixel (x, y) is at address
    //
    //  base + x * xStride + y * yStride
    //
    //-----------------------------------------------------

    IMF_EXPORT
    void			setFrameBuffer (Rgba *base,
						size_t xStride,
						size_t yStride);


    //----------------------------------------------------------------
    // Switch to a different layer -- subsequent calls to readPixels()
    // will read channels layerName.R, layerName.G, etc.
    // After each call to setLayerName(), setFrameBuffer() must be
    // called at least once before the next call to readPixels().
    //----------------------------------------------------------------

    IMF_EXPORT
    void			setLayerName (const std::string &layerName);


    //-------------------------------------------
    // Read pixel data (see class Imf::InputFile)
    //-------------------------------------------

    IMF_EXPORT
    void			readPixels (int scanLine1, int scanLine2);
    IMF_EXPORT
    void			readPixels (int scanLine);


    //--------------------------
    // Access to the file header
    //--------------------------

    IMF_EXPORT
    const Header &		header () const;
    IMF_EXPORT
    const FrameBuffer &		frameBuffer () const;
    IMF_EXPORT
    const IMATH_NAMESPACE::Box2i &	displayWindow () const;
    IMF_EXPORT
    const IMATH_NAMESPACE::Box2i &	dataWindow () const;
    IMF_EXPORT
    float			pixelAspectRatio () const;
    IMF_EXPORT
    const IMATH_NAMESPACE::V2f		screenWindowCenter () const;
    IMF_EXPORT
    float			screenWindowWidth () const;
    IMF_EXPORT
    LineOrder			lineOrder () const;
    IMF_EXPORT
    Compression			compression () const;
    IMF_EXPORT
    RgbaChannels		channels () const;
    IMF_EXPORT
    const char *                fileName () const;
    IMF_EXPORT
    bool			isComplete () const;


    //----------------------------------
    // Access to the file format version
    //----------------------------------

    IMF_EXPORT
    int				version () const;

  private:

    RgbaInputFile (const RgbaInputFile &) = delete;
    RgbaInputFile & operator = (const RgbaInputFile &) = delete;
    RgbaInputFile (RgbaInputFile &&) = delete;
    RgbaInputFile & operator = (RgbaInputFile &&) = delete;

    class IMF_HIDDEN FromYca;

    InputFile *			_inputFile;
    FromYca *			_fromYca;
    std::string			_channelNamePrefix;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif
