//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMFDEEPSCANLINEOUTPUTPART_H_
#define IMFDEEPSCANLINEOUTPUTPART_H_

#include "ImfDeepScanLineOutputFile.h"
#include "ImfMultiPartOutputFile.h"
#include "ImfNamespace.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class DeepScanLineOutputPart
{
  public:

    IMF_EXPORT
    DeepScanLineOutputPart(MultiPartOutputFile& multiPartFile, int partNumber);

    //------------------------
    // Access to the file name
    //------------------------

    IMF_EXPORT
    const char *        fileName () const;


    //--------------------------
    // Access to the file header
    //--------------------------

    IMF_EXPORT
    const Header &      header () const;


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

    IMF_EXPORT
    void                setFrameBuffer (const DeepFrameBuffer &frameBuffer);


    //-----------------------------------
    // Access to the current frame buffer
    //-----------------------------------

    IMF_EXPORT
    const DeepFrameBuffer & frameBuffer () const;


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

    IMF_EXPORT
    void                writePixels (int numScanLines = 1);


    //------------------------------------------------------------------
    // Access to the current scan line:
    //
    // currentScanLine() returns the y coordinate of the first scan line
    // that will be read from the current frame buffer during the next
    // call to writePixels().
    //
    // If header.lineOrder() == INCREASING_Y:
    //
    //  The current scan line before the first call to writePixels()
    //  is header().dataWindow().min.y.  After writing each scan line,
    //  the current scan line is incremented by 1.
    //
    // If header.lineOrder() == DECREASING_Y:
    //
    //  The current scan line before the first call to writePixels()
    //  is header().dataWindow().max.y.  After writing each scan line,
    //  the current scan line is decremented by 1.
    //
    //------------------------------------------------------------------

    IMF_EXPORT
    int                 currentScanLine () const;


    //--------------------------------------------------------------
    // Shortcut to copy all pixels from an InputFile into this file,
    // without uncompressing and then recompressing the pixel data.
    // This file's header must be compatible with the InputFile's
    // header:  The two header's "dataWindow", "compression",
    // "lineOrder" and "channels" attributes must be the same.
    //--------------------------------------------------------------

    IMF_EXPORT
    void                copyPixels (DeepScanLineInputFile &in);
    IMF_EXPORT
    void                copyPixels (DeepScanLineInputPart &in);


    //--------------------------------------------------------------
    // Updating the preview image:
    //
    // updatePreviewImage() supplies a new set of pixels for the
    // preview image attribute in the file's header.  If the header
    // does not contain a preview image, updatePreviewImage() throws
    // an IEX_NAMESPACE::LogicExc.
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

    IMF_EXPORT
    void                updatePreviewImage (const PreviewRgba newPixels[]);

  private:
      DeepScanLineOutputFile* file;
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif /* IMFDEEPSCANLINEOUTPUTPART_H_ */
