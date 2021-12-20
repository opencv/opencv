//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_SCAN_LINE_INPUT_FILE_H
#define INCLUDED_IMF_SCAN_LINE_INPUT_FILE_H

//-----------------------------------------------------------------------------
//
//	class ScanLineInputFile
//
//-----------------------------------------------------------------------------

#include "ImfForward.h"

#include "ImfGenericInputFile.h"
#include "ImfThreading.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


class IMF_EXPORT_TYPE ScanLineInputFile : public GenericInputFile
{
  public:

    //------------
    // Constructor
    //------------

    IMF_EXPORT
    ScanLineInputFile (const Header &header, OPENEXR_IMF_INTERNAL_NAMESPACE::IStream *is,
                       int numThreads = globalThreadCount());


    //-----------------------------------------
    // Destructor -- deallocates internal data
    // structures, but does not close the file.
    //-----------------------------------------

    IMF_EXPORT
    virtual ~ScanLineInputFile ();

    ScanLineInputFile (const ScanLineInputFile& other) = delete;
    ScanLineInputFile& operator = (const ScanLineInputFile& other) = delete;
    ScanLineInputFile (ScanLineInputFile&& other) = delete;
    ScanLineInputFile& operator = (ScanLineInputFile&& other) = delete;


    //------------------------
    // Access to the file name
    //------------------------

    IMF_EXPORT
    const char *	fileName () const;


    //--------------------------
    // Access to the file header
    //--------------------------

    IMF_EXPORT
    const Header &	header () const;


    //----------------------------------
    // Access to the file format version
    //----------------------------------

    IMF_EXPORT
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

    IMF_EXPORT
    void		setFrameBuffer (const FrameBuffer &frameBuffer);


    //-----------------------------------
    // Access to the current frame buffer
    //-----------------------------------

    IMF_EXPORT
    const FrameBuffer &	frameBuffer () const;


    //---------------------------------------------------------------
    // Check if the file is complete:
    //
    // isComplete() returns true if all pixels in the data window are
    // present in the input file, or false if any pixels are missing.
    // (Another program may still be busy writing the file, or file
    // writing may have been aborted prematurely.)
    //---------------------------------------------------------------

    IMF_EXPORT
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
    
    IMF_EXPORT
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

    IMF_EXPORT
    void		readPixels (int scanLine1, int scanLine2);
    IMF_EXPORT
    void		readPixels (int scanLine);


    //----------------------------------------------
    // Read a block of raw pixel data from the file,
    // without uncompressing it (this function is
    // used to implement OutputFile::copyPixels()).
    //----------------------------------------------

    IMF_EXPORT
    void		rawPixelData (int firstScanLine,
				      const char *&pixelData,
				      int &pixelDataSize);

   
    //----------------------------------------------
    // Read a scanline's worth of raw pixel data 
    // from the file, without uncompressing it, and 
    // store in an external buffer, pixelData. 
    // pixelData should be pre-allocated with space 
    // for pixelDataSize chars. 
    //
    // This function can be used to separate the 
    // reading of a raw scan line from the 
    // decompression of that scan line, for
    // example to allow multiple scan lines to be
    // decompressed in parallel by an application's
    // own threads, where it is not convenient to 
    // use the threading within the library.
    //----------------------------------------------

    IMF_EXPORT
    void                rawPixelDataToBuffer(int scanLine,
					     char *pixelData,
					     int &pixelDataSize) const;
    
  
    struct IMF_HIDDEN Data;

  private:

    Data *		_data;

    InputStreamMutex*   _streamData;

    IMF_HIDDEN ScanLineInputFile   (InputPartData* part);

    IMF_HIDDEN void initialize (const Header& header);

    friend class MultiPartInputFile;
    friend class InputFile;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif
