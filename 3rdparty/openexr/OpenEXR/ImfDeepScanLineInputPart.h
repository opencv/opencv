//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMFDEEPSCANLINEINPUTPART_H_
#define IMFDEEPSCANLINEINPUTPART_H_

#include "ImfForward.h"

#include <cstdint>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class IMF_EXPORT_TYPE DeepScanLineInputPart
{
  public:

    IMF_EXPORT
    DeepScanLineInputPart(MultiPartInputFile& file, int partNumber);

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


    //----------------------------------
    // Access to the file format version
    //----------------------------------

    IMF_EXPORT
    int                 version () const;


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
    void                setFrameBuffer (const DeepFrameBuffer &frameBuffer);


    //-----------------------------------
    // Access to the current frame buffer
    //-----------------------------------

    IMF_EXPORT
    const DeepFrameBuffer & frameBuffer () const;


    //---------------------------------------------------------------
    // Check if the file is complete:
    //
    // isComplete() returns true if all pixels in the data window are
    // present in the input file, or false if any pixels are missing.
    // (Another program may still be busy writing the file, or file
    // writing may have been aborted prematurely.)
    //---------------------------------------------------------------

    IMF_EXPORT
    bool                isComplete () const;


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
    void                readPixels (int scanLine1, int scanLine2);
    IMF_EXPORT
    void                readPixels (int scanLine);
    IMF_EXPORT
    void                readPixels (const char * rawPixelData,const DeepFrameBuffer & frameBuffer,
                                    int scanLine1,int scanLine2) const;

    //----------------------------------------------
    // Read a block of raw pixel data from the file,
    // without uncompressing it (this function is
    // used to implement OutputFile::copyPixels()).
    //----------------------------------------------

    IMF_EXPORT
    void                rawPixelData (int firstScanLine,
                                      char * pixelData,
                                      uint64_t &pixelDataSize);
                             
                                      
    //-----------------------------------------------------------
    // Read pixel sample counts into a slice in the frame buffer.
    //
    // readPixelSampleCounts(s1, s2) reads all the counts of
    // pixel samples with y coordinates in the interval
    // [min (s1, s2), max (s1, s2)] from the file, and stores
    // them in the slice naming "sample count".
    //
    // Both s1 and s2 must be within the interval
    // [header().dataWindow().min.y, header.dataWindow().max.y]
    //
    // readPixelSampleCounts(s) calls readPixelSampleCounts(s,s).
    //-----------------------------------------------------------

    IMF_EXPORT
    void                readPixelSampleCounts(int scanline1,
                                              int scanline2);
    IMF_EXPORT
    void                readPixelSampleCounts(int scanline);
    
    IMF_EXPORT
    void                readPixelSampleCounts( const char * rawdata , const DeepFrameBuffer & frameBuffer,
                                               int scanLine1 , int scanLine2) const;
                                               
    IMF_EXPORT
    int                 firstScanLineInChunk(int y) const;
    IMF_EXPORT
    int                 lastScanLineInChunk (int y) const;
  private:
    DeepScanLineInputFile *file;
    
    // needed for copyPixels 
    friend class DeepScanLineOutputFile;
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif /* IMFDEEPSCANLINEINPUTPART_H_ */
