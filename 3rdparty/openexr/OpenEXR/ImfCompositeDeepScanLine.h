//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Weta Digital, Ltd and Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_COMPOSITEDEEPSCANLINE_H
#define INCLUDED_IMF_COMPOSITEDEEPSCANLINE_H

//-----------------------------------------------------------------------------
//
//	Class to composite deep samples into a frame buffer
//      Initialise with a deep input part or deep inputfile
//      (also supports multiple files and parts, and will
//       composite them together, as long as their sizes and channelmaps agree)
//
//      Then call setFrameBuffer, and readPixels, exactly as for reading
//      regular scanline images.
//
//      Restrictions - source file(s) must contain at least Z and alpha channels
//                   - if multiple files/parts are provided, sizes must match
//                   - all requested channels will be composited as premultiplied
//                   - only half and float channels can be requested
//
//      This object should not be considered threadsafe
//
//      The default compositing engine will give spurious results with overlapping
//      volumetric samples - you may derive from DeepCompositing class, override the
//      sort_pixel() and composite_pixel() functions, and pass an instance to
//      setCompositing().
//
//-----------------------------------------------------------------------------

#include "ImfForward.h"

#include <ImathBox.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class IMF_EXPORT_TYPE CompositeDeepScanLine
{
public:
    IMF_EXPORT
    CompositeDeepScanLine ();
    IMF_EXPORT
    virtual ~CompositeDeepScanLine ();

    /// set the source data as a part
    ///@note all parts must remain valid until after last interaction with DeepComp
    IMF_EXPORT
    void addSource (DeepScanLineInputPart* part);

    /// set the source data as a file
    ///@note all file must remain valid until after last interaction with DeepComp
    IMF_EXPORT
    void addSource (DeepScanLineInputFile* file);

    /////////////////////////////////////////
    //
    // set the frame buffer for output values
    // the buffers specified must be large enough
    // to handle the dataWindow()
    //
    /////////////////////////////////////////
    IMF_EXPORT
    void setFrameBuffer (const FrameBuffer& fr);

    /////////////////////////////////////////
    //
    // retrieve frameBuffer
    //
    ////////////////////////////////////////
    IMF_EXPORT
    const FrameBuffer& frameBuffer () const;

    //////////////////////////////////////////////////
    //
    // read scanlines start to end from the source(s)
    // storing the result in the frame buffer provided
    //
    //////////////////////////////////////////////////

    IMF_EXPORT
    void readPixels (int start, int end);

    IMF_EXPORT
    int sources () const; // return number of sources

    /////////////////////////////////////////////////
    //
    // retrieve the datawindow
    // If multiple parts are specified, this will
    // be the union of the dataWindow of all parts
    //
    ////////////////////////////////////////////////

    IMF_EXPORT
    const IMATH_NAMESPACE::Box2i& dataWindow () const;

    //
    // override default sorting/compositing operation
    // (otherwise an instance of the base class will be used)
    //

    IMF_EXPORT
    void setCompositing (DeepCompositing*);

    struct IMF_HIDDEN Data;


    //
    // set the maximum number of samples that will be composited.
    // If a single scanline has more samples, readPixels will throw
    // an exception. This mechanism prevents the library allocating
    // excessive memory to composite deep scanline images.
    // A value of 0 will cause deep compositing to be disabled entirely
    // A negative value disables the limit, allowing images with
    // arbitrarily large sample counts to be composited
    //
    IMF_EXPORT
    static void setMaximumSampleCount(int64_t sampleCount);

    IMF_EXPORT
    static int64_t getMaximumSampleCount();


private:
    struct Data* _Data;

    CompositeDeepScanLine (const CompositeDeepScanLine&) = delete;
    CompositeDeepScanLine& operator= (const CompositeDeepScanLine&) = delete;
    CompositeDeepScanLine (CompositeDeepScanLine&&)                 = delete;
    CompositeDeepScanLine& operator= (CompositeDeepScanLine&&) = delete;
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
