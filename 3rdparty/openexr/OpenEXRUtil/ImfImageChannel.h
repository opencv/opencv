//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_IMAGE_CHANNEL_H
#define INCLUDED_IMF_IMAGE_CHANNEL_H

//----------------------------------------------------------------------------
//
//      class ImageChannel
//
//      For an explanation of images, levels and channels,
//      see the comments in header file Image.h.
//
//----------------------------------------------------------------------------

#include "ImfUtilExport.h"

#include "IexBaseExc.h"
#include "ImfChannelList.h"
#include "ImfFrameBuffer.h"
#include "ImfPixelType.h"
#include <ImathBox.h>
#include <half.h>

#include <cstring>
#include <typeinfo>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class ImageLevel;

//
// Image channels:
//
// An image channel holds the pixel data for a single channel of one level
// of an image.  Separate classes for flat and deep channels are derived
// from the ImageChannel base class.
//

class ImageLevel;

class IMFUTIL_EXPORT_TYPE ImageChannel
{
public:
    //
    // The OpenEXR pixel type of this channel (HALF, FLOAT or UINT).
    //

    virtual PixelType pixelType () const = 0;

    //
    // Generate an OpenEXR channel for this image channel.
    //

    IMFUTIL_EXPORT
    Channel channel () const;

    //
    // Access to x and y sampling rates, "perceptually linear" flag,
    // and the number of pixels that are stored in this channel.
    //

    int    xSampling () const { return _xSampling; }
    int    ySampling () const { return _ySampling; }
    bool   pLinear () const { return _pLinear; }
    int    pixelsPerRow () const { return _pixelsPerRow; }
    int    pixelsPerColumn () const { return _pixelsPerColumn; }
    size_t numPixels () const { return _numPixels; }

    //
    // Access to the image level to which this channel belongs.
    //

    ImageLevel&       level () { return _level; }
    const ImageLevel& level () const { return _level; }

protected:
    IMFUTIL_EXPORT
    ImageChannel (
        ImageLevel& level, int xSampling, int ySampling, bool pLinear);

    IMFUTIL_EXPORT
    virtual ~ImageChannel ();

    IMFUTIL_EXPORT
    virtual void resize ();

    IMFUTIL_EXPORT
    void boundsCheck (int x, int y) const;

private:
    ImageChannel (const ImageChannel&) = delete;
    ImageChannel& operator= (const ImageChannel&) = delete;
    ImageChannel (ImageChannel&&)                 = delete;
    ImageChannel& operator= (ImageChannel&&) = delete;

    ImageLevel& _level;
    int         _xSampling;
    int         _ySampling;
    bool        _pLinear;
    int         _pixelsPerRow;
    int         _pixelsPerColumn;
    size_t      _numPixels;
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
