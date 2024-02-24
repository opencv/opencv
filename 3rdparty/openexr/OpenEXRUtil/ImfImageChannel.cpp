//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      class ImageChannel
//
//----------------------------------------------------------------------------

#include "ImfImageChannel.h"
#include "ImfImageLevel.h"
#include <Iex.h>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

ImageChannel::ImageChannel (
    ImageLevel& level, int xSampling, int ySampling, bool pLinear)
    : _level (level)
    , _xSampling (xSampling)
    , _ySampling (ySampling)
    , _pLinear (pLinear)
    , _pixelsPerRow (0)
    , _pixelsPerColumn (0)
    , _numPixels (0)
{
    // empty
}

ImageChannel::~ImageChannel ()
{
    // empty
}

Channel
ImageChannel::channel () const
{
    return Channel (pixelType (), xSampling (), ySampling (), pLinear ());
}

void
ImageChannel::resize ()
{
    const Box2i& dataWindow = level ().dataWindow ();

    if (dataWindow.min.x % _xSampling || dataWindow.min.y % _ySampling)
    {
        throw ArgExc ("The minimum x and y coordinates of the data window "
                      "of an image level must be multiples of the x and y "
                      "subsampling factors of all channels in the image.");
    }

    int width  = dataWindow.max.x - dataWindow.min.x + 1;
    int height = dataWindow.max.y - dataWindow.min.y + 1;

    if (width % _xSampling || height % _ySampling)
    {
        throw ArgExc ("The width and height of the data window of an image "
                      "level must be multiples of the x and y subsampling "
                      "factors of all channels in the image.");
    }

    _pixelsPerRow    = width / _xSampling;
    _pixelsPerColumn = height / _ySampling;
    _numPixels       = _pixelsPerRow * _pixelsPerColumn;
}

void
ImageChannel::boundsCheck (int x, int y) const
{
    const Box2i& dataWindow = level ().dataWindow ();

    if (x < dataWindow.min.x || x > dataWindow.max.x || y < dataWindow.min.y ||
        y > dataWindow.max.y)
    {
        THROW (
            ArgExc,
            "Attempt to access a pixel at location "
            "(" << x
                << ", " << y
                << ") in an image whose data window is "
                   "("
                << dataWindow.min.x << ", " << dataWindow.min.y
                << ") - "
                   "("
                << dataWindow.max.x << ", " << dataWindow.max.y << ").");
    }

    if (x % _xSampling || y % _ySampling)
    {
        THROW (
            ArgExc,
            "Attempt to access a pixel at location "
            "(" << x
                << ", " << y
                << ") in a channel whose x and y sampling "
                   "rates are "
                << _xSampling << " and " << _ySampling
                << ".  The "
                   "pixel coordinates are not divisible by the sampling rates.");
    }
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
