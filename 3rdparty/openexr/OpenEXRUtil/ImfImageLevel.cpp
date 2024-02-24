//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      class ImageLevel
//
//----------------------------------------------------------------------------

#include "ImfImageLevel.h"
#include <Iex.h>
#include <cassert>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

ImageLevel::ImageLevel (Image& image, int xLevelNumber, int yLevelNumber)
    : _image (image)
    , _xLevelNumber (xLevelNumber)
    , _yLevelNumber (yLevelNumber)
    , _dataWindow (Box2i (V2i (0, 0), V2i (-1, -1)))
{
    // empty
}

ImageLevel::~ImageLevel ()
{
    // empty
}

void
ImageLevel::resize (const Box2i& dataWindow)
{
    if (dataWindow.max.x < dataWindow.min.x - 1 ||
        dataWindow.max.y < dataWindow.min.y - 1)
    {
        THROW (
            ArgExc,
            "Cannot reset data window for image level to "
            "(" << dataWindow.min.x
                << ", " << dataWindow.min.y
                << ") - "
                   "("
                << dataWindow.max.x << ", " << dataWindow.max.y
                << "). "
                   "The new data window is invalid.");
    }

    _dataWindow = dataWindow;
}

void
ImageLevel::shiftPixels (int dx, int dy)
{
    _dataWindow.min.x += dx;
    _dataWindow.min.y += dy;
    _dataWindow.max.x += dx;
    _dataWindow.max.y += dy;
}

void
ImageLevel::throwChannelExists (const string& name) const
{
    THROW (
        ArgExc,
        "Cannot insert a new image channel with "
        "name \""
            << name
            << "\" into an image level. "
               "A channel with the same name exists already.");
}

void
ImageLevel::throwBadChannelName (const string& name) const
{
    THROW (
        ArgExc,
        "Attempt to access non-existent "
        "image channel \""
            << name << "\".");
}

void
ImageLevel::throwBadChannelNameOrType (const string& name) const
{
    THROW (
        ArgExc,
        "Image channel \"" << name
                           << "\" does not exist "
                              "or is not of the expected type.");
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
