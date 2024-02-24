//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_IMAGE_DATA_WINDOW_H
#define INCLUDED_IMF_IMAGE_DATA_WINDOW_H

//----------------------------------------------------------------------------
//
//      enum DataWindowSource,
//      function dataWindowForFile()
//
//----------------------------------------------------------------------------

#include "ImfNamespace.h"
#include "ImfUtilExport.h"
#include <ImathBox.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

enum IMFUTIL_EXPORT_ENUM DataWindowSource
{
    USE_IMAGE_DATA_WINDOW,
    USE_HEADER_DATA_WINDOW
};

//
// Given the an image, i, an OpenEXR file header, h, and a data window
// source flag, d, dataWindowForFile(i,h,d) returns i.dataWindow() if d
// is USE_IMAGE_DATA_WINDOW, or the intersection of i.dataWindow() and
// h.dataWindow() if d is USE_HEADER_DATA_WINDOW.
//

class Image;
class Header;

IMFUTIL_EXPORT
IMATH_NAMESPACE::Box2i
dataWindowForFile (const Header& hdr, const Image& img, DataWindowSource dws);

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
