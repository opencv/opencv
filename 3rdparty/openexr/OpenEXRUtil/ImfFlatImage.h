//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_FLAT_IMAGE_H
#define INCLUDED_IMF_FLAT_IMAGE_H

//----------------------------------------------------------------------------
//
//      class FlatImage
//
//      For an explanation of images, levels and channels,
//      see the comments in header file Image.h.
//
//----------------------------------------------------------------------------

#include "ImfFlatImageLevel.h"
#include "ImfImage.h"
#include "ImfUtilExport.h"

#include "ImfTileDescription.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class IMFUTIL_EXPORT_TYPE FlatImage : public Image
{
public:
    //
    // Constructors and destructor.
    // The default constructor constructs an image with an empty data
    // window level mode ONE_LEVEL and level rounding mode ROUND_DOWN.
    //

    IMFUTIL_EXPORT FlatImage ();

    IMFUTIL_EXPORT
    FlatImage (
        const IMATH_NAMESPACE::Box2i& dataWindow,
        LevelMode                     levelMode         = ONE_LEVEL,
        LevelRoundingMode             levelRoundingMode = ROUND_DOWN);

    IMFUTIL_EXPORT virtual ~FlatImage ();

    //
    // Accessing image levels by level number
    //

    IMFUTIL_EXPORT virtual FlatImageLevel&       level (int l = 0);
    IMFUTIL_EXPORT virtual const FlatImageLevel& level (int l = 0) const;

    IMFUTIL_EXPORT virtual FlatImageLevel&       level (int lx, int ly);
    IMFUTIL_EXPORT virtual const FlatImageLevel& level (int lx, int ly) const;

protected:
    IMFUTIL_EXPORT virtual FlatImageLevel*
    newLevel (int lx, int ly, const IMATH_NAMESPACE::Box2i& dataWindow);
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
