//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      class FlatImage
//
//----------------------------------------------------------------------------

#include "ImfFlatImage.h"
#include <Iex.h>
#include <cassert>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

FlatImage::FlatImage () : Image ()
{
    resize (Box2i (V2i (0, 0), V2i (-1, -1)), ONE_LEVEL, ROUND_DOWN);
}

FlatImage::FlatImage (
    const Box2i&      dataWindow,
    LevelMode         levelMode,
    LevelRoundingMode levelRoundingMode)
    : Image ()
{
    resize (dataWindow, levelMode, levelRoundingMode);
}

FlatImage::~FlatImage ()
{
    // empty
}

FlatImageLevel&
FlatImage::level (int l)
{
    return static_cast<FlatImageLevel&> (Image::level (l));
}

const FlatImageLevel&
FlatImage::level (int l) const
{
    return static_cast<const FlatImageLevel&> (Image::level (l));
}

FlatImageLevel&
FlatImage::level (int lx, int ly)
{
    return static_cast<FlatImageLevel&> (Image::level (lx, ly));
}

const FlatImageLevel&
FlatImage::level (int lx, int ly) const
{
    return static_cast<const FlatImageLevel&> (Image::level (lx, ly));
}

FlatImageLevel*
FlatImage::newLevel (int lx, int ly, const Box2i& dataWindow)
{
    return new FlatImageLevel (*this, lx, ly, dataWindow);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
