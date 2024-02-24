//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      class DeepImage
//
//----------------------------------------------------------------------------

#include "ImfDeepImage.h"
#include <Iex.h>
#include <cassert>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

DeepImage::DeepImage () : Image ()
{
    resize (Box2i (V2i (0, 0), V2i (-1, -1)), ONE_LEVEL, ROUND_DOWN);
}

DeepImage::DeepImage (
    const Box2i&      dataWindow,
    LevelMode         levelMode,
    LevelRoundingMode levelRoundingMode)
    : Image ()
{
    resize (dataWindow, levelMode, levelRoundingMode);
}

DeepImage::~DeepImage ()
{
    // empty
}

DeepImageLevel&
DeepImage::level (int l)
{
    return static_cast<DeepImageLevel&> (Image::level (l));
}

const DeepImageLevel&
DeepImage::level (int l) const
{
    return static_cast<const DeepImageLevel&> (Image::level (l));
}

DeepImageLevel&
DeepImage::level (int lx, int ly)
{
    return static_cast<DeepImageLevel&> (Image::level (lx, ly));
}

const DeepImageLevel&
DeepImage::level (int lx, int ly) const
{
    return static_cast<const DeepImageLevel&> (Image::level (lx, ly));
}

DeepImageLevel*
DeepImage::newLevel (int lx, int ly, const Box2i& dataWindow)
{
    return new DeepImageLevel (*this, lx, ly, dataWindow);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
