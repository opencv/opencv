//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      Function dataWindowForFile()
//
//----------------------------------------------------------------------------

#include "ImfImageDataWindow.h"
#include "ImfImage.h"
#include <Iex.h>
#include <ImfHeader.h>
#include <algorithm>
#include <cassert>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

Box2i
dataWindowForFile (const Header& hdr, const Image& img, DataWindowSource dws)
{
    switch (dws)
    {
        case USE_IMAGE_DATA_WINDOW: return img.dataWindow ();

        case USE_HEADER_DATA_WINDOW:

        {
            if (img.levelMode () != ONE_LEVEL)
                throw ArgExc ("Cannot crop multi-resolution images.");

            const Box2i& hdw = hdr.dataWindow ();
            const Box2i& idw = img.dataWindow ();

            return Box2i (
                V2i (max (hdw.min.x, idw.min.x), max (hdw.min.y, idw.min.y)),
                V2i (min (hdw.max.x, idw.max.x), min (hdw.max.y, idw.max.y)));
        }

        default: throw ArgExc ("Unsupported DataWindowSource.");
    }
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
