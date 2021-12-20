//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_RGBA_H
#define INCLUDED_IMF_RGBA_H

//-----------------------------------------------------------------------------
//
//	class Rgba
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include <half.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


//
// RGBA pixel
//

struct Rgba
{
    half	r;
    half	g;
    half	b;
    half	a;
    
    Rgba () {}
    Rgba (half r, half g, half b, half a = 1.f): r (r), g (g), b (b), a (a) {}
};


//
// Channels in an RGBA file
//

enum IMF_EXPORT_ENUM RgbaChannels
{
    WRITE_R	= 0x01,		// Red
    WRITE_G	= 0x02,		// Green
    WRITE_B	= 0x04,		// Blue
    WRITE_A	= 0x08,		// Alpha

    WRITE_Y	= 0x10,		// Luminance, for black-and-white images,
    				// or in combination with chroma

    WRITE_C	= 0x20,		// Chroma (two subsampled channels, RY and BY,
    				// supported only for scanline-based files)

    WRITE_RGB	= 0x07,		// Red, green, blue
    WRITE_RGBA	= 0x0f,		// Red, green, blue, alpha

    WRITE_YC	= 0x30,		// Luminance, chroma
    WRITE_YA	= 0x18,		// Luminance, alpha
    WRITE_YCA	= 0x38		// Luminance, chroma, alpha
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif
