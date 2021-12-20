//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_PIXEL_TYPE_H
#define INCLUDED_IMF_PIXEL_TYPE_H

//-----------------------------------------------------------------------------
//
//	enum PixelType
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


enum IMF_EXPORT_ENUM PixelType
{
    UINT   = 0,		// unsigned int (32 bit)
    HALF   = 1,		// half (16 bit floating point)
    FLOAT  = 2,		// float (32 bit floating point)

    NUM_PIXELTYPES	// number of different pixel types
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif
