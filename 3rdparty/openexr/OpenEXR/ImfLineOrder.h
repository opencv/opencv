//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_LINE_ORDER_H
#define INCLUDED_IMF_LINE_ORDER_H

//-----------------------------------------------------------------------------
//
//	enum LineOrder
//
//-----------------------------------------------------------------------------
#include "ImfExport.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


enum IMF_EXPORT_ENUM LineOrder
{
    INCREASING_Y = 0,	// first scan line has lowest y coordinate

    DECREASING_Y = 1,	// first scan line has highest y coordinate

    RANDOM_Y = 2,       // only for tiled files; tiles are written
    			// in random order

    NUM_LINEORDERS	// number of different line orders
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif
