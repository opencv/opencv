//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_TILE_DESCRIPTION_H
#define INCLUDED_IMF_TILE_DESCRIPTION_H

//-----------------------------------------------------------------------------
//
//	class TileDescription and enum LevelMode
//
//-----------------------------------------------------------------------------
#include "ImfExport.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


enum IMF_EXPORT_ENUM LevelMode
{
    ONE_LEVEL = 0,
    MIPMAP_LEVELS = 1,
    RIPMAP_LEVELS = 2,
    
    NUM_LEVELMODES	// number of different level modes
};


enum IMF_EXPORT_ENUM LevelRoundingMode
{
    ROUND_DOWN = 0,
    ROUND_UP = 1,

    NUM_ROUNDINGMODES	// number of different rounding modes
};


class IMF_EXPORT_TYPE TileDescription
{
  public:

    unsigned int	xSize;		// size of a tile in the x dimension
    unsigned int	ySize;		// size of a tile in the y dimension
    LevelMode		mode;
    LevelRoundingMode	roundingMode;

    TileDescription (unsigned int xs = 32,
		     unsigned int ys = 32,
                     LevelMode m = ONE_LEVEL,
		     LevelRoundingMode r = ROUND_DOWN)
    :
        xSize (xs),
	ySize (ys),
	mode (m),
	roundingMode (r)
    {
	// empty
    }

    bool
    operator == (const TileDescription &other) const
    {
	return xSize        == other.xSize &&
	       ySize        == other.ySize &&
	       mode         == other.mode &&
	       roundingMode == other.roundingMode;
    }
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif
