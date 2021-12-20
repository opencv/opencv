//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_TILED_MISC_H
#define INCLUDED_IMF_TILED_MISC_H

//-----------------------------------------------------------------------------
//
//	Miscellaneous stuff related to tiled files
//
//-----------------------------------------------------------------------------

#include "ImfForward.h"

#include "ImfTileDescription.h"
#include <ImathBox.h>

#include <stdio.h>
#include <vector>


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

IMF_EXPORT 
int levelSize (int min, int max, int l, LevelRoundingMode rmode);

IMF_EXPORT 
IMATH_NAMESPACE::Box2i dataWindowForLevel (const TileDescription &tileDesc,
				 int minX, int maxX,
				 int minY, int maxY,
				 int lx, int ly);

IMF_EXPORT 
IMATH_NAMESPACE::Box2i dataWindowForTile (const TileDescription &tileDesc,
				int minX, int maxX,
				int minY, int maxY,
				int dx, int dy,
				int lx, int ly);

IMF_EXPORT 
size_t calculateBytesPerPixel (const Header &header);

//
// Calculate the count of bytes for each lines in range [minY, maxY],
// and pixels in range [minX, maxX].
// Data will be saved in bytesPerLine.
// sampleCountBase, sampleCountXStride and sampleCountYStride are
// used to get the sample count values.
//

IMF_EXPORT 
void calculateBytesPerLine (const Header &header,
                            char* sampleCountBase,
                            int sampleCountXStride,
                            int sampleCountYStride,
                            int minX, int maxX,
                            int minY, int maxY,
                            std::vector<int>& xOffsets,
                            std::vector<int>& yOffsets,
                            std::vector<uint64_t>& bytesPerLine);

IMF_EXPORT 
void precalculateTileInfo (const TileDescription& tileDesc,
			   int minX, int maxX,
			   int minY, int maxY,
			   int *&numXTiles, int *&numYTiles,
			   int &numXLevels, int &numYLevels);

IMF_EXPORT 
int getTiledChunkOffsetTableSize(const Header& header);


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
