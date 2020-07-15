///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////


#ifndef INCLUDED_IMF_TILED_MISC_H
#define INCLUDED_IMF_TILED_MISC_H

//-----------------------------------------------------------------------------
//
//	Miscellaneous stuff related to tiled files
//
//-----------------------------------------------------------------------------

#include "ImathBox.h"
#include "ImfHeader.h"
#include "ImfNamespace.h"

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
                            std::vector<Int64>& bytesPerLine);

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
