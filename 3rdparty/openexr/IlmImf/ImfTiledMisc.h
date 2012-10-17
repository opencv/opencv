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
#include <ImfHeader.h>
#include <stdio.h>

namespace Imf {

int levelSize (int min, int max, int l, LevelRoundingMode rmode);

Imath::Box2i dataWindowForLevel (const TileDescription &tileDesc,
                 int minX, int maxX,
                 int minY, int maxY,
                 int lx, int ly);

Imath::Box2i dataWindowForTile (const TileDescription &tileDesc,
                int minX, int maxX,
                int minY, int maxY,
                int dx, int dy,
                int lx, int ly);

size_t calculateBytesPerPixel (const Header &header);

void precalculateTileInfo (const TileDescription& tileDesc,
               int minX, int maxX,
               int minY, int maxY,
               int *&numXTiles, int *&numYTiles,
               int &numXLevels, int &numYLevels);


} // namespace Imf

#endif
