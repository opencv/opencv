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


#ifndef INCLUDED_IMF_TILE_DESCRIPTION_H
#define INCLUDED_IMF_TILE_DESCRIPTION_H

//-----------------------------------------------------------------------------
//
//	class TileDescription and enum LevelMode
//
//-----------------------------------------------------------------------------

namespace Imf {


enum LevelMode
{
    ONE_LEVEL = 0,
    MIPMAP_LEVELS = 1,
    RIPMAP_LEVELS = 2,
    
    NUM_LEVELMODES	// number of different level modes
};


enum LevelRoundingMode
{
    ROUND_DOWN = 0,
    ROUND_UP = 1,

    NUM_ROUNDINGMODES	// number of different rounding modes
};


class TileDescription
{
  public:

    unsigned int    xSize;          // size of a tile in the x dimension
    unsigned int    ySize;          // size of a tile in the y dimension
    LevelMode       mode;
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


} // namespace Imf

#endif
