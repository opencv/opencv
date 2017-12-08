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


#ifndef INCLUDED_IMF_TILE_OFFSETS_H
#define INCLUDED_IMF_TILE_OFFSETS_H

//-----------------------------------------------------------------------------
//
//	class TileOffsets
//
//-----------------------------------------------------------------------------

#include "ImfTileDescription.h"
#include "ImfInt64.h"
#include <vector>
#include "ImfNamespace.h"
#include "ImfForward.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


class IMF_EXPORT TileOffsets
{
  public:

    TileOffsets (LevelMode mode = ONE_LEVEL,
		 int numXLevels = 0,
		 int numYLevels = 0,
		 const int *numXTiles = 0,
		 const int *numYTiles = 0);    

    // --------
    // File I/O
    // --------

    void		readFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,  bool &complete,bool isMultiPart,bool isDeep);
    void                readFrom (std::vector<Int64> chunkOffsets,bool &complete);
    Int64		writeTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os) const;


    //-----------------------------------------------------------
    // Test if the tileOffsets array is empty (all entries are 0)
    //-----------------------------------------------------------

    bool		isEmpty () const;
    
    
    
    //-----------------------------------------------------------
    // populate 'list' with tiles coordinates in the order they appear
    // in the offset table (assumes full table!
    // each array myst be at leat totalTiles long
    //-----------------------------------------------------------
    void getTileOrder(int dx_table[], int dy_table[], int lx_table[], int ly_table[]) const;
    
    
    //-----------------------
    // Access to the elements
    //-----------------------

    Int64 &		operator () (int dx, int dy, int lx, int ly);
    Int64 &		operator () (int dx, int dy, int l);
    const Int64 &	operator () (int dx, int dy, int lx, int ly) const;
    const Int64 &	operator () (int dx, int dy, int l) const;
    bool        isValidTile (int dx, int dy, int lx, int ly) const;
    const std::vector<std::vector<std::vector <Int64> > >& getOffsets() const;
    
  private:

    void		findTiles (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, bool isMultiPartFile,
                                   bool isDeep,
        		           bool skipOnly);
    void		reconstructFromFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,bool isMultiPartFile,bool isDeep);
    bool		readTile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is);
    bool		anyOffsetsAreInvalid () const;

    LevelMode		_mode;
    int			_numXLevels;
    int			_numYLevels;

    std::vector<std::vector<std::vector <Int64> > > _offsets;
    
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT





#endif
