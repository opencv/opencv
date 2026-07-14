///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2011, Industrial Light & Magic, a division of Lucas
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

#ifndef IMFTILEDINPUTPART_H_
#define IMFTILEDINPUTPART_H_

#include "ImfMultiPartInputFile.h"
#include "ImfTiledInputFile.h"
#include "ImfNamespace.h"
#include "ImfForward.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//-----------------------------------------------------------------------------
// class TiledInputPart:
//
// Same interface as TiledInputFile. Please have a reference to TiledInputFile.
//-----------------------------------------------------------------------------

class TiledInputPart
{
    public:
        IMF_EXPORT
        TiledInputPart(MultiPartInputFile& multiPartFile, int partNumber);

        IMF_EXPORT
        const char *        fileName () const;
        IMF_EXPORT
        const Header &      header () const;
        IMF_EXPORT
        int                 version () const;
        IMF_EXPORT
        void                setFrameBuffer (const FrameBuffer &frameBuffer);
        IMF_EXPORT
        const FrameBuffer & frameBuffer () const;
        IMF_EXPORT
        bool                isComplete () const;
        IMF_EXPORT
        unsigned int        tileXSize () const;
        IMF_EXPORT
        unsigned int        tileYSize () const;
        IMF_EXPORT
        LevelMode           levelMode () const;
        IMF_EXPORT
        LevelRoundingMode   levelRoundingMode () const;
        IMF_EXPORT
        int                 numLevels () const;
        IMF_EXPORT
        int                 numXLevels () const;
        IMF_EXPORT
        int                 numYLevels () const;
        IMF_EXPORT
        bool                isValidLevel (int lx, int ly) const;
        IMF_EXPORT
        int                 levelWidth  (int lx) const;
        IMF_EXPORT
        int                 levelHeight (int ly) const;
        IMF_EXPORT
        int                 numXTiles (int lx = 0) const;
        IMF_EXPORT
        int                 numYTiles (int ly = 0) const;
        IMF_EXPORT
        IMATH_NAMESPACE::Box2i        dataWindowForLevel (int l = 0) const;
        IMF_EXPORT
        IMATH_NAMESPACE::Box2i        dataWindowForLevel (int lx, int ly) const;
        IMF_EXPORT
        IMATH_NAMESPACE::Box2i        dataWindowForTile (int dx, int dy, int l = 0) const;
        IMF_EXPORT
        IMATH_NAMESPACE::Box2i        dataWindowForTile (int dx, int dy,
                                               int lx, int ly) const;
        IMF_EXPORT
        void                readTile  (int dx, int dy, int l = 0);
        IMF_EXPORT
        void                readTile  (int dx, int dy, int lx, int ly);
        IMF_EXPORT
        void                readTiles (int dx1, int dx2, int dy1, int dy2,
                                       int lx, int ly);
        IMF_EXPORT
        void                readTiles (int dx1, int dx2, int dy1, int dy2,
                                       int l = 0);
        IMF_EXPORT
        void                rawTileData (int &dx, int &dy,
                                         int &lx, int &ly,
                                         const char *&pixelData,
                                         int &pixelDataSize);

    private:
        TiledInputFile* file;
      // for internal use - allow TiledOutputFile access to file for copyPixels
      friend class TiledOutputFile;
      
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif /* IMFTILEDINPUTPART_H_ */
