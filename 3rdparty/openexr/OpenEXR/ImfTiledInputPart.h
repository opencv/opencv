//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMFTILEDINPUTPART_H_
#define IMFTILEDINPUTPART_H_

#include "ImfForward.h"

#include "ImfTileDescription.h"
#include <ImathBox.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//-----------------------------------------------------------------------------
// class TiledInputPart:
//
// Same interface as TiledInputFile. Please have a reference to TiledInputFile.
//-----------------------------------------------------------------------------

class IMF_EXPORT_TYPE TiledInputPart
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
