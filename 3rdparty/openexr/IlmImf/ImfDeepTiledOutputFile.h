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


#ifndef INCLUDED_IMF_DEEP_TILED_OUTPUT_FILE_H
#define INCLUDED_IMF_DEEP_TILED_OUTPUT_FILE_H

//-----------------------------------------------------------------------------
//
//      class DeepTiledOutputFile
//
//-----------------------------------------------------------------------------

#include "ImfHeader.h"
#include "ImfFrameBuffer.h"
#include "ImathBox.h"
#include "ImfThreading.h"
#include "ImfGenericOutputFile.h"
#include "ImfNamespace.h"
#include "ImfForward.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


class IMF_EXPORT DeepTiledOutputFile : public GenericOutputFile
{
  public:

    //-------------------------------------------------------------------
    // A constructor that opens the file with the specified name, and
    // writes the file header.  The file header is also copied into the
    // TiledOutputFile object, and can later be accessed via the header()
    // method.
    //
    // Destroying TiledOutputFile constructed with this constructor
    // automatically closes the corresponding files.
    //
    // The header must contain a TileDescriptionAttribute called "tiles".
    //
    // The x and y subsampling factors for all image channels must be 1;
    // subsampling is not supported.
    //
    // Tiles can be written to the file in arbitrary order.  The line
    // order attribute can be used to cause the tiles to be sorted in
    // the file.  When the file is read later, reading the tiles in the
    // same order as they are in the file tends to be significantly
    // faster than reading the tiles in random order (see writeTile,
    // below).
    //-------------------------------------------------------------------

    DeepTiledOutputFile (const char fileName[],
                         const Header &header,
                         int numThreads = globalThreadCount ());


    // ----------------------------------------------------------------
    // A constructor that attaches the new TiledOutputFile object to
    // a file that has already been opened.  Destroying TiledOutputFile
    // objects constructed with this constructor does not automatically
    // close the corresponding files.
    // ----------------------------------------------------------------

    DeepTiledOutputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os,
                         const Header &header,
                         int numThreads = globalThreadCount ());


    //-----------------------------------------------------
    // Destructor
    //
    // Destroying a TiledOutputFile object before all tiles
    // have been written results in an incomplete file.
    //-----------------------------------------------------

    virtual ~DeepTiledOutputFile ();


    //------------------------
    // Access to the file name
    //------------------------

    const char *        fileName () const;


    //--------------------------
    // Access to the file header
    //--------------------------

    const Header &      header () const;


    //-------------------------------------------------------
    // Set the current frame buffer -- copies the FrameBuffer
    // object into the TiledOutputFile object.
    //
    // The current frame buffer is the source of the pixel
    // data written to the file.  The current frame buffer
    // must be set at least once before writeTile() is
    // called.  The current frame buffer can be changed
    // after each call to writeTile().
    //-------------------------------------------------------

    void                setFrameBuffer (const DeepFrameBuffer &frameBuffer);


    //-----------------------------------
    // Access to the current frame buffer
    //-----------------------------------

    const DeepFrameBuffer & frameBuffer () const;


    //-------------------
    // Utility functions:
    //-------------------

    //---------------------------------------------------------
    // Multiresolution mode and tile size:
    // The following functions return the xSize, ySize and mode
    // fields of the file header's TileDescriptionAttribute.
    //---------------------------------------------------------

    unsigned int        tileXSize () const;
    unsigned int        tileYSize () const;
    LevelMode           levelMode () const;
    LevelRoundingMode   levelRoundingMode () const;


    //--------------------------------------------------------------------
    // Number of levels:
    //
    // numXLevels() returns the file's number of levels in x direction.
    //
    //  if levelMode() == ONE_LEVEL:
    //      return value is: 1
    //
    //  if levelMode() == MIPMAP_LEVELS:
    //      return value is: rfunc (log (max (w, h)) / log (2)) + 1
    //
    //  if levelMode() == RIPMAP_LEVELS:
    //      return value is: rfunc (log (w) / log (2)) + 1
    //
    //  where
    //      w is the width of the image's data window,  max.x - min.x + 1,
    //      y is the height of the image's data window, max.y - min.y + 1,
    //      and rfunc(x) is either floor(x), or ceil(x), depending on
    //      whether levelRoundingMode() returns ROUND_DOWN or ROUND_UP.
    //
    // numYLevels() returns the file's number of levels in y direction.
    //
    //  if levelMode() == ONE_LEVEL or levelMode() == MIPMAP_LEVELS:
    //      return value is the same as for numXLevels()
    //
    //  if levelMode() == RIPMAP_LEVELS:
    //      return value is: rfunc (log (h) / log (2)) + 1
    //
    //
    // numLevels() is a convenience function for use with MIPMAP_LEVELS
    // files.
    //
    //  if levelMode() == ONE_LEVEL or levelMode() == MIPMAP_LEVELS:
    //      return value is the same as for numXLevels()
    //
    //  if levelMode() == RIPMAP_LEVELS:
    //      an IEX_NAMESPACE::LogicExc exception is thrown
    //
    // isValidLevel(lx, ly) returns true if the file contains
    // a level with level number (lx, ly), false if not.
    //
    //--------------------------------------------------------------------

    int                 numLevels () const;
    int                 numXLevels () const;
    int                 numYLevels () const;
    bool                isValidLevel (int lx, int ly) const;


    //---------------------------------------------------------
    // Dimensions of a level:
    //
    // levelWidth(lx) returns the width of a level with level
    // number (lx, *), where * is any number.
    //
    //  return value is:
    //      max (1, rfunc (w / pow (2, lx)))
    //
    //
    // levelHeight(ly) returns the height of a level with level
    // number (*, ly), where * is any number.
    //
    //  return value is:
    //      max (1, rfunc (h / pow (2, ly)))
    //
    //---------------------------------------------------------

    int                 levelWidth  (int lx) const;
    int                 levelHeight (int ly) const;


    //----------------------------------------------------------
    // Number of tiles:
    //
    // numXTiles(lx) returns the number of tiles in x direction
    // that cover a level with level number (lx, *), where * is
    // any number.
    //
    //  return value is:
    //      (levelWidth(lx) + tileXSize() - 1) / tileXSize()
    //
    //
    // numYTiles(ly) returns the number of tiles in y direction
    // that cover a level with level number (*, ly), where * is
    // any number.
    //
    //  return value is:
    //      (levelHeight(ly) + tileXSize() - 1) / tileXSize()
    //
    //----------------------------------------------------------

    int                 numXTiles (int lx = 0) const;
    int                 numYTiles (int ly = 0) const;


    //---------------------------------------------------------
    // Level pixel ranges:
    //
    // dataWindowForLevel(lx, ly) returns a 2-dimensional
    // region of valid pixel coordinates for a level with
    // level number (lx, ly)
    //
    //  return value is a Box2i with min value:
    //      (dataWindow.min.x, dataWindow.min.y)
    //
    //  and max value:
    //      (dataWindow.min.x + levelWidth(lx) - 1,
    //       dataWindow.min.y + levelHeight(ly) - 1)
    //
    // dataWindowForLevel(level) is a convenience function used
    // for ONE_LEVEL and MIPMAP_LEVELS files.  It returns
    // dataWindowForLevel(level, level).
    //
    //---------------------------------------------------------

    IMATH_NAMESPACE::Box2i        dataWindowForLevel (int l = 0) const;
    IMATH_NAMESPACE::Box2i        dataWindowForLevel (int lx, int ly) const;


    //-------------------------------------------------------------------
    // Tile pixel ranges:
    //
    // dataWindowForTile(dx, dy, lx, ly) returns a 2-dimensional
    // region of valid pixel coordinates for a tile with tile coordinates
    // (dx,dy) and level number (lx, ly).
    //
    //  return value is a Box2i with min value:
    //      (dataWindow.min.x + dx * tileXSize(),
    //       dataWindow.min.y + dy * tileYSize())
    //
    //  and max value:
    //      (dataWindow.min.x + (dx + 1) * tileXSize() - 1,
    //       dataWindow.min.y + (dy + 1) * tileYSize() - 1)
    //
    // dataWindowForTile(dx, dy, level) is a convenience function
    // used for ONE_LEVEL and MIPMAP_LEVELS files.  It returns
    // dataWindowForTile(dx, dy, level, level).
    //
    //-------------------------------------------------------------------

    IMATH_NAMESPACE::Box2i        dataWindowForTile (int dx, int dy,
                                           int l = 0) const;

    IMATH_NAMESPACE::Box2i        dataWindowForTile (int dx, int dy,
                                           int lx, int ly) const;

    //------------------------------------------------------------------
    // Write pixel data:
    //
    // writeTile(dx, dy, lx, ly) writes the tile with tile
    // coordinates (dx, dy), and level number (lx, ly) to
    // the file.
    //
    //   dx must lie in the interval [0, numXTiles(lx) - 1]
    //   dy must lie in the interval [0, numYTiles(ly) - 1]
    //
    //   lx must lie in the interval [0, numXLevels() - 1]
    //   ly must lie in the inverval [0, numYLevels() - 1]
    //
    // writeTile(dx, dy, level) is a convenience function
    // used for ONE_LEVEL and MIPMAP_LEVEL files.  It calls
    // writeTile(dx, dy, level, level).
    //
    // The two writeTiles(dx1, dx2, dy1, dy2, ...) functions allow
    // writing multiple tiles at once.  If multi-threading is used
    // multiple tiles are written concurrently.  The tile coordinates,
    // dx1, dx2 and dy1, dy2, specify inclusive ranges of tile
    // coordinates.  It is valid for dx1 < dx2 or dy1 < dy2; the
    // tiles are always written in the order specified by the line
    // order attribute.  Hence, it is not possible to specify an
    // "invalid" or empty tile range.
    //
    // Pixels that are outside the pixel coordinate range for the tile's
    // level, are never accessed by writeTile().
    //
    // Each tile in the file must be written exactly once.
    //
    // The file's line order attribute determines the order of the tiles
    // in the file:
    //
    //   INCREASING_Y   In the file, the tiles for each level are stored
    //                  in a contiguous block.  The levels are ordered
    //                  like this:
    //
    //                      (0, 0)   (1, 0)   ... (nx-1, 0)
    //                      (0, 1)   (1, 1)   ... (nx-1, 1)
    //                       ...
    //                      (0,ny-1) (1,ny-1) ... (nx-1,ny-1)
    //
    //                  where nx = numXLevels(), and ny = numYLevels().
    //                  In an individual level, (lx, ly), the tiles
    //                  are stored in the following order:
    //
    //                      (0, 0)   (1, 0)   ... (tx-1, 0)
    //                      (0, 1)   (1, 1)   ... (tx-1, 1)
    //                       ...
    //                      (0,ty-1) (1,ty-1) ... (tx-1,ty-1)
    //
    //                  where tx = numXTiles(lx),
    //                  and   ty = numYTiles(ly).
    //
    //   DECREASING_Y   As for INCREASING_Y, the tiles for each level
    //                  are stored in a contiguous block.  The levels
    //                  are ordered the same way as for INCREASING_Y,
    //                  but within an individual level, the tiles
    //                  are stored in this order:
    //
    //                      (0,ty-1) (1,ty-1) ... (tx-1,ty-1)
    //                       ...
    //                      (0, 1)   (1, 1)   ... (tx-1, 1)
    //                      (0, 0)   (1, 0)   ... (tx-1, 0)
    //
    //
    //   RANDOM_Y       The order of the calls to writeTile() determines
    //                  the order of the tiles in the file.
    //
    //------------------------------------------------------------------

    void                writeTile  (int dx, int dy, int l = 0);
    void                writeTile  (int dx, int dy, int lx, int ly);

    void                writeTiles (int dx1, int dx2, int dy1, int dy2,
                                    int lx, int ly);

    void                writeTiles (int dx1, int dx2, int dy1, int dy2,
                                    int l = 0);


    //------------------------------------------------------------------
    // Shortcut to copy all pixels from a TiledInputFile into this file,
    // without uncompressing and then recompressing the pixel data.
    // This file's header must be compatible with the TiledInputFile's
    // header:  The two header's "dataWindow", "compression",
    // "lineOrder", "channels", and "tiles" attributes must be the same.
    //------------------------------------------------------------------

    void                copyPixels (DeepTiledInputFile &in);
    void                copyPixels (DeepTiledInputPart &in);



    //--------------------------------------------------------------
    // Updating the preview image:
    //
    // updatePreviewImage() supplies a new set of pixels for the
    // preview image attribute in the file's header.  If the header
    // does not contain a preview image, updatePreviewImage() throws
    // an IEX_NAMESPACE::LogicExc.
    //
    // Note: updatePreviewImage() is necessary because images are
    // often stored in a file incrementally, a few tiles at a time,
    // while the image is being generated.  Since the preview image
    // is an attribute in the file's header, it gets stored in the
    // file as soon as the file is opened, but we may not know what
    // the preview image should look like until we have written the
    // last tile of the main image.
    //
    //--------------------------------------------------------------

    void                updatePreviewImage (const PreviewRgba newPixels[]);


    //-------------------------------------------------------------
    // Break a tile -- for testing and debugging only:
    //
    // breakTile(dx,dy,lx,ly,p,n,c) introduces an error into the
    // output file by writing n copies of character c, starting
    // p bytes from the beginning of the tile with tile coordinates
    // (dx, dy) and level number (lx, ly).
    //
    // Warning: Calling this function usually results in a broken
    // image file.  The file or parts of it may not be readable,
    // or the file may contain bad data.
    //
    //-------------------------------------------------------------

    void                breakTile  (int dx, int dy,
                                    int lx, int ly,
                                    int offset,
                                    int length,
                                    char c);
    struct Data;

  private:

    // ----------------------------------------------------------------
    // A constructor attaches the OutputStreamMutex to the
    // given one from MultiPartOutputFile. Set the previewPosition
    // and lineOffsetsPosition which have been acquired from
    // the constructor of MultiPartOutputFile as well.
    // ----------------------------------------------------------------
    DeepTiledOutputFile (const OutputPartData* part);

    DeepTiledOutputFile (const DeepTiledOutputFile &);              // not implemented
    DeepTiledOutputFile & operator = (const DeepTiledOutputFile &); // not implemented

    void                initialize (const Header &header);

    bool                isValidTile (int dx, int dy,
                                     int lx, int ly) const;

    size_t              bytesPerLineForTile (int dx, int dy,
                                             int lx, int ly) const;

    Data *              _data;


    friend class MultiPartOutputFile;

};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
