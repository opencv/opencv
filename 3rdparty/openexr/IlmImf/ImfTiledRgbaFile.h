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


#ifndef INCLUDED_IMF_TILED_RGBA_FILE_H
#define INCLUDED_IMF_TILED_RGBA_FILE_H

//-----------------------------------------------------------------------------
//
//	Simplified RGBA image I/O for tiled files
//
//	class TiledRgbaOutputFile
//	class TiledRgbaInputFile
//
//-----------------------------------------------------------------------------

#include <ImfHeader.h>
#include <ImfFrameBuffer.h>
#include "ImathVec.h"
#include "ImathBox.h"
#include "half.h"
#include <ImfTileDescription.h>
#include <ImfRgba.h>
#include <ImfThreading.h>
#include <string>

namespace Imf {

class TiledOutputFile;
class TiledInputFile;
struct PreviewRgba;


//
// Tiled RGBA output file.
//

class TiledRgbaOutputFile
{
  public:

    //---------------------------------------------------
    // Constructor -- rgbaChannels, tileXSize, tileYSize,
    // levelMode, and levelRoundingMode overwrite the
    // channel list and tile description attribute in the
    // header that is passed as an argument to the
    // constructor.
    //---------------------------------------------------

    TiledRgbaOutputFile (const char name[],
			 const Header &header,
			 RgbaChannels rgbaChannels,
			 int tileXSize,
			 int tileYSize,
			 LevelMode mode,
			 LevelRoundingMode rmode = ROUND_DOWN,
                         int numThreads = globalThreadCount ());


    //---------------------------------------------------
    // Constructor -- like the previous one, but the new
    // TiledRgbaOutputFile is attached to a file that has
    // already been opened by the caller.  Destroying
    // TiledRgbaOutputFileObjects constructed with this
    // constructor does not automatically close the
    // corresponding files.
    //---------------------------------------------------

    TiledRgbaOutputFile (OStream &os,
			 const Header &header,
			 RgbaChannels rgbaChannels,
			 int tileXSize,
			 int tileYSize,
			 LevelMode mode,
			 LevelRoundingMode rmode = ROUND_DOWN,
                         int numThreads = globalThreadCount ());


    //------------------------------------------------------
    // Constructor -- header data are explicitly specified
    // as function call arguments (an empty dataWindow means
    // "same as displayWindow")
    //------------------------------------------------------

    TiledRgbaOutputFile (const char name[],
			 int tileXSize,
			 int tileYSize,
			 LevelMode mode,
			 LevelRoundingMode rmode,
			 const Imath::Box2i &displayWindow,
			 const Imath::Box2i &dataWindow = Imath::Box2i(),
			 RgbaChannels rgbaChannels = WRITE_RGBA,
			 float pixelAspectRatio = 1,
			 const Imath::V2f screenWindowCenter =
						    Imath::V2f (0, 0),
			 float screenWindowWidth = 1,
			 LineOrder lineOrder = INCREASING_Y,
			 Compression compression = ZIP_COMPRESSION,
                         int numThreads = globalThreadCount ());


    //-----------------------------------------------
    // Constructor -- like the previous one, but both
    // the display window and the data window are
    // Box2i (V2i (0, 0), V2i (width - 1, height -1))
    //-----------------------------------------------

    TiledRgbaOutputFile (const char name[],
			 int width,
			 int height,
			 int tileXSize,
			 int tileYSize,
			 LevelMode mode,
			 LevelRoundingMode rmode = ROUND_DOWN,
			 RgbaChannels rgbaChannels = WRITE_RGBA,
			 float pixelAspectRatio = 1,
			 const Imath::V2f screenWindowCenter =
						    Imath::V2f (0, 0),
			 float screenWindowWidth = 1,
			 LineOrder lineOrder = INCREASING_Y,
			 Compression compression = ZIP_COMPRESSION,
                         int numThreads = globalThreadCount ());


    virtual ~TiledRgbaOutputFile ();


    //------------------------------------------------
    // Define a frame buffer as the pixel data source:
    // Pixel (x, y) is at address
    //
    //  base + x * xStride + y * yStride
    //
    //------------------------------------------------

    void		setFrameBuffer (const Rgba *base,
					size_t xStride,
					size_t yStride);

    //--------------------------
    // Access to the file header
    //--------------------------

    const Header &		header () const;
    const FrameBuffer &		frameBuffer () const;
    const Imath::Box2i &	displayWindow () const;
    const Imath::Box2i &	dataWindow () const;
    float			pixelAspectRatio () const;
    const Imath::V2f		screenWindowCenter () const;
    float			screenWindowWidth () const;
    LineOrder			lineOrder () const;
    Compression			compression () const;
    RgbaChannels		channels () const;


    //----------------------------------------------------
    // Utility functions (same as in Imf::TiledOutputFile)
    //----------------------------------------------------

    unsigned int	tileXSize () const;
    unsigned int	tileYSize () const;
    LevelMode		levelMode () const;
    LevelRoundingMode	levelRoundingMode () const;

    int			numLevels () const;
    int			numXLevels () const;
    int			numYLevels () const;
    bool		isValidLevel (int lx, int ly) const;

    int			levelWidth  (int lx) const;
    int			levelHeight (int ly) const;

    int			numXTiles (int lx = 0) const;
    int			numYTiles (int ly = 0) const;

    Imath::Box2i	dataWindowForLevel (int l = 0) const;
    Imath::Box2i	dataWindowForLevel (int lx, int ly) const;

    Imath::Box2i	dataWindowForTile (int dx, int dy,
					   int l = 0) const;

    Imath::Box2i	dataWindowForTile (int dx, int dy,
					   int lx, int ly) const;

    //------------------------------------------------------------------
    // Write pixel data:
    //
    // writeTile(dx, dy, lx, ly) writes the tile with tile
    // coordinates (dx, dy), and level number (lx, ly) to
    // the file.
    //
    //   dx must lie in the interval [0, numXTiles(lx)-1]
    //   dy must lie in the interval [0, numYTiles(ly)-1]
    //
    //   lx must lie in the interval [0, numXLevels()-1]
    //   ly must lie in the inverval [0, numYLevels()-1]
    //
    // writeTile(dx, dy, level) is a convenience function
    // used for ONE_LEVEL and MIPMAP_LEVEL files.  It calls
    // writeTile(dx, dy, level, level).
    //
    // The two writeTiles(dx1, dx2, dy1, dy2, ...) functions allow
    // writing multiple tiles at once.  If multi-threading is used
    // multiple tiles are written concurrently.
    //
    // Pixels that are outside the pixel coordinate range for the tile's
    // level, are never accessed by writeTile().
    //
    // Each tile in the file must be written exactly once.
    //
    //------------------------------------------------------------------

    void		writeTile (int dx, int dy, int l = 0);
    void		writeTile (int dx, int dy, int lx, int ly);

    void		writeTiles (int dxMin, int dxMax, int dyMin, int dyMax,
                                    int lx, int ly);

    void		writeTiles (int dxMin, int dxMax, int dyMin, int dyMax,
                                    int l = 0);


    // -------------------------------------------------------------------------
    // Update the preview image (see Imf::TiledOutputFile::updatePreviewImage())
    // -------------------------------------------------------------------------

    void		updatePreviewImage (const PreviewRgba[]);


    //------------------------------------------------
    // Break a tile -- for testing and debugging only
    // (see Imf::TiledOutputFile::breakTile())
    //
    // Warning: Calling this function usually results
    // in a broken image file.  The file or parts of
    // it may not be readable, or the file may contain
    // bad data.
    //
    //------------------------------------------------

    void		breakTile  (int dx, int dy,
				    int lx, int ly,
				    int offset,
				    int length,
				    char c);
  private:

    //
    // Copy constructor and assignment are not implemented
    //

    TiledRgbaOutputFile (const TiledRgbaOutputFile &);	
    TiledRgbaOutputFile & operator = (const TiledRgbaOutputFile &);

    class ToYa;

    TiledOutputFile *            _outputFile;
    ToYa *			_toYa;
};



//
// Tiled RGBA input file
//

class TiledRgbaInputFile
{
  public:

    //--------------------------------------------------------
    // Constructor -- opens the file with the specified name.
    // Destroying TiledRgbaInputFile objects constructed with
    // this constructor automatically closes the corresponding
    // files.
    //--------------------------------------------------------

    TiledRgbaInputFile (const char name[],
                        int numThreads = globalThreadCount ());


    //-------------------------------------------------------
    // Constructor -- attaches the new TiledRgbaInputFile
    // object to a file that has already been opened by the
    // caller.
    // Destroying TiledRgbaInputFile objects constructed with
    // this constructor does not automatically close the
    // corresponding files.
    //-------------------------------------------------------

    TiledRgbaInputFile (IStream &is, int numThreads = globalThreadCount ());


    //------------------------------------------------------------
    // Constructors -- the same as the previous two, but the names
    // of the red, green, blue, alpha, and luminance channels are
    // expected to be layerName.R, layerName.G, etc.
    //------------------------------------------------------------

    TiledRgbaInputFile (const char name[],
		        const std::string &layerName,
		        int numThreads = globalThreadCount());

    TiledRgbaInputFile (IStream &is,
		        const std::string &layerName,
		        int numThreads = globalThreadCount());

    //-----------
    // Destructor
    //-----------

    virtual ~TiledRgbaInputFile ();


    //-----------------------------------------------------
    // Define a frame buffer as the pixel data destination:
    // Pixel (x, y) is at address
    //
    //  base + x * xStride + y * yStride
    //
    //-----------------------------------------------------

    void			setFrameBuffer (Rgba *base,
						size_t xStride,
						size_t yStride);

    //-------------------------------------------------------------------
    // Switch to a different layer -- subsequent calls to readTile()
    // and readTiles() will read channels layerName.R, layerName.G, etc.
    // After each call to setLayerName(), setFrameBuffer() must be called
    // at least once before the next call to readTile() or readTiles().
    //-------------------------------------------------------------------

    void			setLayerName (const std::string &layerName);


    //--------------------------
    // Access to the file header
    //--------------------------

    const Header &		header () const;
    const FrameBuffer &		frameBuffer () const;
    const Imath::Box2i &	displayWindow () const;
    const Imath::Box2i &	dataWindow () const;
    float			pixelAspectRatio () const;
    const Imath::V2f		screenWindowCenter () const;
    float			screenWindowWidth () const;
    LineOrder			lineOrder () const;
    Compression			compression () const;
    RgbaChannels		channels () const;
    const char *                fileName () const;
    bool			isComplete () const;

    //----------------------------------
    // Access to the file format version
    //----------------------------------

    int				version () const;


    //---------------------------------------------------
    // Utility functions (same as in Imf::TiledInputFile)
    //---------------------------------------------------

    unsigned int	tileXSize () const;
    unsigned int	tileYSize () const;
    LevelMode		levelMode () const;
    LevelRoundingMode	levelRoundingMode () const;

    int			numLevels () const;
    int			numXLevels () const;
    int			numYLevels () const;
    bool		isValidLevel (int lx, int ly) const;

    int			levelWidth  (int lx) const;
    int			levelHeight (int ly) const;

    int			numXTiles (int lx = 0) const;
    int			numYTiles (int ly = 0) const;

    Imath::Box2i	dataWindowForLevel (int l = 0) const;
    Imath::Box2i	dataWindowForLevel (int lx, int ly) const;

    Imath::Box2i	dataWindowForTile (int dx, int dy,
					   int l = 0) const;

    Imath::Box2i	dataWindowForTile (int dx, int dy,
					   int lx, int ly) const;
					   

    //----------------------------------------------------------------
    // Read pixel data:
    //
    // readTile(dx, dy, lx, ly) reads the tile with tile
    // coordinates (dx, dy), and level number (lx, ly),
    // and stores it in the current frame buffer.
    //
    //   dx must lie in the interval [0, numXTiles(lx)-1]
    //   dy must lie in the interval [0, numYTiles(ly)-1]
    //
    //   lx must lie in the interval [0, numXLevels()-1]
    //   ly must lie in the inverval [0, numYLevels()-1]
    //
    // readTile(dx, dy, level) is a convenience function used
    // for ONE_LEVEL and MIPMAP_LEVELS files.  It calls
    // readTile(dx, dy, level, level).
    //
    // The two readTiles(dx1, dx2, dy1, dy2, ...) functions allow
    // reading multiple tiles at once.  If multi-threading is used
    // multiple tiles are read concurrently.
    //
    // Pixels that are outside the pixel coordinate range for the
    // tile's level, are never accessed by readTile().
    //
    // Attempting to access a tile that is not present in the file
    // throws an InputExc exception.
    //
    //----------------------------------------------------------------

    void           	readTile (int dx, int dy, int l = 0);
    void           	readTile (int dx, int dy, int lx, int ly);

    void		readTiles (int dxMin, int dxMax,
                                   int dyMin, int dyMax, int lx, int ly);

    void		readTiles (int dxMin, int dxMax,
                                   int dyMin, int dyMax, int l = 0);

  private:

    //
    // Copy constructor and assignment are not implemented
    //

    TiledRgbaInputFile (const TiledRgbaInputFile &);
    TiledRgbaInputFile & operator = (const TiledRgbaInputFile &);

    class FromYa;

    TiledInputFile *	_inputFile;
    FromYa *		_fromYa;
    std::string		_channelNamePrefix;
};


} // namespace Imf

#endif
