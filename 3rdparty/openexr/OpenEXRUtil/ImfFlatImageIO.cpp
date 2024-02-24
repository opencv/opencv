//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      OpenEXR file I/O for flat images.
//
//----------------------------------------------------------------------------

#include "ImfFlatImageIO.h"
#include <Iex.h>
#include <ImfChannelList.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfTestFile.h>
#include <ImfTiledInputFile.h>
#include <ImfTiledOutputFile.h>
#include <cassert>
#include <cstring>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

void
saveFlatImage (
    const string&    fileName,
    const Header&    hdr,
    const FlatImage& img,
    DataWindowSource dws)
{
    if (img.levelMode () != ONE_LEVEL || hdr.hasTileDescription ())
        saveFlatTiledImage (fileName, hdr, img, dws);
    else
        saveFlatScanLineImage (fileName, hdr, img, dws);
}

void
saveFlatImage (const string& fileName, const FlatImage& img)
{
    Header hdr;
    hdr.displayWindow () = img.dataWindow ();
    saveFlatImage (fileName, hdr, img);
}

void
loadFlatImage (const string& fileName, Header& hdr, FlatImage& img)
{
    bool tiled, deep, multiPart;

    if (!isOpenExrFile (fileName.c_str (), tiled, deep, multiPart))
    {
        THROW (
            ArgExc,
            "Cannot load image file " << fileName
                                      << ".  "
                                         "The file is not an OpenEXR file.");
    }

    if (multiPart)
    {
        THROW (
            ArgExc,
            "Cannot load image file "
                << fileName
                << ".  "
                   "Multi-part file loading is not supported.");
    }

    if (deep)
    {
        THROW (
            ArgExc,
            "Cannot load deep image file " << fileName
                                           << " "
                                              "as a flat image.");
    }

    if (tiled)
        loadFlatTiledImage (fileName, hdr, img);
    else
        loadFlatScanLineImage (fileName, hdr, img);
}

void
loadFlatImage (const string& fileName, FlatImage& img)
{
    Header hdr;
    loadFlatImage (fileName, hdr, img);
}

void
saveFlatScanLineImage (
    const string&    fileName,
    const Header&    hdr,
    const FlatImage& img,
    DataWindowSource dws)
{
    Header newHdr;

    for (Header::ConstIterator i = hdr.begin (); i != hdr.end (); ++i)
    {
        if (strcmp (i.name (), "dataWindow") && strcmp (i.name (), "tiles") &&
            strcmp (i.name (), "channels"))
        {
            newHdr.insert (i.name (), i.attribute ());
        }
    }

    newHdr.dataWindow () = dataWindowForFile (hdr, img, dws);

    const FlatImageLevel& level = img.level ();
    FrameBuffer           fb;

    for (FlatImageLevel::ConstIterator i = level.begin (); i != level.end ();
         ++i)
    {
        newHdr.channels ().insert (i.name (), i.channel ().channel ());
        fb.insert (i.name (), i.channel ().slice ());
    }

    OutputFile out (fileName.c_str (), newHdr);
    out.setFrameBuffer (fb);
    out.writePixels (
        newHdr.dataWindow ().max.y - newHdr.dataWindow ().min.y + 1);
}

void
saveFlatScanLineImage (const string& fileName, const FlatImage& img)
{
    Header hdr;
    hdr.displayWindow () = img.dataWindow ();
    saveFlatScanLineImage (fileName, hdr, img);
}

void
loadFlatScanLineImage (const string& fileName, Header& hdr, FlatImage& img)
{
    InputFile in (fileName.c_str ());

    const ChannelList& cl = in.header ().channels ();

    img.clearChannels ();

    for (ChannelList::ConstIterator i = cl.begin (); i != cl.end (); ++i)
        img.insertChannel (i.name (), i.channel ());

    img.resize (in.header ().dataWindow (), ONE_LEVEL, ROUND_DOWN);

    FlatImageLevel& level = img.level ();
    FrameBuffer     fb;

    for (FlatImageLevel::ConstIterator i = level.begin (); i != level.end ();
         ++i)
        fb.insert (i.name (), i.channel ().slice ());

    in.setFrameBuffer (fb);
    in.readPixels (level.dataWindow ().min.y, level.dataWindow ().max.y);

    for (Header::ConstIterator i = in.header ().begin ();
         i != in.header ().end ();
         ++i)
    {
        if (strcmp (i.name (), "tiles")) hdr.insert (i.name (), i.attribute ());
    }
}

void
loadFlatScanLineImage (const string& fileName, FlatImage& img)
{
    Header hdr;
    loadFlatScanLineImage (fileName, hdr, img);
}

namespace
{

void
saveLevel (TiledOutputFile& out, const FlatImage& img, int x, int y)
{
    const FlatImageLevel& level = img.level (x, y);
    FrameBuffer           fb;

    for (FlatImageLevel::ConstIterator i = level.begin (); i != level.end ();
         ++i)
        fb.insert (i.name (), i.channel ().slice ());

    out.setFrameBuffer (fb);
    out.writeTiles (0, out.numXTiles (x) - 1, 0, out.numYTiles (y) - 1, x, y);
}

} // namespace

void
saveFlatTiledImage (
    const string&    fileName,
    const Header&    hdr,
    const FlatImage& img,
    DataWindowSource dws)
{
    Header newHdr;

    for (Header::ConstIterator i = hdr.begin (); i != hdr.end (); ++i)
    {
        if (strcmp (i.name (), "dataWindow") && strcmp (i.name (), "tiles") &&
            strcmp (i.name (), "channels"))
        {
            newHdr.insert (i.name (), i.attribute ());
        }
    }

    if (hdr.hasTileDescription ())
    {
        newHdr.setTileDescription (TileDescription (
            hdr.tileDescription ().xSize,
            hdr.tileDescription ().ySize,
            img.levelMode (),
            img.levelRoundingMode ()));
    }
    else
    {
        newHdr.setTileDescription (TileDescription (
            64, // xSize
            64, // ySize
            img.levelMode (),
            img.levelRoundingMode ()));
    }

    newHdr.dataWindow () = dataWindowForFile (hdr, img, dws);

    const FlatImageLevel& level = img.level (0, 0);

    for (FlatImageLevel::ConstIterator i = level.begin (); i != level.end ();
         ++i)
        newHdr.channels ().insert (i.name (), i.channel ().channel ());

    TiledOutputFile out (fileName.c_str (), newHdr);

    switch (img.levelMode ())
    {
        case ONE_LEVEL: saveLevel (out, img, 0, 0); break;

        case MIPMAP_LEVELS:

            for (int x = 0; x < out.numLevels (); ++x)
                saveLevel (out, img, x, x);

            break;

        case RIPMAP_LEVELS:

            for (int y = 0; y < out.numYLevels (); ++y)
                for (int x = 0; x < out.numXLevels (); ++x)
                    saveLevel (out, img, x, y);

            break;

        default: assert (false);
    }
}

void
saveFlatTiledImage (const string& fileName, const FlatImage& img)
{
    Header hdr;
    hdr.displayWindow () = img.dataWindow ();
    saveFlatTiledImage (fileName, hdr, img);
}

namespace
{

void
loadLevel (TiledInputFile& in, FlatImage& img, int x, int y)
{
    FlatImageLevel& level = img.level (x, y);
    FrameBuffer     fb;

    for (FlatImageLevel::ConstIterator i = level.begin (); i != level.end ();
         ++i)
        fb.insert (i.name (), i.channel ().slice ());

    in.setFrameBuffer (fb);
    in.readTiles (0, in.numXTiles (x) - 1, 0, in.numYTiles (y) - 1, x, y);
}

} // namespace

void
loadFlatTiledImage (const string& fileName, Header& hdr, FlatImage& img)
{
    TiledInputFile in (fileName.c_str ());

    const ChannelList& cl = in.header ().channels ();

    img.clearChannels ();

    for (ChannelList::ConstIterator i = cl.begin (); i != cl.end (); ++i)
        img.insertChannel (i.name (), i.channel ());

    img.resize (
        in.header ().dataWindow (),
        in.header ().tileDescription ().mode,
        in.header ().tileDescription ().roundingMode);

    switch (img.levelMode ())
    {
        case ONE_LEVEL: loadLevel (in, img, 0, 0); break;

        case MIPMAP_LEVELS:

            for (int x = 0; x < img.numLevels (); ++x)
                loadLevel (in, img, x, x);

            break;

        case RIPMAP_LEVELS:

            for (int y = 0; y < img.numYLevels (); ++y)
                for (int x = 0; x < img.numXLevels (); ++x)
                    loadLevel (in, img, x, y);

            break;

        default: assert (false);
    }

    for (Header::ConstIterator i = in.header ().begin ();
         i != in.header ().end ();
         ++i)
    {
        hdr.insert (i.name (), i.attribute ());
    }
}

void
loadFlatTiledImage (const string& fileName, FlatImage& img)
{
    Header hdr;
    loadFlatTiledImage (fileName, hdr, img);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
