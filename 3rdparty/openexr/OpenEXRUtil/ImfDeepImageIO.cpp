//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      OpenEXR file I/O for deep images.
//
//----------------------------------------------------------------------------

#include "ImfDeepImageIO.h"
#include <Iex.h>
#include <ImfChannelList.h>
#include <ImfDeepScanLineInputFile.h>
#include <ImfDeepScanLineOutputFile.h>
#include <ImfDeepTiledInputFile.h>
#include <ImfDeepTiledOutputFile.h>
#include <ImfHeader.h>
#include <ImfMultiPartInputFile.h>
#include <ImfPartType.h>
#include <ImfTestFile.h>
#include <cassert>
#include <cstring>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

void
saveDeepImage (
    const string&    fileName,
    const Header&    hdr,
    const DeepImage& img,
    DataWindowSource dws)
{
    if (img.levelMode () != ONE_LEVEL || hdr.hasTileDescription ())
        saveDeepTiledImage (fileName, hdr, img, dws);
    else
        saveDeepScanLineImage (fileName, hdr, img, dws);
}

void
saveDeepImage (const string& fileName, const DeepImage& img)
{
    Header hdr;
    hdr.displayWindow () = img.dataWindow ();
    saveDeepImage (fileName, hdr, img);
}

void
loadDeepImage (const string& fileName, Header& hdr, DeepImage& img)
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

    if (!deep)
    {
        THROW (
            ArgExc,
            "Cannot load flat image file " << fileName
                                           << " "
                                              "as a deep image.");
    }

    //XXX TODO: the tiled flag obtained above is unreliable;
    // open the file as a multi-part file and inspect the header.
    // Can the OpenEXR library be fixed?

    {
        MultiPartInputFile mpi (fileName.c_str ());

        tiled =
            (mpi.parts () > 0 && mpi.header (0).hasType () &&
             isTiled (mpi.header (0).type ()));
    }

    if (tiled)
        loadDeepTiledImage (fileName, hdr, img);
    else
        loadDeepScanLineImage (fileName, hdr, img);
}

void
loadDeepImage (const string& fileName, DeepImage& img)
{
    Header hdr;
    loadDeepImage (fileName, hdr, img);
}

void
saveDeepScanLineImage (
    const string&    fileName,
    const Header&    hdr,
    const DeepImage& img,
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

    //XXX TODO: setting the compression to, for example,  ZIP_COMPRESSION,
    //then the OpenEXR library will save the file, but later it will not be
    //able to read it.  Fix the library!

    newHdr.compression () = ZIPS_COMPRESSION;

    const DeepImageLevel& level = img.level ();
    DeepFrameBuffer       fb;

    fb.insertSampleCountSlice (level.sampleCounts ().slice ());

    for (DeepImageLevel::ConstIterator i = level.begin (); i != level.end ();
         ++i)
    {
        newHdr.channels ().insert (i.name (), i.channel ().channel ());
        fb.insert (i.name (), i.channel ().slice ());
    }

    DeepScanLineOutputFile out (fileName.c_str (), newHdr);
    out.setFrameBuffer (fb);
    out.writePixels (
        newHdr.dataWindow ().max.y - newHdr.dataWindow ().min.y + 1);
}

void
saveDeepScanLineImage (const string& fileName, const DeepImage& img)
{
    Header hdr;
    hdr.displayWindow () = img.dataWindow ();
    saveDeepScanLineImage (fileName, hdr, img);
}

void
loadDeepScanLineImage (const string& fileName, Header& hdr, DeepImage& img)
{
    DeepScanLineInputFile in (fileName.c_str ());

    const ChannelList& cl = in.header ().channels ();

    img.clearChannels ();

    for (ChannelList::ConstIterator i = cl.begin (); i != cl.end (); ++i)
        img.insertChannel (i.name (), i.channel ());

    img.resize (in.header ().dataWindow (), ONE_LEVEL, ROUND_DOWN);

    DeepImageLevel& level = img.level ();
    DeepFrameBuffer fb;

    fb.insertSampleCountSlice (level.sampleCounts ().slice ());

    for (DeepImageLevel::ConstIterator i = level.begin (); i != level.end ();
         ++i)
        fb.insert (i.name (), i.channel ().slice ());

    in.setFrameBuffer (fb);

    {
        SampleCountChannel::Edit edit (level.sampleCounts ());

        in.readPixelSampleCounts (
            level.dataWindow ().min.y, level.dataWindow ().max.y);
    }

    in.readPixels (level.dataWindow ().min.y, level.dataWindow ().max.y);

    for (Header::ConstIterator i = in.header ().begin ();
         i != in.header ().end ();
         ++i)
    {
        if (strcmp (i.name (), "tiles")) hdr.insert (i.name (), i.attribute ());
    }
}

void
loadDeepScanLineImage (const string& fileName, DeepImage& img)
{
    Header hdr;
    loadDeepScanLineImage (fileName, hdr, img);
}

namespace
{

void
saveLevel (DeepTiledOutputFile& out, const DeepImage& img, int x, int y)
{
    const DeepImageLevel& level = img.level (x, y);
    DeepFrameBuffer       fb;

    fb.insertSampleCountSlice (level.sampleCounts ().slice ());

    for (DeepImageLevel::ConstIterator i = level.begin (); i != level.end ();
         ++i)
        fb.insert (i.name (), i.channel ().slice ());

    out.setFrameBuffer (fb);
    out.writeTiles (0, out.numXTiles (x) - 1, 0, out.numYTiles (y) - 1, x, y);
}

} // namespace

void
saveDeepTiledImage (
    const string&    fileName,
    const Header&    hdr,
    const DeepImage& img,
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

    //XXX TODO: setting the compression to, for example,  ZIP_COMPRESSION,
    //then the OpenEXR library will save the file, but later it will not be
    //able to read it.  Fix the library!

    newHdr.compression () = ZIPS_COMPRESSION;

    const DeepImageLevel& level = img.level (0, 0);

    for (DeepImageLevel::ConstIterator i = level.begin (); i != level.end ();
         ++i)
        newHdr.channels ().insert (i.name (), i.channel ().channel ());

    DeepTiledOutputFile out (fileName.c_str (), newHdr);

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
saveDeepTiledImage (const string& fileName, const DeepImage& img)
{
    Header hdr;
    hdr.displayWindow () = img.dataWindow ();
    saveDeepTiledImage (fileName, hdr, img);
}

namespace
{

void
loadLevel (DeepTiledInputFile& in, DeepImage& img, int x, int y)
{
    DeepImageLevel& level = img.level (x, y);
    DeepFrameBuffer fb;

    fb.insertSampleCountSlice (level.sampleCounts ().slice ());

    for (DeepImageLevel::ConstIterator i = level.begin (); i != level.end ();
         ++i)
        fb.insert (i.name (), i.channel ().slice ());

    in.setFrameBuffer (fb);

    {
        SampleCountChannel::Edit edit (level.sampleCounts ());

        in.readPixelSampleCounts (
            0, in.numXTiles (x) - 1, 0, in.numYTiles (y) - 1, x, y);
    }

    in.readTiles (0, in.numXTiles (x) - 1, 0, in.numYTiles (y) - 1, x, y);
}

} // namespace

void
loadDeepTiledImage (const string& fileName, Header& hdr, DeepImage& img)
{
    DeepTiledInputFile in (fileName.c_str ());

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
loadDeepTiledImage (const string& fileName, DeepImage& img)
{
    Header hdr;
    loadDeepTiledImage (fileName, hdr, img);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
