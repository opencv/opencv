//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfDeepTiledInputPart.h"

#include "ImfMultiPartInputFile.h"
#include "ImfDeepTiledInputFile.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

DeepTiledInputPart::DeepTiledInputPart(MultiPartInputFile& multiPartFile, int partNumber)
{
    file = multiPartFile.getInputPart<DeepTiledInputFile>(partNumber);
}


const char *
DeepTiledInputPart::fileName () const
{
    return file->fileName();
}


const Header &
DeepTiledInputPart::header () const
{
    return file->header();
}


int
DeepTiledInputPart::version () const
{
    return file->version();
}


void
DeepTiledInputPart::setFrameBuffer (const DeepFrameBuffer &frameBuffer)
{
    file->setFrameBuffer(frameBuffer);
}


const DeepFrameBuffer &
DeepTiledInputPart::frameBuffer () const
{
    return file->frameBuffer();
}


bool
DeepTiledInputPart::isComplete () const
{
    return file->isComplete();
}


unsigned int
DeepTiledInputPart::tileXSize () const
{
    return file->tileXSize();
}


unsigned int
DeepTiledInputPart::tileYSize () const
{
    return file->tileYSize();
}


LevelMode
DeepTiledInputPart::levelMode () const
{
    return file->levelMode();
}


LevelRoundingMode
DeepTiledInputPart::levelRoundingMode () const
{
    return file->levelRoundingMode();
}


int
DeepTiledInputPart::numLevels () const
{
    return file->numLevels();
}


int
DeepTiledInputPart::numXLevels () const
{
    return file->numXLevels();
}


int
DeepTiledInputPart::numYLevels () const
{
    return file->numYLevels();
}


bool
DeepTiledInputPart::isValidLevel (int lx, int ly) const
{
    return file->isValidLevel(lx, ly);
}


int
DeepTiledInputPart::levelWidth  (int lx) const
{
    return file->levelWidth(lx);
}


int
DeepTiledInputPart::levelHeight (int ly) const
{
    return file->levelHeight(ly);
}


int
DeepTiledInputPart::numXTiles (int lx) const
{
    return file->numXTiles(lx);
}


int
DeepTiledInputPart::numYTiles (int ly) const
{
    return file->numYTiles(ly);
}


IMATH_NAMESPACE::Box2i
DeepTiledInputPart::dataWindowForLevel (int l) const
{
    return file->dataWindowForLevel(l);
}

IMATH_NAMESPACE::Box2i
DeepTiledInputPart::dataWindowForLevel (int lx, int ly) const
{
    return file->dataWindowForLevel(lx, ly);
}


IMATH_NAMESPACE::Box2i
DeepTiledInputPart::dataWindowForTile (int dx, int dy, int l) const
{
    return file->dataWindowForTile(dx, dy, l);
}


IMATH_NAMESPACE::Box2i
DeepTiledInputPart::dataWindowForTile (int dx, int dy,
                                       int lx, int ly) const
{
    return file->dataWindowForTile(dx, dy, lx, ly);
}


void
DeepTiledInputPart::readTile  (int dx, int dy, int l)
{
    file->readTile(dx, dy, l);
}


void
DeepTiledInputPart::readTile  (int dx, int dy, int lx, int ly)
{
    file->readTile(dx, dy, lx, ly);
}


void
DeepTiledInputPart::readTiles (int dx1, int dx2, int dy1, int dy2,
                               int lx, int ly)
{
    file->readTiles(dx1, dx2, dy1, dy2, lx, ly);
}


void
DeepTiledInputPart::readTiles (int dx1, int dx2, int dy1, int dy2,
                               int l)
{
    file->readTiles(dx1, dx2, dy1, dy2, l);
}


void
DeepTiledInputPart::rawTileData (int &dx, int &dy,
                                 int &lx, int &ly,
                                 char * pixelData,
                                 uint64_t & dataSize) const
{
    file->rawTileData(dx, dy, lx, ly, pixelData, dataSize );
}


void
DeepTiledInputPart::readPixelSampleCount  (int dx, int dy, int l)
{
    file->readPixelSampleCount(dx, dy, l);
}


void
DeepTiledInputPart::readPixelSampleCount  (int dx, int dy, int lx, int ly)
{
    file->readPixelSampleCount(dx, dy, lx, ly);
}


void
DeepTiledInputPart::readPixelSampleCounts (int dx1, int dx2,
                                          int dy1, int dy2,
                                          int lx, int ly)
{
    file->readPixelSampleCounts(dx1, dx2, dy1, dy2, lx, ly);
}

void
DeepTiledInputPart::readPixelSampleCounts (int dx1, int dx2,
                                          int dy1, int dy2,
                                          int l)
{
    file->readPixelSampleCounts(dx1, dx2, dy1, dy2, l);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
