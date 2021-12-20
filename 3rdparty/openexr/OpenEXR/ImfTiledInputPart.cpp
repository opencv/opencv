//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfTiledInputPart.h"

#include "ImfMultiPartInputFile.h"
#include "ImfTiledInputFile.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

TiledInputPart::TiledInputPart(MultiPartInputFile& multiPartFile, int partNumber)
{
    file = multiPartFile.getInputPart<TiledInputFile>(partNumber);
}

const char *
TiledInputPart::fileName () const
{
    return file->fileName();
}

const Header &
TiledInputPart::header () const
{
    return file->header();
}

int
TiledInputPart::version () const
{
    return file->version();
}

void
TiledInputPart::setFrameBuffer (const FrameBuffer &frameBuffer)
{
    file->setFrameBuffer(frameBuffer);
}

const FrameBuffer &
TiledInputPart::frameBuffer () const
{
    return file->frameBuffer();
}

bool
TiledInputPart::isComplete () const
{
    return file->isComplete();
}

unsigned int
TiledInputPart::tileXSize () const
{
    return file->tileXSize();
}

unsigned int
TiledInputPart::tileYSize () const
{
    return file->tileYSize();
}

LevelMode
TiledInputPart::levelMode () const
{
    return file->levelMode();
}

LevelRoundingMode
TiledInputPart::levelRoundingMode () const
{
    return file->levelRoundingMode();
}

int
TiledInputPart::numLevels () const
{
    return file->numLevels();
}

int
TiledInputPart::numXLevels () const
{
    return file->numXLevels();
}

int
TiledInputPart::numYLevels () const
{
    return file->numYLevels();
}

bool
TiledInputPart::isValidLevel (int lx, int ly) const
{
    return file->isValidLevel(lx, ly);
}

int
TiledInputPart::levelWidth  (int lx) const
{
    return file->levelWidth(lx);
}

int
TiledInputPart::levelHeight (int ly) const
{
    return file->levelHeight(ly);
}

int
TiledInputPart::numXTiles (int lx) const
{
    return file->numXTiles(lx);
}

int
TiledInputPart::numYTiles (int ly) const
{
    return file->numYTiles(ly);
}

IMATH_NAMESPACE::Box2i
TiledInputPart::dataWindowForLevel (int l) const
{
    return file->dataWindowForLevel(l);
}

IMATH_NAMESPACE::Box2i
TiledInputPart::dataWindowForLevel (int lx, int ly) const
{
    return file->dataWindowForLevel(lx, ly);
}

IMATH_NAMESPACE::Box2i
TiledInputPart::dataWindowForTile (int dx, int dy, int l) const
{
    return file->dataWindowForTile(dx, dy, l);
}

IMATH_NAMESPACE::Box2i
TiledInputPart::dataWindowForTile (int dx, int dy, int lx, int ly) const
{
    return file->dataWindowForTile(dx, dy, lx, ly);
}

void
TiledInputPart::readTile  (int dx, int dy, int l)
{
    file->readTile(dx, dy, l);
}

void
TiledInputPart::readTile  (int dx, int dy, int lx, int ly)
{
    file->readTile(dx, dy, lx, ly);
}

void
TiledInputPart::readTiles (int dx1, int dx2, int dy1, int dy2, int lx, int ly)
{
    file->readTiles(dx1, dx2, dy1, dy2, lx, ly);
}

void
TiledInputPart::readTiles (int dx1, int dx2, int dy1, int dy2, int l)
{
    file->readTiles(dx1, dx2, dy1, dy2, l);
}

void
TiledInputPart::rawTileData (int &dx, int &dy, int &lx, int &ly,
             const char *&pixelData, int &pixelDataSize)
{
    file->rawTileData(dx, dy, lx, ly, pixelData, pixelDataSize);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
