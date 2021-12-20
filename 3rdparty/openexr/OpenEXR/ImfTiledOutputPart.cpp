//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfTiledOutputPart.h"

#include "ImfMultiPartOutputFile.h"
#include "ImfTiledOutputFile.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

TiledOutputPart::TiledOutputPart(MultiPartOutputFile& multiPartFile, int partNumber)
{
    file = multiPartFile.getOutputPart<TiledOutputFile>(partNumber);
}

const char *
TiledOutputPart::fileName () const
{
    return file->fileName();
}

const Header &
TiledOutputPart::header () const
{
    return file->header();
}

void
TiledOutputPart::setFrameBuffer (const FrameBuffer &frameBuffer)
{
    file->setFrameBuffer(frameBuffer);
}

const FrameBuffer &
TiledOutputPart::frameBuffer () const
{
    return file->frameBuffer();
}

unsigned int
TiledOutputPart::tileXSize () const
{
    return file->tileXSize();
}

unsigned int
TiledOutputPart::tileYSize () const
{
    return file->tileYSize();
}

LevelMode
TiledOutputPart::levelMode () const
{
    return file->levelMode();
}

LevelRoundingMode
TiledOutputPart::levelRoundingMode () const
{
    return file->levelRoundingMode();
}

int
TiledOutputPart::numLevels () const
{
    return file->numLevels();
}

int
TiledOutputPart::numXLevels () const
{
    return file->numXLevels();
}

int
TiledOutputPart::numYLevels () const
{
    return file->numYLevels();
}

bool
TiledOutputPart::isValidLevel (int lx, int ly) const
{
    return file->isValidLevel(lx, ly);
}

int
TiledOutputPart::levelWidth  (int lx) const
{
    return file->levelWidth(lx);
}

int
TiledOutputPart::levelHeight (int ly) const
{
    return file->levelHeight(ly);
}

int
TiledOutputPart::numXTiles (int lx) const
{
    return file->numXTiles(lx);
}

int
TiledOutputPart::numYTiles (int ly) const
{
    return file->numYTiles(ly);
}

IMATH_NAMESPACE::Box2i
TiledOutputPart::dataWindowForLevel (int l) const
{
    return file->dataWindowForLevel(l);
}

IMATH_NAMESPACE::Box2i
TiledOutputPart::dataWindowForLevel (int lx, int ly) const
{
    return file->dataWindowForLevel(lx, ly);
}

IMATH_NAMESPACE::Box2i
TiledOutputPart::dataWindowForTile (int dx, int dy, int l) const
{
    return file->dataWindowForTile(dx, dy, l);
}

IMATH_NAMESPACE::Box2i
TiledOutputPart::dataWindowForTile (int dx, int dy, int lx, int ly) const
{
    return file->dataWindowForTile(dx, dy, lx, ly);
}

void
TiledOutputPart::writeTile  (int dx, int dy, int l)
{
    file->writeTile(dx, dy, l);
}

void
TiledOutputPart::writeTile  (int dx, int dy, int lx, int ly)
{
    file->writeTile(dx, dy, lx, ly);
}

void
TiledOutputPart::writeTiles (int dx1, int dx2, int dy1, int dy2, int lx, int ly)
{
    file->writeTiles(dx1, dx2, dy1, dy2, lx, ly);
}

void
TiledOutputPart::writeTiles (int dx1, int dx2, int dy1, int dy2, int l)
{
    file->writeTiles(dx1, dx2, dy1, dy2, l);
}

void
TiledOutputPart::copyPixels (TiledInputFile &in)
{
    file->copyPixels(in);
}

void
TiledOutputPart::copyPixels (InputFile &in)
{
    file->copyPixels(in);
}

void
TiledOutputPart::copyPixels (TiledInputPart &in)
{
    file->copyPixels(in);
}

void
TiledOutputPart::copyPixels (InputPart &in)
{
    file->copyPixels(in);
}



void
TiledOutputPart::updatePreviewImage (const PreviewRgba newPixels[])
{
    file->updatePreviewImage(newPixels);
}

void
TiledOutputPart::breakTile  (int dx, int dy, int lx, int ly, int offset, int length, char c)
{
    file->breakTile(dx, dy, lx, ly, offset, length, c);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
