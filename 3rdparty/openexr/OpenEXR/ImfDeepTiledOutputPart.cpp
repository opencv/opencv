//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfDeepTiledOutputPart.h"

#include "ImfMultiPartOutputFile.h"

#include "ImfDeepTiledOutputFile.h"
#include "ImfDeepTiledInputPart.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

DeepTiledOutputPart::DeepTiledOutputPart(MultiPartOutputFile& multiPartFile, int partNumber)
{
    file = multiPartFile.getOutputPart<DeepTiledOutputFile>(partNumber);
}

const char *
DeepTiledOutputPart::fileName () const
{
    return file->fileName();
}


const Header &
DeepTiledOutputPart::header () const
{
    return file->header();
}


void
DeepTiledOutputPart::setFrameBuffer (const DeepFrameBuffer &frameBuffer)
{
    file->setFrameBuffer(frameBuffer);
}


const DeepFrameBuffer &
DeepTiledOutputPart::frameBuffer () const
{
    return file->frameBuffer();
}


unsigned int
DeepTiledOutputPart::tileXSize () const
{
    return file->tileXSize();
}


unsigned int
DeepTiledOutputPart::tileYSize () const
{
    return file->tileYSize();
}


LevelMode
DeepTiledOutputPart::levelMode () const
{
    return file->levelMode();
}


LevelRoundingMode
DeepTiledOutputPart::levelRoundingMode () const
{
    return file->levelRoundingMode();
}


int
DeepTiledOutputPart::numLevels () const
{
    return file->numLevels();
}


int
DeepTiledOutputPart::numXLevels () const
{
    return file->numXLevels();
}


int
DeepTiledOutputPart::numYLevels () const
{
    return file->numYLevels();
}


bool
DeepTiledOutputPart::isValidLevel (int lx, int ly) const
{
    return file->isValidLevel(lx, ly);
}


int
DeepTiledOutputPart::levelWidth  (int lx) const
{
    return file->levelWidth(lx);
}


int
DeepTiledOutputPart::levelHeight (int ly) const
{
    return file->levelHeight(ly);
}


int
DeepTiledOutputPart::numXTiles (int lx) const
{
    return file->numXTiles(lx);
}


int
DeepTiledOutputPart::numYTiles (int ly) const
{
    return file->numYTiles(ly);
}



IMATH_NAMESPACE::Box2i
DeepTiledOutputPart::dataWindowForLevel (int l) const
{
    return file->dataWindowForLevel(l);
}


IMATH_NAMESPACE::Box2i
DeepTiledOutputPart::dataWindowForLevel (int lx, int ly) const
{
    return file->dataWindowForLevel(lx, ly);
}


IMATH_NAMESPACE::Box2i
DeepTiledOutputPart::dataWindowForTile (int dx, int dy,
                                        int l) const
{
    return file->dataWindowForTile(dx, dy, l);
}


IMATH_NAMESPACE::Box2i
DeepTiledOutputPart::dataWindowForTile (int dx, int dy,
                                        int lx, int ly) const
{
    return file->dataWindowForTile(dx, dy, lx, ly);
}


void
DeepTiledOutputPart::writeTile  (int dx, int dy, int l)
{
    file->writeTile(dx, dy, l);
}


void
DeepTiledOutputPart::writeTile  (int dx, int dy, int lx, int ly)
{
    file->writeTile(dx, dy, lx, ly);
}


void
DeepTiledOutputPart::writeTiles (int dx1, int dx2, int dy1, int dy2,
                                 int lx, int ly)
{
    file->writeTiles(dx1, dx2, dy1, dy2, lx, ly);
}


void
DeepTiledOutputPart::writeTiles (int dx1, int dx2, int dy1, int dy2,
                                 int l)
{
    file->writeTiles(dx1, dx2, dy1, dy2, l);
}


void
DeepTiledOutputPart::copyPixels (DeepTiledInputFile &in)
{
    file->copyPixels(in);
}


void
DeepTiledOutputPart::copyPixels (DeepTiledInputPart &in)
{
    file->copyPixels(in);
}


void
DeepTiledOutputPart::updatePreviewImage (const PreviewRgba newPixels[])
{
    file->updatePreviewImage(newPixels);
}


void
DeepTiledOutputPart::breakTile  (int dx, int dy,
                                 int lx, int ly,
                                 int offset,
                                 int length,
                                 char c)
{
    file->breakTile(dx, dy, lx, ly, offset, length, c);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
