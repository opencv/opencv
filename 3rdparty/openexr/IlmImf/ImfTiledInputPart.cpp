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

#include "ImfTiledInputPart.h"
#include "ImfNamespace.h"

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
