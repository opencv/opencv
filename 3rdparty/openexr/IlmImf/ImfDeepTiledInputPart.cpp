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


#include "ImfDeepTiledInputPart.h"
#include "ImfMultiPartInputFile.h"
#include "ImfNamespace.h"

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
                                 Int64 & dataSize) const
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
