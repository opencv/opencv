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

#include "ImfInputPart.h"
#include "ImfNamespace.h"

#include "ImfMultiPartInputFile.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

InputPart::InputPart(MultiPartInputFile& multiPartFile, int partNumber)
{
    file = multiPartFile.getInputPart<InputFile>(partNumber);
}

const char *
InputPart::fileName () const
{
    return file->fileName();
}

const Header &
InputPart::header () const
{
    return file->header();
}

int
InputPart::version () const
{
    return file->version();
}

void
InputPart::setFrameBuffer (const FrameBuffer &frameBuffer)
{
    file->setFrameBuffer(frameBuffer);
}

const FrameBuffer &
InputPart::frameBuffer () const
{
    return file->frameBuffer();
}

bool
InputPart::isComplete () const
{
    return file->isComplete();
}

bool
InputPart::isOptimizationEnabled() const
{
   return file->isOptimizationEnabled();
}

void
InputPart::readPixels (int scanLine1, int scanLine2)
{
    file->readPixels(scanLine1, scanLine2);
}

void
InputPart::readPixels (int scanLine)
{
    file->readPixels(scanLine);
}

void
InputPart::rawPixelData (int firstScanLine, const char *&pixelData, int &pixelDataSize)
{
    file->rawPixelData(firstScanLine, pixelData, pixelDataSize);
}


void
InputPart::rawPixelDataToBuffer (int scanLine, char *pixelData, int &pixelDataSize) const
{
    file->rawPixelDataToBuffer(scanLine, pixelData, pixelDataSize);
}


void
InputPart::rawTileData (int &dx, int &dy, int &lx, int &ly,
             const char *&pixelData, int &pixelDataSize)
{
    file->rawTileData(dx, dy, lx, ly, pixelData, pixelDataSize);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
