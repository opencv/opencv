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


#include "ImfDeepScanLineInputPart.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

DeepScanLineInputPart::DeepScanLineInputPart(MultiPartInputFile& multiPartFile, int partNumber)
{
    file = multiPartFile.getInputPart<DeepScanLineInputFile>(partNumber);
}


const char *
DeepScanLineInputPart::fileName () const
{
    return file->fileName();
}


const Header &
DeepScanLineInputPart::header () const
{
    return file->header();
}


int
DeepScanLineInputPart::version () const
{
    return file->version();
}


void
DeepScanLineInputPart::setFrameBuffer (const DeepFrameBuffer &frameBuffer)
{
    file->setFrameBuffer(frameBuffer);
}


const DeepFrameBuffer &
DeepScanLineInputPart::frameBuffer () const
{
    return file->frameBuffer();
}


bool
DeepScanLineInputPart::isComplete () const
{
    return file->isComplete();
}


void
DeepScanLineInputPart::readPixels (int scanLine1, int scanLine2)
{
    file->readPixels(scanLine1, scanLine2);
}


void
DeepScanLineInputPart::readPixels (int scanLine)
{
    file->readPixels(scanLine);
}


void
DeepScanLineInputPart::rawPixelData (int firstScanLine,
                                     char *pixelData,
                                     Int64 &pixelDataSize)
{
    file->rawPixelData(firstScanLine, pixelData, pixelDataSize);
}


void
DeepScanLineInputPart::readPixelSampleCounts(int scanline1,
                                            int scanline2)
{
    file->readPixelSampleCounts(scanline1, scanline2);
}


void
DeepScanLineInputPart::readPixelSampleCounts(int scanline)
{
    file->readPixelSampleCounts(scanline);
}

int 
DeepScanLineInputPart::firstScanLineInChunk(int y) const
{
    return file->firstScanLineInChunk(y);
}

int 
DeepScanLineInputPart::lastScanLineInChunk(int y) const
{
    return file->lastScanLineInChunk(y);
}

void 
DeepScanLineInputPart::readPixels(const char* rawPixelData, const DeepFrameBuffer& frameBuffer, int scanLine1, int scanLine2) const
{
    return file->readPixels(rawPixelData,frameBuffer,scanLine1,scanLine2);
}
void 
DeepScanLineInputPart::readPixelSampleCounts(const char* rawdata, const DeepFrameBuffer& frameBuffer, int scanLine1, int scanLine2) const
{
   return file->readPixelSampleCounts(rawdata,frameBuffer,scanLine1,scanLine2);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
