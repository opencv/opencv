//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfDeepScanLineInputPart.h"

#include "ImfMultiPartInputFile.h"
#include "ImfDeepScanLineInputFile.h"
#include "ImfDeepScanLineOutputFile.h"

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
                                     uint64_t &pixelDataSize)
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
