//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfInputPart.h"
#include "ImfNamespace.h"

#include "ImfMultiPartInputFile.h"
#include "ImfInputFile.h"

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
