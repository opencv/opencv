//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfOutputPart.h"

#include "ImfMultiPartOutputFile.h"
#include "ImfOutputFile.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

OutputPart::OutputPart(MultiPartOutputFile& multiPartFile, int partNumber)
{
    file = multiPartFile.getOutputPart<OutputFile>(partNumber);
}

const char *
OutputPart::fileName () const
{
    return file->fileName();
}

const Header &
OutputPart::header () const
{
    return file->header();
}

void
OutputPart::setFrameBuffer (const FrameBuffer &frameBuffer)
{
    file->setFrameBuffer(frameBuffer);
}

const FrameBuffer &
OutputPart::frameBuffer () const
{
    return file->frameBuffer();
}

void
OutputPart::writePixels (int numScanLines)
{
    file->writePixels(numScanLines);
}

int
OutputPart::currentScanLine () const
{
    return file->currentScanLine();
}

void
OutputPart::copyPixels (InputFile &in)
{
    file->copyPixels(in);
}

void
OutputPart::copyPixels (InputPart &in)
{
    file->copyPixels(in);
}

void
OutputPart::updatePreviewImage (const PreviewRgba newPixels[])
{
    file->updatePreviewImage(newPixels);
}

void
OutputPart::breakScanLine  (int y, int offset, int length, char c)
{
    file->breakScanLine(y, offset, length, c);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
