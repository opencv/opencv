//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfDeepScanLineOutputPart.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

DeepScanLineOutputPart::DeepScanLineOutputPart(MultiPartOutputFile& multiPartFile, int partNumber)
{
    file = multiPartFile.getOutputPart<DeepScanLineOutputFile>(partNumber);
}


const char *
DeepScanLineOutputPart::fileName () const
{
    return file->fileName();
}


const Header &
DeepScanLineOutputPart::header () const
{
    return file->header();
}


void
DeepScanLineOutputPart::setFrameBuffer (const DeepFrameBuffer &frameBuffer)
{
    file->setFrameBuffer(frameBuffer);
}


const DeepFrameBuffer &
DeepScanLineOutputPart::frameBuffer () const
{
    return file->frameBuffer();
}


void
DeepScanLineOutputPart::writePixels (int numScanLines)
{
    file->writePixels(numScanLines);
}


int
DeepScanLineOutputPart::currentScanLine () const
{
    return file->currentScanLine();
}


void
DeepScanLineOutputPart::copyPixels (DeepScanLineInputFile &in)
{
    file->copyPixels(in);
}

void
DeepScanLineOutputPart::copyPixels (DeepScanLineInputPart &in)
{
    file->copyPixels(in);
}

void
DeepScanLineOutputPart::updatePreviewImage (const PreviewRgba newPixels[])
{
    file->updatePreviewImage(newPixels);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT

