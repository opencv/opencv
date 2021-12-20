//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMFOUTPUTPART_H_
#define IMFOUTPUTPART_H_

#include "ImfForward.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


//---------------------------------------------------------------------
// class OutputPart:
//
// Same interface as OutputFile. Please refer to OutputFile.
//---------------------------------------------------------------------

class IMF_EXPORT_TYPE OutputPart
{
    public:
        IMF_EXPORT
        OutputPart(MultiPartOutputFile& multiPartFile, int partNumber);

        IMF_EXPORT
        const char *        fileName () const;
        IMF_EXPORT
        const Header &      header () const;
        IMF_EXPORT
        void                setFrameBuffer (const FrameBuffer &frameBuffer);
        IMF_EXPORT
        const FrameBuffer & frameBuffer () const;
        IMF_EXPORT
        void                writePixels (int numScanLines = 1);
        IMF_EXPORT
        int                 currentScanLine () const;
        IMF_EXPORT
        void                copyPixels (InputFile &in);
        IMF_EXPORT
        void                copyPixels (InputPart &in);
        
        IMF_EXPORT
        void                updatePreviewImage (const PreviewRgba newPixels[]);
        IMF_EXPORT
        void                breakScanLine  (int y, int offset, int length, char c);

    private:
        OutputFile* file;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif /* IMFOUTPUTPART_H_ */
