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

#ifndef IMFINPUTPART_H_
#define IMFINPUTPART_H_

#include "ImfInputFile.h"
#include "ImfOutputPart.h"
#include "ImfForward.h"
#include "ImfNamespace.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

//-------------------------------------------------------------------
// class InputPart:
//
// Same interface as InputFile. Please refer to InputFile.
//-------------------------------------------------------------------

class InputPart
{
    public:
        IMF_EXPORT
        InputPart(MultiPartInputFile& multiPartFile, int partNumber);

        IMF_EXPORT
        const char *        fileName () const;
        IMF_EXPORT
        const Header &      header () const;
        IMF_EXPORT
        int                 version () const;
        IMF_EXPORT
        void                setFrameBuffer (const FrameBuffer &frameBuffer);
        IMF_EXPORT
        const FrameBuffer & frameBuffer () const;
        IMF_EXPORT
        bool                isComplete () const;
        IMF_EXPORT
        bool                isOptimizationEnabled () const;
        IMF_EXPORT
        void                readPixels (int scanLine1, int scanLine2);
        IMF_EXPORT
        void                readPixels (int scanLine);
        IMF_EXPORT
        void                rawPixelData (int firstScanLine,
                                          const char *&pixelData,
                                          int &pixelDataSize);

 
        IMF_EXPORT
        void                rawPixelDataToBuffer (int scanLine,
                                                  char *pixelData,
                                                  int &pixelDataSize) const;


        IMF_EXPORT
        void                rawTileData (int &dx, int &dy,
                                         int &lx, int &ly,
                                         const char *&pixelData,
                                         int &pixelDataSize);

    private:
        InputFile* file;
    // for internal use - give OutputFile and TiledOutputFile access to file for copyPixels
    friend void OutputFile::copyPixels(InputPart&);
    friend void TiledOutputFile::copyPixels(InputPart&);
    
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif /* IMFINPUTPART_H_ */
