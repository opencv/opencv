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

#ifndef IMFMULTIPARTINPUTFILE_H_
#define IMFMULTIPARTINPUTFILE_H_

#include "ImfGenericInputFile.h"
#include "ImfNamespace.h"
#include "ImfForward.h"
#include "ImfThreading.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


class MultiPartInputFile : public GenericInputFile
{
  public:
    IMF_EXPORT
    MultiPartInputFile(const char fileName[],
                       int numThreads = globalThreadCount(),
                       bool reconstructChunkOffsetTable = true);

    IMF_EXPORT
    MultiPartInputFile(IStream& is,
                       int numThreads = globalThreadCount(),
                       bool reconstructChunkOffsetTable = true);

    IMF_EXPORT
    virtual ~MultiPartInputFile();

    // ----------------------
    // Count of number of parts in file
    // ---------------------
    IMF_EXPORT
    int parts() const;
    
    
    //----------------------
    // Access to the headers
    //----------------------

    IMF_EXPORT
    const Header &  header(int n) const;
    

    //----------------------------------
    // Access to the file format version
    //----------------------------------

    IMF_EXPORT
    int			    version () const;


    // =----------------------------------------
    // Check whether the entire chunk offset
    // table for the part is written correctly
    // -----------------------------------------
    IMF_EXPORT
    bool partComplete(int part) const;


    struct Data;


  private:
    Data*                           _data;

    MultiPartInputFile(const MultiPartInputFile &); // not implemented

    
    //
    // used internally by 'Part' types to access individual parts of the multipart file
    //
    template<class T> T*    getInputPart(int partNumber);
    InputPartData*          getPart(int);
    
    void                    initialize();


    

    friend class InputPart;
    friend class ScanLineInputPart;
    friend class TiledInputPart;
    friend class DeepScanLineInputPart;
    friend class DeepTiledInputPart;

    //
    // For backward compatibility.
    //

    friend class InputFile;
    friend class TiledInputFile;
    friend class ScanLineInputFile;
    friend class DeepScanLineInputFile;
    friend class DeepTiledInputFile;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif /* IMFMULTIPARTINPUTFILE_H_ */
