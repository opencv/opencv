///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2011, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
//
// Portions (c) 2012 Weta Digital Ltd
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

#ifndef MULTIPARTOUTPUTFILE_H_
#define MULTIPARTOUTPUTFILE_H_

#include "ImfHeader.h"
#include "ImfGenericOutputFile.h"
#include "ImfForward.h"
#include "ImfThreading.h"
#include "ImfNamespace.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


//
// Class responsible for handling the writing of multipart images.
//
// Note: Certain attributes are 'common' to all parts. Notably:
// * Display Window
// * Pixel Aspect Ratio
// * Time Code
// * Chromaticities
// The first header forms the basis for the set of attributes that are shared 
// across the constituent parts.
//
// Parameters
//  headers - pointer to array of headers; one for each part of the image file
//  parts - count of number of parts
//  overrideSharedAttributes - toggle for the handling of shared attributes.
//                             set false to check for inconsistencies, true
//                             to copy the values over from the first header.
//  numThreads - number of threads that should be used in encoding the data.
//
    
class IMF_EXPORT MultiPartOutputFile : public GenericOutputFile
{
    public:
        MultiPartOutputFile(const char fileName[],
                            const Header * headers,
                            int parts,
                            bool overrideSharedAttributes = false,
                            int numThreads = globalThreadCount());
                            
        MultiPartOutputFile(OStream & os,
                            const Header * headers,
                            int parts,
                            bool overrideSharedAttributes = false,
                            int numThreads = globalThreadCount());                            

        //
        // return number of parts in file
        //
        int parts() const ;
        
        
        //
        // return header for part n
        // (note: may have additional attributes compared to that passed to constructor)
        //
        const Header & header(int n) const;
                            
        ~MultiPartOutputFile();

        struct Data;

    private:
        Data*                           _data;

        MultiPartOutputFile(const MultiPartOutputFile &); // not implemented
        
        template<class T>         T*  getOutputPart(int partNumber);

    
    friend class OutputPart;
    friend class TiledOutputPart;
    friend class DeepScanLineOutputPart;
    friend class DeepTiledOutputPart;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif /* MULTIPARTOUTPUTFILE_H_ */
