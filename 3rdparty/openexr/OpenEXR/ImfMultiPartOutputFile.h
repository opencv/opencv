//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef MULTIPARTOUTPUTFILE_H_
#define MULTIPARTOUTPUTFILE_H_

#include "ImfForward.h"

#include "ImfGenericOutputFile.h"
#include "ImfThreading.h"

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
    
class IMF_EXPORT_TYPE MultiPartOutputFile : public GenericOutputFile
{
    public:
        IMF_EXPORT
        MultiPartOutputFile(const char fileName[],
                            const Header * headers,
                            int parts,
                            bool overrideSharedAttributes = false,
                            int numThreads = globalThreadCount());
                            
        IMF_EXPORT
        MultiPartOutputFile(OStream & os,
                            const Header * headers,
                            int parts,
                            bool overrideSharedAttributes = false,
                            int numThreads = globalThreadCount());                            

        //
        // return number of parts in file
        //
        IMF_EXPORT
        int parts() const;
        
        //
        // return header for part n
        // (note: may have additional attributes compared to that passed to constructor)
        //
        IMF_EXPORT
        const Header & header(int n) const;
                            
        IMF_EXPORT
        ~MultiPartOutputFile();

        MultiPartOutputFile(const MultiPartOutputFile& other) = delete;
        MultiPartOutputFile& operator = (const MultiPartOutputFile& other) = delete;
        MultiPartOutputFile(MultiPartOutputFile&& other) = delete;
        MultiPartOutputFile& operator = (MultiPartOutputFile&& other) = delete;

        struct IMF_HIDDEN Data;

    private:
        Data*                           _data;

        template<class T> IMF_HIDDEN T*  getOutputPart(int partNumber);

    
    friend class OutputPart;
    friend class TiledOutputPart;
    friend class DeepScanLineOutputPart;
    friend class DeepTiledOutputPart;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif /* MULTIPARTOUTPUTFILE_H_ */
