/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

namespace cv {
namespace ocl {

class ProgramCache
{
protected:
    ProgramCache();
    ~ProgramCache();
    friend class std::auto_ptr<ProgramCache>;
public:
    static ProgramCache *getProgramCache();

    cl_program getProgram(const Context *ctx, const cv::ocl::ProgramEntry* source,
                          const char *build_options);

    void releaseProgram();
protected:
    //lookup the binary given the file name
    // (with acquired mutexCache)
    cl_program progLookup(const String& srcsign);

    //add program to the cache
    // (with acquired mutexCache)
    void addProgram(const String& srcsign, cl_program program);

    std::map <String, cl_program> codeCache;
    unsigned int cacheSize;

    //The presumed watermark for the cache volume (256MB). Is it enough?
    //We may need more delicate algorithms when necessary later.
    //Right now, let's just leave it along.
    static const unsigned MAX_PROG_CACHE_SIZE = 1024;

    // acquire both mutexes in this order: 1) mutexFiles 2) mutexCache
    static cv::Mutex mutexFiles;
    static cv::Mutex mutexCache;
};

}//namespace ocl
}//namespace cv
