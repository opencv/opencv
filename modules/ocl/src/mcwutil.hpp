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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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

#ifndef _OPENCV_MCWUTIL_
#define _OPENCV_MCWUTIL_

#include "precomp.hpp"

#if defined (HAVE_OPENCL)

using namespace std;

namespace cv
{
    namespace ocl
    {
        enum FLUSH_MODE
        {
            CLFINISH = 0,
            CLFLUSH,
            DISABLE
        };
        void openCLExecuteKernel2(Context *clCxt , const char **source, string kernelName, size_t globalThreads[3],
                                  size_t localThreads[3],  vector< pair<size_t, const void *> > &args, int channels, int depth, FLUSH_MODE finish_mode = DISABLE);
        void openCLExecuteKernel2(Context *clCxt , const char **source, string kernelName, size_t globalThreads[3],
                                  size_t localThreads[3],  vector< pair<size_t, const void *> > &args, int channels,
                                  int depth, char *build_options, FLUSH_MODE finish_mode = DISABLE);
        // bind oclMat to OpenCL image textures
        // note:
        //   1. there is no memory management. User need to explicitly release the resource
        //   2. for faster clamping, there is no buffer padding for the constructed texture
        cl_mem bindTexture(const oclMat &mat);
        void releaseTexture(cl_mem& texture);
    }//namespace ocl

}//namespace cv
#endif // HAVE_OPENCL
#endif //_OPENCV_MCWUTIL_
