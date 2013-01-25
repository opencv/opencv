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
//    Guoping Long, longguoping@gmail.com
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

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#if defined _MSC_VER && _MSC_VER >= 1200
#pragma warning( disable: 4267 4324 4244 4251 4710 4711 4514 4996 )
#endif

#ifdef HAVE_CVCONFIG_H
#include "cvconfig.h"
#endif

#include <map>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <sstream>
#include <exception>
#include <stdio.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ocl/ocl.hpp"

#include "opencv2/core/internal.hpp"
//#include "opencv2/highgui/highgui.hpp"

#define __ATI__

#if defined (HAVE_OPENCL)

#if defined __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/opencl.h>
#endif

#include "safe_call.hpp"

using namespace std;

namespace cv
{
    namespace ocl
    {
        ///////////////////////////OpenCL call wrappers////////////////////////////
        void openCLMallocPitch(Context *clCxt, void **dev_ptr, size_t *pitch,
                               size_t widthInBytes, size_t height);
        void openCLMemcpy2D(Context *clCxt, void *dst, size_t dpitch,
                            const void *src, size_t spitch,
                            size_t width, size_t height, enum openCLMemcpyKind kind, int channels = -1);
        void openCLCopyBuffer2D(Context *clCxt, void *dst, size_t dpitch, int dst_offset,
                                const void *src, size_t spitch,
                                size_t width, size_t height, int src_offset);
        void openCLFree(void *devPtr);
        cl_mem openCLCreateBuffer(Context *clCxt, size_t flag, size_t size);
        void openCLReadBuffer(Context *clCxt, cl_mem dst_buffer, void *host_buffer, size_t size);
        cl_kernel openCLGetKernelFromSource(const Context *clCxt,
                                            const char **source, string kernelName);
        cl_kernel openCLGetKernelFromSource(const Context *clCxt,
                                            const char **source, string kernelName, const char *build_options);
        void openCLVerifyKernel(const Context *clCxt, cl_kernel kernel, size_t *localThreads);
        void openCLExecuteKernel(Context *clCxt , const char **source, string kernelName, vector< std::pair<size_t, const void *> > &args,
                                 int globalcols , int globalrows, size_t blockSize = 16, int kernel_expand_depth = -1, int kernel_expand_channel = -1);
        void openCLExecuteKernel_(Context *clCxt , const char **source, string kernelName,
                                  size_t globalThreads[3], size_t localThreads[3],
                                  vector< pair<size_t, const void *> > &args, int channels, int depth, const char *build_options);
        void openCLExecuteKernel(Context *clCxt , const char **source, string kernelName, size_t globalThreads[3],
                                 size_t localThreads[3],  vector< pair<size_t, const void *> > &args, int channels, int depth);
        void openCLExecuteKernel(Context *clCxt , const char **source, string kernelName, size_t globalThreads[3],
                                 size_t localThreads[3],  vector< pair<size_t, const void *> > &args, int channels,
                                 int depth, const char *build_options);

        cl_mem load_constant(cl_context context, cl_command_queue command_queue, const void *value,
                             const size_t size);

        cl_mem openCLMalloc(cl_context clCxt, size_t size, cl_mem_flags flags, void *host_ptr);

        //void openCLMemcpy2DWithNoPadding(cl_command_queue command_queue, cl_mem buffer, size_t size, size_t offset, void *ptr,
        //                                 enum openCLMemcpyKind kind, cl_bool blocking_write);
        int savetofile(const Context *clcxt,  cl_program &program, const char *fileName);
        struct Context::Impl
        {
            //Information of the OpenCL context
            cl_context clContext;
            cl_command_queue clCmdQueue;
            cl_device_id *devices;
            string devName;
            cl_uint maxDimensions;
            size_t maxWorkGroupSize;
            size_t *maxWorkItemSizes;
            cl_uint maxComputeUnits;
            int double_support;
            //extra options to recognize vendor specific fp64 extensions
            char *extra_options;
            string Binpath;
        };
    }
}


#else /* defined(HAVE_OPENCL) */

static inline void throw_nogpu()
{
    CV_Error(CV_GpuNotSupported, "The library is compilled without OpenCL support.\n");
}

#endif /* defined(HAVE_OPENCL) */

#endif /* __OPENCV_PRECOMP_H__ */
