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

#ifndef __OPENCV_OCL_PRIVATE_UTIL__
#define __OPENCV_OCL_PRIVATE_UTIL__

#include "opencv2/ocl/ocl.hpp"

#if defined __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/opencl.h>
#endif

namespace cv
{
    namespace ocl
    {
        enum openCLMemcpyKind
        {
            clMemcpyHostToDevice = 0,
            clMemcpyDeviceToHost,
            clMemcpyDeviceToDevice
        };
        ///////////////////////////OpenCL call wrappers////////////////////////////
        void CV_EXPORTS openCLMallocPitch(Context *clCxt, void **dev_ptr, size_t *pitch,
                                          size_t widthInBytes, size_t height);
        void CV_EXPORTS openCLMallocPitchEx(Context *clCxt, void **dev_ptr, size_t *pitch,
                                            size_t widthInBytes, size_t height, DevMemRW rw_type, DevMemType mem_type);
        void CV_EXPORTS openCLMemcpy2D(Context *clCxt, void *dst, size_t dpitch,
                                       const void *src, size_t spitch,
                                       size_t width, size_t height, openCLMemcpyKind kind, int channels = -1);
        void CV_EXPORTS openCLCopyBuffer2D(Context *clCxt, void *dst, size_t dpitch, int dst_offset,
                                           const void *src, size_t spitch,
                                           size_t width, size_t height, int src_offset);
        void CV_EXPORTS openCLFree(void *devPtr);
        cl_mem CV_EXPORTS openCLCreateBuffer(Context *clCxt, size_t flag, size_t size);
        void CV_EXPORTS openCLReadBuffer(Context *clCxt, cl_mem dst_buffer, void *host_buffer, size_t size);
        cl_kernel CV_EXPORTS openCLGetKernelFromSource(const Context *clCxt,
                                                       const char **source, std::string kernelName);
        cl_kernel CV_EXPORTS openCLGetKernelFromSource(const Context *clCxt,
                                                       const char **source, std::string kernelName, const char *build_options);
        void CV_EXPORTS openCLVerifyKernel(const Context *clCxt, cl_kernel kernel, size_t *localThreads);
        void CV_EXPORTS openCLExecuteKernel(Context *clCxt , const char **source, string kernelName, std::vector< std::pair<size_t, const void *> > &args,
                                 int globalcols , int globalrows, size_t blockSize = 16, int kernel_expand_depth = -1, int kernel_expand_channel = -1);
        void CV_EXPORTS openCLExecuteKernel_(Context *clCxt , const char **source, std::string kernelName,
                                  size_t globalThreads[3], size_t localThreads[3],
                                  std::vector< std::pair<size_t, const void *> > &args, int channels, int depth, const char *build_options);
        void CV_EXPORTS openCLExecuteKernel(Context *clCxt , const char **source, std::string kernelName, size_t globalThreads[3],
                                 size_t localThreads[3],  std::vector< std::pair<size_t, const void *> > &args, int channels, int depth);
        void CV_EXPORTS openCLExecuteKernel(Context *clCxt , const char **source, std::string kernelName, size_t globalThreads[3],
                                 size_t localThreads[3],  std::vector< std::pair<size_t, const void *> > &args, int channels,
                                 int depth, const char *build_options);

        cl_mem CV_EXPORTS load_constant(cl_context context, cl_command_queue command_queue, const void *value,
                             const size_t size);

        cl_mem CV_EXPORTS openCLMalloc(cl_context clCxt, size_t size, cl_mem_flags flags, void *host_ptr);

        int CV_EXPORTS savetofile(const Context *clcxt,  cl_program &program, const char *fileName);

        enum FLUSH_MODE
        {
            CLFINISH = 0,
            CLFLUSH,
            DISABLE
        };

        void CV_EXPORTS openCLExecuteKernel2(Context *clCxt , const char **source, std::string kernelName, size_t globalThreads[3],
                                  size_t localThreads[3],  std::vector< std::pair<size_t, const void *> > &args, int channels, int depth, FLUSH_MODE finish_mode = DISABLE);
        void CV_EXPORTS openCLExecuteKernel2(Context *clCxt , const char **source, std::string kernelName, size_t globalThreads[3],
                                  size_t localThreads[3],  std::vector< std::pair<size_t, const void *> > &args, int channels,
                                  int depth, char *build_options, FLUSH_MODE finish_mode = DISABLE);
        // bind oclMat to OpenCL image textures
        // note:
        //   1. there is no memory management. User need to explicitly release the resource
        //   2. for faster clamping, there is no buffer padding for the constructed texture
        cl_mem CV_EXPORTS bindTexture(const oclMat &mat);
        void CV_EXPORTS releaseTexture(cl_mem& texture);

        // returns whether the current context supports image2d_t format or not
        bool CV_EXPORTS support_image2d(Context *clCxt = Context::getContext());

        // the enums are used to query device information
        // currently only support wavefront size queries
        enum DEVICE_INFO
        {
             WAVEFRONT_SIZE,            //in AMD speak
             WARP_SIZE = WAVEFRONT_SIZE //in nvidia speak
        };
        //info should have been pre-allocated
        void CV_EXPORTS queryDeviceInfo(DEVICE_INFO info_type, void* info);

    }//namespace ocl

}//namespace cv

#endif //__OPENCV_OCL_PRIVATE_UTIL__
