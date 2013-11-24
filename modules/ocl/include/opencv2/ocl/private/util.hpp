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
//     and/or other materials provided with the distribution.
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

#include "opencv2/ocl/cl_runtime/cl_runtime.hpp"
#include "opencv2/core/ocl_genbase.hpp"

#include "opencv2/ocl.hpp"

namespace cv
{
namespace ocl
{

inline cl_device_id getClDeviceID(const Context *ctx)
{
    return *(cl_device_id*)(ctx->getOpenCLDeviceIDPtr());
}

inline cl_context getClContext(const Context *ctx)
{
    return *(cl_context*)(ctx->getOpenCLContextPtr());
}

inline cl_command_queue getClCommandQueue(const Context *ctx)
{
    return *(cl_command_queue*)(ctx->getOpenCLCommandQueuePtr());
}

CV_EXPORTS cv::Mutex& getInitializationMutex();

enum openCLMemcpyKind
{
    clMemcpyHostToDevice = 0,
    clMemcpyDeviceToHost,
    clMemcpyDeviceToDevice
};
///////////////////////////OpenCL call wrappers////////////////////////////
CV_EXPORTS void openCLMallocPitch(Context *clCxt, void **dev_ptr, size_t *pitch,
        size_t widthInBytes, size_t height);
CV_EXPORTS void openCLMallocPitchEx(Context *clCxt, void **dev_ptr, size_t *pitch,
        size_t widthInBytes, size_t height, DevMemRW rw_type, DevMemType mem_type);
CV_EXPORTS void openCLMemcpy2D(Context *clCxt, void *dst, size_t dpitch,
        const void *src, size_t spitch,
        size_t width, size_t height, openCLMemcpyKind kind, int channels = -1);
CV_EXPORTS void openCLCopyBuffer2D(Context *clCxt, void *dst, size_t dpitch, int dst_offset,
        const void *src, size_t spitch,
        size_t width, size_t height, int src_offset);
CV_EXPORTS void openCLFree(void *devPtr);
CV_EXPORTS cl_mem openCLCreateBuffer(Context *clCxt, size_t flag, size_t size);
CV_EXPORTS void openCLReadBuffer(Context *clCxt, cl_mem dst_buffer, void *host_buffer, size_t size);
CV_EXPORTS cl_kernel openCLGetKernelFromSource(const Context *clCxt,
        const cv::ocl::ProgramEntry* source, String kernelName);
CV_EXPORTS cl_kernel openCLGetKernelFromSource(const Context *clCxt,
        const cv::ocl::ProgramEntry* source, String kernelName, const char *build_options);
CV_EXPORTS cl_kernel openCLGetKernelFromSource(Context *ctx, const cv::ocl::ProgramEntry* source,
        String kernelName, int channels, int depth, const char *build_options);
CV_EXPORTS void openCLVerifyKernel(const Context *clCxt, cl_kernel kernel, size_t *localThreads);
CV_EXPORTS void openCLExecuteKernel(Context *ctx, cl_kernel kernel, size_t globalThreads[3],
                          size_t localThreads[3], std::vector< std::pair<size_t, const void *> > &args);
CV_EXPORTS void openCLExecuteKernel(Context *clCxt , const cv::ocl::ProgramEntry* source, String kernelName, std::vector< std::pair<size_t, const void *> > &args,
        int globalcols , int globalrows, size_t blockSize = 16, int kernel_expand_depth = -1, int kernel_expand_channel = -1);
CV_EXPORTS void openCLExecuteKernel_(Context *clCxt, const cv::ocl::ProgramEntry* source, String kernelName,
        size_t globalThreads[3], size_t localThreads[3],
        std::vector< std::pair<size_t, const void *> > &args, int channels, int depth, const char *build_options);
CV_EXPORTS void openCLExecuteKernel(Context *clCxt, const cv::ocl::ProgramEntry* source, String kernelName, size_t globalThreads[3],
        size_t localThreads[3],  std::vector< std::pair<size_t, const void *> > &args, int channels, int depth);
CV_EXPORTS void openCLExecuteKernel(Context *clCxt, const cv::ocl::ProgramEntry* source, String kernelName, size_t globalThreads[3],
        size_t localThreads[3],  std::vector< std::pair<size_t, const void *> > &args, int channels,
        int depth, const char *build_options);

CV_EXPORTS cl_mem load_constant(cl_context context, cl_command_queue command_queue, const void *value,
        const size_t size);

CV_EXPORTS cl_mem openCLMalloc(cl_context clCxt, size_t size, cl_mem_flags flags, void *host_ptr);

enum FLUSH_MODE
{
    CLFINISH = 0,
    CLFLUSH,
    DISABLE
};

CV_EXPORTS void openCLExecuteKernel2(Context *clCxt, const cv::ocl::ProgramEntry* source, String kernelName, size_t globalThreads[3],
        size_t localThreads[3],  std::vector< std::pair<size_t, const void *> > &args, int channels, int depth, FLUSH_MODE finish_mode = DISABLE);
CV_EXPORTS void openCLExecuteKernel2(Context *clCxt, const cv::ocl::ProgramEntry* source, String kernelName, size_t globalThreads[3],
        size_t localThreads[3],  std::vector< std::pair<size_t, const void *> > &args, int channels,
        int depth, const char *build_options, FLUSH_MODE finish_mode = DISABLE);

// bind oclMat to OpenCL image textures
// note:
//   1. there is no memory management. User need to explicitly release the resource
//   2. for faster clamping, there is no buffer padding for the constructed texture
CV_EXPORTS cl_mem bindTexture(const oclMat &mat);
CV_EXPORTS void releaseTexture(cl_mem& texture);

//Represents an image texture object
class CV_EXPORTS TextureCL
{
public:
    TextureCL(cl_mem tex, int r, int c, int t)
        : tex_(tex), rows(r), cols(c), type(t) {}
    ~TextureCL()
    {
        openCLFree(tex_);
    }
    operator cl_mem()
    {
        return tex_;
    }
    cl_mem const tex_;
    const int rows;
    const int cols;
    const int type;
private:
    //disable assignment
    void operator=(const TextureCL&);
};
// bind oclMat to OpenCL image textures and retunrs an TextureCL object
// note:
//   for faster clamping, there is no buffer padding for the constructed texture
CV_EXPORTS Ptr<TextureCL> bindTexturePtr(const oclMat &mat);

CV_EXPORTS bool isCpuDevice();

CV_EXPORTS size_t queryWaveFrontSize(cl_kernel kernel);


inline size_t divUp(size_t total, size_t grain)
{
    return (total + grain - 1) / grain;
}

inline size_t roundUp(size_t sz, size_t n)
{
    // we don't assume that n is a power of 2 (see alignSize)
    // equal to divUp(sz, n) * n
    size_t t = sz + n - 1;
    size_t rem = t % n;
    size_t result = t - rem;
    return result;
}

}//namespace ocl
}//namespace cv

#endif //__OPENCV_OCL_PRIVATE_UTIL__
