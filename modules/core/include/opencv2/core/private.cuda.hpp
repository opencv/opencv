/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
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

#ifndef OPENCV_CORE_PRIVATE_CUDA_HPP
#define OPENCV_CORE_PRIVATE_CUDA_HPP

#ifndef __OPENCV_BUILD
#  error this is a private header which should not be used from outside of the OpenCV library
#endif

#include "cvconfig.h"

#include "opencv2/core/cvdef.h"
#include "opencv2/core/base.hpp"

#include "opencv2/core/cuda.hpp"

#ifdef HAVE_CUDA
#  include <cuda.h>
#  include <cuda_runtime.h>
#  if defined(__CUDACC_VER_MAJOR__) && (8 <= __CUDACC_VER_MAJOR__)
#    if defined (__GNUC__) && !defined(__CUDACC__)
#     pragma GCC diagnostic push
#     pragma GCC diagnostic ignored "-Wstrict-aliasing"
#     include <cuda_fp16.h>
#     pragma GCC diagnostic pop
#    else
#     include <cuda_fp16.h>
#    endif
#  endif // defined(__CUDACC_VER_MAJOR__) && (8 <= __CUDACC_VER_MAJOR__)
#  include <npp.h>
#  include "opencv2/core/cuda_stream_accessor.hpp"
#  include "opencv2/core/cuda/common.hpp"

# ifndef NPP_VERSION
#  define NPP_VERSION (NPP_VERSION_MAJOR * 1000 + NPP_VERSION_MINOR * 100 + NPP_VERSION_BUILD)
# endif

#  define CUDART_MINIMUM_REQUIRED_VERSION 6050

#  if (CUDART_VERSION < CUDART_MINIMUM_REQUIRED_VERSION)
#    error "Insufficient Cuda Runtime library version, please update it."
#  endif

#endif

//! @cond IGNORED

namespace cv { namespace cuda {
    CV_EXPORTS cv::String getNppErrorMessage(int code);
    CV_EXPORTS cv::String getCudaDriverApiErrorMessage(int code);

    CV_EXPORTS GpuMat getInputMat(InputArray _src, Stream& stream);

    CV_EXPORTS GpuMat getOutputMat(OutputArray _dst, int rows, int cols, int type, Stream& stream);
    static inline GpuMat getOutputMat(OutputArray _dst, Size size, int type, Stream& stream)
    {
        return getOutputMat(_dst, size.height, size.width, type, stream);
    }

    CV_EXPORTS void syncOutput(const GpuMat& dst, OutputArray _dst, Stream& stream);
}}

#ifndef HAVE_CUDA

static inline CV_NORETURN void throw_no_cuda() { CV_Error(cv::Error::GpuNotSupported, "The library is compiled without CUDA support"); }

#else // HAVE_CUDA

#define nppSafeSetStream(oldStream, newStream) { if(oldStream != newStream) { cudaStreamSynchronize(oldStream); nppSetStream(newStream); } }

static inline CV_NORETURN void throw_no_cuda() { CV_Error(cv::Error::StsNotImplemented, "The called functionality is disabled for current build or platform"); }

namespace cv { namespace cuda
{
    static inline void checkNppError(int code, const char* file, const int line, const char* func)
    {
        if (code < 0)
            cv::error(cv::Error::GpuApiCallError, getNppErrorMessage(code), func, file, line);
    }

    static inline void checkCudaDriverApiError(int code, const char* file, const int line, const char* func)
    {
        if (code != CUDA_SUCCESS)
            cv::error(cv::Error::GpuApiCallError, getCudaDriverApiErrorMessage(code), func, file, line);
    }

    template<int n> struct NPPTypeTraits;
    template<> struct NPPTypeTraits<CV_8U>  { typedef Npp8u npp_type; };
    template<> struct NPPTypeTraits<CV_8S>  { typedef Npp8s npp_type; };
    template<> struct NPPTypeTraits<CV_16U> { typedef Npp16u npp_type; };
    template<> struct NPPTypeTraits<CV_16S> { typedef Npp16s npp_type; };
    template<> struct NPPTypeTraits<CV_32S> { typedef Npp32s npp_type; };
    template<> struct NPPTypeTraits<CV_32F> { typedef Npp32f npp_type; };
    template<> struct NPPTypeTraits<CV_64F> { typedef Npp64f npp_type; };

#define nppSafeCall(expr)  cv::cuda::checkNppError(expr, __FILE__, __LINE__, CV_Func)
// NppStreamContext is introduced in NPP version 10100 included in CUDA toolkit 10.1 (CUDA_VERSION == 10010) however not all of the NPP functions called internally by OpenCV
// - have an NppStreamContext argument (e.g. nppiHistogramEvenGetBufferSize_8u_C1R_Ctx in CUDA 12.3) and/or
// - have a corresponding function in the supplied library (e.g. nppiEvenLevelsHost_32s_Ctx is not present in nppist.lib or libnppist.so as of CUDA 12.6)
// Because support for these functions has gradually been introduced without being mentioned in the release notes this flag is set to a version of NPP (version 12205 included in CUDA toolkit 12.4) which is known to work.
#define USE_NPP_STREAM_CTX NPP_VERSION >= 12205
#if USE_NPP_STREAM_CTX
    class NppStreamHandler
    {
    public:
        inline explicit NppStreamHandler(cudaStream_t newStream)
        {
            nppStreamContext = {};
            #if CUDA_VERSION < 12090
                nppSafeCall(nppGetStreamContext(&nppStreamContext));
            #else
                int device = 0;
                cudaSafeCall(cudaGetDevice(&device));

                cudaDeviceProp prop{};
                cudaSafeCall(cudaGetDeviceProperties(&prop, device));

                nppStreamContext.nCudaDeviceId = device;
                nppStreamContext.nMultiProcessorCount = prop.multiProcessorCount;
                nppStreamContext.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
                nppStreamContext.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
                nppStreamContext.nSharedMemPerBlock = prop.sharedMemPerBlock;
                nppStreamContext.nCudaDevAttrComputeCapabilityMajor = prop.major;
                nppStreamContext.nCudaDevAttrComputeCapabilityMinor = prop.minor;
            #endif
            nppStreamContext.hStream = newStream;
            cudaSafeCall(cudaStreamGetFlags(nppStreamContext.hStream, &nppStreamContext.nStreamFlags));
        }

        inline explicit NppStreamHandler(Stream& newStream) : NppStreamHandler(StreamAccessor::getStream(newStream)) {}

        inline operator NppStreamContext() const {
            return nppStreamContext;
        }

        inline NppStreamContext get() { return nppStreamContext; }

    private:
        NppStreamContext nppStreamContext;
    };
#else
    class NppStreamHandler
    {
    public:
        inline explicit NppStreamHandler(Stream& newStream)
        {
            oldStream = nppGetStream();
            nppSafeSetStream(oldStream, StreamAccessor::getStream(newStream));
        }

        inline explicit NppStreamHandler(cudaStream_t newStream)
        {
            oldStream = nppGetStream();
            nppSafeSetStream(oldStream, newStream);
        }

        inline ~NppStreamHandler()
        {
            nppSafeSetStream(nppGetStream(), oldStream);
        }

    private:
        cudaStream_t oldStream;
    };
#endif
}}

#define cuSafeCall(expr)  cv::cuda::checkCudaDriverApiError(expr, __FILE__, __LINE__, CV_Func)

#endif // HAVE_CUDA

//! @endcond

#endif // OPENCV_CORE_PRIVATE_CUDA_HPP
