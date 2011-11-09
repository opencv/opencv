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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_CUDA_SAFE_CALL_HPP__
#define __OPENCV_CUDA_SAFE_CALL_HPP__

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cublas.h>
#include "NCV.hpp"

#if defined(__GNUC__)
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
    #define nppSafeCall(expr)  ___nppSafeCall(expr, __FILE__, __LINE__, __func__)
    #define ncvSafeCall(expr)  ___ncvSafeCall(expr, __FILE__, __LINE__, __func__)
    #define cufftSafeCall(expr)  ___cufftSafeCall(expr, __FILE__, __LINE__, __func__)
    #define cublasSafeCall(expr)  ___cublasSafeCall(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__)
    #define nppSafeCall(expr)  ___nppSafeCall(expr, __FILE__, __LINE__)
    #define ncvSafeCall(expr)  ___ncvSafeCall(expr, __FILE__, __LINE__)
    #define cufftSafeCall(expr)  ___cufftSafeCall(expr, __FILE__, __LINE__)
    #define cublasSafeCall(expr)  ___cublasSafeCall(expr, __FILE__, __LINE__)
#endif

namespace cv { namespace gpu {

void error(const char *error_string, const char *file, const int line, const char *func = "");
void nppError(int err, const char *file, const int line, const char *func = "");
void ncvError(int err, const char *file, const int line, const char *func = "");
void cufftError(int err, const char *file, const int line, const char *func = "");
void cublasError(int err, const char *file, const int line, const char *func = "");

static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err)
        cv::gpu::error(cudaGetErrorString(err), file, line, func);
}

static inline void ___nppSafeCall(int err, const char *file, const int line, const char *func = "")
{
    if (err < 0)
        cv::gpu::nppError(err, file, line, func);
}

static inline void ___ncvSafeCall(int err, const char *file, const int line, const char *func = "")
{
    if (NCV_SUCCESS != err)
        cv::gpu::ncvError(err, file, line, func);
}

static inline void ___cufftSafeCall(cufftResult_t err, const char *file, const int line, const char *func = "")
{
    if (CUFFT_SUCCESS != err)
        cv::gpu::cufftError(err, file, line, func);
}

static inline void ___cublasSafeCall(cublasStatus_t err, const char *file, const int line, const char *func = "")
{
    if (CUBLAS_STATUS_SUCCESS != err)
        cv::gpu::cublasError(err, file, line, func);
}

}}

#endif /* __OPENCV_CUDA_SAFE_CALL_HPP__ */