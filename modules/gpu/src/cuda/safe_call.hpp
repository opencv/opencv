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

#include "cvconfig.h"

#include <cuda_runtime_api.h>

#ifdef HAVE_CUFFT
#   include <cufft.h>
#endif

#ifdef HAVE_CUBLAS
#   include <cublas.h>
#endif

#include "NCV.hpp"

namespace cv { namespace gpu {

void nppError(int err, const char *file, const int line, const char *func = "");

void ncvError(int err, const char *file, const int line, const char *func = "");

#ifdef HAVE_CUFFT
    void cufftError(int err, const char *file, const int line, const char *func = "");
#endif

#ifdef HAVE_CUBLAS
    void cublasError(int err, const char *file, const int line, const char *func = "");
#endif

}}

// nppSafeCall

static inline void ___nppSafeCall(int err, const char *file, const int line, const char *func = "")
{
    if (err < 0)
        cv::gpu::nppError(err, file, line, func);
}

#define nppSafeCall(expr)  ___nppSafeCall(expr, __FILE__, __LINE__, CV_Func)

// ncvSafeCall

static inline void ___ncvSafeCall(int err, const char *file, const int line, const char *func = "")
{
    if (NCV_SUCCESS != err)
        cv::gpu::ncvError(err, file, line, func);
}

#define ncvSafeCall(expr)  ___ncvSafeCall(expr, __FILE__, __LINE__, CV_Func)

// cufftSafeCall

#ifdef HAVE_CUFFT
    static inline void ___cufftSafeCall(cufftResult_t err, const char *file, const int line, const char *func = "")
    {
        if (CUFFT_SUCCESS != err)
            cv::gpu::cufftError(err, file, line, func);
    }

#define cufftSafeCall(expr)  ___cufftSafeCall(expr, __FILE__, __LINE__, CV_Func)
#endif

// cublasSafeCall

#ifdef HAVE_CUBLAS
    static inline void ___cublasSafeCall(cublasStatus_t err, const char *file, const int line, const char *func = "")
    {
        if (CUBLAS_STATUS_SUCCESS != err)
            cv::gpu::cublasError(err, file, line, func);
    }

#define cublasSafeCall(expr)  ___cublasSafeCall(expr, __FILE__, __LINE__, CV_Func)
#endif

#endif /* __OPENCV_CUDA_SAFE_CALL_HPP__ */
