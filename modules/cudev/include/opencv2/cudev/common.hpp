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

#pragma once

#ifndef __OPENCV_CUDEV_COMMON_HPP__
#define __OPENCV_CUDEV_COMMON_HPP__

#include <cuda_runtime.h>
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"

namespace cv { namespace cudev {

using namespace cv::cuda;

// CV_CUDEV_ARCH

#ifndef __CUDA_ARCH__
#   define CV_CUDEV_ARCH 0
#else
#   define CV_CUDEV_ARCH __CUDA_ARCH__
#endif

// CV_CUDEV_SAFE_CALL

__host__ __forceinline__ void checkCudaError(cudaError_t err, const char* file, const int line, const char* func)
{
    if (cudaSuccess != err)
        cv::error(cv::Error::GpuApiCallError, cudaGetErrorString(err), func, file, line);
}

#define CV_CUDEV_SAFE_CALL(expr) cv::cudev::checkCudaError((expr), __FILE__, __LINE__, CV_Func)

// divUp

__host__ __device__ __forceinline__ int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

// math constants

#define CV_PI_F   ((float)CV_PI)
#define CV_LOG2_F ((float)CV_LOG2)

}}

#endif
