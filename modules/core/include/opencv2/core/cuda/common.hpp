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

#ifndef __OPENCV_CUDA_COMMON_HPP__
#define __OPENCV_CUDA_COMMON_HPP__

#include <cuda_runtime.h>
#include "opencv2/core/cuda_types.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/base.hpp"

#ifndef CV_PI_F
    #ifndef CV_PI
        #define CV_PI_F 3.14159265f
    #else
        #define CV_PI_F ((float)CV_PI)
    #endif
#endif

namespace cv { namespace cuda {
    static inline void checkCudaError(cudaError_t err, const char* file, const int line, const char* func)
    {
        if (cudaSuccess != err)
            cv::error(cv::Error::GpuApiCallError, cudaGetErrorString(err), func, file, line);
    }
}}

#ifndef cudaSafeCall
    #define cudaSafeCall(expr)  cv::cuda::checkCudaError(expr, __FILE__, __LINE__, CV_Func)
#endif

namespace cv { namespace cuda
{
    template <typename T> static inline bool isAligned(const T* ptr, size_t size)
    {
        return reinterpret_cast<size_t>(ptr) % size == 0;
    }

    static inline bool isAligned(size_t step, size_t size)
    {
        return step % size == 0;
    }
}}

namespace cv { namespace cuda
{
    namespace device
    {
        __host__ __device__ __forceinline__ int divUp(int total, int grain)
        {
            return (total + grain - 1) / grain;
        }

        template<class T> inline void bindTexture(const textureReference* tex, const PtrStepSz<T>& img)
        {
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
            cudaSafeCall( cudaBindTexture2D(0, tex, img.ptr(), &desc, img.cols, img.rows, img.step) );
        }
    }
}}



#endif // __OPENCV_CUDA_COMMON_HPP__
