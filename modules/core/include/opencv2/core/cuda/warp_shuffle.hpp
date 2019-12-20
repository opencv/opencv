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

#ifndef OPENCV_CUDA_WARP_SHUFFLE_HPP
#define OPENCV_CUDA_WARP_SHUFFLE_HPP

/** @file
 * @deprecated Use @ref cudev instead.
 */

//! @cond IGNORED

namespace cv { namespace cuda { namespace device
{
#if __CUDACC_VER_MAJOR__ >= 9
#  define __shfl(x, y, z) __shfl_sync(0xFFFFFFFFU, x, y, z)
#  define __shfl_up(x, y, z) __shfl_up_sync(0xFFFFFFFFU, x, y, z)
#  define __shfl_down(x, y, z) __shfl_down_sync(0xFFFFFFFFU, x, y, z)
#endif
    template <typename T>
    __device__ __forceinline__ T shfl(T val, int srcLane, int width = warpSize)
    {
    #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
        return __shfl(val, srcLane, width);
    #else
        return T();
    #endif
    }
    __device__ __forceinline__ unsigned int shfl(unsigned int val, int srcLane, int width = warpSize)
    {
    #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
        return (unsigned int) __shfl((int) val, srcLane, width);
    #else
        return 0;
    #endif
    }
    __device__ __forceinline__ double shfl(double val, int srcLane, int width = warpSize)
    {
    #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
        int lo = __double2loint(val);
        int hi = __double2hiint(val);

        lo = __shfl(lo, srcLane, width);
        hi = __shfl(hi, srcLane, width);

        return __hiloint2double(hi, lo);
    #else
        return 0.0;
    #endif
    }

    template <typename T>
    __device__ __forceinline__ T shfl_down(T val, unsigned int delta, int width = warpSize)
    {
    #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
        return __shfl_down(val, delta, width);
    #else
        return T();
    #endif
    }
    __device__ __forceinline__ unsigned int shfl_down(unsigned int val, unsigned int delta, int width = warpSize)
    {
    #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
        return (unsigned int) __shfl_down((int) val, delta, width);
    #else
        return 0;
    #endif
    }
    __device__ __forceinline__ double shfl_down(double val, unsigned int delta, int width = warpSize)
    {
    #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
        int lo = __double2loint(val);
        int hi = __double2hiint(val);

        lo = __shfl_down(lo, delta, width);
        hi = __shfl_down(hi, delta, width);

        return __hiloint2double(hi, lo);
    #else
        return 0.0;
    #endif
    }

    template <typename T>
    __device__ __forceinline__ T shfl_up(T val, unsigned int delta, int width = warpSize)
    {
    #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
        return __shfl_up(val, delta, width);
    #else
        return T();
    #endif
    }
    __device__ __forceinline__ unsigned int shfl_up(unsigned int val, unsigned int delta, int width = warpSize)
    {
    #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
        return (unsigned int) __shfl_up((int) val, delta, width);
    #else
        return 0;
    #endif
    }
    __device__ __forceinline__ double shfl_up(double val, unsigned int delta, int width = warpSize)
    {
    #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
        int lo = __double2loint(val);
        int hi = __double2hiint(val);

        lo = __shfl_up(lo, delta, width);
        hi = __shfl_up(hi, delta, width);

        return __hiloint2double(hi, lo);
    #else
        return 0.0;
    #endif
    }
}}}

#  undef __shfl
#  undef __shfl_up
#  undef __shfl_down

//! @endcond

#endif // OPENCV_CUDA_WARP_SHUFFLE_HPP
