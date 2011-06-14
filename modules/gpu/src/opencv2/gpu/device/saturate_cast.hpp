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

#ifndef __OPENCV_GPU_SATURATE_CAST_HPP__
#define __OPENCV_GPU_SATURATE_CAST_HPP__

#include "internal_shared.hpp"

namespace cv
{
    namespace gpu
    {
        namespace device
        {
            template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(uchar v) { return _Tp(v); }
            template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(schar v) { return _Tp(v); }
            template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(ushort v) { return _Tp(v); }
            template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(short v) { return _Tp(v); }
            template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(uint v) { return _Tp(v); }
            template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(int v) { return _Tp(v); }
            template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(float v) { return _Tp(v); }
            template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(double v) { return _Tp(v); }

            template<> static __device__ __forceinline__ uchar saturate_cast<uchar>(schar v)
            { return (uchar)max((int)v, 0); }
            template<> static __device__ __forceinline__ uchar saturate_cast<uchar>(ushort v)
            { return (uchar)min((uint)v, (uint)UCHAR_MAX); }
            template<> static __device__ __forceinline__ uchar saturate_cast<uchar>(int v)
            { return (uchar)((uint)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
            template<> static __device__ __forceinline__ uchar saturate_cast<uchar>(uint v)
            { return (uchar)min(v, (uint)UCHAR_MAX); }
            template<> static __device__ __forceinline__ uchar saturate_cast<uchar>(short v)
            { return saturate_cast<uchar>((uint)v); }

            template<> static __device__ __forceinline__ uchar saturate_cast<uchar>(float v)
            { int iv = __float2int_rn(v); return saturate_cast<uchar>(iv); }
            template<> static __device__ __forceinline__ uchar saturate_cast<uchar>(double v)
            {
            #if defined (__CUDA_ARCH__) && __CUDA_ARCH__ >= 130
                int iv = __double2int_rn(v); return saturate_cast<uchar>(iv);
            #else
                return saturate_cast<uchar>((float)v);
            #endif
            }

            template<> static __device__ __forceinline__ schar saturate_cast<schar>(uchar v)
            { return (schar)min((int)v, SCHAR_MAX); }
            template<> static __device__ __forceinline__ schar saturate_cast<schar>(ushort v)
            { return (schar)min((uint)v, (uint)SCHAR_MAX); }
            template<> static __device__ __forceinline__ schar saturate_cast<schar>(int v)
            {
                return (schar)((uint)(v-SCHAR_MIN) <= (uint)UCHAR_MAX ?
                            v : v > 0 ? SCHAR_MAX : SCHAR_MIN);
            }
            template<> static __device__ __forceinline__ schar saturate_cast<schar>(short v)
            { return saturate_cast<schar>((int)v); }
            template<> static __device__ __forceinline__ schar saturate_cast<schar>(uint v)
            { return (schar)min(v, (uint)SCHAR_MAX); }

            template<> static __device__ __forceinline__ schar saturate_cast<schar>(float v)
            { int iv = __float2int_rn(v); return saturate_cast<schar>(iv); }
            template<> static __device__ __forceinline__ schar saturate_cast<schar>(double v)
            {             
            #if defined (__CUDA_ARCH__) && __CUDA_ARCH__ >= 130
                int iv = __double2int_rn(v); return saturate_cast<schar>(iv);
            #else
                return saturate_cast<schar>((float)v);
            #endif
            }

            template<> static __device__ __forceinline__ ushort saturate_cast<ushort>(schar v)
            { return (ushort)max((int)v, 0); }
            template<> static __device__ __forceinline__ ushort saturate_cast<ushort>(short v)
            { return (ushort)max((int)v, 0); }
            template<> static __device__ __forceinline__ ushort saturate_cast<ushort>(int v)
            { return (ushort)((uint)v <= (uint)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
            template<> static __device__ __forceinline__ ushort saturate_cast<ushort>(uint v)
            { return (ushort)min(v, (uint)USHRT_MAX); }
            template<> static __device__ __forceinline__ ushort saturate_cast<ushort>(float v)
            { int iv = __float2int_rn(v); return saturate_cast<ushort>(iv); }
            template<> static __device__ __forceinline__ ushort saturate_cast<ushort>(double v)
            {             
            #if defined (__CUDA_ARCH__) && __CUDA_ARCH__ >= 130
                int iv = __double2int_rn(v); return saturate_cast<ushort>(iv);
            #else
                return saturate_cast<ushort>((float)v);
            #endif
            }

            template<> static __device__ __forceinline__ short saturate_cast<short>(ushort v)
            { return (short)min((int)v, SHRT_MAX); }
            template<> static __device__ __forceinline__ short saturate_cast<short>(int v)
            {
                return (short)((uint)(v - SHRT_MIN) <= (uint)USHRT_MAX ?
                        v : v > 0 ? SHRT_MAX : SHRT_MIN);
            }
            template<> static __device__ __forceinline__ short saturate_cast<short>(uint v)
            { return (short)min(v, (uint)SHRT_MAX); }
            template<> static __device__ __forceinline__ short saturate_cast<short>(float v)
            { int iv = __float2int_rn(v); return saturate_cast<short>(iv); }
            template<> static __device__ __forceinline__ short saturate_cast<short>(double v)
            {            
            #if defined (__CUDA_ARCH__) && __CUDA_ARCH__ >= 130
                int iv = __double2int_rn(v); return saturate_cast<short>(iv);
            #else
                return saturate_cast<short>((float)v);
            #endif
            }

            template<> static __device__ __forceinline__ int saturate_cast<int>(float v) { return __float2int_rn(v); }
            template<> static __device__ __forceinline__ int saturate_cast<int>(double v) 
            {
            #if defined (__CUDA_ARCH__) && __CUDA_ARCH__ >= 130 
                return __double2int_rn(v);
            #else
                return saturate_cast<int>((float)v);
            #endif
            }

            template<> static __device__ __forceinline__ uint saturate_cast<uint>(float v){ return __float2uint_rn(v); }
            template<> static __device__ __forceinline__ uint saturate_cast<uint>(double v) 
            {            
            #if defined (__CUDA_ARCH__) && __CUDA_ARCH__ >= 130
                return __double2uint_rn(v);
            #else
                return saturate_cast<uint>((float)v);
            #endif
            }
        }
    }
}

#endif /* __OPENCV_GPU_SATURATE_CAST_HPP__ */