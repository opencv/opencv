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

#ifndef __OPENCV_GPU_VECMATH_HPP__
#define __OPENCV_GPU_VECMATH_HPP__

#include "cuda_shared.hpp"

namespace cv
{
    namespace gpu
    {
        template<typename T, int N> struct TypeVec;
        template<typename T> struct TypeVec<T, 1> { typedef T vec_t; };
        template<> struct TypeVec<unsigned char, 2> { typedef uchar2 vec_t; };
        template<> struct TypeVec<uchar2, 2> { typedef uchar2 vec_t; };
        template<> struct TypeVec<unsigned char, 3> { typedef uchar3 vec_t; };;
        template<> struct TypeVec<uchar3, 3> { typedef uchar3 vec_t; };
        template<> struct TypeVec<unsigned char, 4> { typedef uchar4 vec_t; };;
        template<> struct TypeVec<uchar4, 4> { typedef uchar4 vec_t; };
        template<> struct TypeVec<char, 2> { typedef char2 vec_t; };
        template<> struct TypeVec<char2, 2> { typedef char2 vec_t; };
        template<> struct TypeVec<char, 3> { typedef char3 vec_t; };
        template<> struct TypeVec<char3, 3> { typedef char3 vec_t; };
        template<> struct TypeVec<char, 4> { typedef char4 vec_t; };
        template<> struct TypeVec<char4, 4> { typedef char4 vec_t; };
        template<> struct TypeVec<unsigned short, 2> { typedef ushort2 vec_t; };
        template<> struct TypeVec<ushort2, 2> { typedef ushort2 vec_t; };
        template<> struct TypeVec<unsigned short, 3> { typedef ushort3 vec_t; };
        template<> struct TypeVec<ushort3, 3> { typedef ushort3 vec_t; };
        template<> struct TypeVec<unsigned short, 4> { typedef ushort4 vec_t; };
        template<> struct TypeVec<ushort4, 4> { typedef ushort4 vec_t; };
        template<> struct TypeVec<short, 2> { typedef short2 vec_t; };
        template<> struct TypeVec<short2, 2> { typedef short2 vec_t; };
        template<> struct TypeVec<short, 3> { typedef short3 vec_t; };
        template<> struct TypeVec<short3, 3> { typedef short3 vec_t; };
        template<> struct TypeVec<short, 4> { typedef short4 vec_t; };
        template<> struct TypeVec<short4, 4> { typedef short4 vec_t; };
        template<> struct TypeVec<unsigned int, 2> { typedef uint2 vec_t; };
        template<> struct TypeVec<uint2, 2> { typedef uint2 vec_t; };
        template<> struct TypeVec<unsigned int, 3> { typedef uint3 vec_t; };
        template<> struct TypeVec<uint3, 3> { typedef uint3 vec_t; };
        template<> struct TypeVec<unsigned int, 4> { typedef uint4 vec_t; };
        template<> struct TypeVec<uint4, 4> { typedef uint4 vec_t; };
        template<> struct TypeVec<int, 2> { typedef int2 vec_t; };
        template<> struct TypeVec<int2, 2> { typedef int2 vec_t; };
        template<> struct TypeVec<int, 3> { typedef int3 vec_t; };
        template<> struct TypeVec<int3, 3> { typedef int3 vec_t; };
        template<> struct TypeVec<int, 4> { typedef int4 vec_t; };
        template<> struct TypeVec<int4, 4> { typedef int4 vec_t; };
        template<> struct TypeVec<float, 2> { typedef float2 vec_t; };
        template<> struct TypeVec<float2, 2> { typedef float2 vec_t; };
        template<> struct TypeVec<float, 3> { typedef float3 vec_t; };
        template<> struct TypeVec<float3, 3> { typedef float3 vec_t; };
        template<> struct TypeVec<float, 4> { typedef float4 vec_t; };
        template<> struct TypeVec<float4, 4> { typedef float4 vec_t; };        

        static __device__ uchar4 operator+(const uchar4& a, const uchar4& b)
        {
            return make_uchar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        }
        static __device__ uchar4 operator-(const uchar4& a, const uchar4& b)
        {
            return make_uchar4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
        }
        static __device__ uchar4 operator*(const uchar4& a, const uchar4& b)
        {
            return make_uchar4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
        }
        static __device__ uchar4 operator/(const uchar4& a, const uchar4& b)
        {
            return make_uchar4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
        }
        template <typename T>
        static __device__ uchar4 operator*(const uchar4& a, T s)
        {
            return make_uchar4(a.x * s, a.y * s, a.z * s, a.w * s);
        }
        template <typename T>
        static __device__ uchar4 operator*(T s, const uchar4& a)
        {
            return a * s;
        }
    }
}

#endif // __OPENCV_GPU_VECMATH_HPP__