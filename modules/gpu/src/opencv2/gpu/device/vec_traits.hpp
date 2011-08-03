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

#ifndef __OPENCV_GPU_VEC_TRAITS_HPP__
#define __OPENCV_GPU_VEC_TRAITS_HPP__

#include "internal_shared.hpp"

namespace cv { namespace gpu { namespace device
{
    template<typename T, int N> struct TypeVec;

#define OPENCV_GPU_IMPLEMENT_TYPE_VEC(type) \
    template<> struct TypeVec<type, 1> { typedef type vec_type; }; \
    template<> struct TypeVec<type ## 1, 1> { typedef type ## 1 vec_type; }; \
    template<> struct TypeVec<type, 2> { typedef type ## 2 vec_type; }; \
    template<> struct TypeVec<type ## 2, 2> { typedef type ## 2 vec_type; }; \
    template<> struct TypeVec<type, 3> { typedef type ## 3 vec_type; }; \
    template<> struct TypeVec<type ## 3, 3> { typedef type ## 3 vec_type; }; \
    template<> struct TypeVec<type, 4> { typedef type ## 4 vec_type; }; \
    template<> struct TypeVec<type ## 4, 4> { typedef type ## 4 vec_type; };

    OPENCV_GPU_IMPLEMENT_TYPE_VEC(uchar)
    OPENCV_GPU_IMPLEMENT_TYPE_VEC(char)
    OPENCV_GPU_IMPLEMENT_TYPE_VEC(ushort)
    OPENCV_GPU_IMPLEMENT_TYPE_VEC(short)
    OPENCV_GPU_IMPLEMENT_TYPE_VEC(int)
    OPENCV_GPU_IMPLEMENT_TYPE_VEC(uint)
    OPENCV_GPU_IMPLEMENT_TYPE_VEC(float)
    OPENCV_GPU_IMPLEMENT_TYPE_VEC(double)

#undef OPENCV_GPU_IMPLEMENT_TYPE_VEC

    template<> struct TypeVec<schar, 1> { typedef schar vec_type; };
    template<> struct TypeVec<schar, 2> { typedef char2 vec_type; };
    template<> struct TypeVec<schar, 3> { typedef char3 vec_type; };
    template<> struct TypeVec<schar, 4> { typedef char4 vec_type; };

    template<> struct TypeVec<bool, 1> { typedef uchar vec_type; };
    template<> struct TypeVec<bool, 2> { typedef uchar2 vec_type; };
    template<> struct TypeVec<bool, 3> { typedef uchar3 vec_type; };
    template<> struct TypeVec<bool, 4> { typedef uchar4 vec_type; };

    template<typename T> struct VecTraits;

#define OPENCV_GPU_IMPLEMENT_VEC_TRAITS(type) \
    template<> struct VecTraits<type> \
    { \
        typedef type elem_type; \
        enum {cn=1}; \
        static __device__ __host__ type all(type v) {return v;} \
        static __device__ __host__ type make(type x) {return x;} \
    }; \
    template<> struct VecTraits<type ## 1> \
    { \
        typedef type elem_type; \
        enum {cn=1}; \
        static __device__ __host__ type ## 1 all(type v) {return make_ ## type ## 1(v);} \
        static __device__ __host__ type ## 1 make(type x) {return make_ ## type ## 1(x);} \
    }; \
    template<> struct VecTraits<type ## 2> \
    { \
        typedef type elem_type; \
        enum {cn=2}; \
        static __device__ __host__ type ## 2 all(type v) {return make_ ## type ## 2(v, v);} \
        static __device__ __host__ type ## 2 make(type x, type y) {return make_ ## type ## 2(x, y);} \
    }; \
    template<> struct VecTraits<type ## 3> \
    { \
        typedef type elem_type; \
        enum {cn=3}; \
        static __device__ __host__ type ## 3 all(type v) {return make_ ## type ## 3(v, v, v);} \
        static __device__ __host__ type ## 3 make(type x, type y, type z) {return make_ ## type ## 3(x, y, z);} \
    }; \
    template<> struct VecTraits<type ## 4> \
    { \
        typedef type elem_type; \
        enum {cn=4}; \
        static __device__ __host__ type ## 4 all(type v) {return make_ ## type ## 4(v, v, v, v);} \
        static __device__ __host__ type ## 4 make(type x, type y, type z, type w) {return make_ ## type ## 4(x, y, z, w);} \
    };

    OPENCV_GPU_IMPLEMENT_VEC_TRAITS(uchar)
    OPENCV_GPU_IMPLEMENT_VEC_TRAITS(char)
    OPENCV_GPU_IMPLEMENT_VEC_TRAITS(ushort)
    OPENCV_GPU_IMPLEMENT_VEC_TRAITS(short)
    OPENCV_GPU_IMPLEMENT_VEC_TRAITS(int)
    OPENCV_GPU_IMPLEMENT_VEC_TRAITS(uint)
    OPENCV_GPU_IMPLEMENT_VEC_TRAITS(float)
    OPENCV_GPU_IMPLEMENT_VEC_TRAITS(double)

#undef OPENCV_GPU_IMPLEMENT_VEC_TRAITS

    template<> struct VecTraits<schar> 
    { 
        typedef schar elem_type; 
        enum {cn=1}; 
        static __device__ __host__ schar all(schar v) {return v;}
        static __device__ __host__ schar make(schar x) {return x;}
    };
}}}

#endif // __OPENCV_GPU_VEC_TRAITS_HPP__
