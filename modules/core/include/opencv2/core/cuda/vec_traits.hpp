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

#ifndef OPENCV_CUDA_VEC_TRAITS_HPP
#define OPENCV_CUDA_VEC_TRAITS_HPP

#include "common.hpp"
#include "cuda_compat.hpp"

/** @file
 * @deprecated Use @ref cudev instead.
 */

//! @cond IGNORED

namespace cv { namespace cuda { namespace device
{
    using cv::cuda::device::compat::double4;
    using cv::cuda::device::compat::make_double4;

    template<typename T, int N> struct TypeVec;

    struct __align__(8) uchar8
    {
        uchar a0, a1, a2, a3, a4, a5, a6, a7;
    };
    static __host__ __device__ __forceinline__ uchar8 make_uchar8(uchar a0, uchar a1, uchar a2, uchar a3, uchar a4, uchar a5, uchar a6, uchar a7)
    {
        uchar8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
        return val;
    }
    struct __align__(8) char8
    {
        schar a0, a1, a2, a3, a4, a5, a6, a7;
    };
    static __host__ __device__ __forceinline__ char8 make_char8(schar a0, schar a1, schar a2, schar a3, schar a4, schar a5, schar a6, schar a7)
    {
        char8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
        return val;
    }
    struct __align__(16) ushort8
    {
        ushort a0, a1, a2, a3, a4, a5, a6, a7;
    };
    static __host__ __device__ __forceinline__ ushort8 make_ushort8(ushort a0, ushort a1, ushort a2, ushort a3, ushort a4, ushort a5, ushort a6, ushort a7)
    {
        ushort8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
        return val;
    }
    struct __align__(16) short8
    {
        short a0, a1, a2, a3, a4, a5, a6, a7;
    };
    static __host__ __device__ __forceinline__ short8 make_short8(short a0, short a1, short a2, short a3, short a4, short a5, short a6, short a7)
    {
        short8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
        return val;
    }
    struct __align__(32) uint8
    {
        uint a0, a1, a2, a3, a4, a5, a6, a7;
    };
    static __host__ __device__ __forceinline__ uint8 make_uint8(uint a0, uint a1, uint a2, uint a3, uint a4, uint a5, uint a6, uint a7)
    {
        uint8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
        return val;
    }
    struct __align__(32) int8
    {
        int a0, a1, a2, a3, a4, a5, a6, a7;
    };
    static __host__ __device__ __forceinline__ int8 make_int8(int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7)
    {
        int8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
        return val;
    }
    struct __align__(32) float8
    {
        float a0, a1, a2, a3, a4, a5, a6, a7;
    };
    static __host__ __device__ __forceinline__ float8 make_float8(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7)
    {
        float8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
        return val;
    }
    struct double8
    {
        double a0, a1, a2, a3, a4, a5, a6, a7;
    };
    static __host__ __device__ __forceinline__ double8 make_double8(double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7)
    {
        double8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
        return val;
    }

#define OPENCV_CUDA_IMPLEMENT_TYPE_VEC(type) \
    template<> struct TypeVec<type, 1> { typedef type vec_type; }; \
    template<> struct TypeVec<type ## 1, 1> { typedef type ## 1 vec_type; }; \
    template<> struct TypeVec<type, 2> { typedef type ## 2 vec_type; }; \
    template<> struct TypeVec<type ## 2, 2> { typedef type ## 2 vec_type; }; \
    template<> struct TypeVec<type, 3> { typedef type ## 3 vec_type; }; \
    template<> struct TypeVec<type ## 3, 3> { typedef type ## 3 vec_type; }; \
    template<> struct TypeVec<type, 4> { typedef type ## 4 vec_type; }; \
    template<> struct TypeVec<type ## 4, 4> { typedef type ## 4 vec_type; }; \
    template<> struct TypeVec<type, 8> { typedef type ## 8 vec_type; }; \
    template<> struct TypeVec<type ## 8, 8> { typedef type ## 8 vec_type; };

    OPENCV_CUDA_IMPLEMENT_TYPE_VEC(uchar)
    OPENCV_CUDA_IMPLEMENT_TYPE_VEC(char)
    OPENCV_CUDA_IMPLEMENT_TYPE_VEC(ushort)
    OPENCV_CUDA_IMPLEMENT_TYPE_VEC(short)
    OPENCV_CUDA_IMPLEMENT_TYPE_VEC(int)
    OPENCV_CUDA_IMPLEMENT_TYPE_VEC(uint)
    OPENCV_CUDA_IMPLEMENT_TYPE_VEC(float)
    OPENCV_CUDA_IMPLEMENT_TYPE_VEC(double)

    #undef OPENCV_CUDA_IMPLEMENT_TYPE_VEC

    template<> struct TypeVec<schar, 1> { typedef schar vec_type; };
    template<> struct TypeVec<schar, 2> { typedef char2 vec_type; };
    template<> struct TypeVec<schar, 3> { typedef char3 vec_type; };
    template<> struct TypeVec<schar, 4> { typedef char4 vec_type; };
    template<> struct TypeVec<schar, 8> { typedef char8 vec_type; };

    template<> struct TypeVec<bool, 1> { typedef uchar vec_type; };
    template<> struct TypeVec<bool, 2> { typedef uchar2 vec_type; };
    template<> struct TypeVec<bool, 3> { typedef uchar3 vec_type; };
    template<> struct TypeVec<bool, 4> { typedef uchar4 vec_type; };
    template<> struct TypeVec<bool, 8> { typedef uchar8 vec_type; };

    template<typename T> struct VecTraits;

#define OPENCV_CUDA_IMPLEMENT_VEC_TRAITS(type) \
    template<> struct VecTraits<type> \
    { \
        typedef type elem_type; \
        enum {cn=1}; \
        static __device__ __host__ __forceinline__ type all(type v) {return v;} \
        static __device__ __host__ __forceinline__ type make(type x) {return x;} \
        static __device__ __host__ __forceinline__ type make(const type* v) {return *v;} \
    }; \
    template<> struct VecTraits<type ## 1> \
    { \
        typedef type elem_type; \
        enum {cn=1}; \
        static __device__ __host__ __forceinline__ type ## 1 all(type v) {return make_ ## type ## 1(v);} \
        static __device__ __host__ __forceinline__ type ## 1 make(type x) {return make_ ## type ## 1(x);} \
        static __device__ __host__ __forceinline__ type ## 1 make(const type* v) {return make_ ## type ## 1(*v);} \
    }; \
    template<> struct VecTraits<type ## 2> \
    { \
        typedef type elem_type; \
        enum {cn=2}; \
        static __device__ __host__ __forceinline__ type ## 2 all(type v) {return make_ ## type ## 2(v, v);} \
        static __device__ __host__ __forceinline__ type ## 2 make(type x, type y) {return make_ ## type ## 2(x, y);} \
        static __device__ __host__ __forceinline__ type ## 2 make(const type* v) {return make_ ## type ## 2(v[0], v[1]);} \
    }; \
    template<> struct VecTraits<type ## 3> \
    { \
        typedef type elem_type; \
        enum {cn=3}; \
        static __device__ __host__ __forceinline__ type ## 3 all(type v) {return make_ ## type ## 3(v, v, v);} \
        static __device__ __host__ __forceinline__ type ## 3 make(type x, type y, type z) {return make_ ## type ## 3(x, y, z);} \
        static __device__ __host__ __forceinline__ type ## 3 make(const type* v) {return make_ ## type ## 3(v[0], v[1], v[2]);} \
    }; \
    template<> struct VecTraits<type ## 4> \
    { \
        typedef type elem_type; \
        enum {cn=4}; \
        static __device__ __host__ __forceinline__ type ## 4 all(type v) {return make_ ## type ## 4(v, v, v, v);} \
        static __device__ __host__ __forceinline__ type ## 4 make(type x, type y, type z, type w) {return make_ ## type ## 4(x, y, z, w);} \
        static __device__ __host__ __forceinline__ type ## 4 make(const type* v) {return make_ ## type ## 4(v[0], v[1], v[2], v[3]);} \
    }; \
    template<> struct VecTraits<type ## 8> \
    { \
        typedef type elem_type; \
        enum {cn=8}; \
        static __device__ __host__ __forceinline__ type ## 8 all(type v) {return make_ ## type ## 8(v, v, v, v, v, v, v, v);} \
        static __device__ __host__ __forceinline__ type ## 8 make(type a0, type a1, type a2, type a3, type a4, type a5, type a6, type a7) {return make_ ## type ## 8(a0, a1, a2, a3, a4, a5, a6, a7);} \
        static __device__ __host__ __forceinline__ type ## 8 make(const type* v) {return make_ ## type ## 8(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);} \
    };

    OPENCV_CUDA_IMPLEMENT_VEC_TRAITS(uchar)
    OPENCV_CUDA_IMPLEMENT_VEC_TRAITS(ushort)
    OPENCV_CUDA_IMPLEMENT_VEC_TRAITS(short)
    OPENCV_CUDA_IMPLEMENT_VEC_TRAITS(int)
    OPENCV_CUDA_IMPLEMENT_VEC_TRAITS(uint)
    OPENCV_CUDA_IMPLEMENT_VEC_TRAITS(float)
    OPENCV_CUDA_IMPLEMENT_VEC_TRAITS(double)

    #undef OPENCV_CUDA_IMPLEMENT_VEC_TRAITS

    template<> struct VecTraits<char>
    {
        typedef char elem_type;
        enum {cn=1};
        static __device__ __host__ __forceinline__ char all(char v) {return v;}
        static __device__ __host__ __forceinline__ char make(char x) {return x;}
        static __device__ __host__ __forceinline__ char make(const char* x) {return *x;}
    };
    template<> struct VecTraits<schar>
    {
        typedef schar elem_type;
        enum {cn=1};
        static __device__ __host__ __forceinline__ schar all(schar v) {return v;}
        static __device__ __host__ __forceinline__ schar make(schar x) {return x;}
        static __device__ __host__ __forceinline__ schar make(const schar* x) {return *x;}
    };
    template<> struct VecTraits<char1>
    {
        typedef schar elem_type;
        enum {cn=1};
        static __device__ __host__ __forceinline__ char1 all(schar v) {return make_char1(v);}
        static __device__ __host__ __forceinline__ char1 make(schar x) {return make_char1(x);}
        static __device__ __host__ __forceinline__ char1 make(const schar* v) {return make_char1(v[0]);}
    };
    template<> struct VecTraits<char2>
    {
        typedef schar elem_type;
        enum {cn=2};
        static __device__ __host__ __forceinline__ char2 all(schar v) {return make_char2(v, v);}
        static __device__ __host__ __forceinline__ char2 make(schar x, schar y) {return make_char2(x, y);}
        static __device__ __host__ __forceinline__ char2 make(const schar* v) {return make_char2(v[0], v[1]);}
    };
    template<> struct VecTraits<char3>
    {
        typedef schar elem_type;
        enum {cn=3};
        static __device__ __host__ __forceinline__ char3 all(schar v) {return make_char3(v, v, v);}
        static __device__ __host__ __forceinline__ char3 make(schar x, schar y, schar z) {return make_char3(x, y, z);}
        static __device__ __host__ __forceinline__ char3 make(const schar* v) {return make_char3(v[0], v[1], v[2]);}
    };
    template<> struct VecTraits<char4>
    {
        typedef schar elem_type;
        enum {cn=4};
        static __device__ __host__ __forceinline__ char4 all(schar v) {return make_char4(v, v, v, v);}
        static __device__ __host__ __forceinline__ char4 make(schar x, schar y, schar z, schar w) {return make_char4(x, y, z, w);}
        static __device__ __host__ __forceinline__ char4 make(const schar* v) {return make_char4(v[0], v[1], v[2], v[3]);}
    };
    template<> struct VecTraits<char8>
    {
        typedef schar elem_type;
        enum {cn=8};
        static __device__ __host__ __forceinline__ char8 all(schar v) {return make_char8(v, v, v, v, v, v, v, v);}
        static __device__ __host__ __forceinline__ char8 make(schar a0, schar a1, schar a2, schar a3, schar a4, schar a5, schar a6, schar a7) {return make_char8(a0, a1, a2, a3, a4, a5, a6, a7);}
        static __device__ __host__ __forceinline__ char8 make(const schar* v) {return make_char8(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);}
    };
}}} // namespace cv { namespace cuda { namespace cudev

//! @endcond

#endif // OPENCV_CUDA_VEC_TRAITS_HPP
