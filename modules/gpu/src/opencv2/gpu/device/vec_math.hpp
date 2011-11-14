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

#include "internal_shared.hpp"
#include "saturate_cast.hpp"
#include "vec_traits.hpp"
#include "functional.hpp"

namespace cv { namespace gpu { namespace device 
{
    namespace vec_math_detail
    {
        template <int cn, typename VecD> struct SatCastHelper;
        template <typename VecD> struct SatCastHelper<1, VecD>
        {
            template <typename VecS> static __device__ __forceinline__ VecD cast(const VecS& v)
            {
                typedef typename VecTraits<VecD>::elem_type D;
                return VecTraits<VecD>::make(saturate_cast<D>(v.x));
            }
        };
        template <typename VecD> struct SatCastHelper<2, VecD>
        {
            template <typename VecS> static __device__ __forceinline__ VecD cast(const VecS& v)
            {
                typedef typename VecTraits<VecD>::elem_type D;
                return VecTraits<VecD>::make(saturate_cast<D>(v.x), saturate_cast<D>(v.y));
            }
        };
        template <typename VecD> struct SatCastHelper<3, VecD>
        {
            template <typename VecS> static __device__ __forceinline__ VecD cast(const VecS& v)
            {
                typedef typename VecTraits<VecD>::elem_type D;
                return VecTraits<VecD>::make(saturate_cast<D>(v.x), saturate_cast<D>(v.y), saturate_cast<D>(v.z));
            }
        };
        template <typename VecD> struct SatCastHelper<4, VecD>
        {
            template <typename VecS> static __device__ __forceinline__ VecD cast(const VecS& v)
            {
                typedef typename VecTraits<VecD>::elem_type D;
                return VecTraits<VecD>::make(saturate_cast<D>(v.x), saturate_cast<D>(v.y), saturate_cast<D>(v.z), saturate_cast<D>(v.w));
            }
        };

        template <typename VecD, typename VecS> static __device__ __forceinline__ VecD saturate_cast_caller(const VecS& v)
        {
            return SatCastHelper<VecTraits<VecD>::cn, VecD>::cast(v);
        }
    }

    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const uchar1& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const char1& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const ushort1& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const short1& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const uint1& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const int1& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const float1& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const double1& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}

    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const uchar2& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const char2& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const ushort2& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const short2& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const uint2& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const int2& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const float2& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const double2& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}

    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const uchar3& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const char3& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const ushort3& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const short3& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const uint3& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const int3& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const float3& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const double3& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}

    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const uchar4& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const char4& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const ushort4& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const short4& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const uint4& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const int4& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const float4& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}
    template<typename _Tp> static __device__ __forceinline__ _Tp saturate_cast(const double4& v) {return vec_math_detail::saturate_cast_caller<_Tp>(v);}

#define OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, op, func) \
    __device__ __forceinline__ TypeVec<func<type>::result_type, 1>::vec_type op(const type ## 1 & a) \
    { \
        func<type> f; \
        return VecTraits<TypeVec<func<type>::result_type, 1>::vec_type>::make(f(a.x)); \
    } \
    __device__ __forceinline__ TypeVec<func<type>::result_type, 2>::vec_type op(const type ## 2 & a) \
    { \
        func<type> f; \
        return VecTraits<TypeVec<func<type>::result_type, 2>::vec_type>::make(f(a.x), f(a.y)); \
    } \
    __device__ __forceinline__ TypeVec<func<type>::result_type, 3>::vec_type op(const type ## 3 & a) \
    { \
        func<type> f; \
        return VecTraits<TypeVec<func<type>::result_type, 3>::vec_type>::make(f(a.x), f(a.y), f(a.z)); \
    } \
    __device__ __forceinline__ TypeVec<func<type>::result_type, 4>::vec_type op(const type ## 4 & a) \
    { \
        func<type> f; \
        return VecTraits<TypeVec<func<type>::result_type, 4>::vec_type>::make(f(a.x), f(a.y), f(a.z), f(a.w)); \
    }

    namespace vec_math_detail
    {    
        template <typename T1, typename T2> struct BinOpTraits
        {
            typedef int argument_type;
        };
        template <typename T> struct BinOpTraits<T, T>
        {
            typedef T argument_type;
        };
        template <typename T> struct BinOpTraits<T, double>
        {
            typedef double argument_type;
        };
        template <typename T> struct BinOpTraits<double, T>
        {
            typedef double argument_type;
        };
        template <> struct BinOpTraits<double, double>
        {
            typedef double argument_type;
        };
        template <typename T> struct BinOpTraits<T, float>
        {
            typedef float argument_type;
        };
        template <typename T> struct BinOpTraits<float, T>
        {
            typedef float argument_type;
        };
        template <> struct BinOpTraits<float, float>
        {
            typedef float argument_type;
        };
        template <> struct BinOpTraits<double, float>
        {
            typedef double argument_type;
        };
        template <> struct BinOpTraits<float, double>
        {
            typedef double argument_type;
        };
    }

#define OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, op, func) \
    __device__ __forceinline__ TypeVec<func<type>::result_type, 1>::vec_type op(const type ## 1 & a, const type ## 1 & b) \
    { \
        func<type> f; \
        return VecTraits<TypeVec<func<type>::result_type, 1>::vec_type>::make(f(a.x, b.x)); \
    } \
    template <typename T> \
    __device__ __forceinline__ typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 1>::vec_type op(const type ## 1 & v, T s) \
    { \
        func<typename vec_math_detail::BinOpTraits<type, T>::argument_type> f; \
        return VecTraits<typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 1>::vec_type>::make(f(v.x, s)); \
    } \
    template <typename T> \
    __device__ __forceinline__ typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 1>::vec_type op(T s, const type ## 1 & v) \
    { \
        func<typename vec_math_detail::BinOpTraits<type, T>::argument_type> f; \
        return VecTraits<typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 1>::vec_type>::make(f(s, v.x)); \
    } \
    __device__ __forceinline__ TypeVec<func<type>::result_type, 2>::vec_type op(const type ## 2 & a, const type ## 2 & b) \
    { \
        func<type> f; \
        return VecTraits<TypeVec<func<type>::result_type, 2>::vec_type>::make(f(a.x, b.x), f(a.y, b.y)); \
    } \
    template <typename T> \
    __device__ __forceinline__ typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 2>::vec_type op(const type ## 2 & v, T s) \
    { \
        func<typename vec_math_detail::BinOpTraits<type, T>::argument_type> f; \
        return VecTraits<typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 2>::vec_type>::make(f(v.x, s), f(v.y, s)); \
    } \
    template <typename T> \
    __device__ __forceinline__ typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 2>::vec_type op(T s, const type ## 2 & v) \
    { \
        func<typename vec_math_detail::BinOpTraits<type, T>::argument_type> f; \
        return VecTraits<typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 2>::vec_type>::make(f(s, v.x), f(s, v.y)); \
    } \
    __device__ __forceinline__ TypeVec<func<type>::result_type, 3>::vec_type op(const type ## 3 & a, const type ## 3 & b) \
    { \
        func<type> f; \
        return VecTraits<TypeVec<func<type>::result_type, 3>::vec_type>::make(f(a.x, b.x), f(a.y, b.y), f(a.z, b.z)); \
    } \
    template <typename T> \
    __device__ __forceinline__ typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 3>::vec_type op(const type ## 3 & v, T s) \
    { \
        func<typename vec_math_detail::BinOpTraits<type, T>::argument_type> f; \
        return VecTraits<typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 3>::vec_type>::make(f(v.x, s), f(v.y, s), f(v.z, s)); \
    } \
    template <typename T> \
    __device__ __forceinline__ typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 3>::vec_type op(T s, const type ## 3 & v) \
    { \
        func<typename vec_math_detail::BinOpTraits<type, T>::argument_type> f; \
        return VecTraits<typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 3>::vec_type>::make(f(s, v.x), f(s, v.y), f(s, v.z)); \
    } \
    __device__ __forceinline__ TypeVec<func<type>::result_type, 4>::vec_type op(const type ## 4 & a, const type ## 4 & b) \
    { \
        func<type> f; \
        return VecTraits<TypeVec<func<type>::result_type, 4>::vec_type>::make(f(a.x, b.x), f(a.y, b.y), f(a.z, b.z), f(a.w, b.w)); \
    } \
    template <typename T> \
    __device__ __forceinline__ typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 4>::vec_type op(const type ## 4 & v, T s) \
    { \
        func<typename vec_math_detail::BinOpTraits<type, T>::argument_type> f; \
        return VecTraits<typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 4>::vec_type>::make(f(v.x, s), f(v.y, s), f(v.z, s), f(v.w, s)); \
    } \
    template <typename T> \
    __device__ __forceinline__ typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 4>::vec_type op(T s, const type ## 4 & v) \
    { \
        func<typename vec_math_detail::BinOpTraits<T, type>::argument_type> f; \
        return VecTraits<typename TypeVec<typename func<typename vec_math_detail::BinOpTraits<type, T>::argument_type>::result_type, 4>::vec_type>::make(f(s, v.x), f(s, v.y), f(s, v.z), f(s, v.w)); \
    }

#define OPENCV_GPU_IMPLEMENT_VEC_OP(type) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator +, plus) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator -, minus) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator *, multiplies) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator /, divides) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP (type, operator -, negate) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator ==, equal_to) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator !=, not_equal_to) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator > , greater) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator < , less) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator >=, greater_equal) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator <=, less_equal) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator &&, logical_and) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator ||, logical_or) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP (type, operator ! , logical_not) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, max, maximum) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, min, minimum) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, fabs, fabs_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, sqrt, sqrt_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, exp, exp_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, exp2, exp2_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, exp10, exp10_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, log, log_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, log2, log2_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, log10, log10_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, sin, sin_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, cos, cos_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, tan, tan_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, asin, asin_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, acos, acos_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, atan, atan_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, sinh, sinh_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, cosh, cosh_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, tanh, tanh_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, asinh, asinh_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, acosh, acosh_func) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP(type, atanh, atanh_func) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, hypot, hypot_func) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, atan2, atan2_func) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, pow, pow_func) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, hypot_sqr, hypot_sqr_func)

#define OPENCV_GPU_IMPLEMENT_VEC_INT_OP(type) \
    OPENCV_GPU_IMPLEMENT_VEC_OP(type) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator &, bit_and) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator |, bit_or) \
    OPENCV_GPU_IMPLEMENT_VEC_BINOP(type, operator ^, bit_xor) \
    OPENCV_GPU_IMPLEMENT_VEC_UNOP (type, operator ~, bit_not)

    OPENCV_GPU_IMPLEMENT_VEC_INT_OP(uchar)
    OPENCV_GPU_IMPLEMENT_VEC_INT_OP(char)
    OPENCV_GPU_IMPLEMENT_VEC_INT_OP(ushort)
    OPENCV_GPU_IMPLEMENT_VEC_INT_OP(short)
    OPENCV_GPU_IMPLEMENT_VEC_INT_OP(int)
    OPENCV_GPU_IMPLEMENT_VEC_INT_OP(uint)
    OPENCV_GPU_IMPLEMENT_VEC_OP(float)
    OPENCV_GPU_IMPLEMENT_VEC_OP(double)

    #undef OPENCV_GPU_IMPLEMENT_VEC_UNOP
    #undef OPENCV_GPU_IMPLEMENT_VEC_BINOP
    #undef OPENCV_GPU_IMPLEMENT_VEC_OP
    #undef OPENCV_GPU_IMPLEMENT_VEC_INT_OP
}}} // namespace cv { namespace gpu { namespace device
        
#endif // __OPENCV_GPU_VECMATH_HPP__