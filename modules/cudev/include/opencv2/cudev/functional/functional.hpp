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

#ifndef __OPENCV_CUDEV_FUNCTIONAL_FUNCTIONAL_HPP__
#define __OPENCV_CUDEV_FUNCTIONAL_FUNCTIONAL_HPP__

#include "../common.hpp"
#include "../util/saturate_cast.hpp"
#include "../util/vec_traits.hpp"
#include "../util/vec_math.hpp"
#include "../util/type_traits.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// Function Objects

template <typename _Arg, typename _Result> struct unary_function
{
    typedef _Arg    argument_type;
    typedef _Result result_type;
};

template <typename _Arg1, typename _Arg2, typename _Result> struct binary_function
{
    typedef _Arg1   first_argument_type;
    typedef _Arg2   second_argument_type;
    typedef _Result result_type;
};

// Arithmetic Operations

template <typename T> struct plus : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a,
                                             typename TypeTraits<T>::parameter_type b) const
    {
        return saturate_cast<T>(a + b);
    }
};

template <typename T> struct minus : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a,
                                             typename TypeTraits<T>::parameter_type b) const
    {
        return saturate_cast<T>(a - b);
    }
};

template <typename T> struct multiplies : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a,
                                             typename TypeTraits<T>::parameter_type b) const
    {
        return saturate_cast<T>(a * b);
    }
};

template <typename T> struct divides : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a,
                                             typename TypeTraits<T>::parameter_type b) const
    {
        return saturate_cast<T>(a / b);
    }
};

template <typename T> struct modulus : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a,
                                             typename TypeTraits<T>::parameter_type b) const
    {
        return saturate_cast<T>(a % b);
    }
};

template <typename T> struct negate : unary_function<T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a) const
    {
        return saturate_cast<T>(-a);
    }
};

// Comparison Operations

template <typename T> struct equal_to : binary_function<T, T, typename MakeVec<uchar, VecTraits<T>::cn>::type>
{
    __device__ __forceinline__ typename MakeVec<uchar, VecTraits<T>::cn>::type
                                    operator ()(typename TypeTraits<T>::parameter_type a,
                                                typename TypeTraits<T>::parameter_type b) const
    {
        return a == b;
    }
};

template <typename T> struct not_equal_to : binary_function<T, T, typename MakeVec<uchar, VecTraits<T>::cn>::type>
{
    __device__ __forceinline__ typename MakeVec<uchar, VecTraits<T>::cn>::type
                                    operator ()(typename TypeTraits<T>::parameter_type a,
                                                typename TypeTraits<T>::parameter_type b) const
    {
        return a != b;
    }
};

template <typename T> struct greater : binary_function<T, T, typename MakeVec<uchar, VecTraits<T>::cn>::type>
{
    __device__ __forceinline__ typename MakeVec<uchar, VecTraits<T>::cn>::type
                                    operator ()(typename TypeTraits<T>::parameter_type a,
                                                typename TypeTraits<T>::parameter_type b) const
    {
        return a > b;
    }
};

template <typename T> struct less : binary_function<T, T, typename MakeVec<uchar, VecTraits<T>::cn>::type>
{
    __device__ __forceinline__ typename MakeVec<uchar, VecTraits<T>::cn>::type
                                    operator ()(typename TypeTraits<T>::parameter_type a,
                                                typename TypeTraits<T>::parameter_type b) const
    {
        return a < b;
    }
};

template <typename T> struct greater_equal : binary_function<T, T, typename MakeVec<uchar, VecTraits<T>::cn>::type>
{
    __device__ __forceinline__ typename MakeVec<uchar, VecTraits<T>::cn>::type
                                    operator ()(typename TypeTraits<T>::parameter_type a,
                                                typename TypeTraits<T>::parameter_type b) const
    {
        return a >= b;
    }
};

template <typename T> struct less_equal : binary_function<T, T, typename MakeVec<uchar, VecTraits<T>::cn>::type>
{
    __device__ __forceinline__ typename MakeVec<uchar, VecTraits<T>::cn>::type
                                    operator ()(typename TypeTraits<T>::parameter_type a,
                                                typename TypeTraits<T>::parameter_type b) const
    {
        return a <= b;
    }
};

// Logical Operations

template <typename T> struct logical_and : binary_function<T, T, typename MakeVec<uchar, VecTraits<T>::cn>::type>
{
    __device__ __forceinline__ typename MakeVec<uchar, VecTraits<T>::cn>::type
                                    operator ()(typename TypeTraits<T>::parameter_type a,
                                                typename TypeTraits<T>::parameter_type b) const
    {
        return a && b;
    }
};

template <typename T> struct logical_or : binary_function<T, T, typename MakeVec<uchar, VecTraits<T>::cn>::type>
{
    __device__ __forceinline__ typename MakeVec<uchar, VecTraits<T>::cn>::type
                                    operator ()(typename TypeTraits<T>::parameter_type a,
                                                typename TypeTraits<T>::parameter_type b) const
    {
        return a || b;
    }
};

template <typename T> struct logical_not : unary_function<T, typename MakeVec<uchar, VecTraits<T>::cn>::type>
{
    __device__ __forceinline__ typename MakeVec<uchar, VecTraits<T>::cn>::type
                                    operator ()(typename TypeTraits<T>::parameter_type a) const
    {
        return !a;
    }
};

// Bitwise Operations

template <typename T> struct bit_and : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a,
                                             typename TypeTraits<T>::parameter_type b) const
    {
        return a & b;
    }
};

template <typename T> struct bit_or : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a,
                                             typename TypeTraits<T>::parameter_type b) const
    {
        return a | b;
    }
};

template <typename T> struct bit_xor : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a,
                                             typename TypeTraits<T>::parameter_type b) const
    {
        return a ^ b;
    }
};

template <typename T> struct bit_not : unary_function<T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type v) const
    {
        return ~v;
    }
};

template <typename T> struct bit_lshift : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a,
                                             typename TypeTraits<T>::parameter_type b) const
    {
        return a << b;
    }
};

template <typename T> struct bit_rshift : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a,
                                             typename TypeTraits<T>::parameter_type b) const
    {
        return a >> b;
    }
};

// Generalized Identity Operations

template <typename T> struct identity : unary_function<T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type x) const
    {
        return x;
    }
};

template <typename T1, typename T2> struct project1st : binary_function<T1, T2, T1>
{
    __device__ __forceinline__ T1
                    operator ()(typename TypeTraits<T1>::parameter_type lhs,
                                typename TypeTraits<T2>::parameter_type) const
    {
        return lhs;
    }
};

template <typename T1, typename T2> struct project2nd : binary_function<T1, T2, T2>
{
    __device__ __forceinline__ T2
                    operator ()(typename TypeTraits<T1>::parameter_type,
                                typename TypeTraits<T2>::parameter_type rhs) const
    {
        return rhs;
    }
};

// Min/Max Operations

template <typename T> struct maximum : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a,
                                             typename TypeTraits<T>::parameter_type b) const
    {
        return max(a, b);
    }
};

template <typename T> struct minimum : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a,
                                             typename TypeTraits<T>::parameter_type b) const
    {
        return min(a, b);
    }
};

#define CV_CUDEV_MINMAX_INST(type, maxop, minop) \
    template <> struct maximum<type> : binary_function<type, type, type> \
    { \
        __device__ __forceinline__ type operator ()(type a, type b) const {return maxop(a, b);} \
    }; \
    template <> struct minimum<type> : binary_function<type, type, type> \
    { \
        __device__ __forceinline__ type operator ()(type a, type b) const {return minop(a, b);} \
    };


CV_CUDEV_MINMAX_INST(uchar, ::max, ::min)
CV_CUDEV_MINMAX_INST(schar, ::max, ::min)
CV_CUDEV_MINMAX_INST(ushort, ::max, ::min)
CV_CUDEV_MINMAX_INST(short, ::max, ::min)
CV_CUDEV_MINMAX_INST(int, ::max, ::min)
CV_CUDEV_MINMAX_INST(uint, ::max, ::min)
CV_CUDEV_MINMAX_INST(float, ::fmaxf, ::fminf)
CV_CUDEV_MINMAX_INST(double, ::fmax, ::fmin)

#undef CV_CUDEV_MINMAX_INST

// abs_func

template <typename T> struct abs_func : unary_function<T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type x) const
    {
        return abs(x);
    }
};

template <> struct abs_func<uchar> : unary_function<uchar, uchar>
{
    __device__ __forceinline__ uchar operator ()(uchar x) const
    {
        return x;
    }
};

template <> struct abs_func<schar> : unary_function<schar, schar>
{
    __device__ __forceinline__ schar operator ()(schar x) const
    {
        return ::abs((int) x);
    }
};

template <> struct abs_func<ushort> : unary_function<ushort, ushort>
{
    __device__ __forceinline__ ushort operator ()(ushort x) const
    {
        return x;
    }
};

template <> struct abs_func<short> : unary_function<short, short>
{
    __device__ __forceinline__ short operator ()(short x) const
    {
        return ::abs((int) x);
    }
};

template <> struct abs_func<uint> : unary_function<uint, uint>
{
    __device__ __forceinline__ uint operator ()(uint x) const
    {
        return x;
    }
};

template <> struct abs_func<int> : unary_function<int, int>
{
    __device__ __forceinline__ int operator ()(int x) const
    {
        return ::abs(x);
    }
};

template <> struct abs_func<float> : unary_function<float, float>
{
    __device__ __forceinline__ float operator ()(float x) const
    {
        return ::fabsf(x);
    }
};

template <> struct abs_func<double> : unary_function<double, double>
{
    __device__ __forceinline__ double operator ()(double x) const
    {
        return ::fabs(x);
    }
};

// absdiff_func

template <typename T> struct absdiff_func : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type a, typename TypeTraits<T>::parameter_type b) const
    {
        abs_func<T> f;
        return f(a - b);
    }
};

// Math functions

template <typename T> struct sqr_func : unary_function<T, T>
{
    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type x) const
    {
        return x * x;
    }
};

namespace functional_detail
{
    template <typename T> struct FloatType
    {
        typedef typename MakeVec<
            typename LargerType<float, typename VecTraits<T>::elem_type>::type,
            VecTraits<T>::cn
        >::type type;
    };
}

#define CV_CUDEV_UNARY_FUNCTION_INST(name, func) \
    template <typename T> struct name ## _func : unary_function<T, typename functional_detail::FloatType<T>::type> \
    { \
        __device__ __forceinline__ typename functional_detail::FloatType<T>::type operator ()(typename TypeTraits<T>::parameter_type a) const \
        { \
            return name(a); \
        } \
    }; \
    template <> struct name ## _func<uchar> : unary_function<uchar, float> \
    { \
        __device__ __forceinline__ float operator ()(uchar a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<schar> : unary_function<schar, float> \
    { \
        __device__ __forceinline__ float operator ()(schar a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<ushort> : unary_function<ushort, float> \
    { \
        __device__ __forceinline__ float operator ()(ushort a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<short> : unary_function<short, float> \
    { \
        __device__ __forceinline__ float operator ()(short a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<uint> : unary_function<uint, float> \
    { \
        __device__ __forceinline__ float operator ()(uint a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<int> : unary_function<int, float> \
    { \
        __device__ __forceinline__ float operator ()(int a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<float> : unary_function<float, float> \
    { \
        __device__ __forceinline__ float operator ()(float a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<double> : unary_function<double, double> \
    { \
        __device__ __forceinline__ double operator ()(double a) const \
        { \
            return func(a); \
        } \
    };

CV_CUDEV_UNARY_FUNCTION_INST(sqrt, ::sqrt)
CV_CUDEV_UNARY_FUNCTION_INST(exp, ::exp)
CV_CUDEV_UNARY_FUNCTION_INST(exp2, ::exp2)
CV_CUDEV_UNARY_FUNCTION_INST(exp10, ::exp10)
CV_CUDEV_UNARY_FUNCTION_INST(log, ::log)
CV_CUDEV_UNARY_FUNCTION_INST(log2, ::log2)
CV_CUDEV_UNARY_FUNCTION_INST(log10, ::log10)
CV_CUDEV_UNARY_FUNCTION_INST(sin, ::sin)
CV_CUDEV_UNARY_FUNCTION_INST(cos, ::cos)
CV_CUDEV_UNARY_FUNCTION_INST(tan, ::tan)
CV_CUDEV_UNARY_FUNCTION_INST(asin, ::asin)
CV_CUDEV_UNARY_FUNCTION_INST(acos, ::acos)
CV_CUDEV_UNARY_FUNCTION_INST(atan, ::atan)
CV_CUDEV_UNARY_FUNCTION_INST(sinh, ::sinh)
CV_CUDEV_UNARY_FUNCTION_INST(cosh, ::cosh)
CV_CUDEV_UNARY_FUNCTION_INST(tanh, ::tanh)
CV_CUDEV_UNARY_FUNCTION_INST(asinh, ::asinh)
CV_CUDEV_UNARY_FUNCTION_INST(acosh, ::acosh)
CV_CUDEV_UNARY_FUNCTION_INST(atanh, ::atanh)

#undef CV_CUDEV_UNARY_FUNCTION_INST

#define CV_CUDEV_BINARY_FUNCTION_INST(name, func) \
    template <typename T> struct name ## _func : binary_function<T, T, typename functional_detail::FloatType<T>::type> \
    { \
        __device__ __forceinline__ typename functional_detail::FloatType<T>::type operator ()(typename TypeTraits<T>::parameter_type a, typename TypeTraits<T>::parameter_type b) const \
        { \
            return name(a, b); \
        } \
    }; \
    template <> struct name ## _func<uchar> : binary_function<uchar, uchar, float> \
    { \
        __device__ __forceinline__ float operator ()(uchar a, uchar b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<schar> : binary_function<schar, schar, float> \
    { \
        __device__ __forceinline__ float operator ()(schar a, schar b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<ushort> : binary_function<ushort, ushort, float> \
    { \
        __device__ __forceinline__ float operator ()(ushort a, ushort b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<short> : binary_function<short, short, float> \
    { \
        __device__ __forceinline__ float operator ()(short a, short b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<uint> : binary_function<uint, uint, float> \
    { \
        __device__ __forceinline__ float operator ()(uint a, uint b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<int> : binary_function<int, int, float> \
    { \
        __device__ __forceinline__ float operator ()(int a, int b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<float> : binary_function<float, float, float> \
    { \
        __device__ __forceinline__ float operator ()(float a, float b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<double> : binary_function<double, double, double> \
    { \
        __device__ __forceinline__ double operator ()(double a, double b) const \
        { \
            return func(a, b); \
        } \
    };

CV_CUDEV_BINARY_FUNCTION_INST(hypot, ::hypot)
CV_CUDEV_BINARY_FUNCTION_INST(atan2, ::atan2)

#undef CV_CUDEV_BINARY_FUNCTION_INST

template <typename T> struct magnitude_func : binary_function<T, T, typename functional_detail::FloatType<T>::type>
{
    __device__ __forceinline__ typename functional_detail::FloatType<T>::type operator ()(typename TypeTraits<T>::parameter_type a, typename TypeTraits<T>::parameter_type b) const
    {
        sqrt_func<typename functional_detail::FloatType<T>::type> f;
        return f(a * a + b * b);
    }
};

template <typename T> struct magnitude_sqr_func : binary_function<T, T, typename functional_detail::FloatType<T>::type>
{
    __device__ __forceinline__ typename functional_detail::FloatType<T>::type operator ()(typename TypeTraits<T>::parameter_type a, typename TypeTraits<T>::parameter_type b) const
    {
        return a * a + b * b;
    }
};

template <typename T, bool angleInDegrees> struct direction_func : binary_function<T, T, T>
{
    __device__ T operator ()(T x, T y) const
    {
        atan2_func<T> f;
        typename atan2_func<T>::result_type angle = f(y, x);

        angle += (angle < 0) * (2.0f * CV_PI_F);

        if (angleInDegrees)
            angle *= (180.0f / CV_PI_F);

        return saturate_cast<T>(angle);
    }
};

template <typename T> struct pow_func : binary_function<T, float, float>
{
    __device__ __forceinline__ float operator ()(T val, float power) const
    {
        return ::powf(val, power);
    }
};
template <> struct pow_func<double> : binary_function<double, double, double>
{
    __device__ __forceinline__ double operator ()(double val, double power) const
    {
        return ::pow(val, power);
    }
};

// Saturate Cast Functor

template <typename T, typename D> struct saturate_cast_func : unary_function<T, D>
{
    __device__ __forceinline__ D operator ()(typename TypeTraits<T>::parameter_type v) const
    {
        return saturate_cast<D>(v);
    }
};

// Threshold Functors

template <typename T> struct ThreshBinaryFunc : unary_function<T, T>
{
    T thresh;
    T maxVal;

    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type src) const
    {
        return saturate_cast<T>(src > thresh) * maxVal;
    }
};

template <typename T>
__host__ __device__ ThreshBinaryFunc<T> thresh_binary_func(T thresh, T maxVal)
{
    ThreshBinaryFunc<T> f;
    f.thresh = thresh;
    f.maxVal = maxVal;
    return f;
}

template <typename T> struct ThreshBinaryInvFunc : unary_function<T, T>
{
    T thresh;
    T maxVal;

    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type src) const
    {
        return saturate_cast<T>(src <= thresh) * maxVal;
    }
};

template <typename T>
__host__ __device__ ThreshBinaryInvFunc<T> thresh_binary_inv_func(T thresh, T maxVal)
{
    ThreshBinaryInvFunc<T> f;
    f.thresh = thresh;
    f.maxVal = maxVal;
    return f;
}

template <typename T> struct ThreshTruncFunc : unary_function<T, T>
{
    T thresh;

    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type src) const
    {
        minimum<T> minOp;
        return minOp(src, thresh);
    }
};

template <typename T>
__host__ __device__ ThreshTruncFunc<T> thresh_trunc_func(T thresh)
{
    ThreshTruncFunc<T> f;
    f.thresh = thresh;
    return f;
}

template <typename T> struct ThreshToZeroFunc : unary_function<T, T>
{
    T thresh;

    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type src) const
    {
        return saturate_cast<T>(src > thresh) * src;
    }
};

template <typename T>
__host__ __device__ ThreshToZeroFunc<T> thresh_to_zero_func(T thresh)
{
    ThreshToZeroFunc<T> f;
    f.thresh = thresh;
    return f;
}

template <typename T> struct ThreshToZeroInvFunc : unary_function<T, T>
{
    T thresh;

    __device__ __forceinline__ T operator ()(typename TypeTraits<T>::parameter_type src) const
    {
        return saturate_cast<T>(src <= thresh) * src;
    }
};

template <typename T>
__host__ __device__ ThreshToZeroInvFunc<T> thresh_to_zero_inv_func(T thresh)
{
    ThreshToZeroInvFunc<T> f;
    f.thresh = thresh;
    return f;
}

// Function Object Adaptors

template <class Predicate> struct UnaryNegate : unary_function<typename Predicate::argument_type, typename Predicate::result_type>
{
    Predicate pred;

    __device__ __forceinline__ typename Predicate::result_type operator ()(
            typename TypeTraits<typename Predicate::argument_type>::parameter_type x) const
    {
        return !pred(x);
    }
};

template <class Predicate>
__host__ __device__ UnaryNegate<Predicate> not1(const Predicate& pred)
{
    UnaryNegate<Predicate> n;
    n.pred = pred;
    return n;
}

template <class Predicate> struct BinaryNegate : binary_function<typename Predicate::first_argument_type, typename Predicate::second_argument_type, typename Predicate::result_type>
{
    Predicate pred;

    __device__ __forceinline__ typename Predicate::result_type operator ()(
            typename TypeTraits<typename Predicate::first_argument_type>::parameter_type x,
            typename TypeTraits<typename Predicate::second_argument_type>::parameter_type y) const
    {
        return !pred(x, y);
    }
};

template <class Predicate>
__host__ __device__ BinaryNegate<Predicate> not2(const Predicate& pred)
{
    BinaryNegate<Predicate> n;
    n.pred = pred;
    return n;
}

template <class Op> struct Binder1st : unary_function<typename Op::second_argument_type, typename Op::result_type>
{
    Op op;
    typename Op::first_argument_type arg1;

    __device__ __forceinline__ typename Op::result_type operator ()(
            typename TypeTraits<typename Op::second_argument_type>::parameter_type a) const
    {
        return op(arg1, a);
    }
};

template <class Op>
__host__ __device__ Binder1st<Op> bind1st(const Op& op, const typename Op::first_argument_type& arg1)
{
    Binder1st<Op> b;
    b.op = op;
    b.arg1 = arg1;
    return b;
}

template <class Op> struct Binder2nd : unary_function<typename Op::first_argument_type, typename Op::result_type>
{
    Op op;
    typename Op::second_argument_type arg2;

    __device__ __forceinline__ typename Op::result_type operator ()(
            typename TypeTraits<typename Op::first_argument_type>::parameter_type a) const
    {
        return op(a, arg2);
    }
};

template <class Op>
__host__ __device__ Binder2nd<Op> bind2nd(const Op& op, const typename Op::second_argument_type& arg2)
{
    Binder2nd<Op> b;
    b.op = op;
    b.arg2 = arg2;
    return b;
}

// Functor Traits

template <typename F> struct IsUnaryFunction
{
    typedef char Yes;
    struct No {Yes a[2];};

    template <typename T, typename D> static Yes check(unary_function<T, D>);
    static No check(...);

    static F makeF();

    enum { value = (sizeof(check(makeF())) == sizeof(Yes)) };
};

template <typename F> struct IsBinaryFunction
{
    typedef char Yes;
    struct No {Yes a[2];};

    template <typename T1, typename T2, typename D> static Yes check(binary_function<T1, T2, D>);
    static No check(...);

    static F makeF();

    enum { value = (sizeof(check(makeF())) == sizeof(Yes)) };
};

//! @}

}}

#endif
