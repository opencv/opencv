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

#ifndef __OPENCV_GPU_FUNCTIONAL_HPP__
#define __OPENCV_GPU_FUNCTIONAL_HPP__

#include <thrust/functional.h>
#include "internal_shared.hpp"
#include "saturate_cast.hpp"
#include "vec_traits.hpp"

namespace cv { namespace gpu { namespace device
{
    // Function Objects

    using thrust::unary_function;
    using thrust::binary_function;

    // Arithmetic Operations

    using thrust::plus;
    using thrust::minus;
    using thrust::multiplies;
    using thrust::divides;
    using thrust::modulus;
    using thrust::negate;

    // Comparison Operations
    
    using thrust::equal_to;
    using thrust::not_equal_to;
    using thrust::greater;
    using thrust::less;
    using thrust::greater_equal;
    using thrust::less_equal;

    // Logical Operations
    
    using thrust::logical_and;
    using thrust::logical_or;
    using thrust::logical_not;

    // Bitwise Operations

    using thrust::bit_and;
    using thrust::bit_or;
    using thrust::bit_xor;
    template <typename T> struct bit_not : unary_function<T, T>
    {
        __forceinline__ __device__ T operator ()(const T& v) const {return ~v;}
    };

    // Generalized Identity Operations

    using thrust::identity;    
    using thrust::project1st;
    using thrust::project2nd;

    // Min/Max Operations

#define OPENCV_GPU_IMPLEMENT_MINMAX(name, type, op) \
    template <> struct name<type> : binary_function<type, type, type> \
    { \
        __forceinline__ __device__ type operator()(type lhs, type rhs) const {return op(lhs, rhs);} \
    };

    template <typename T> struct maximum : binary_function<T, T, T>
    {
        __forceinline__ __device__ T operator()(const T& lhs, const T& rhs) const {return lhs < rhs ? rhs : lhs;}
    };
    OPENCV_GPU_IMPLEMENT_MINMAX(maximum, uchar, max)
    OPENCV_GPU_IMPLEMENT_MINMAX(maximum, schar, max)
    OPENCV_GPU_IMPLEMENT_MINMAX(maximum, char, max)
    OPENCV_GPU_IMPLEMENT_MINMAX(maximum, ushort, max)
    OPENCV_GPU_IMPLEMENT_MINMAX(maximum, short, max)
    OPENCV_GPU_IMPLEMENT_MINMAX(maximum, int, max)
    OPENCV_GPU_IMPLEMENT_MINMAX(maximum, uint, max)
    OPENCV_GPU_IMPLEMENT_MINMAX(maximum, float, fmax)
    OPENCV_GPU_IMPLEMENT_MINMAX(maximum, double, fmax)

    template <typename T> struct minimum : binary_function<T, T, T>
    {
        __forceinline__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs < rhs ? lhs : rhs;}
    };
    OPENCV_GPU_IMPLEMENT_MINMAX(minimum, uchar, min)
    OPENCV_GPU_IMPLEMENT_MINMAX(minimum, schar, min)
    OPENCV_GPU_IMPLEMENT_MINMAX(minimum, char, min)
    OPENCV_GPU_IMPLEMENT_MINMAX(minimum, ushort, min)
    OPENCV_GPU_IMPLEMENT_MINMAX(minimum, short, min)
    OPENCV_GPU_IMPLEMENT_MINMAX(minimum, int, min)
    OPENCV_GPU_IMPLEMENT_MINMAX(minimum, uint, min)
    OPENCV_GPU_IMPLEMENT_MINMAX(minimum, float, fmin)
    OPENCV_GPU_IMPLEMENT_MINMAX(minimum, double, fmin)

#undef OPENCV_GPU_IMPLEMENT_MINMAX

    // Math functions

#define OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(func) \
    template <typename T> struct func ## _func : unary_function<T, float> \
    { \
        __forceinline__ __device__ float operator ()(const T& v) const \
        { \
            return func ## f(v); \
        } \
    }; \
    template <> struct func ## _func<double> : unary_function<double, double> \
    { \
        __forceinline__ __device__ double operator ()(double v) const \
        { \
            return func(v); \
        } \
    };
#define OPENCV_GPU_IMPLEMENT_BIN_FUNCTOR(func) \
    template <typename T> struct func ## _func : binary_function<T, T, float> \
    { \
        __forceinline__ __device__ float operator ()(const T& v1, const T& v2) const \
        { \
            return func ## f(v1, v2); \
        } \
    }; \
    template <> struct func ## _func<double> : binary_function<double, double, double> \
    { \
        __forceinline__ __device__ double operator ()(double v1, double v2) const \
        { \
            return func(v1, v2); \
        } \
    };

    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(fabs)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(sqrt)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(exp)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(exp2)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(exp10)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(log)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(log2)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(log10)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(sin)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(cos)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(tan)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(asin)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(acos)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(atan)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(sinh)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(cosh)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(tanh)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(asinh)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(acosh)
    OPENCV_GPU_IMPLEMENT_UN_FUNCTOR(atanh)

    OPENCV_GPU_IMPLEMENT_BIN_FUNCTOR(hypot)
    OPENCV_GPU_IMPLEMENT_BIN_FUNCTOR(atan2)
    OPENCV_GPU_IMPLEMENT_BIN_FUNCTOR(pow)

#undef OPENCV_GPU_IMPLEMENT_UN_FUNCTOR
#undef OPENCV_GPU_IMPLEMENT_BIN_FUNCTOR

    template<typename T> struct hypot_sqr_func : binary_function<T, T, float> 
    {
        __forceinline__ __device__ T operator ()(T src1, T src2) const
        {
            return src1 * src1 + src2 * src2;
        }
    };

    // Saturate Cast Functor

    template <typename T, typename D> struct saturate_cast_func : unary_function<T, D>
    {
        __forceinline__ __device__ D operator ()(const T& v) const
        {
            return saturate_cast<D>(v);
        }
    };

    // Threshold Functors

    template <typename T> struct thresh_binary_func : unary_function<T, T>
    {
        __forceinline__ __host__ __device__ thresh_binary_func(T thresh_, T maxVal_) : thresh(thresh_), maxVal(maxVal_) {}

        __forceinline__ __device__ T operator()(const T& src) const
        {
            return src > thresh ? maxVal : 0;
        }

        const T thresh;
        const T maxVal;
    };
    template <typename T> struct thresh_binary_inv_func : unary_function<T, T>
    {
        __forceinline__ __host__ __device__ thresh_binary_inv_func(T thresh_, T maxVal_) : thresh(thresh_), maxVal(maxVal_) {}

        __forceinline__ __device__ T operator()(const T& src) const
        {
            return src > thresh ? 0 : maxVal;
        }

        const T thresh;
        const T maxVal;
    };
    template <typename T> struct thresh_trunc_func : unary_function<T, T>
    {
        explicit __forceinline__ __host__ __device__ thresh_trunc_func(T thresh_, T maxVal_ = 0) : thresh(thresh_) {}

        __forceinline__ __device__ T operator()(const T& src) const
        {
            return minimum<T>()(src, thresh);
        }

        const T thresh;
    };
    template <typename T> struct thresh_to_zero_func : unary_function<T, T>
    {
        explicit __forceinline__ __host__ __device__ thresh_to_zero_func(T thresh_, T maxVal_ = 0) : thresh(thresh_) {}

        __forceinline__ __device__ T operator()(const T& src) const
        {
            return src > thresh ? src : 0;
        }

        const T thresh;
    };
    template <typename T> struct thresh_to_zero_inv_func : unary_function<T, T>
    {
        explicit __forceinline__ __host__ __device__ thresh_to_zero_inv_func(T thresh_, T maxVal_ = 0) : thresh(thresh_) {}

        __forceinline__ __device__ T operator()(const T& src) const
        {
            return src > thresh ? 0 : src;
        }

        const T thresh;
    };    

    // Function Object Adaptors

    using thrust::unary_negate;
    using thrust::not1;

    using thrust::binary_negate;
    using thrust::not2;

    template <typename Op> struct binder1st : unary_function<typename Op::second_argument_type, typename Op::result_type> 
    {
        __forceinline__ __host__ __device__ binder1st(const Op& op_, const typename Op::first_argument_type& arg1_) : op(op_), arg1(arg1_) {}

        __forceinline__ __device__ typename Op::result_type operator ()(const typename Op::second_argument_type& a) const
        {
            return op(arg1, a);
        }

        const Op op;
        const typename Op::first_argument_type arg1;
    };
    template <typename Op, typename T> static __forceinline__ __host__ __device__ binder1st<Op> bind1st(const Op& op, const T& x)
    {
        return binder1st<Op>(op, typename Op::first_argument_type(x));
    }
    template <typename Op> struct binder2nd : unary_function<typename Op::first_argument_type, typename Op::result_type> 
    {
        __forceinline__ __host__ __device__ binder2nd(const Op& op_, const typename Op::second_argument_type& arg2_) : op(op_), arg2(arg2_) {}

        __forceinline__ __device__ typename Op::result_type operator ()(const typename Op::first_argument_type& a) const
        {
            return op(a, arg2);
        }

        const Op op;
        const typename Op::second_argument_type arg2;
    };
    template <typename Op, typename T> static __forceinline__ __host__ __device__ binder2nd<Op> bind2nd(const Op& op, const T& x)
    {
        return binder2nd<Op>(op, typename Op::second_argument_type(x));
    }

    // Functor Traits

    template <typename F> struct IsUnaryFunction
    {
        struct Yes {};
        struct No {Yes a[2];};

        template <typename T, typename D> static Yes check(unary_function<T, D>*);
        static No check(...);

        enum { value = (sizeof(check((F*)0)) == sizeof(Yes)) };
    };

    template <typename F> struct IsBinaryFunction
    {
        struct Yes {};
        struct No {Yes a[2];};

        template <typename T1, typename T2, typename D> static Yes check(binary_function<T1, T2, D>*);
        static No check(...);

        enum { value = (sizeof(check((F*)0)) == sizeof(Yes)) };
    };

    namespace detail
    {
        template <size_t src_elem_size, size_t dst_elem_size> struct UnOpShift { enum { shift = 1 }; };
        template <size_t src_elem_size> struct UnOpShift<src_elem_size, 1> { enum { shift = 4 }; };
        template <size_t src_elem_size> struct UnOpShift<src_elem_size, 2> { enum { shift = 2 }; };

        template <typename T, typename D> struct DefaultUnaryShift
        {
            enum { shift = detail::UnOpShift<sizeof(T), sizeof(D)>::shift };
        };
        
        template <size_t src_elem_size1, size_t src_elem_size2, size_t dst_elem_size> struct BinOpShift { enum { shift = 1 }; };
        template <size_t src_elem_size1, size_t src_elem_size2> struct BinOpShift<src_elem_size1, src_elem_size2, 1> { enum { shift = 4 }; };
        template <size_t src_elem_size1, size_t src_elem_size2> struct BinOpShift<src_elem_size1, src_elem_size2, 2> { enum { shift = 2 }; };

        template <typename T1, typename T2, typename D> struct DefaultBinaryShift
        {
            enum { shift = detail::BinOpShift<sizeof(T1), sizeof(T2), sizeof(D)>::shift };
        };

        template <typename Func, bool unary = IsUnaryFunction<Func>::value> struct ShiftDispatcher;
        template <typename Func> struct ShiftDispatcher<Func, true>
        {
            enum { shift = DefaultUnaryShift<typename Func::argument_type, typename Func::result_type>::shift };
        };
        template <typename Func> struct ShiftDispatcher<Func, false>
        {
            enum { shift = DefaultBinaryShift<typename Func::first_argument_type, typename Func::second_argument_type, typename Func::result_type>::shift };
        };
    }

    template <typename Func> struct DefaultTransformShift
    {
        enum { shift = detail::ShiftDispatcher<Func>::shift };
    };

    template <typename Func> struct DefaultTransformFunctorTraits
    {
        enum { simple_block_dim_x = 16 };
        enum { simple_block_dim_y = 16 };

        enum { smart_block_dim_x = 16 };
        enum { smart_block_dim_y = 16 };
        enum { smart_shift = DefaultTransformShift<Func>::shift };
    };

    template <typename Func> struct TransformFunctorTraits : DefaultTransformFunctorTraits<Func> {};
}}}

#endif // __OPENCV_GPU_FUNCTIONAL_HPP__
