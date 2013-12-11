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

#ifndef __OPENCV_CUDEV_EXPR_BINARY_OP_HPP__
#define __OPENCV_CUDEV_EXPR_BINARY_OP_HPP__

#include "../common.hpp"
#include "../util/type_traits.hpp"
#include "../ptr2d/traits.hpp"
#include "../ptr2d/transform.hpp"
#include "../ptr2d/gpumat.hpp"
#include "../ptr2d/texture.hpp"
#include "../ptr2d/glob.hpp"
#include "../functional/functional.hpp"
#include "expr.hpp"

namespace cv { namespace cudev {

// Binary Operations

#define CV_CUDEV_EXPR_BINOP_INST(op, functor) \
    template <typename T> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<GpuMat_<T> >::ptr_type, typename PtrTraits<GpuMat_<T> >::ptr_type, functor<T> > > \
    operator op(const GpuMat_<T>& src1, const GpuMat_<T>& src2) \
    { \
        return makeExpr(transformPtr(src1, src2, functor<T>())); \
    } \
    template <typename T> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<GpuMat_<T> >::ptr_type, typename PtrTraits<GlobPtrSz<T> >::ptr_type, functor<T> > > \
    operator op(const GpuMat_<T>& src1, const GlobPtrSz<T>& src2) \
    { \
        return makeExpr(transformPtr(src1, src2, functor<T>())); \
    } \
    template <typename T> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<GlobPtrSz<T> >::ptr_type, typename PtrTraits<GpuMat_<T> >::ptr_type, functor<T> > > \
    operator op(const GlobPtrSz<T>& src1, const GpuMat_<T>& src2) \
    { \
        return makeExpr(transformPtr(src1, src2, functor<T>())); \
    } \
    template <typename T> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<GpuMat_<T> >::ptr_type, typename PtrTraits<Texture<T> >::ptr_type, functor<T> > > \
    operator op(const GpuMat_<T>& src1, const Texture<T>& src2) \
    { \
        return makeExpr(transformPtr(src1, src2, functor<T>())); \
    } \
    template <typename T> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<Texture<T> >::ptr_type, typename PtrTraits<GpuMat_<T> >::ptr_type, functor<T> > > \
    operator op(const Texture<T>& src1, const GpuMat_<T>& src2) \
    { \
        return makeExpr(transformPtr(src1, src2, functor<T>())); \
    } \
    template <typename T, class Body> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<GpuMat_<T> >::ptr_type, typename PtrTraits<Body>::ptr_type, functor<typename LargerType<T, typename PtrTraits<Body>::value_type>::type> > > \
    operator op(const GpuMat_<T>& src1, const Expr<Body>& src2) \
    { \
        return makeExpr(transformPtr(src1, src2.body, functor<typename LargerType<T, typename PtrTraits<Body>::value_type>::type>())); \
    } \
    template <typename T, class Body> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<Body>::ptr_type, typename PtrTraits<GpuMat_<T> >::ptr_type, functor<typename LargerType<T, typename PtrTraits<Body>::value_type>::type> > > \
    operator op(const Expr<Body>& src1, const GpuMat_<T>& src2) \
    { \
        return makeExpr(transformPtr(src1.body, src2, functor<typename LargerType<T, typename PtrTraits<Body>::value_type>::type>())); \
    } \
    template <typename T> \
    __host__ Expr<UnaryTransformPtrSz<typename PtrTraits<GpuMat_<T> >::ptr_type, Binder2nd< functor<T> > > > \
    operator op(const GpuMat_<T>& src, T val) \
    { \
        return makeExpr(transformPtr(src, bind2nd(functor<T>(), val))); \
    } \
    template <typename T> \
    __host__ Expr<UnaryTransformPtrSz<typename PtrTraits<GpuMat_<T> >::ptr_type, Binder1st< functor<T> > > > \
    operator op(T val, const GpuMat_<T>& src) \
    { \
        return makeExpr(transformPtr(src, bind1st(functor<T>(), val))); \
    } \
    template <typename T> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<GlobPtrSz<T> >::ptr_type, typename PtrTraits<GlobPtrSz<T> >::ptr_type, functor<T> > > \
    operator op(const GlobPtrSz<T>& src1, const GlobPtrSz<T>& src2) \
    { \
        return makeExpr(transformPtr(src1, src2, functor<T>())); \
    } \
    template <typename T> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<GlobPtrSz<T> >::ptr_type, typename PtrTraits<Texture<T> >::ptr_type, functor<T> > > \
    operator op(const GlobPtrSz<T>& src1, const Texture<T>& src2) \
    { \
        return makeExpr(transformPtr(src1, src2, functor<T>())); \
    } \
    template <typename T> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<Texture<T> >::ptr_type, typename PtrTraits<GlobPtrSz<T> >::ptr_type, functor<T> > > \
    operator op(const Texture<T>& src1, const GlobPtrSz<T>& src2) \
    { \
        return makeExpr(transformPtr(src1, src2, functor<T>())); \
    } \
    template <typename T, class Body> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<GlobPtrSz<T> >::ptr_type, typename PtrTraits<Body>::ptr_type, functor<typename LargerType<T, typename PtrTraits<Body>::value_type>::type> > > \
    operator op(const GlobPtrSz<T>& src1, const Expr<Body>& src2) \
    { \
        return makeExpr(transformPtr(src1, src2.body, functor<typename LargerType<T, typename PtrTraits<Body>::value_type>::type>())); \
    } \
    template <typename T, class Body> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<Body>::ptr_type, typename PtrTraits<GlobPtrSz<T> >::ptr_type, functor<typename LargerType<T, typename PtrTraits<Body>::value_type>::type> > > \
    operator op(const Expr<Body>& src1, const GlobPtrSz<T>& src2) \
    { \
        return makeExpr(transformPtr(src1.body, src2, functor<typename LargerType<T, typename PtrTraits<Body>::value_type>::type>())); \
    } \
    template <typename T> \
    __host__ Expr<UnaryTransformPtrSz<typename PtrTraits<GlobPtrSz<T> >::ptr_type, Binder2nd< functor<T> > > > \
    operator op(const GlobPtrSz<T>& src, T val) \
    { \
        return makeExpr(transformPtr(src, bind2nd(functor<T>(), val))); \
    } \
    template <typename T> \
    __host__ Expr<UnaryTransformPtrSz<typename PtrTraits<GlobPtrSz<T> >::ptr_type, Binder1st< functor<T> > > > \
    operator op(T val, const GlobPtrSz<T>& src) \
    { \
        return makeExpr(transformPtr(src, bind1st(functor<T>(), val))); \
    } \
    template <typename T> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<Texture<T> >::ptr_type, typename PtrTraits<Texture<T> >::ptr_type, functor<T> > > \
    operator op(const Texture<T>& src1, const Texture<T>& src2) \
    { \
        return makeExpr(transformPtr(src1, src2, functor<T>())); \
    } \
    template <typename T, class Body> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<Texture<T> >::ptr_type, typename PtrTraits<Body>::ptr_type, functor<typename LargerType<T, typename PtrTraits<Body>::value_type>::type> > > \
    operator op(const Texture<T>& src1, const Expr<Body>& src2) \
    { \
        return makeExpr(transformPtr(src1, src2.body, functor<typename LargerType<T, typename PtrTraits<Body>::value_type>::type>())); \
    } \
    template <typename T, class Body> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<Body>::ptr_type, typename PtrTraits<Texture<T> >::ptr_type, functor<typename LargerType<T, typename PtrTraits<Body>::value_type>::type> > > \
    operator op(const Expr<Body>& src1, const Texture<T>& src2) \
    { \
        return makeExpr(transformPtr(src1.body, src2, functor<typename LargerType<T, typename PtrTraits<Body>::value_type>::type>())); \
    } \
    template <typename T> \
    __host__ Expr<UnaryTransformPtrSz<typename PtrTraits<Texture<T> >::ptr_type, Binder2nd< functor<T> > > > \
    operator op(const Texture<T>& src, T val) \
    { \
        return makeExpr(transformPtr(src, bind2nd(functor<T>(), val))); \
    } \
    template <typename T> \
    __host__ Expr<UnaryTransformPtrSz<typename PtrTraits<Texture<T> >::ptr_type, Binder1st< functor<T> > > > \
    operator op(T val, const Texture<T>& src) \
    { \
        return makeExpr(transformPtr(src, bind1st(functor<T>(), val))); \
    } \
    template <class Body1, class Body2> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<Body1>::ptr_type, typename PtrTraits<Body2>::ptr_type, functor<typename LargerType<typename PtrTraits<Body1>::value_type, typename PtrTraits<Body2>::value_type>::type> > > \
    operator op(const Expr<Body1>& a, const Expr<Body2>& b) \
    { \
        return makeExpr(transformPtr(a.body, b.body, functor<typename LargerType<typename PtrTraits<Body1>::value_type, typename PtrTraits<Body2>::value_type>::type>())); \
    } \
    template <class Body> \
    __host__ Expr<UnaryTransformPtrSz<typename PtrTraits<Body>::ptr_type, Binder2nd< functor<typename Body::value_type> > > > \
    operator op(const Expr<Body>& a, typename Body::value_type val) \
    { \
        return makeExpr(transformPtr(a.body, bind2nd(functor<typename Body::value_type>(), val))); \
    } \
    template <class Body> \
    __host__ Expr<UnaryTransformPtrSz<typename PtrTraits<Body>::ptr_type, Binder1st< functor<typename Body::value_type> > > > \
    operator op(typename Body::value_type val, const Expr<Body>& a) \
    { \
        return makeExpr(transformPtr(a.body, bind1st(functor<typename Body::value_type>(), val))); \
    }

CV_CUDEV_EXPR_BINOP_INST(+, plus)
CV_CUDEV_EXPR_BINOP_INST(-, minus)
CV_CUDEV_EXPR_BINOP_INST(*, multiplies)
CV_CUDEV_EXPR_BINOP_INST(/, divides)
CV_CUDEV_EXPR_BINOP_INST(%, modulus)

CV_CUDEV_EXPR_BINOP_INST(==, equal_to)
CV_CUDEV_EXPR_BINOP_INST(!=, not_equal_to)
CV_CUDEV_EXPR_BINOP_INST(>, greater)
CV_CUDEV_EXPR_BINOP_INST(<, less)
CV_CUDEV_EXPR_BINOP_INST(>=, greater_equal)
CV_CUDEV_EXPR_BINOP_INST(<=, less_equal)

CV_CUDEV_EXPR_BINOP_INST(&&, logical_and)
CV_CUDEV_EXPR_BINOP_INST(||, logical_or)

CV_CUDEV_EXPR_BINOP_INST(&, bit_and)
CV_CUDEV_EXPR_BINOP_INST(|, bit_or)
CV_CUDEV_EXPR_BINOP_INST(^, bit_xor)
CV_CUDEV_EXPR_BINOP_INST(<<, bit_lshift)
CV_CUDEV_EXPR_BINOP_INST(>>, bit_rshift)

#undef CV_CUDEV_EXPR_BINOP_INST

}}

#endif
