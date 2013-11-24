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

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/cudev.hpp"

using namespace cv::cudev;

void cmpScalar(const GpuMat& src, cv::Scalar val, bool inv, GpuMat& dst, const GpuMat&, double, Stream& stream, int cmpop);

namespace
{
    template <class Op, typename T> struct CmpOp : binary_function<T, T, uchar>
    {
        __device__ __forceinline__ uchar operator()(T a, T b) const
        {
            Op op;
            return -op(a, b);
        }
    };

#define MAKE_VEC(_type, _cn) typename MakeVec<_type, _cn>::type

    template <class Op, typename T, int cn> struct CmpScalarOp;

    template <class Op, typename T>
    struct CmpScalarOp<Op, T, 1> : unary_function<T, uchar>
    {
        T val;

        __device__ __forceinline__ uchar operator()(T src) const
        {
            CmpOp<Op, T> op;
            return op(src, val);
        }
    };

    template <class Op, typename T>
    struct CmpScalarOp<Op, T, 2> : unary_function<MAKE_VEC(T, 2), MAKE_VEC(uchar, 2)>
    {
        MAKE_VEC(T, 2) val;

        __device__ __forceinline__ MAKE_VEC(uchar, 2) operator()(const MAKE_VEC(T, 2) & src) const
        {
            CmpOp<Op, T> op;
            return VecTraits<MAKE_VEC(uchar, 2)>::make(op(src.x, val.x), op(src.y, val.y));
        }
    };

    template <class Op, typename T>
    struct CmpScalarOp<Op, T, 3> : unary_function<MAKE_VEC(T, 3), MAKE_VEC(uchar, 3)>
    {
        MAKE_VEC(T, 3) val;

        __device__ __forceinline__ MAKE_VEC(uchar, 3) operator()(const MAKE_VEC(T, 3) & src) const
        {
            CmpOp<Op, T> op;
            return VecTraits<MAKE_VEC(uchar, 3)>::make(op(src.x, val.x), op(src.y, val.y), op(src.z, val.z));
        }
    };

    template <class Op, typename T>
    struct CmpScalarOp<Op, T, 4> : unary_function<MAKE_VEC(T, 4), MAKE_VEC(uchar, 4)>
    {
        MAKE_VEC(T, 4) val;

        __device__ __forceinline__ MAKE_VEC(uchar, 4) operator()(const MAKE_VEC(T, 4) & src) const
        {
            CmpOp<Op, T> op;
            return VecTraits<MAKE_VEC(uchar, 4)>::make(op(src.x, val.x), op(src.y, val.y), op(src.z, val.z), op(src.w, val.w));
        }
    };

#undef TYPE_VEC

    template <typename ScalarDepth> struct TransformPolicy : DefaultTransformPolicy
    {
    };
    template <> struct TransformPolicy<double> : DefaultTransformPolicy
    {
        enum {
            shift = 1
        };
    };

    template <template <typename> class Op, typename T, int cn>
    void cmpScalarImpl(const GpuMat& src, cv::Scalar value, GpuMat& dst, Stream& stream)
    {
        typedef typename MakeVec<T, cn>::type src_type;
        typedef typename MakeVec<uchar, cn>::type dst_type;

        cv::Scalar_<T> value_ = value;

        CmpScalarOp<Op<T>, T, cn> op;
        op.val = VecTraits<src_type>::make(value_.val);

        gridTransformUnary_< TransformPolicy<T> >(globPtr<src_type>(src), globPtr<dst_type>(dst), op, stream);
    }
}

void cmpScalar(const GpuMat& src, cv::Scalar val, bool inv, GpuMat& dst, const GpuMat&, double, Stream& stream, int cmpop)
{
    typedef void (*func_t)(const GpuMat& src, cv::Scalar value, GpuMat& dst, Stream& stream);
    static const func_t funcs[7][6][4] =
    {
        {
            {cmpScalarImpl<equal_to,      uchar, 1>, cmpScalarImpl<equal_to,      uchar, 2>, cmpScalarImpl<equal_to,      uchar, 3>, cmpScalarImpl<equal_to,      uchar, 4>},
            {cmpScalarImpl<greater,       uchar, 1>, cmpScalarImpl<greater,       uchar, 2>, cmpScalarImpl<greater,       uchar, 3>, cmpScalarImpl<greater,       uchar, 4>},
            {cmpScalarImpl<greater_equal, uchar, 1>, cmpScalarImpl<greater_equal, uchar, 2>, cmpScalarImpl<greater_equal, uchar, 3>, cmpScalarImpl<greater_equal, uchar, 4>},
            {cmpScalarImpl<less,          uchar, 1>, cmpScalarImpl<less,          uchar, 2>, cmpScalarImpl<less,          uchar, 3>, cmpScalarImpl<less,          uchar, 4>},
            {cmpScalarImpl<less_equal,    uchar, 1>, cmpScalarImpl<less_equal,    uchar, 2>, cmpScalarImpl<less_equal,    uchar, 3>, cmpScalarImpl<less_equal,    uchar, 4>},
            {cmpScalarImpl<not_equal_to,  uchar, 1>, cmpScalarImpl<not_equal_to,  uchar, 2>, cmpScalarImpl<not_equal_to,  uchar, 3>, cmpScalarImpl<not_equal_to,  uchar, 4>}
        },
        {
            {cmpScalarImpl<equal_to,      schar, 1>, cmpScalarImpl<equal_to,      schar, 2>, cmpScalarImpl<equal_to,      schar, 3>, cmpScalarImpl<equal_to,      schar, 4>},
            {cmpScalarImpl<greater,       schar, 1>, cmpScalarImpl<greater,       schar, 2>, cmpScalarImpl<greater,       schar, 3>, cmpScalarImpl<greater,       schar, 4>},
            {cmpScalarImpl<greater_equal, schar, 1>, cmpScalarImpl<greater_equal, schar, 2>, cmpScalarImpl<greater_equal, schar, 3>, cmpScalarImpl<greater_equal, schar, 4>},
            {cmpScalarImpl<less,          schar, 1>, cmpScalarImpl<less,          schar, 2>, cmpScalarImpl<less,          schar, 3>, cmpScalarImpl<less,          schar, 4>},
            {cmpScalarImpl<less_equal,    schar, 1>, cmpScalarImpl<less_equal,    schar, 2>, cmpScalarImpl<less_equal,    schar, 3>, cmpScalarImpl<less_equal,    schar, 4>},
            {cmpScalarImpl<not_equal_to,  schar, 1>, cmpScalarImpl<not_equal_to,  schar, 2>, cmpScalarImpl<not_equal_to,  schar, 3>, cmpScalarImpl<not_equal_to,  schar, 4>}
        },
        {
            {cmpScalarImpl<equal_to,      ushort, 1>, cmpScalarImpl<equal_to,      ushort, 2>, cmpScalarImpl<equal_to,      ushort, 3>, cmpScalarImpl<equal_to,      ushort, 4>},
            {cmpScalarImpl<greater,       ushort, 1>, cmpScalarImpl<greater,       ushort, 2>, cmpScalarImpl<greater,       ushort, 3>, cmpScalarImpl<greater,       ushort, 4>},
            {cmpScalarImpl<greater_equal, ushort, 1>, cmpScalarImpl<greater_equal, ushort, 2>, cmpScalarImpl<greater_equal, ushort, 3>, cmpScalarImpl<greater_equal, ushort, 4>},
            {cmpScalarImpl<less,          ushort, 1>, cmpScalarImpl<less,          ushort, 2>, cmpScalarImpl<less,          ushort, 3>, cmpScalarImpl<less,          ushort, 4>},
            {cmpScalarImpl<less_equal,    ushort, 1>, cmpScalarImpl<less_equal,    ushort, 2>, cmpScalarImpl<less_equal,    ushort, 3>, cmpScalarImpl<less_equal,    ushort, 4>},
            {cmpScalarImpl<not_equal_to,  ushort, 1>, cmpScalarImpl<not_equal_to,  ushort, 2>, cmpScalarImpl<not_equal_to,  ushort, 3>, cmpScalarImpl<not_equal_to,  ushort, 4>}
        },
        {
            {cmpScalarImpl<equal_to,      short, 1>, cmpScalarImpl<equal_to,      short, 2>, cmpScalarImpl<equal_to,      short, 3>, cmpScalarImpl<equal_to,      short, 4>},
            {cmpScalarImpl<greater,       short, 1>, cmpScalarImpl<greater,       short, 2>, cmpScalarImpl<greater,       short, 3>, cmpScalarImpl<greater,       short, 4>},
            {cmpScalarImpl<greater_equal, short, 1>, cmpScalarImpl<greater_equal, short, 2>, cmpScalarImpl<greater_equal, short, 3>, cmpScalarImpl<greater_equal, short, 4>},
            {cmpScalarImpl<less,          short, 1>, cmpScalarImpl<less,          short, 2>, cmpScalarImpl<less,          short, 3>, cmpScalarImpl<less,          short, 4>},
            {cmpScalarImpl<less_equal,    short, 1>, cmpScalarImpl<less_equal,    short, 2>, cmpScalarImpl<less_equal,    short, 3>, cmpScalarImpl<less_equal,    short, 4>},
            {cmpScalarImpl<not_equal_to,  short, 1>, cmpScalarImpl<not_equal_to,  short, 2>, cmpScalarImpl<not_equal_to,  short, 3>, cmpScalarImpl<not_equal_to,  short, 4>}
        },
        {
            {cmpScalarImpl<equal_to,      int, 1>, cmpScalarImpl<equal_to,      int, 2>, cmpScalarImpl<equal_to,      int, 3>, cmpScalarImpl<equal_to,      int, 4>},
            {cmpScalarImpl<greater,       int, 1>, cmpScalarImpl<greater,       int, 2>, cmpScalarImpl<greater,       int, 3>, cmpScalarImpl<greater,       int, 4>},
            {cmpScalarImpl<greater_equal, int, 1>, cmpScalarImpl<greater_equal, int, 2>, cmpScalarImpl<greater_equal, int, 3>, cmpScalarImpl<greater_equal, int, 4>},
            {cmpScalarImpl<less,          int, 1>, cmpScalarImpl<less,          int, 2>, cmpScalarImpl<less,          int, 3>, cmpScalarImpl<less,          int, 4>},
            {cmpScalarImpl<less_equal,    int, 1>, cmpScalarImpl<less_equal,    int, 2>, cmpScalarImpl<less_equal,    int, 3>, cmpScalarImpl<less_equal,    int, 4>},
            {cmpScalarImpl<not_equal_to,  int, 1>, cmpScalarImpl<not_equal_to,  int, 2>, cmpScalarImpl<not_equal_to,  int, 3>, cmpScalarImpl<not_equal_to,  int, 4>}
        },
        {
            {cmpScalarImpl<equal_to,      float, 1>, cmpScalarImpl<equal_to,      float, 2>, cmpScalarImpl<equal_to,      float, 3>, cmpScalarImpl<equal_to,      float, 4>},
            {cmpScalarImpl<greater,       float, 1>, cmpScalarImpl<greater,       float, 2>, cmpScalarImpl<greater,       float, 3>, cmpScalarImpl<greater,       float, 4>},
            {cmpScalarImpl<greater_equal, float, 1>, cmpScalarImpl<greater_equal, float, 2>, cmpScalarImpl<greater_equal, float, 3>, cmpScalarImpl<greater_equal, float, 4>},
            {cmpScalarImpl<less,          float, 1>, cmpScalarImpl<less,          float, 2>, cmpScalarImpl<less,          float, 3>, cmpScalarImpl<less,          float, 4>},
            {cmpScalarImpl<less_equal,    float, 1>, cmpScalarImpl<less_equal,    float, 2>, cmpScalarImpl<less_equal,    float, 3>, cmpScalarImpl<less_equal,    float, 4>},
            {cmpScalarImpl<not_equal_to,  float, 1>, cmpScalarImpl<not_equal_to,  float, 2>, cmpScalarImpl<not_equal_to,  float, 3>, cmpScalarImpl<not_equal_to,  float, 4>}
        },
        {
            {cmpScalarImpl<equal_to,      double, 1>, cmpScalarImpl<equal_to,      double, 2>, cmpScalarImpl<equal_to,      double, 3>, cmpScalarImpl<equal_to,      double, 4>},
            {cmpScalarImpl<greater,       double, 1>, cmpScalarImpl<greater,       double, 2>, cmpScalarImpl<greater,       double, 3>, cmpScalarImpl<greater,       double, 4>},
            {cmpScalarImpl<greater_equal, double, 1>, cmpScalarImpl<greater_equal, double, 2>, cmpScalarImpl<greater_equal, double, 3>, cmpScalarImpl<greater_equal, double, 4>},
            {cmpScalarImpl<less,          double, 1>, cmpScalarImpl<less,          double, 2>, cmpScalarImpl<less,          double, 3>, cmpScalarImpl<less,          double, 4>},
            {cmpScalarImpl<less_equal,    double, 1>, cmpScalarImpl<less_equal,    double, 2>, cmpScalarImpl<less_equal,    double, 3>, cmpScalarImpl<less_equal,    double, 4>},
            {cmpScalarImpl<not_equal_to,  double, 1>, cmpScalarImpl<not_equal_to,  double, 2>, cmpScalarImpl<not_equal_to,  double, 3>, cmpScalarImpl<not_equal_to,  double, 4>}
        }
    };

    if (inv)
    {
        // src1 is a scalar; swap it with src2
        cmpop = cmpop == cv::CMP_LT ? cv::CMP_GT : cmpop == cv::CMP_LE ? cv::CMP_GE :
            cmpop == cv::CMP_GE ? cv::CMP_LE : cmpop == cv::CMP_GT ? cv::CMP_LT : cmpop;
    }

    const int depth = src.depth();
    const int cn = src.channels();

    CV_DbgAssert( depth <= CV_64F && cn <= 4 );

    funcs[depth][cmpop][cn - 1](src, val, dst, stream);
}

#endif
