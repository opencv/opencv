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

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/transform.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/simd_functions.hpp"
#include "opencv2/core/cuda/vec_math.hpp"

#include "arithm_func_traits.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace arithm
{
    template <class Op, typename T>
    struct Cmp : binary_function<T, T, uchar>
    {
        __device__ __forceinline__ uchar operator()(T a, T b) const
        {
            Op op;
            return -op(a, b);
        }
    };

#define TYPE_VEC(type, cn) typename TypeVec<type, cn>::vec_type

    template <class Op, typename T, int cn> struct CmpScalar;
    template <class Op, typename T>
    struct CmpScalar<Op, T, 1> : unary_function<T, uchar>
    {
        T val;

        __host__ explicit CmpScalar(T val_) : val(val_) {}

        __device__ __forceinline__ uchar operator()(T src) const
        {
            Cmp<Op, T> op;
            return op(src, val);
        }
    };
    template <class Op, typename T>
    struct CmpScalar<Op, T, 2> : unary_function<TYPE_VEC(T, 2), TYPE_VEC(uchar, 2)>
    {
        TYPE_VEC(T, 2) val;

        __host__ explicit CmpScalar(TYPE_VEC(T, 2) val_) : val(val_) {}

        __device__ __forceinline__ TYPE_VEC(uchar, 2) operator()(const TYPE_VEC(T, 2) & src) const
        {
            Cmp<Op, T> op;
            return VecTraits<TYPE_VEC(uchar, 2)>::make(op(src.x, val.x), op(src.y, val.y));
        }
    };
    template <class Op, typename T>
    struct CmpScalar<Op, T, 3> : unary_function<TYPE_VEC(T, 3), TYPE_VEC(uchar, 3)>
    {
        TYPE_VEC(T, 3) val;

        __host__ explicit CmpScalar(TYPE_VEC(T, 3) val_) : val(val_) {}

        __device__ __forceinline__ TYPE_VEC(uchar, 3) operator()(const TYPE_VEC(T, 3) & src) const
        {
            Cmp<Op, T> op;
            return VecTraits<TYPE_VEC(uchar, 3)>::make(op(src.x, val.x), op(src.y, val.y), op(src.z, val.z));
        }
    };
    template <class Op, typename T>
    struct CmpScalar<Op, T, 4> : unary_function<TYPE_VEC(T, 4), TYPE_VEC(uchar, 4)>
    {
        TYPE_VEC(T, 4) val;

        __host__ explicit CmpScalar(TYPE_VEC(T, 4) val_) : val(val_) {}

        __device__ __forceinline__ TYPE_VEC(uchar, 4) operator()(const TYPE_VEC(T, 4) & src) const
        {
            Cmp<Op, T> op;
            return VecTraits<TYPE_VEC(uchar, 4)>::make(op(src.x, val.x), op(src.y, val.y), op(src.z, val.z), op(src.w, val.w));
        }
    };

#undef TYPE_VEC
}

namespace cv { namespace cuda { namespace device
{
    template <class Op, typename T> struct TransformFunctorTraits< arithm::CmpScalar<Op, T, 1> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(uchar)>
    {
    };
}}}

namespace arithm
{
    template <template <typename> class Op, typename T, int cn>
    void cmpScalar(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef typename TypeVec<T, cn>::vec_type src_t;
        typedef typename TypeVec<uchar, cn>::vec_type dst_t;

        T sval[] = {static_cast<T>(val[0]), static_cast<T>(val[1]), static_cast<T>(val[2]), static_cast<T>(val[3])};
        src_t val1 = VecTraits<src_t>::make(sval);

        CmpScalar<Op<T>, T, cn> op(val1);
        device::transform((PtrStepSz<src_t>) src, (PtrStepSz<dst_t>) dst, op, WithOutMask(), stream);
    }

    template <typename T> void cmpScalarEq(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            cmpScalar<equal_to, T, 1>,
            cmpScalar<equal_to, T, 2>,
            cmpScalar<equal_to, T, 3>,
            cmpScalar<equal_to, T, 4>
        };

        funcs[cn](src, val, dst, stream);
    }
    template <typename T> void cmpScalarNe(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            cmpScalar<not_equal_to, T, 1>,
            cmpScalar<not_equal_to, T, 2>,
            cmpScalar<not_equal_to, T, 3>,
            cmpScalar<not_equal_to, T, 4>
        };

        funcs[cn](src, val, dst, stream);
    }
    template <typename T> void cmpScalarLt(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            cmpScalar<less, T, 1>,
            cmpScalar<less, T, 2>,
            cmpScalar<less, T, 3>,
            cmpScalar<less, T, 4>
        };

        funcs[cn](src, val, dst, stream);
    }
    template <typename T> void cmpScalarLe(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            cmpScalar<less_equal, T, 1>,
            cmpScalar<less_equal, T, 2>,
            cmpScalar<less_equal, T, 3>,
            cmpScalar<less_equal, T, 4>
        };

        funcs[cn](src, val, dst, stream);
    }
    template <typename T> void cmpScalarGt(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            cmpScalar<greater, T, 1>,
            cmpScalar<greater, T, 2>,
            cmpScalar<greater, T, 3>,
            cmpScalar<greater, T, 4>
        };

        funcs[cn](src, val, dst, stream);
    }
    template <typename T> void cmpScalarGe(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            cmpScalar<greater_equal, T, 1>,
            cmpScalar<greater_equal, T, 2>,
            cmpScalar<greater_equal, T, 3>,
            cmpScalar<greater_equal, T, 4>
        };

        funcs[cn](src, val, dst, stream);
    }

    template void cmpScalarEq<uchar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarEq<schar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarEq<ushort>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarEq<short >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarEq<int   >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarEq<float >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarEq<double>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);

    template void cmpScalarNe<uchar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarNe<schar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarNe<ushort>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarNe<short >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarNe<int   >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarNe<float >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarNe<double>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);

    template void cmpScalarLt<uchar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLt<schar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLt<ushort>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLt<short >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLt<int   >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLt<float >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLt<double>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);

    template void cmpScalarLe<uchar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLe<schar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLe<ushort>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLe<short >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLe<int   >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLe<float >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLe<double>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);

    template void cmpScalarGt<uchar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGt<schar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGt<ushort>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGt<short >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGt<int   >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGt<float >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGt<double>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);

    template void cmpScalarGe<uchar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGe<schar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGe<ushort>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGe<short >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGe<int   >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGe<float >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGe<double>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
}

#endif // CUDA_DISABLER
