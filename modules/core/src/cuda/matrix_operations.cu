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

#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/transform.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/type_traits.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"

#include "matrix_operations.hpp"

namespace cv { namespace gpu { namespace cudev
{
    ///////////////////////////////////////////////////////////////////////////
    // copyWithMask

    template <typename T>
    void copyWithMask(PtrStepSzb src, PtrStepSzb dst, int cn, PtrStepSzb mask, bool multiChannelMask, cudaStream_t stream)
    {
        if (multiChannelMask)
            cv::gpu::cudev::transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, identity<T>(), SingleMask(mask), stream);
        else
            cv::gpu::cudev::transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, identity<T>(), SingleMaskChannels(mask, cn), stream);
    }

    void copyWithMask(PtrStepSzb src, PtrStepSzb dst, size_t elemSize1, int cn, PtrStepSzb mask, bool multiChannelMask, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, int cn, PtrStepSzb mask, bool multiChannelMask, cudaStream_t stream);

        static const func_t tab[] =
        {
            0,
            copyWithMask<uchar>,
            copyWithMask<ushort>,
            0,
            copyWithMask<int>,
            0,
            0,
            0,
            copyWithMask<double>
        };

        const func_t func = tab[elemSize1];
        CV_DbgAssert( func != 0 );

        func(src, dst, cn, mask, multiChannelMask, stream);
    }

    ///////////////////////////////////////////////////////////////////////////
    // set

    template<typename T, class Mask>
    __global__ void set(PtrStepSz<T> mat, const Mask mask, const int channels, const typename TypeVec<T, 4>::vec_type value)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= mat.cols * channels || y >= mat.rows)
            return;

        const T scalar[4] = {value.x, value.y, value.z, value.w};

        if (mask(y, x / channels))
            mat(y, x) = scalar[x % channels];
    }

    template <typename T>
    void set(PtrStepSz<T> mat, const T* scalar, int channels, cudaStream_t stream)
    {
        typedef typename TypeVec<T, 4>::vec_type scalar_t;

        dim3 block(32, 8);
        dim3 grid(divUp(mat.cols * channels, block.x), divUp(mat.rows, block.y));

        set<T><<<grid, block, 0, stream>>>(mat, WithOutMask(), channels, VecTraits<scalar_t>::make(scalar));
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall ( cudaDeviceSynchronize() );
    }

    template void set<uchar >(PtrStepSz<uchar > mat, const uchar*  scalar, int channels, cudaStream_t stream);
    template void set<schar >(PtrStepSz<schar > mat, const schar*  scalar, int channels, cudaStream_t stream);
    template void set<ushort>(PtrStepSz<ushort> mat, const ushort* scalar, int channels, cudaStream_t stream);
    template void set<short >(PtrStepSz<short > mat, const short*  scalar, int channels, cudaStream_t stream);
    template void set<int   >(PtrStepSz<int   > mat, const int*    scalar, int channels, cudaStream_t stream);
    template void set<float >(PtrStepSz<float > mat, const float*  scalar, int channels, cudaStream_t stream);
    template void set<double>(PtrStepSz<double> mat, const double* scalar, int channels, cudaStream_t stream);

    template <typename T>
    void set(PtrStepSz<T> mat, const T* scalar, PtrStepSzb mask, int channels, cudaStream_t stream)
    {
        typedef typename TypeVec<T, 4>::vec_type scalar_t;

        dim3 block(32, 8);
        dim3 grid(divUp(mat.cols * channels, block.x), divUp(mat.rows, block.y));

        set<T><<<grid, block, 0, stream>>>(mat, SingleMask(mask), channels, VecTraits<scalar_t>::make(scalar));
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall ( cudaDeviceSynchronize() );
    }

    template void set<uchar >(PtrStepSz<uchar > mat, const uchar*  scalar, PtrStepSzb mask, int channels, cudaStream_t stream);
    template void set<schar >(PtrStepSz<schar > mat, const schar*  scalar, PtrStepSzb mask, int channels, cudaStream_t stream);
    template void set<ushort>(PtrStepSz<ushort> mat, const ushort* scalar, PtrStepSzb mask, int channels, cudaStream_t stream);
    template void set<short >(PtrStepSz<short > mat, const short*  scalar, PtrStepSzb mask, int channels, cudaStream_t stream);
    template void set<int   >(PtrStepSz<int   > mat, const int*    scalar, PtrStepSzb mask, int channels, cudaStream_t stream);
    template void set<float >(PtrStepSz<float > mat, const float*  scalar, PtrStepSzb mask, int channels, cudaStream_t stream);
    template void set<double>(PtrStepSz<double> mat, const double* scalar, PtrStepSzb mask, int channels, cudaStream_t stream);

    ///////////////////////////////////////////////////////////////////////////
    // convert

    template <typename T, typename D, typename S> struct Convertor : unary_function<T, D>
    {
        Convertor(S alpha_, S beta_) : alpha(alpha_), beta(beta_) {}

        __device__ __forceinline__ D operator()(typename TypeTraits<T>::ParameterType src) const
        {
            return saturate_cast<D>(alpha * src + beta);
        }

        S alpha, beta;
    };

    namespace detail
    {
        template <size_t src_size, size_t dst_size, typename F> struct ConvertTraitsDispatcher : DefaultTransformFunctorTraits<F>
        {
        };
        template <typename F> struct ConvertTraitsDispatcher<1, 1, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 8 };
        };
        template <typename F> struct ConvertTraitsDispatcher<1, 2, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 4 };
        };
        template <typename F> struct ConvertTraitsDispatcher<1, 4, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_block_dim_y = 8 };
            enum { smart_shift = 4 };
        };

        template <typename F> struct ConvertTraitsDispatcher<2, 2, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 4 };
        };
        template <typename F> struct ConvertTraitsDispatcher<2, 4, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 2 };
        };

        template <typename F> struct ConvertTraitsDispatcher<4, 2, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_block_dim_y = 8 };
            enum { smart_shift = 4 };
        };
        template <typename F> struct ConvertTraitsDispatcher<4, 4, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_block_dim_y = 8 };
            enum { smart_shift = 2 };
        };

        template <typename F> struct ConvertTraits : ConvertTraitsDispatcher<sizeof(typename F::argument_type), sizeof(typename F::result_type), F>
        {
        };
    }

    template <typename T, typename D, typename S> struct TransformFunctorTraits< Convertor<T, D, S> > : detail::ConvertTraits< Convertor<T, D, S> >
    {
    };

    template<typename T, typename D, typename S>
    void cvt_(PtrStepSzb src, PtrStepSzb dst, double alpha, double beta, cudaStream_t stream)
    {
        Convertor<T, D, S> op(static_cast<S>(alpha), static_cast<S>(beta));
        cv::gpu::cudev::transform((PtrStepSz<T>)src, (PtrStepSz<D>)dst, op, WithOutMask(), stream);
    }

    void convert(PtrStepSzb src, int sdepth, PtrStepSzb dst, int ddepth, double alpha, double beta, cudaStream_t stream)
    {
        typedef void (*caller_t)(PtrStepSzb src, PtrStepSzb dst, double alpha, double beta, cudaStream_t stream);

        static const caller_t tab[7][7] =
        {
            {
                cvt_<uchar, uchar, float>,
                cvt_<uchar, schar, float>,
                cvt_<uchar, ushort, float>,
                cvt_<uchar, short, float>,
                cvt_<uchar, int, float>,
                cvt_<uchar, float, float>,
                cvt_<uchar, double, double>
            },
            {
                cvt_<schar, uchar, float>,
                cvt_<schar, schar, float>,
                cvt_<schar, ushort, float>,
                cvt_<schar, short, float>,
                cvt_<schar, int, float>,
                cvt_<schar, float, float>,
                cvt_<schar, double, double>
            },
            {
                cvt_<ushort, uchar, float>,
                cvt_<ushort, schar, float>,
                cvt_<ushort, ushort, float>,
                cvt_<ushort, short, float>,
                cvt_<ushort, int, float>,
                cvt_<ushort, float, float>,
                cvt_<ushort, double, double>
            },
            {
                cvt_<short, uchar, float>,
                cvt_<short, schar, float>,
                cvt_<short, ushort, float>,
                cvt_<short, short, float>,
                cvt_<short, int, float>,
                cvt_<short, float, float>,
                cvt_<short, double, double>
            },
            {
                cvt_<int, uchar, float>,
                cvt_<int, schar, float>,
                cvt_<int, ushort, float>,
                cvt_<int, short, float>,
                cvt_<int, int, double>,
                cvt_<int, float, double>,
                cvt_<int, double, double>
            },
            {
                cvt_<float, uchar, float>,
                cvt_<float, schar, float>,
                cvt_<float, ushort, float>,
                cvt_<float, short, float>,
                cvt_<float, int, float>,
                cvt_<float, float, float>,
                cvt_<float, double, double>
            },
            {
                cvt_<double, uchar, double>,
                cvt_<double, schar, double>,
                cvt_<double, ushort, double>,
                cvt_<double, short, double>,
                cvt_<double, int, double>,
                cvt_<double, float, double>,
                cvt_<double, double, double>
            }
        };

        const caller_t func = tab[sdepth][ddepth];
        func(src, dst, alpha, beta, stream);
    }
}}} // namespace cv { namespace gpu { namespace cudev
