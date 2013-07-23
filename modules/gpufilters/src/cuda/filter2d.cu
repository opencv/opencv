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
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"

namespace cv { namespace cuda { namespace device
{
    template <class SrcPtr, typename D>
    __global__ void filter2D(const SrcPtr src, PtrStepSz<D> dst,
                             const float* __restrict__ kernel,
                             const int kWidth, const int kHeight,
                             const int anchorX, const int anchorY)
    {
        typedef typename TypeVec<float, VecTraits<D>::cn>::vec_type sum_t;

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= dst.cols || y >= dst.rows)
            return;

        sum_t res = VecTraits<sum_t>::all(0);
        int kInd = 0;

        for (int i = 0; i < kHeight; ++i)
        {
            for (int j = 0; j < kWidth; ++j)
                res = res + src(y - anchorY + i, x - anchorX + j) * kernel[kInd++];
        }

        dst(y, x) = saturate_cast<D>(res);
    }

    template <typename T, typename D, template <typename> class Brd> struct Filter2DCaller;

    #define IMPLEMENT_FILTER2D_TEX_READER(type) \
        texture< type , cudaTextureType2D, cudaReadModeElementType> tex_filter2D_ ## type (0, cudaFilterModePoint, cudaAddressModeClamp); \
        struct tex_filter2D_ ## type ## _reader \
        { \
            typedef type elem_type; \
            typedef int index_type; \
            const int xoff; \
            const int yoff; \
            tex_filter2D_ ## type ## _reader (int xoff_, int yoff_) : xoff(xoff_), yoff(yoff_) {} \
            __device__ __forceinline__ elem_type operator ()(index_type y, index_type x) const \
            { \
                return tex2D(tex_filter2D_ ## type , x + xoff, y + yoff); \
            } \
        }; \
        template <typename D, template <typename> class Brd> struct Filter2DCaller< type , D, Brd> \
        { \
            static void call(const PtrStepSz< type > srcWhole, int xoff, int yoff, PtrStepSz<D> dst, const float* kernel, \
                int kWidth, int kHeight, int anchorX, int anchorY, const float* borderValue, cudaStream_t stream) \
            { \
                typedef typename TypeVec<float, VecTraits< type >::cn>::vec_type work_type; \
                dim3 block(16, 16); \
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y)); \
                bindTexture(&tex_filter2D_ ## type , srcWhole); \
                tex_filter2D_ ## type ##_reader texSrc(xoff, yoff); \
                Brd<work_type> brd(dst.rows, dst.cols, VecTraits<work_type>::make(borderValue)); \
                BorderReader< tex_filter2D_ ## type ##_reader, Brd<work_type> > brdSrc(texSrc, brd); \
                filter2D<<<grid, block, 0, stream>>>(brdSrc, dst, kernel, kWidth, kHeight, anchorX, anchorY); \
                cudaSafeCall( cudaGetLastError() ); \
                if (stream == 0) \
                    cudaSafeCall( cudaDeviceSynchronize() ); \
            } \
        };

    IMPLEMENT_FILTER2D_TEX_READER(uchar);
    IMPLEMENT_FILTER2D_TEX_READER(uchar4);

    IMPLEMENT_FILTER2D_TEX_READER(ushort);
    IMPLEMENT_FILTER2D_TEX_READER(ushort4);

    IMPLEMENT_FILTER2D_TEX_READER(float);
    IMPLEMENT_FILTER2D_TEX_READER(float4);

    #undef IMPLEMENT_FILTER2D_TEX_READER

    template <typename T, typename D>
    void filter2D(PtrStepSzb srcWhole, int ofsX, int ofsY, PtrStepSzb dst, const float* kernel,
                  int kWidth, int kHeight, int anchorX, int anchorY,
                  int borderMode, const float* borderValue, cudaStream_t stream)
    {
        typedef void (*func_t)(const PtrStepSz<T> srcWhole, int xoff, int yoff, PtrStepSz<D> dst, const float* kernel,
                               int kWidth, int kHeight, int anchorX, int anchorY, const float* borderValue, cudaStream_t stream);
        static const func_t funcs[] =
        {
            Filter2DCaller<T, D, BrdConstant>::call,
            Filter2DCaller<T, D, BrdReplicate>::call,
            Filter2DCaller<T, D, BrdReflect>::call,
            Filter2DCaller<T, D, BrdWrap>::call,
            Filter2DCaller<T, D, BrdReflect101>::call
        };

        funcs[borderMode]((PtrStepSz<T>) srcWhole, ofsX, ofsY, (PtrStepSz<D>) dst, kernel,
                          kWidth, kHeight, anchorX, anchorY, borderValue, stream);
    }

    template void filter2D<uchar  , uchar  >(PtrStepSzb srcWhole, int ofsX, int ofsY, PtrStepSzb dst, const float* kernel, int kWidth, int kHeight, int anchorX, int anchorY, int borderMode, const float* borderValue, cudaStream_t stream);
    template void filter2D<uchar4 , uchar4 >(PtrStepSzb srcWhole, int ofsX, int ofsY, PtrStepSzb dst, const float* kernel, int kWidth, int kHeight, int anchorX, int anchorY, int borderMode, const float* borderValue, cudaStream_t stream);
    template void filter2D<ushort , ushort >(PtrStepSzb srcWhole, int ofsX, int ofsY, PtrStepSzb dst, const float* kernel, int kWidth, int kHeight, int anchorX, int anchorY, int borderMode, const float* borderValue, cudaStream_t stream);
    template void filter2D<ushort4, ushort4>(PtrStepSzb srcWhole, int ofsX, int ofsY, PtrStepSzb dst, const float* kernel, int kWidth, int kHeight, int anchorX, int anchorY, int borderMode, const float* borderValue, cudaStream_t stream);
    template void filter2D<float  , float  >(PtrStepSzb srcWhole, int ofsX, int ofsY, PtrStepSzb dst, const float* kernel, int kWidth, int kHeight, int anchorX, int anchorY, int borderMode, const float* borderValue, cudaStream_t stream);
    template void filter2D<float4 , float4 >(PtrStepSzb srcWhole, int ofsX, int ofsY, PtrStepSzb dst, const float* kernel, int kWidth, int kHeight, int anchorX, int anchorY, int borderMode, const float* borderValue, cudaStream_t stream);
}}}

#endif // CUDA_DISABLER
