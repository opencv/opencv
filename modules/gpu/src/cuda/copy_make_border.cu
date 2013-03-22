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

#include "internal_shared.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template <typename Ptr2D, typename T> __global__ void copyMakeBorder(const Ptr2D src, PtrStepSz<T> dst, int top, int left)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
                dst.ptr(y)[x] = src(y - top, x - left);
        }

        template <template <typename> class B, typename T> struct CopyMakeBorderDispatcher
        {
            static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& dst, int top, int left,
                const typename VecTraits<T>::elem_type* borderValue, cudaStream_t stream)
            {
                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                B<T> brd(src.rows, src.cols, VecTraits<T>::make(borderValue));
                BorderReader< PtrStep<T>, B<T> > brdSrc(src, brd);

                copyMakeBorder<<<grid, block, 0, stream>>>(brdSrc, dst, top, left);
                cudaSafeCall( cudaGetLastError() );

                if (stream == 0)
                    cudaSafeCall( cudaDeviceSynchronize() );
            }
        };

        template <typename T, int cn> void copyMakeBorder_gpu(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode,
            const T* borderValue, cudaStream_t stream)
        {
            typedef typename TypeVec<T, cn>::vec_type vec_type;

            typedef void (*caller_t)(const PtrStepSz<vec_type>& src, const PtrStepSz<vec_type>& dst, int top, int left, const T* borderValue, cudaStream_t stream);

            static const caller_t callers[5] =
            {
                CopyMakeBorderDispatcher<BrdReflect101, vec_type>::call,
                CopyMakeBorderDispatcher<BrdReplicate, vec_type>::call,
                CopyMakeBorderDispatcher<BrdConstant, vec_type>::call,
                CopyMakeBorderDispatcher<BrdReflect, vec_type>::call,
                CopyMakeBorderDispatcher<BrdWrap, vec_type>::call
            };

            callers[borderMode](PtrStepSz<vec_type>(src), PtrStepSz<vec_type>(dst), top, left, borderValue, stream);
        }

        template void copyMakeBorder_gpu<uchar, 1>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const uchar* borderValue, cudaStream_t stream);
        template void copyMakeBorder_gpu<uchar, 2>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const uchar* borderValue, cudaStream_t stream);
        template void copyMakeBorder_gpu<uchar, 3>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const uchar* borderValue, cudaStream_t stream);
        template void copyMakeBorder_gpu<uchar, 4>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const uchar* borderValue, cudaStream_t stream);

        //template void copyMakeBorder_gpu<schar, 1>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const schar* borderValue, cudaStream_t stream);
        //template void copyMakeBorder_gpu<schar, 2>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const schar* borderValue, cudaStream_t stream);
        //template void copyMakeBorder_gpu<schar, 3>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const schar* borderValue, cudaStream_t stream);
        //template void copyMakeBorder_gpu<schar, 4>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const schar* borderValue, cudaStream_t stream);

        template void copyMakeBorder_gpu<ushort, 1>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const ushort* borderValue, cudaStream_t stream);
        //template void copyMakeBorder_gpu<ushort, 2>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const ushort* borderValue, cudaStream_t stream);
        template void copyMakeBorder_gpu<ushort, 3>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const ushort* borderValue, cudaStream_t stream);
        template void copyMakeBorder_gpu<ushort, 4>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const ushort* borderValue, cudaStream_t stream);

        template void copyMakeBorder_gpu<short, 1>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const short* borderValue, cudaStream_t stream);
        //template void copyMakeBorder_gpu<short, 2>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const short* borderValue, cudaStream_t stream);
        template void copyMakeBorder_gpu<short, 3>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const short* borderValue, cudaStream_t stream);
        template void copyMakeBorder_gpu<short, 4>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const short* borderValue, cudaStream_t stream);

        //template void copyMakeBorder_gpu<int, 1>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const int* borderValue, cudaStream_t stream);
        //template void copyMakeBorder_gpu<int, 2>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const int* borderValue, cudaStream_t stream);
        //template void copyMakeBorder_gpu<int, 3>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const int* borderValue, cudaStream_t stream);
        //template void copyMakeBorder_gpu<int, 4>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const int* borderValue, cudaStream_t stream);

        template void copyMakeBorder_gpu<float, 1>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const float* borderValue, cudaStream_t stream);
        //template void copyMakeBorder_gpu<float, 2>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const float* borderValue, cudaStream_t stream);
        template void copyMakeBorder_gpu<float, 3>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const float* borderValue, cudaStream_t stream);
        template void copyMakeBorder_gpu<float, 4>(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const float* borderValue, cudaStream_t stream);
    } // namespace imgproc
}}} // namespace cv { namespace gpu { namespace device

#endif /* CUDA_DISABLER */
