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

#include "internal_shared.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"
#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"

BEGIN_OPENCV_DEVICE_NAMESPACE

namespace pyr_down {

template <typename T, typename B> __global__ void pyrDown(const PtrStep<T> src, PtrStep<T> dst, const B b, int dst_cols)
{
    typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type value_type;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    __shared__ value_type smem[256 + 4];

    value_type sum;
    
    const int src_y = 2*y;

    sum = VecTraits<value_type>::all(0);
    
    sum = sum + 0.0625f * b.at(src_y - 2, x, src.data, src.step);
    sum = sum + 0.25f   * b.at(src_y - 1, x, src.data, src.step);
    sum = sum + 0.375f  * b.at(src_y    , x, src.data, src.step);
    sum = sum + 0.25f   * b.at(src_y + 1, x, src.data, src.step);
    sum = sum + 0.0625f * b.at(src_y + 2, x, src.data, src.step);

    smem[2 + threadIdx.x] = sum;

    if (threadIdx.x < 2)
    {
        const int left_x = x - 2 + threadIdx.x;

        sum = VecTraits<value_type>::all(0);
    
        sum = sum + 0.0625f * b.at(src_y - 2, left_x, src.data, src.step);
        sum = sum + 0.25f   * b.at(src_y - 1, left_x, src.data, src.step);
        sum = sum + 0.375f  * b.at(src_y    , left_x, src.data, src.step);
        sum = sum + 0.25f   * b.at(src_y + 1, left_x, src.data, src.step);
        sum = sum + 0.0625f * b.at(src_y + 2, left_x, src.data, src.step);

        smem[threadIdx.x] = sum;
    }

    if (threadIdx.x > 253)
    {
        const int right_x = x + threadIdx.x + 2;

        sum = VecTraits<value_type>::all(0);
    
        sum = sum + 0.0625f * b.at(src_y - 2, right_x, src.data, src.step);
        sum = sum + 0.25f   * b.at(src_y - 1, right_x, src.data, src.step);
        sum = sum + 0.375f  * b.at(src_y    , right_x, src.data, src.step);
        sum = sum + 0.25f   * b.at(src_y + 1, right_x, src.data, src.step);
        sum = sum + 0.0625f * b.at(src_y + 2, right_x, src.data, src.step);

        smem[4 + threadIdx.x] = sum;
    }

    __syncthreads();

    if (threadIdx.x < 128)
    {
        const int tid2 = threadIdx.x * 2;

        sum = VecTraits<value_type>::all(0);

        sum = sum + 0.0625f * smem[2 + tid2 - 2];
        sum = sum + 0.25f   * smem[2 + tid2 - 1];
        sum = sum + 0.375f  * smem[2 + tid2    ];
        sum = sum + 0.25f   * smem[2 + tid2 + 1];
        sum = sum + 0.0625f * smem[2 + tid2 + 2];

        const int dst_x = (blockIdx.x * blockDim.x + tid2) / 2;

        if (dst_x < dst_cols)
            dst.ptr(y)[dst_x] = saturate_cast<T>(sum);
    }
}

template <typename T, template <typename> class B> void pyrDown_caller(const DevMem2D_<T>& src, const DevMem2D_<T>& dst, cudaStream_t stream)
{
    const dim3 block(256);
    const dim3 grid(divUp(src.cols, block.x), dst.rows);

    B<T> b(src.rows, src.cols);

    pyrDown<T><<<grid, block, 0, stream>>>(src, dst, b, dst.cols);
    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

template <typename T, int cn> void pyrDown_gpu(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream)
{
    typedef typename TypeVec<T, cn>::vec_type type;

    typedef void (*caller_t)(const DevMem2D_<type>& src, const DevMem2D_<type>& dst, cudaStream_t stream);

    static const caller_t callers[] = 
    {
        pyrDown_caller<type, BrdReflect101>, pyrDown_caller<type, BrdReplicate>, pyrDown_caller<type, BrdConstant>, pyrDown_caller<type, BrdReflect>, pyrDown_caller<type, BrdWrap>
    };

    callers[borderType](static_cast< DevMem2D_<type> >(src), static_cast< DevMem2D_<type> >(dst), stream);
}

template void pyrDown_gpu<uchar, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<uchar, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<uchar, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<uchar, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

template void pyrDown_gpu<schar, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<schar, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<schar, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<schar, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

template void pyrDown_gpu<ushort, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<ushort, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<ushort, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<ushort, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

template void pyrDown_gpu<short, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<short, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<short, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<short, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

template void pyrDown_gpu<int, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<int, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<int, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<int, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

template void pyrDown_gpu<float, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<float, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<float, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrDown_gpu<float, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

} // namespace pyr_down

END_OPENCV_DEVICE_NAMESPACE
