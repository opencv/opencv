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

namespace imgproc {

template <typename T, typename B> __global__ void pyrUp(const PtrStep<T> src, DevMem2D_<T> dst, const B b)
{
    typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type value_type;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ T smem1[10][10];
    __shared__ value_type smem2[20][16];

    value_type sum;

    if (threadIdx.x < 10 && threadIdx.y < 10)
        smem1[threadIdx.y][threadIdx.x] = b.at(blockIdx.y * blockDim.y / 2 + threadIdx.y - 1, blockIdx.x * blockDim.x / 2 + threadIdx.x - 1, src.data, src.step);

    __syncthreads();

    const int tidx = threadIdx.x;

    sum = VecTraits<value_type>::all(0);

    sum = sum + (tidx % 2 == 0) * 0.0625f * smem1[1 + threadIdx.y / 2][1 + ((tidx - 2) >> 1)];
    sum = sum + (tidx % 2 != 0) * 0.25f   * smem1[1 + threadIdx.y / 2][1 + ((tidx - 1) >> 1)];
    sum = sum + (tidx % 2 == 0) * 0.375f  * smem1[1 + threadIdx.y / 2][1 + ((tidx    ) >> 1)];
    sum = sum + (tidx % 2 != 0) * 0.25f   * smem1[1 + threadIdx.y / 2][1 + ((tidx + 1) >> 1)];
    sum = sum + (tidx % 2 == 0) * 0.0625f * smem1[1 + threadIdx.y / 2][1 + ((tidx + 2) >> 1)];

    smem2[2 + threadIdx.y][tidx] = sum;

    if (threadIdx.y < 2)
    {
        sum = VecTraits<value_type>::all(0);

        sum = sum + (tidx % 2 == 0) * 0.0625f * smem1[0][1 + ((tidx - 2) >> 1)];
        sum = sum + (tidx % 2 != 0) * 0.25f   * smem1[0][1 + ((tidx - 1) >> 1)];
        sum = sum + (tidx % 2 == 0) * 0.375f  * smem1[0][1 + ((tidx    ) >> 1)];
        sum = sum + (tidx % 2 != 0) * 0.25f   * smem1[0][1 + ((tidx + 1) >> 1)];
        sum = sum + (tidx % 2 == 0) * 0.0625f * smem1[0][1 + ((tidx + 2) >> 1)];

        smem2[threadIdx.y][tidx] = sum;
    }

    if (threadIdx.y > 13)
    {
        sum = VecTraits<value_type>::all(0);

        sum = sum + (tidx % 2 == 0) * 0.0625f * smem1[9][1 + ((tidx - 2) >> 1)];
        sum = sum + (tidx % 2 != 0) * 0.25f   * smem1[9][1 + ((tidx - 1) >> 1)];
        sum = sum + (tidx % 2 == 0) * 0.375f  * smem1[9][1 + ((tidx    ) >> 1)];
        sum = sum + (tidx % 2 != 0) * 0.25f   * smem1[9][1 + ((tidx + 1) >> 1)];
        sum = sum + (tidx % 2 == 0) * 0.0625f * smem1[9][1 + ((tidx + 2) >> 1)];

        smem2[4 + threadIdx.y][tidx] = sum;
    }

    __syncthreads();

    sum = VecTraits<value_type>::all(0);

    sum = sum + (tidx % 2 == 0) * 0.0625f * smem2[2 + threadIdx.y - 2][tidx];
    sum = sum + (tidx % 2 != 0) * 0.25f   * smem2[2 + threadIdx.y - 1][tidx];
    sum = sum + (tidx % 2 == 0) * 0.375f  * smem2[2 + threadIdx.y    ][tidx];
    sum = sum + (tidx % 2 != 0) * 0.25f   * smem2[2 + threadIdx.y + 1][tidx];
    sum = sum + (tidx % 2 == 0) * 0.0625f * smem2[2 + threadIdx.y + 2][tidx];

    if (x < dst.cols && y < dst.rows)
        dst.ptr(y)[x] = saturate_cast<T>(4.0f * sum);
}

template <typename T, template <typename> class B> void pyrUp_caller(const DevMem2D_<T>& src, const DevMem2D_<T>& dst, cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    B<T> b(src.rows, src.cols);

    pyrUp<T><<<grid, block, 0, stream>>>(src, dst, b);
    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

template <typename T, int cn> void pyrUp_gpu(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream)
{
    typedef typename TypeVec<T, cn>::vec_type type;

    typedef void (*caller_t)(const DevMem2D_<type>& src, const DevMem2D_<type>& dst, cudaStream_t stream);

    static const caller_t callers[] = 
    {
        pyrUp_caller<type, BrdReflect101>, pyrUp_caller<type, BrdReplicate>, pyrUp_caller<type, BrdConstant>, pyrUp_caller<type, BrdReflect>, pyrUp_caller<type, BrdWrap>
    };

    callers[borderType](static_cast< DevMem2D_<type> >(src), static_cast< DevMem2D_<type> >(dst), stream);
}

template void pyrUp_gpu<uchar, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<uchar, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<uchar, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<uchar, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

template void pyrUp_gpu<schar, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<schar, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<schar, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<schar, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

template void pyrUp_gpu<ushort, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<ushort, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<ushort, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<ushort, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

template void pyrUp_gpu<short, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<short, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<short, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<short, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

template void pyrUp_gpu<int, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<int, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<int, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<int, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

template void pyrUp_gpu<float, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<float, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<float, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
template void pyrUp_gpu<float, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

} // namespace imgproc

END_OPENCV_DEVICE_NAMESPACE
