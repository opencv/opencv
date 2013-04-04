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

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/utility.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace gfft
    {
        texture<float, cudaTextureType2D, cudaReadModeElementType> eigTex(0, cudaFilterModePoint, cudaAddressModeClamp);

        __device__ uint g_counter = 0;

        template <class Mask> __global__ void findCorners(float threshold, const Mask mask, float2* corners, uint max_count, int rows, int cols)
        {
            #if __CUDA_ARCH__ >= 110

            const int j = blockIdx.x * blockDim.x + threadIdx.x;
            const int i = blockIdx.y * blockDim.y + threadIdx.y;

            if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1 && mask(i, j))
            {
                float val = tex2D(eigTex, j, i);

                if (val > threshold)
                {
                    float maxVal = val;

                    maxVal = ::fmax(tex2D(eigTex, j - 1, i - 1), maxVal);
                    maxVal = ::fmax(tex2D(eigTex, j    , i - 1), maxVal);
                    maxVal = ::fmax(tex2D(eigTex, j + 1, i - 1), maxVal);

                    maxVal = ::fmax(tex2D(eigTex, j - 1, i), maxVal);
                    maxVal = ::fmax(tex2D(eigTex, j + 1, i), maxVal);

                    maxVal = ::fmax(tex2D(eigTex, j - 1, i + 1), maxVal);
                    maxVal = ::fmax(tex2D(eigTex, j    , i + 1), maxVal);
                    maxVal = ::fmax(tex2D(eigTex, j + 1, i + 1), maxVal);

                    if (val == maxVal)
                    {
                        const uint ind = atomicInc(&g_counter, (uint)(-1));

                        if (ind < max_count)
                            corners[ind] = make_float2(j, i);
                    }
                }
            }

            #endif // __CUDA_ARCH__ >= 110
        }

        int findCorners_gpu(PtrStepSzf eig, float threshold, PtrStepSzb mask, float2* corners, int max_count)
        {
            void* counter_ptr;
            cudaSafeCall( cudaGetSymbolAddress(&counter_ptr, g_counter) );

            cudaSafeCall( cudaMemset(counter_ptr, 0, sizeof(uint)) );

            bindTexture(&eigTex, eig);

            dim3 block(16, 16);
            dim3 grid(divUp(eig.cols, block.x), divUp(eig.rows, block.y));

            if (mask.data)
                findCorners<<<grid, block>>>(threshold, SingleMask(mask), corners, max_count, eig.rows, eig.cols);
            else
                findCorners<<<grid, block>>>(threshold, WithOutMask(), corners, max_count, eig.rows, eig.cols);

            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            uint count;
            cudaSafeCall( cudaMemcpy(&count, counter_ptr, sizeof(uint), cudaMemcpyDeviceToHost) );

            return min(count, max_count);
        }

        class EigGreater
        {
        public:
            __device__ __forceinline__ bool operator()(float2 a, float2 b) const
            {
                return tex2D(eigTex, a.x, a.y) > tex2D(eigTex, b.x, b.y);
            }
        };


        void sortCorners_gpu(PtrStepSzf eig, float2* corners, int count)
        {
            bindTexture(&eigTex, eig);

            thrust::device_ptr<float2> ptr(corners);

            thrust::sort(ptr, ptr + count, EigGreater());
        }
    } // namespace optical_flow
}}}


#endif /* CUDA_DISABLER */
