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

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/functional.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace orb
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // cull

        int cull_gpu(int* loc, float* response, int size, int n_points)
        {
            thrust::device_ptr<int> loc_ptr(loc);
            thrust::device_ptr<float> response_ptr(response);

            thrust::sort_by_key(response_ptr, response_ptr + size, loc_ptr, thrust::greater<float>());

            return n_points;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // HarrisResponses

        __global__ void HarrisResponses(const PtrStepb img, const short2* loc_, float* response, const int npoints, const int blockSize, const float harris_k)
        {
            __shared__ int smem0[8 * 32];
            __shared__ int smem1[8 * 32];
            __shared__ int smem2[8 * 32];

            const int ptidx = blockIdx.x * blockDim.y + threadIdx.y;

            if (ptidx < npoints)
            {
                const short2 loc = loc_[ptidx];

                const int r = blockSize / 2;
                const int x0 = loc.x - r;
                const int y0 = loc.y - r;

                int a = 0, b = 0, c = 0;

                for (int ind = threadIdx.x; ind < blockSize * blockSize; ind += blockDim.x)
                {
                    const int i = ind / blockSize;
                    const int j = ind % blockSize;

                    int Ix = (img(y0 + i, x0 + j + 1) - img(y0 + i, x0 + j - 1)) * 2 +
                        (img(y0 + i - 1, x0 + j + 1) - img(y0 + i - 1, x0 + j - 1)) +
                        (img(y0 + i + 1, x0 + j + 1) - img(y0 + i + 1, x0 + j - 1));

                    int Iy = (img(y0 + i + 1, x0 + j) - img(y0 + i - 1, x0 + j)) * 2 +
                        (img(y0 + i + 1, x0 + j - 1) - img(y0 + i - 1, x0 + j - 1)) +
                        (img(y0 + i + 1, x0 + j + 1) - img(y0 + i - 1, x0 + j + 1));

                    a += Ix * Ix;
                    b += Iy * Iy;
                    c += Ix * Iy;
                }

                int* srow0 = smem0 + threadIdx.y * blockDim.x;
                int* srow1 = smem1 + threadIdx.y * blockDim.x;
                int* srow2 = smem2 + threadIdx.y * blockDim.x;

                plus<int> op;
                reduce<32>(smem_tuple(srow0, srow1, srow2), thrust::tie(a, b, c), threadIdx.x, thrust::make_tuple(op, op, op));

                if (threadIdx.x == 0)
                {
                    float scale = (1 << 2) * blockSize * 255.0f;
                    scale = 1.0f / scale;
                    const float scale_sq_sq = scale * scale * scale * scale;

                    response[ptidx] = ((float)a * b - (float)c * c - harris_k * ((float)a + b) * ((float)a + b)) * scale_sq_sq;
                }
            }
        }

        void HarrisResponses_gpu(PtrStepSzb img, const short2* loc, float* response, const int npoints, int blockSize, float harris_k, cudaStream_t stream)
        {
            dim3 block(32, 8);

            dim3 grid;
            grid.x = divUp(npoints, block.y);

            HarrisResponses<<<grid, block, 0, stream>>>(img, loc, response, npoints, blockSize, harris_k);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // IC_Angle

        __constant__ int c_u_max[32];

        void loadUMax(const int* u_max, int count)
        {
            cudaSafeCall( cudaMemcpyToSymbol(c_u_max, u_max, count * sizeof(int)) );
        }

        __global__ void IC_Angle(const PtrStepb image, const short2* loc_, float* angle, const int npoints, const int half_k)
        {
            __shared__ int smem0[8 * 32];
            __shared__ int smem1[8 * 32];

            int* srow0 = smem0 + threadIdx.y * blockDim.x;
            int* srow1 = smem1 + threadIdx.y * blockDim.x;

            plus<int> op;

            const int ptidx = blockIdx.x * blockDim.y + threadIdx.y;

            if (ptidx < npoints)
            {
                int m_01 = 0, m_10 = 0;

                const short2 loc = loc_[ptidx];

                // Treat the center line differently, v=0
                for (int u = threadIdx.x - half_k; u <= half_k; u += blockDim.x)
                    m_10 += u * image(loc.y, loc.x + u);

                reduce<32>(srow0, m_10, threadIdx.x, op);

                for (int v = 1; v <= half_k; ++v)
                {
                    // Proceed over the two lines
                    int v_sum = 0;
                    int m_sum = 0;
                    const int d = c_u_max[v];

                    for (int u = threadIdx.x - d; u <= d; u += blockDim.x)
                    {
                        int val_plus = image(loc.y + v, loc.x + u);
                        int val_minus = image(loc.y - v, loc.x + u);

                        v_sum += (val_plus - val_minus);
                        m_sum += u * (val_plus + val_minus);
                    }

                    reduce<32>(smem_tuple(srow0, srow1), thrust::tie(v_sum, m_sum), threadIdx.x, thrust::make_tuple(op, op));

                    m_10 += m_sum;
                    m_01 += v * v_sum;
                }

                if (threadIdx.x == 0)
                {
                    float kp_dir = ::atan2f((float)m_01, (float)m_10);
                    kp_dir += (kp_dir < 0) * (2.0f * CV_PI_F);
                    kp_dir *= 180.0f / CV_PI_F;

                    angle[ptidx] = kp_dir;
                }
            }
        }

        void IC_Angle_gpu(PtrStepSzb image, const short2* loc, float* angle, int npoints, int half_k, cudaStream_t stream)
        {
            dim3 block(32, 8);

            dim3 grid;
            grid.x = divUp(npoints, block.y);

            IC_Angle<<<grid, block, 0, stream>>>(image, loc, angle, npoints, half_k);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // computeOrbDescriptor

        template <int WTA_K> struct OrbDescriptor;

        #define GET_VALUE(idx) \
            img(loc.y + __float2int_rn(pattern_x[idx] * sina + pattern_y[idx] * cosa), \
                loc.x + __float2int_rn(pattern_x[idx] * cosa - pattern_y[idx] * sina))

        template <> struct OrbDescriptor<2>
        {
            __device__ static int calc(const PtrStepb& img, short2 loc, const int* pattern_x, const int* pattern_y, float sina, float cosa, int i)
            {
                pattern_x += 16 * i;
                pattern_y += 16 * i;

                int t0, t1, val;

                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                val = t0 < t1;

                t0 = GET_VALUE(2); t1 = GET_VALUE(3);
                val |= (t0 < t1) << 1;

                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                val |= (t0 < t1) << 2;

                t0 = GET_VALUE(6); t1 = GET_VALUE(7);
                val |= (t0 < t1) << 3;

                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                val |= (t0 < t1) << 4;

                t0 = GET_VALUE(10); t1 = GET_VALUE(11);
                val |= (t0 < t1) << 5;

                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                val |= (t0 < t1) << 6;

                t0 = GET_VALUE(14); t1 = GET_VALUE(15);
                val |= (t0 < t1) << 7;

                return val;
            }
        };

        template <> struct OrbDescriptor<3>
        {
            __device__ static int calc(const PtrStepb& img, short2 loc, const int* pattern_x, const int* pattern_y, float sina, float cosa, int i)
            {
                pattern_x += 12 * i;
                pattern_y += 12 * i;

                int t0, t1, t2, val;

                t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
                val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

                t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

                t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

                t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

                return val;
            }
        };

        template <> struct OrbDescriptor<4>
        {
            __device__ static int calc(const PtrStepb& img, short2 loc, const int* pattern_x, const int* pattern_y, float sina, float cosa, int i)
            {
                pattern_x += 16 * i;
                pattern_y += 16 * i;

                int t0, t1, t2, t3, k, val;
                int a, b;

                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                t2 = GET_VALUE(2); t3 = GET_VALUE(3);
                a = 0, b = 2;
                if( t1 > t0 ) t0 = t1, a = 1;
                if( t3 > t2 ) t2 = t3, b = 3;
                k = t0 > t2 ? a : b;
                val = k;

                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                t2 = GET_VALUE(6); t3 = GET_VALUE(7);
                a = 0, b = 2;
                if( t1 > t0 ) t0 = t1, a = 1;
                if( t3 > t2 ) t2 = t3, b = 3;
                k = t0 > t2 ? a : b;
                val |= k << 2;

                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                t2 = GET_VALUE(10); t3 = GET_VALUE(11);
                a = 0, b = 2;
                if( t1 > t0 ) t0 = t1, a = 1;
                if( t3 > t2 ) t2 = t3, b = 3;
                k = t0 > t2 ? a : b;
                val |= k << 4;

                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                t2 = GET_VALUE(14); t3 = GET_VALUE(15);
                a = 0, b = 2;
                if( t1 > t0 ) t0 = t1, a = 1;
                if( t3 > t2 ) t2 = t3, b = 3;
                k = t0 > t2 ? a : b;
                val |= k << 6;

                return val;
            }
        };

        #undef GET_VALUE

        template <int WTA_K>
        __global__ void computeOrbDescriptor(const PtrStepb img, const short2* loc, const float* angle_, const int npoints,
            const int* pattern_x, const int* pattern_y, PtrStepb desc, int dsize)
        {
            const int descidx = blockIdx.x * blockDim.x + threadIdx.x;
            const int ptidx = blockIdx.y * blockDim.y + threadIdx.y;

            if (ptidx < npoints && descidx < dsize)
            {
                float angle = angle_[ptidx];
                angle *= (float)(CV_PI_F / 180.f);

                float sina, cosa;
                ::sincosf(angle, &sina, &cosa);

                desc.ptr(ptidx)[descidx] = OrbDescriptor<WTA_K>::calc(img, loc[ptidx], pattern_x, pattern_y, sina, cosa, descidx);
            }
        }

        void computeOrbDescriptor_gpu(PtrStepb img, const short2* loc, const float* angle, const int npoints,
            const int* pattern_x, const int* pattern_y, PtrStepb desc, int dsize, int WTA_K, cudaStream_t stream)
        {
            dim3 block(32, 8);

            dim3 grid;
            grid.x = divUp(dsize, block.x);
            grid.y = divUp(npoints, block.y);

            switch (WTA_K)
            {
            case 2:
                computeOrbDescriptor<2><<<grid, block, 0, stream>>>(img, loc, angle, npoints, pattern_x, pattern_y, desc, dsize);
                break;

            case 3:
                computeOrbDescriptor<3><<<grid, block, 0, stream>>>(img, loc, angle, npoints, pattern_x, pattern_y, desc, dsize);
                break;

            case 4:
                computeOrbDescriptor<4><<<grid, block, 0, stream>>>(img, loc, angle, npoints, pattern_x, pattern_y, desc, dsize);
                break;
            }

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // mergeLocation

        __global__ void mergeLocation(const short2* loc_, float* x, float* y, const int npoints, float scale)
        {
            const int ptidx = blockIdx.x * blockDim.x + threadIdx.x;

            if (ptidx < npoints)
            {
                short2 loc = loc_[ptidx];

                x[ptidx] = loc.x * scale;
                y[ptidx] = loc.y * scale;
            }
        }

        void mergeLocation_gpu(const short2* loc, float* x, float* y, int npoints, float scale, cudaStream_t stream)
        {
            dim3 block(256);

            dim3 grid;
            grid.x = divUp(npoints, block.x);

            mergeLocation<<<grid, block, 0, stream>>>(loc, x, y, npoints, scale);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    }
}}}

#endif /* CUDA_DISABLER */
