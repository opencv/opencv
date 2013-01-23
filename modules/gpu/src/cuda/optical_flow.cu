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

#include "opencv2/gpu/device/common.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace optical_flow
    {
        #define NEEDLE_MAP_SCALE 16
        #define NUM_VERTS_PER_ARROW 6

        __global__ void NeedleMapAverageKernel(const PtrStepSzf u, const PtrStepf v, PtrStepf u_avg, PtrStepf v_avg)
        {
            __shared__ float smem[2 * NEEDLE_MAP_SCALE];

            volatile float* u_col_sum = smem;
            volatile float* v_col_sum = u_col_sum + NEEDLE_MAP_SCALE;

            const int x = blockIdx.x * NEEDLE_MAP_SCALE + threadIdx.x;
            const int y = blockIdx.y * NEEDLE_MAP_SCALE;

            u_col_sum[threadIdx.x] = 0;
            v_col_sum[threadIdx.x] = 0;

            #pragma unroll
            for(int i = 0; i < NEEDLE_MAP_SCALE; ++i)
            {
                u_col_sum[threadIdx.x] += u(::min(y + i, u.rows - 1), x);
                v_col_sum[threadIdx.x] += v(::min(y + i, u.rows - 1), x);
            }

            if (threadIdx.x < 8)
            {
                // now add the column sums
                const uint X = threadIdx.x;

                if (X | 0xfe == 0xfe)  // bit 0 is 0
                {
                    u_col_sum[threadIdx.x] += u_col_sum[threadIdx.x + 1];
                    v_col_sum[threadIdx.x] += v_col_sum[threadIdx.x + 1];
                }

                if (X | 0xfe == 0xfc) // bits 0 & 1 == 0
                {
                    u_col_sum[threadIdx.x] += u_col_sum[threadIdx.x + 2];
                    v_col_sum[threadIdx.x] += v_col_sum[threadIdx.x + 2];
                }

                if (X | 0xf8 == 0xf8)
                {
                    u_col_sum[threadIdx.x] += u_col_sum[threadIdx.x + 4];
                    v_col_sum[threadIdx.x] += v_col_sum[threadIdx.x + 4];
                }

                if (X == 0)
                {
                    u_col_sum[threadIdx.x] += u_col_sum[threadIdx.x + 8];
                    v_col_sum[threadIdx.x] += v_col_sum[threadIdx.x + 8];
                }
            }

            if (threadIdx.x == 0)
            {
                const float coeff = 1.0f / (NEEDLE_MAP_SCALE * NEEDLE_MAP_SCALE);

                u_col_sum[0] *= coeff;
                v_col_sum[0] *= coeff;

                u_avg(blockIdx.y, blockIdx.x) = u_col_sum[0];
                v_avg(blockIdx.y, blockIdx.x) = v_col_sum[0];
            }
        }

        void NeedleMapAverage_gpu(PtrStepSzf u, PtrStepSzf v, PtrStepSzf u_avg, PtrStepSzf v_avg)
        {
            const dim3 block(NEEDLE_MAP_SCALE);
            const dim3 grid(u_avg.cols, u_avg.rows);

            NeedleMapAverageKernel<<<grid, block>>>(u, v, u_avg, v_avg);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void NeedleMapVertexKernel(const PtrStepSzf u_avg, const PtrStepf v_avg, float* vertex_data, float* color_data, float max_flow, float xscale, float yscale)
        {
            // test - just draw a triangle at each pixel
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            const float arrow_x = x * NEEDLE_MAP_SCALE + NEEDLE_MAP_SCALE / 2.0f;
            const float arrow_y = y * NEEDLE_MAP_SCALE + NEEDLE_MAP_SCALE / 2.0f;

            float3 v[NUM_VERTS_PER_ARROW];

            if (x < u_avg.cols && y < u_avg.rows)
            {
                const float u_avg_val = u_avg(y, x);
                const float v_avg_val = v_avg(y, x);

                const float theta = ::atan2f(v_avg_val, u_avg_val);// + CV_PI;

                float r = ::sqrtf(v_avg_val * v_avg_val + u_avg_val * u_avg_val);
                r = fmin(14.0f * (r / max_flow), 14.0f);

                v[0].z = 1.0f;
                v[1].z = 0.7f;
                v[2].z = 0.7f;
                v[3].z = 0.7f;
                v[4].z = 0.7f;
                v[5].z = 1.0f;

                v[0].x = arrow_x;
                v[0].y = arrow_y;
                v[5].x = arrow_x;
                v[5].y = arrow_y;

                v[2].x = arrow_x + r * ::cosf(theta);
                v[2].y = arrow_y + r * ::sinf(theta);
                v[3].x = v[2].x;
                v[3].y = v[2].y;

                r = ::fmin(r, 2.5f);

                v[1].x = arrow_x + r * ::cosf(theta - CV_PI_F / 2.0f);
                v[1].y = arrow_y + r * ::sinf(theta - CV_PI_F / 2.0f);

                v[4].x = arrow_x + r * ::cosf(theta + CV_PI_F / 2.0f);
                v[4].y = arrow_y + r * ::sinf(theta + CV_PI_F / 2.0f);

                int indx = (y * u_avg.cols + x) * NUM_VERTS_PER_ARROW * 3;

                color_data[indx] = (theta - CV_PI_F) / CV_PI_F * 180.0f;
                vertex_data[indx++] = v[0].x * xscale;
                vertex_data[indx++] = v[0].y * yscale;
                vertex_data[indx++] = v[0].z;

                color_data[indx] = (theta - CV_PI_F) / CV_PI_F * 180.0f;
                vertex_data[indx++] = v[1].x * xscale;
                vertex_data[indx++] = v[1].y * yscale;
                vertex_data[indx++] = v[1].z;

                color_data[indx] = (theta - CV_PI_F) / CV_PI_F * 180.0f;
                vertex_data[indx++] = v[2].x * xscale;
                vertex_data[indx++] = v[2].y * yscale;
                vertex_data[indx++] = v[2].z;

                color_data[indx] = (theta - CV_PI_F) / CV_PI_F * 180.0f;
                vertex_data[indx++] = v[3].x * xscale;
                vertex_data[indx++] = v[3].y * yscale;
                vertex_data[indx++] = v[3].z;

                color_data[indx] = (theta - CV_PI_F) / CV_PI_F * 180.0f;
                vertex_data[indx++] = v[4].x * xscale;
                vertex_data[indx++] = v[4].y * yscale;
                vertex_data[indx++] = v[4].z;

                color_data[indx] = (theta - CV_PI_F) / CV_PI_F * 180.0f;
                vertex_data[indx++] = v[5].x * xscale;
                vertex_data[indx++] = v[5].y * yscale;
                vertex_data[indx++] = v[5].z;
            }
        }

        void CreateOpticalFlowNeedleMap_gpu(PtrStepSzf u_avg, PtrStepSzf v_avg, float* vertex_buffer, float* color_data, float max_flow, float xscale, float yscale)
        {
            const dim3 block(16);
            const dim3 grid(divUp(u_avg.cols, block.x), divUp(u_avg.rows, block.y));

            NeedleMapVertexKernel<<<grid, block>>>(u_avg, v_avg, vertex_buffer, color_data, max_flow, xscale, yscale);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    }
}}}

#endif /* CUDA_DISABLER */
