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
#include "opencv2/gpu/device/vec_math.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace hough
    {
        __device__ int g_counter;

        texture<uchar, cudaTextureType2D, cudaReadModeElementType> tex_mask(false, cudaFilterModePoint, cudaAddressModeClamp);

        __global__ void houghLinesProbabilistic(const PtrStepSzi accum,
                                                int4* out, const int maxSize,
                                                const float rho, const float theta,
                                                const int lineGap, const int lineLength,
                                                const int rows, const int cols)
        {
            const int r = blockIdx.x * blockDim.x + threadIdx.x;
            const int n = blockIdx.y * blockDim.y + threadIdx.y;

            if (r >= accum.cols - 2 || n >= accum.rows - 2)
                return;

            const int curVotes = accum(n + 1, r + 1);

            if (curVotes >= lineLength &&
                curVotes > accum(n, r) &&
                curVotes > accum(n, r + 1) &&
                curVotes > accum(n, r + 2) &&
                curVotes > accum(n + 1, r) &&
                curVotes > accum(n + 1, r + 2) &&
                curVotes > accum(n + 2, r) &&
                curVotes > accum(n + 2, r + 1) &&
                curVotes > accum(n + 2, r + 2))
            {
                const float radius = (r - (accum.cols - 2 - 1) * 0.5f) * rho;
                const float angle = n * theta;

                float cosa;
                float sina;
                sincosf(angle, &sina, &cosa);

                float2 p0 = make_float2(cosa * radius, sina * radius);
                float2 dir = make_float2(-sina, cosa);

                float2 pb[4] = {make_float2(-1, -1), make_float2(-1, -1), make_float2(-1, -1), make_float2(-1, -1)};
                float a;

                if (dir.x != 0)
                {
                    a = -p0.x / dir.x;
                    pb[0].x = 0;
                    pb[0].y = p0.y + a * dir.y;

                    a = (cols - 1 - p0.x) / dir.x;
                    pb[1].x = cols - 1;
                    pb[1].y = p0.y + a * dir.y;
                }
                if (dir.y != 0)
                {
                    a = -p0.y / dir.y;
                    pb[2].x = p0.x + a * dir.x;
                    pb[2].y = 0;

                    a = (rows - 1 - p0.y) / dir.y;
                    pb[3].x = p0.x + a * dir.x;
                    pb[3].y = rows - 1;
                }

                if (pb[0].x == 0 && (pb[0].y >= 0 && pb[0].y < rows))
                {
                    p0 = pb[0];
                    if (dir.x < 0)
                        dir = -dir;
                }
                else if (pb[1].x == cols - 1 && (pb[0].y >= 0 && pb[0].y < rows))
                {
                    p0 = pb[1];
                    if (dir.x > 0)
                        dir = -dir;
                }
                else if (pb[2].y == 0 && (pb[2].x >= 0 && pb[2].x < cols))
                {
                    p0 = pb[2];
                    if (dir.y < 0)
                        dir = -dir;
                }
                else if (pb[3].y == rows - 1 && (pb[3].x >= 0 && pb[3].x < cols))
                {
                    p0 = pb[3];
                    if (dir.y > 0)
                        dir = -dir;
                }

                float2 d;
                if (::fabsf(dir.x) > ::fabsf(dir.y))
                {
                    d.x = dir.x > 0 ? 1 : -1;
                    d.y = dir.y / ::fabsf(dir.x);
                }
                else
                {
                    d.x = dir.x / ::fabsf(dir.y);
                    d.y = dir.y > 0 ? 1 : -1;
                }

                float2 line_end[2];
                int gap;
                bool inLine = false;

                float2 p1 = p0;
                if (p1.x < 0 || p1.x >= cols || p1.y < 0 || p1.y >= rows)
                    return;

                for (;;)
                {
                    if (tex2D(tex_mask, p1.x, p1.y))
                    {
                        gap = 0;

                        if (!inLine)
                        {
                            line_end[0] = p1;
                            line_end[1] = p1;
                            inLine = true;
                        }
                        else
                        {
                            line_end[1] = p1;
                        }
                    }
                    else if (inLine)
                    {
                        if (++gap > lineGap)
                        {
                            bool good_line = ::abs(line_end[1].x - line_end[0].x) >= lineLength ||
                                             ::abs(line_end[1].y - line_end[0].y) >= lineLength;

                            if (good_line)
                            {
                                const int ind = ::atomicAdd(&g_counter, 1);
                                if (ind < maxSize)
                                    out[ind] = make_int4(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
                            }

                            gap = 0;
                            inLine = false;
                        }
                    }

                    p1 = p1 + d;
                    if (p1.x < 0 || p1.x >= cols || p1.y < 0 || p1.y >= rows)
                    {
                        if (inLine)
                        {
                            bool good_line = ::abs(line_end[1].x - line_end[0].x) >= lineLength ||
                                             ::abs(line_end[1].y - line_end[0].y) >= lineLength;

                            if (good_line)
                            {
                                const int ind = ::atomicAdd(&g_counter, 1);
                                if (ind < maxSize)
                                    out[ind] = make_int4(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
                            }

                        }
                        break;
                    }
                }
            }
        }

        int houghLinesProbabilistic_gpu(PtrStepSzb mask, PtrStepSzi accum, int4* out, int maxSize, float rho, float theta, int lineGap, int lineLength)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(accum.cols - 2, block.x), divUp(accum.rows - 2, block.y));

            bindTexture(&tex_mask, mask);

            houghLinesProbabilistic<<<grid, block>>>(accum,
                                                     out, maxSize,
                                                     rho, theta,
                                                     lineGap, lineLength,
                                                     mask.rows, mask.cols);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            totalCount = ::min(totalCount, maxSize);

            return totalCount;
        }
    }
}}}


#endif /* CUDA_DISABLER */
